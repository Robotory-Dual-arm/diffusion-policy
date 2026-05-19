from typing import Dict
import time
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchvision
import torchvision.transforms as T
from omegaconf import DictConfig

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from diffusion_policy.model.force.force_encoder import CausalConvForceEncoder, GRUForceEncoder


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)

class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,   # 예측할 step수
            n_action_steps,   # 실제로 실행할 step수
            n_obs_steps,    # 관찰할 step수 
            # vision_encoder_cfg: DictConfig,
            # force_encoder_cfg: DictConfig,
            obs_encoder: DictConfig,
            num_inference_steps=None,   # denoising step수
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),   # U-Net 모델의 dimension
            kernel_size=5,   
            n_groups=8,
            cond_predict_scale=True,
            pose_repr: dict={},
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta (cfg에 저장되어있음); shape_meta : action과 obs의 종류, shape, type 정보
        action_shape = shape_meta['action']['shape']   
        assert len(action_shape) == 1   # (shape,) 
        action_dim = action_shape[0]   # shape (scalar)
        obs_shape_meta = shape_meta['obs']
        # obs의 종류에 이중 뭐가 있는지 찾기
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'wrench': []   
            }

        self.obs_pose_repr = pose_repr.get('obs_pose_repr', 'abs')
        self.action_pose_repr = pose_repr.get('action_pose_repr', 'abs')

        
        self.num_image = 0 # image수
        self.num_wrench = 0 # wrench수 (wrist, fingers)
        self.num_wrench_component = 0 # wrench의 component 수 
        self.num_low_dim = 0
        self.num_low_dim_component = 0 
        image_resolution = None
        wrench_shape = None

        rgb_keys = []
        low_dim_keys = []
        wrench_keys = []

        key_shape_map = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            type = attr.get('type', 'low_dim')

            # obs relative시에 quat -> rot6d
            if self.obs_pose_repr == 'relative' and 'quat' in key:
                shape = (6,)

            key_shape_map[key] = shape
            
            if type == 'rgb':
                obs_config['rgb'].append(key)
                rgb_keys.append(key)   
                
                self.num_image += 1
                image_resolution = shape[1:]  # (H, W)
                
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
                low_dim_keys.append(key)

                self.num_low_dim += 1
                self.num_low_dim_component += shape[0]

            elif type == 'wrench':
                obs_config['wrench'].append(key)
                wrench_keys.append(key)

                self.num_wrench += 1
                self.num_wrench_component += shape[0]
                wrench_shape = shape  # (wrench, T)

            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        print(f"========= Initialized Observation ==========")
        print(f"Pose representation: obs {self.obs_pose_repr}, action {self.action_pose_repr}")
        print(f"Observation config: {obs_config}")

        vc = obs_encoder.vision_encoder_cfg
        fc = obs_encoder.force_encoder_cfg

        vision_encoder = timm.create_model(
            model_name=vc.model_name,
            pretrained=vc.pretrained,
            global_pool=vc.global_pool,  
            num_classes=0  
        )
        if vc.frozen:
            assert vc.pretrained, "Frozen vision encoder must be pretrained"
            for param in vision_encoder.parameters():
                param.requires_grad = False


        ### Image Preprocess
        randomcrop = T.RandomCrop(size=int(image_resolution[0] * vc.transforms.randomcrop.ratio))
        centercrop = T.CenterCrop(size=int(image_resolution[0] * vc.transforms.randomcrop.ratio))
        resize = T.Resize(size=image_resolution[0], antialias=True)
        colorjitter = T.ColorJitter(brightness=vc.transforms.colorjitter.brightness,
                                                         contrast=vc.transforms.colorjitter.contrast,
                                                         saturation=vc.transforms.colorjitter.saturation,
                                                         hue=vc.transforms.colorjitter.hue,)
        grayscale = T.RandomGrayscale(p=vc.transforms.colorjitter.grayscale)
        
        self.transform_train = T.Compose([randomcrop, resize, colorjitter, grayscale])
        self.transform_eval = T.Compose([centercrop, resize])
        
        
        ### Vision Encoder
        # resnet
        vision_feature_dim = 0
        if vc.model_name.startswith('resnet'):
            if vc.downsample_ratio == 32: # 512 x 7 x 7
                modules = list(vision_encoder.children())[:-2]
                vision_encoder = nn.Sequential(*modules)
                vision_feature_dim = 512

            elif vc.downsample_ratio == 16: # 256 x 14 x 14
                modules = list(vision_encoder.children())[:-3]
                vision_encoder = nn.Sequential(*modules)
                vision_feature_dim = 256

            else:
                raise ValueError(f"Unsupported downsample ratio {vc.downsample_ratio} for ResNet")
            # resnet from scratch  BN -> GN
            if vc.use_group_norm and not vc.pretrained:
                vision_encoder = replace_submodules(
                    root_module=vision_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=((x.num_features // 16) 
                                    if (x.num_features % 16 == 0) 
                                    else (x.num_features // 8)
                                    ),
                        num_channels=x.num_features
                    )
                )
            
            if vc.feature_aggregation == 'attention_pool_2d':
                feature_map_shape = [ x // vc.downsample_ratio for x in image_resolution]
                self.attention_pool_2d = AttentionPool2d(
                    spacial_dim=feature_map_shape[0],
                    embed_dim=vision_feature_dim,
                    num_heads=vision_feature_dim // 64,
                    output_dim=vision_feature_dim
                )


        # ViT
        elif vc.model_name.startswith('vit'):
            vision_feature_dim = 768


        ### Force Encoder
        force_encoder = None
        force_feature_dim = 0
        force_obs_steps = 0
        if self.num_wrench > 0:
            if fc.model_name == 'causalconv':
                force_encoder = CausalConvForceEncoder(
                    input_dim=self.num_wrench_component,
                    feature_dim=fc.feature_dim
                )
            elif fc.model_name == 'gru':
                force_encoder = GRUForceEncoder(
                    input_dim=self.num_wrench_component,
                    feature_dim=fc.feature_dim
                )
            else:
                raise ValueError(f"Unsupported force encoder: {fc.model_name}")
            force_feature_dim = fc.feature_dim
            # Dataset returns the latest wrench history window only.
            force_obs_steps = 1

        # fuse mode
        self.low_dim_encoder = None
        if obs_encoder.fuse_mode == 'modality-attention':
            self.transformer_encoder = torch.nn.TransformerEncoderLayer(
                d_model=vision_feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            # token design:
            # - vision: one token per image per step -> len(rgb_keys) * n_obs_steps
            # - wrench: one encoded history token
            # low_dim stays outside modality attention, matching ACP.
            n_features = len(rgb_keys) * n_obs_steps + force_obs_steps
            self.linear_projection = nn.Linear(vision_feature_dim*n_features, vision_feature_dim)
            
            if obs_encoder.position_encoding == 'learnable':
                self.position_embedding = torch.nn.parameter.Parameter(
                    torch.randn(n_features, vision_feature_dim))
                
        
        # Diffusion Model 시작 ==================================================
        # create diffusion model
        if obs_encoder.fuse_mode == 'modality-attention':
            # Project image/wrench tokens, then concatenate raw low_dim history.
            obs_feature_dim = vision_feature_dim + self.num_low_dim_component * n_obs_steps
        elif obs_encoder.fuse_mode == 'concat':
            # Keep normalized low_dim state direct, matching the original hybrid policy more closely.
            obs_feature_dim = vision_feature_dim * self.num_image * n_obs_steps \
                                + self.num_low_dim_component * n_obs_steps \
                                + force_feature_dim * force_obs_steps
        else:
            raise ValueError(f"Unsupported fuse mode: {obs_encoder.fuse_mode}")

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:   # True
            input_dim = action_dim
            global_cond_dim = obs_feature_dim 

        # U-Net 모델 생성
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_config = obs_config
        self.vision_model_name = vc.model_name
        self.vision_encoder = vision_encoder
        self.force_encoder = force_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.vision_feature_dim = vision_feature_dim
        self.force_feature_dim = force_feature_dim
        self.force_obs_steps = force_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.wrench_keys = wrench_keys
        self.key_shape_map = key_shape_map
        self.fuse_mode = obs_encoder.fuse_mode
        self.position_encoding = obs_encoder.position_encoding

   
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision Encoder params: %e" % sum(p.numel() for p in self.vision_encoder.parameters()))
        if self.force_encoder is not None:
            print("Force Encoder params: %e" % sum(p.numel() for p in self.force_encoder.parameters()))
        else:
            print("Force Encoder params: 0.000000e+00")
   

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model   # Unet
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning; obs_as_global_cond=False일때 condition data 설정
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output; noise 예측(traj랑 같은 dim)
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def _apply_image_transform(self, obs_dict, transform):
        transformed_obs = dict(obs_dict)
        for key in self.rgb_keys:
            obs = obs_dict[key]
            img = obs.reshape(-1, *obs.shape[2:])
            img = transform(img)
            transformed_obs[key] = img.reshape(*obs.shape[:2], *img.shape[1:])
        return transformed_obs

    def _encode_wrench(self, wrench_nobs, batch_size):
        if len(self.wrench_keys) == 0:
            return [], []
        assert self.force_encoder is not None

        wrench_total = torch.cat(
            [wrench_nobs[key] for key in self.wrench_keys],
            dim=-2
        )
        assert wrench_total.ndim == 4, f"Expected wrench shape (B, T, C, H), got {wrench_total.shape}"

        wrench_obs_steps = wrench_total.shape[1]
        force_feature = self.force_encoder(
            wrench_total.reshape(-1, *wrench_total.shape[-2:])
        )
        force_feature = force_feature.reshape(batch_size, wrench_obs_steps, -1)

        return [force_feature.reshape(batch_size, -1)], [force_feature]


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # obs_dict
        # ['image0'] = [[image0_t-1], [image0_t]]
        # ['robot_pose_L'] = [[pose_L_t-1], [pose_L_t]]
        # ['wrench_wrist_R'] = [[wrench_wrist_hist_32]]

        # image crop, resize
        obs_dict = self._apply_image_transform(obs_dict, self.transform_eval)


        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # print('nobs.size:', {k: v.size() for k, v in nobs.items()})
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon    # 예측할 step수
        Da = self.action_dim   # action의 dimension
        Do = self.obs_feature_dim   # 관찰할 것의 dimension
        To = self.n_obs_steps   # 관찰한 step수


        # build input
        device = self.device
        dtype = self.dtype

        # Separate and save wrench data
        wrench_nobs = {}
        for key in self.wrench_keys:
            wrench_nobs[key] = nobs.pop(key)  # Save and remove from nobs
      
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # (B, To, ...) -> (B*To, ...)

        modality_features = list()

        # Image encoding
        vision_features = []
        for key in self.rgb_keys:
            img = this_nobs[key]
            assert img.shape[1:] == self.key_shape_map[key]
            raw_vision_feature = self.vision_encoder(img) # (B*To, vision_feature_dim, n, n)
            
            # resnet
            if self.vision_model_name.startswith('resnet'):
                vision_feature = self.attention_pool_2d(raw_vision_feature) # (B*To, vision_feature_dim)
            # ViT
            else:
                vision_feature = raw_vision_feature[:, 0, :]   # CLS token
            vision_features.append(vision_feature.reshape(B, -1)) # (B, To*vision_feature_dim)
            modality_features.append(vision_feature.reshape(B, To, -1)) # (B, To, vision_feature_dim)

        # low-dim encoding
        if len(self.low_dim_keys) > 0:
            low_dim_features = []
            for t in range(To):
                low_dim_t = torch.cat([nobs[key][:,t,:] for key in self.low_dim_keys], dim=-1)
                low_dim_features.append(low_dim_t.reshape(B, -1))
            low_dim_features = torch.stack(low_dim_features, dim=1)  # (B, To, low_dim_dim)
        else:
            low_dim_features = torch.empty(B, To, 0, device=device, dtype=dtype)

        # Force encoding
        force_features, force_modality_features = self._encode_wrench(wrench_nobs, B)
        modality_features.extend(force_modality_features)

        
        # fuse mode
        if self.fuse_mode == 'modality-attention':
            in_embeds = torch.cat(modality_features, dim=1)
            if self.position_encoding == 'learnable':
                pos_emb = self.position_embedding.to(
                    device=in_embeds.device,
                    dtype=in_embeds.dtype,
                )
                in_embeds = in_embeds + pos_emb.unsqueeze(0)
            out_embeds = self.transformer_encoder(in_embeds)
            projected_embeds = self.linear_projection(out_embeds.flatten(start_dim=1))
            nobs_features = torch.cat([projected_embeds, low_dim_features.reshape(B, -1)], dim=-1)
        elif self.fuse_mode == 'concat':
            nobs_features = torch.cat(vision_features + [low_dim_features.reshape(B, -1)] + force_features, dim=-1)
        else:
            raise ValueError(f"Unsupported fuse mode: {self.fuse_mode}")
        
        assert nobs_features.shape[-1] == Do, f"Expected obs feature dim {Do}, got {nobs_features.shape[-1]}"


        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:   # True
            ### condition through global feature
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else: # False
            ### condition through impainting
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch

        # image crop, resize, colorjitter
        obs_dict = self._apply_image_transform(batch['obs'], self.transform_train)

        nobs = self.normalizer.normalize(obs_dict)
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        T = nactions.shape[1]
        Do = self.obs_feature_dim   # 관찰할 것의 dimension
        To = self.n_obs_steps   # 관찰한 step수


        # Separate and save wrench data
        wrench_nobs = {}
        for key in self.wrench_keys:
            wrench_nobs[key] = nobs.pop(key)  # Save and remove from nobs
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # (B, To, ...) -> (B*To, ...)

        modality_features = list()

        # Image encoding
        vision_features = []
        for key in self.rgb_keys:
            img = this_nobs[key]
            assert img.shape[1:] == self.key_shape_map[key]
            raw_vision_feature = self.vision_encoder(img) # (B*To, vision_feature_dim, n, n)
            
            # resnet
            if self.vision_model_name.startswith('resnet'):
                vision_feature = self.attention_pool_2d(raw_vision_feature) # (B*To, vision_feature_dim)
            # ViT
            else:
                vision_feature = raw_vision_feature[:, 0, :]   # CLS token
            vision_features.append(vision_feature.reshape(B, -1)) # (B, To*vision_feature_dim)
            modality_features.append(vision_feature.reshape(B, To, -1)) # (B, To, vision_feature_dim)

        # low-dim encoding
        if len(self.low_dim_keys) > 0:
            low_dim_features = []
            for t in range(To):
                low_dim_t = torch.cat([nobs[key][:,t,:] for key in self.low_dim_keys], dim=-1)
                low_dim_features.append(low_dim_t.reshape(B, -1))
            low_dim_features = torch.stack(low_dim_features, dim=1)  # (B, To, low_dim_dim)
        else:
            low_dim_features = torch.empty(B, To, 0, device=nactions.device, dtype=nactions.dtype)
        

        # Force encoding
        force_features, force_modality_features = self._encode_wrench(wrench_nobs, B)
        modality_features.extend(force_modality_features)

        
        # fuse mode
        if self.fuse_mode == 'modality-attention':
            in_embeds = torch.cat(modality_features, dim=1) # (B, feature_num, feature_dim)
            if self.position_encoding == 'learnable':
                pos_emb = self.position_embedding.to(
                    device=in_embeds.device,
                    dtype=in_embeds.dtype,
                )
                in_embeds = in_embeds + pos_emb.unsqueeze(0)
            out_embeds = self.transformer_encoder(in_embeds) # (B, feature_num, feature_dim)
            projected_embeds = self.linear_projection(out_embeds.flatten(start_dim=1)) # (B, obs_feature_dim)
            nobs_features = torch.cat([projected_embeds, low_dim_features.reshape(B, -1)], dim=-1)

        elif self.fuse_mode == 'concat':
            nobs_features = torch.cat(vision_features + [low_dim_features.reshape(B, -1)] + force_features, dim=-1)
        else:
            raise ValueError(f"Unsupported fuse mode: {self.fuse_mode}")
        
        assert nobs_features.shape[-1] == Do, f"Expected obs feature dim {Do}, got {nobs_features.shape[-1]}"


        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:   # True
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, T, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            velocity = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
            target = velocity   # |v - v_pred|^2  ->  (SNR + 1) weighting
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
