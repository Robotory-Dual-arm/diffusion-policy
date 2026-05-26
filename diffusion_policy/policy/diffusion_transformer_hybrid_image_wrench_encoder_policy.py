from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchvision.transforms as T
from omegaconf import DictConfig
import timm

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.bae_transformer_for_diffusion_force import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from diffusion_policy.model.force.force_encoder import CausalConvForceEncoder, GRUForceEncoder
from diffusion_policy.model.vision.attention_pool_2d import AttentionPool2d


class PretrainedImageNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x):
        x = (x + 1.0) / 2.0
        mean = self.mean.to(dtype=x.dtype)
        std = self.std.to(dtype=x.dtype)
        return (x - mean) / std


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_encoder: DictConfig,
            num_inference_steps=None,
            # image
            # crop_shape=(76, 76),
            # obs_encoder_group_norm=False,
            # eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            bidirectional_prefix_steps=None,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,   # 실행할 action만 예측
            pose_repr: dict={},
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]   
        obs_shape_meta = shape_meta['obs']
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
                image_resolution = shape[1:] # (H, W)

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
                wrench_shape = shape # (wrench, T)
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
        pretrained_model_norm = bool(vc.get('pretrained_model_norm', False))
        self.pretrained_image_norm = nn.Identity()
        if pretrained_model_norm:
            data_cfg = timm.data.resolve_model_data_config(vision_encoder)
            self.pretrained_image_norm = PretrainedImageNormalize(
                mean=data_cfg['mean'],
                std=data_cfg['std']
            )
            print(
                "Using pretrained image norm: "
                f"mean={data_cfg['mean']}, std={data_cfg['std']}"
            )
        if vc.frozen:
            assert vc.pretrained, "Frozen vision encoder must be pretrained"
            for param in vision_encoder.parameters():
                param.requires_grad = False


        ### Image Preprocess
        if len(rgb_keys) > 0:
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
            if len(rgb_keys) > 0:
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
        if len(wrench_keys) > 0:
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
                raise ValueError(f"Unsupported force encoder {fc.model_name}")
            force_feature_dim = fc.feature_dim

        ### Low-dim encoder
        self.low_dim_encoder = nn.Linear(self.num_low_dim_component, vision_feature_dim)


        # fuse mode
        if obs_encoder.fuse_mode == 'modality-attention':
            self.transformer_encoder = torch.nn.TransformerEncoderLayer(
                d_model=vision_feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            n_force_tokens = 1 if len(wrench_keys) > 0 else 0
            n_features = (len(rgb_keys) + 1) * n_obs_steps + n_force_tokens # vision + low_dim (+ force)
            n_attention_features = len(rgb_keys) * n_obs_steps + n_force_tokens # vision (+ force), low_dim bypasses attention
            self.linear_projection = nn.Linear(vision_feature_dim*n_features, vision_feature_dim)
            
            if obs_encoder.position_encoding == 'learnable':
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(n_attention_features, vision_feature_dim))
            self.modality_attention_token_count = n_attention_features


        # Diffusion Model 시작 ========================================
        # create diffusion model
        if obs_encoder.fuse_mode == 'modality-attention':
            obs_feature_dim = vision_feature_dim 
        # elif obs_encoder.fuse_mode == 'concat':
        #     obs_feature_dim = vision_feature_dim * self.num_image * n_obs_steps \
        #                         + self.num_low_dim_component * n_obs_steps + force_feature_dim

        if obs_as_cond: # True
            input_dim = action_dim
            cond_dim = obs_feature_dim
        else:
            input_dim = obs_feature_dim + action_dim
            cond_dim = 0
        output_dim = input_dim


        # Transformer 
        model = TransformerForDiffusion(
            input_dim=input_dim, # action dim
            output_dim=output_dim, # action dim
            horizon=horizon, # action horizon
            n_obs_steps=n_obs_steps, # To
            # cond_dim=cond_dim, # obs feature dim
            n_layer=n_layer, 
            n_head=n_head,
            n_emb=n_emb, # token dim
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            bidirectional_prefix_steps=bidirectional_prefix_steps,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
            n_cond_tokens=n_features,
        )

        self.obs_encoder = obs_encoder
        self.vision_model_name = vc.model_name
        self.vision_encoder = vision_encoder
        self.force_encoder = force_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator( # for inpainting
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.vision_feature_dim = vision_feature_dim
        self.force_feature_dim = force_feature_dim
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.wrench_keys = wrench_keys
        self.key_shape_map = key_shape_map
        self.fuse_mode = obs_encoder.fuse_mode
        self.position_encoding = obs_encoder.position_encoding
        self.feature_aggregation = vc.feature_aggregation


        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision Encoder params: %e" % sum(p.numel() for p in self.vision_encoder.parameters()))
        if self.force_encoder is not None:
            print("Force Encoder params: %e" % sum(p.numel() for p in self.force_encoder.parameters()))
        else:
            print("Force Encoder disabled: no wrench obs keys")

    
    # ========= inference  ============

    def _apply_image_transform(self, obs_dict, transform):
        transformed_obs = dict(obs_dict)
        for key in self.rgb_keys:
            obs = obs_dict[key]
            img = obs.reshape(-1, *obs.shape[2:])
            img = transform(img)
            transformed_obs[key] = img.reshape(*obs.shape[:2], *img.shape[1:])
        return transformed_obs

    def _apply_pretrained_image_norm(self, obs_dict):
        if isinstance(self.pretrained_image_norm, nn.Identity):
            return obs_dict

        normalized_obs = dict(obs_dict)
        for key in self.rgb_keys:
            obs = obs_dict[key]
            img = obs.reshape(-1, *obs.shape[2:])
            img = self.pretrained_image_norm(img)
            normalized_obs[key] = img.reshape(*obs.shape[:2], *img.shape[1:])
        return normalized_obs

    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model   # Transformer model
        scheduler = self.noise_scheduler

        # 랜덤 trajectory 생성
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values; scheduler.timestep 생성
        scheduler.set_timesteps(self.num_inference_steps)

        # for t in scheduler.timesteps:
        for denoise_step, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask] # inpainting일때, condition 갈아끼우기

            # 2. predict model output
            model_output = model(trajectory, t, cond)   # Transformer

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory   # 최종 trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:   
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        
        # image crop, resize
        obs_dict = self._apply_image_transform(obs_dict, self.transform_eval)
        

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = self._apply_pretrained_image_norm(nobs)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # Separate and save wrench data
        wrench_nobs = {}
        for key in self.wrench_keys:
            wrench_nobs[key] = nobs.pop(key)  # Save and remove from nobs
      
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # (B, To, ...) -> (B*To, ...)

        attention_features = list()

        # Image encoding
        vision_features = []
        for key in self.rgb_keys:
            img = this_nobs[key]
            assert img.shape[1:] == self.key_shape_map[key]
            raw_vision_feature = self.vision_encoder(img) # (B*To, vision_feature_dim, n, n)
            # resnet
            if self.vision_model_name.startswith('resnet'):
                if self.feature_aggregation == 'attention_pool_2d':
                    vision_feature = self.attention_pool_2d(raw_vision_feature) # (B*To, vision_feature_dim)
                # elif self.feature_aggregation == 'adaptive_avg_pool_2d':
                #     AdaptiveAvgPool2d((k, k))  ->  spatial 정보 조금 남길수있음
            # ViT
            else:
                vision_feature = raw_vision_feature[:, 0, :]   # CLS token
            vision_features.append(vision_feature.reshape(B, -1)) # (B, To*vision_feature_dim)
            attention_features.append(vision_feature.reshape(B, To, -1)) # (B, To, vision_feature_dim)

        # low-dim encoding (linear)
        low_dim_features = []
        for t in range(To):
            low_dim_t = torch.cat([nobs[key][:,t,:] for key in self.low_dim_keys], dim=-1)
            low_dim_feature_t = self.low_dim_encoder(low_dim_t)
            low_dim_features.append(low_dim_feature_t.reshape(B, -1))
        low_dim_features = torch.stack(low_dim_features, dim=1)  # (B, To, low_dim_feature_dim)
        # low_dim is kept as condition tokens, but excluded from modality-attention.

        # Force encoding
        if self.force_encoder is not None:
            force_features = []
            combined_wrench_data = []
            for key in self.wrench_keys:
                combined_wrench_data.append(wrench_nobs[key]) # (B, To=1, wrench_axis, wrench_hist)
            wrench_total = torch.cat(combined_wrench_data, dim=-2) # (B, To=1, num_wrench_component, wrench_hist)
            force_feature = self.force_encoder(wrench_total.reshape(-1, *wrench_total.shape[-2:])) # (B, 1, feature_dim)
            force_features.append(force_feature.reshape(B, -1)) # (B, feature_dim)
            attention_features.append(force_feature) # (B, 1, feature_dim)
        # attention_features: image tokens + force token. low_dim bypasses this attention.


        # fuse mode
        if self.fuse_mode == 'modality-attention':
            if len(attention_features) > 0:
                in_embeds = torch.cat(attention_features, dim=1) # (B, image_tokens + force_tokens, feature_dim)
                if self.position_encoding == 'learnable':
                    if self.position_embedding.device != in_embeds.device:
                        self.position_embedding = self.position_embedding.to(in_embeds.device)
                    in_embeds = in_embeds + self.position_embedding
                out_embeds = self.transformer_encoder(in_embeds)

                token_sizes = [x.shape[1] for x in attention_features]
                attended_modalities = list(torch.split(out_embeds, token_sizes, dim=1))
                attended_vision = attended_modalities[:len(self.rgb_keys)]
                attended_low_dim = low_dim_features
                attended_force = attended_modalities[len(self.rgb_keys)] if self.force_encoder is not None else None
            else:
                attended_vision = []
                attended_low_dim = low_dim_features
                attended_force = None

            condition_features = attended_vision + [attended_low_dim]
            if attended_force is not None:
                condition_features.append(attended_force)
            token_num = sum(x.shape[1] for x in condition_features)
            nobs_features = torch.cat(condition_features, dim=1)
            assert nobs_features.shape[1] == token_num

        # elif self.fuse_mode == 'concat':
        #     nobs_features = torch.cat(vision_features + low_dim_features + force_features, dim=-1)
        
        assert nobs_features.shape[-1] == Do, f"Expected obs feature dim {Do}, got {nobs_features.shape[-1]}"


        # handle different ways of passing observation
        cond = None # obs
        if self.obs_as_cond:   # cross attention
            cond = nobs_features
            shape = (B, T, Da) # action dim
            if self.pred_action_steps_only:   # False; 이거 이용하면 가변길이 액션 생성할수있겠다.
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # else:   # inpainting, self attention
        #     nobs_features = nobs_features.reshape(B, -1) # 문제 있다.
        #     shape = (B, T, Da+Do)
        #     cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        #     cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        #     cond_data[:,:To,Da:] = nobs_features
        #     cond_mask[:,:To,Da:] = True   # obs와 action이후 dimension은 masking

        # run sampling; Denoising후 trajectory
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]   # action부분만 사용
        action_pred = self.normalizer['action'].unnormalize(naction_pred)   # 정규화 풀기

        # get action
        if self.pred_action_steps_only:   # False
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]   # 실제 실행할 action만 추출
        
        result = {
            'action': action,   # 실제 사용할 traj
            'action_pred': action_pred   # 예측한 전체 traj
        }
        return result


    def predict_action_horizon(
            self,
            obs_dict: Dict[str, torch.Tensor],
            start_action,
            noise_ratio=0.5) -> Dict[str, torch.Tensor]:
        """
        Re-noise start_action, then denoise it from the selected diffusion step.
        """
        assert 'past_action' not in obs_dict # not implemented yet

        # image crop, resize
        obs_dict = self._apply_image_transform(obs_dict, self.transform_eval)

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = self._apply_pretrained_image_norm(nobs)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype
        start_action = start_action.to(device=device, dtype=dtype)
        T = start_action.shape[1]

        # Separate and save wrench data
        wrench_nobs = {}
        for key in self.wrench_keys:
            wrench_nobs[key] = nobs.pop(key)  # Save and remove from nobs

        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # (B, To, ...) -> (B*To, ...)

        attention_features = list()

        # Image encoding
        vision_features = []
        for key in self.rgb_keys:
            img = this_nobs[key]
            assert img.shape[1:] == self.key_shape_map[key]
            raw_vision_feature = self.vision_encoder(img) # (B*To, vision_feature_dim, n, n)
            # resnet
            if self.vision_model_name.startswith('resnet'):
                if self.feature_aggregation == 'attention_pool_2d':
                    vision_feature = self.attention_pool_2d(raw_vision_feature) # (B*To, vision_feature_dim)
                # elif self.feature_aggregation == 'adaptive_avg_pool_2d':
                #     AdaptiveAvgPool2d((k, k))  ->  spatial 정보 조금 남길수있음
            # ViT
            else:
                vision_feature = raw_vision_feature[:, 0, :]   # CLS token
            vision_features.append(vision_feature.reshape(B, -1)) # (B, To*vision_feature_dim)
            attention_features.append(vision_feature.reshape(B, To, -1)) # (B, To, vision_feature_dim)

        # low-dim encoding (linear)
        low_dim_features = []
        for t in range(To):
            low_dim_t = torch.cat([nobs[key][:,t,:] for key in self.low_dim_keys], dim=-1)
            low_dim_feature_t = self.low_dim_encoder(low_dim_t)
            low_dim_features.append(low_dim_feature_t.reshape(B, -1))
        low_dim_features = torch.stack(low_dim_features, dim=1)  # (B, To, low_dim_feature_dim)
        # low_dim is kept as condition tokens, but excluded from modality-attention.

        # Force encoding
        if self.force_encoder is not None:
            force_features = []
            combined_wrench_data = []
            for key in self.wrench_keys:
                combined_wrench_data.append(wrench_nobs[key]) # (B, To=1, wrench_axis, wrench_hist)
            wrench_total = torch.cat(combined_wrench_data, dim=-2) # (B, To=1, num_wrench_component, wrench_hist)
            force_feature = self.force_encoder(wrench_total.reshape(-1, *wrench_total.shape[-2:])) # (B, 1, feature_dim)
            force_features.append(force_feature.reshape(B, -1)) # (B, feature_dim)
            attention_features.append(force_feature) # (B, 1, feature_dim)
        # attention_features: image tokens + force token. low_dim bypasses this attention.


        # fuse mode
        if self.fuse_mode == 'modality-attention':
            if len(attention_features) > 0:
                in_embeds = torch.cat(attention_features, dim=1) # (B, image_tokens + force_tokens, feature_dim)
                if self.position_encoding == 'learnable':
                    if self.position_embedding.device != in_embeds.device:
                        self.position_embedding = self.position_embedding.to(in_embeds.device)
                    in_embeds = in_embeds + self.position_embedding
                out_embeds = self.transformer_encoder(in_embeds)

                token_sizes = [x.shape[1] for x in attention_features]
                attended_modalities = list(torch.split(out_embeds, token_sizes, dim=1))
                attended_vision = attended_modalities[:len(self.rgb_keys)]
                attended_low_dim = low_dim_features
                attended_force = attended_modalities[len(self.rgb_keys)] if self.force_encoder is not None else None
            else:
                attended_vision = []
                attended_low_dim = low_dim_features
                attended_force = None

            condition_features = attended_vision + [attended_low_dim]
            if attended_force is not None:
                condition_features.append(attended_force)
            token_num = sum(x.shape[1] for x in condition_features)
            nobs_features = torch.cat(condition_features, dim=1)
            assert nobs_features.shape[1] == token_num

        # elif self.fuse_mode == 'concat':
        #     nobs_features = torch.cat(vision_features + low_dim_features + force_features, dim=-1)
        
        assert nobs_features.shape[-1] == Do, f"Expected obs feature dim {Do}, got {nobs_features.shape[-1]}"


        # handle different ways of passing observation
        cond = None # obs
        if self.obs_as_cond:   # cross attention
            cond = nobs_features
            shape = (B, T, Da) # action dim
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # else:   # inpainting, self attention
        #     nobs_features = nobs_features.reshape(B, -1) # 문제 있다.
        #     shape = (B, T, Da+Do)
        #     cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        #     cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        #     cond_data[:,:To,Da:] = nobs_features
        #     cond_mask[:,:To,Da:] = True   # obs와 action이후 dimension은 masking

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps

        start_action_norm = self.normalizer['action'].normalize(start_action)
        start_index = int(round((1.0 - float(noise_ratio)) * (len(timesteps) - 1)))
        noise_t = timesteps[start_index].to(device)
        timestep_batch = noise_t.reshape(1).long().expand(B)
        noise = torch.randn(start_action_norm.shape, dtype=dtype, device=device)
        trajectory = scheduler.add_noise(start_action_norm, noise, timestep_batch)

        old_mask = getattr(self.model, 'mask', None)
        old_memory_mask = getattr(self.model, 'memory_mask', None)
        if old_mask is not None:
            self.model.mask = old_mask[:T, :T]
        if old_memory_mask is not None:
            self.model.memory_mask = old_memory_mask[:T, :]

        # run sampling; Denoising후 trajectory
        try:
            for t in timesteps[start_index:]:
                trajectory[cond_mask] = cond_data[cond_mask]
                model_output = self.model(trajectory, t, cond)
                trajectory = scheduler.step(
                    model_output, t, trajectory,
                    **self.kwargs
                    ).prev_sample
            trajectory[cond_mask] = cond_data[cond_mask]
        finally:
            if old_mask is not None:
                self.model.mask = old_mask
            if old_memory_mask is not None:
                self.model.memory_mask = old_memory_mask

        # unnormalize prediction
        naction_pred = trajectory[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)   # 정규화 풀기

        # get action
        start = To - 1
        action = action_pred[:,start:]   # 첫 action은 현재 obs window의 앞쪽 context로 사용

        result = {
            'action': action,   # 실제 사용할 traj
            'action_pred': action_pred,   # 예측한 전체 traj
            'noise_timestep': int(noise_t.detach().cpu().item())
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            # self, 
            # transformer_weight_decay: float, 
            # obs_encoder_weight_decay: float,
            # learning_rate: float, 
            # betas: Tuple[float, float]
            self,
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float,
            obs_encoder_weight_decay: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups( # transformer 추가
            weight_decay=weight_decay)
        
        # obs encoder from scratch
        modules = [self.low_dim_encoder]
        if self.force_encoder is not None:
            modules.append(self.force_encoder)
        if hasattr(self, 'attention_pool_2d'):
            modules.append(self.attention_pool_2d)
        if hasattr(self, 'transformer_encoder'):
            modules.append(self.transformer_encoder)
        if hasattr(self, 'linear_projection'):
            modules.append(self.linear_projection)

        obs_params = []
        for module in modules:
            obs_params.extend([p for p in module.parameters() if p.requires_grad])
        
        if hasattr(self, 'position_embedding'):
            pe = self.position_embedding
            if isinstance(pe, torch.nn.Parameter) and pe.requires_grad:
                obs_params.append(pe)
        
        optim_groups.append({
            "params": obs_params,
            "weight_decay": obs_encoder_weight_decay
        })

        # vision encoder from pretrained
        optim_groups.append({
            "params": [p for p in self.vision_encoder.parameters() if p.requires_grad],
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr
        })

        
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch

        # image crop, resize, colorjitter
        obs_dict = self._apply_image_transform(batch['obs'], self.transform_train)


        nobs = self.normalizer.normalize(obs_dict)
        nobs = self._apply_pretrained_image_norm(nobs)
        nactions = self.normalizer['action'].normalize(batch['action'])
        To = self.n_obs_steps
        B = nactions.shape[0]
        T = nactions.shape[1]
        Do = self.obs_feature_dim   # 관찰할 것의 dimension


        # Separate and save wrench data
        wrench_nobs = {}
        for key in self.wrench_keys:
            wrench_nobs[key] = nobs.pop(key)  # Save and remove from nobs
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # (B, To, ...) -> (B*To, ...)

        attention_features = list()

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
            attention_features.append(vision_feature.reshape(B, To, -1)) # (B, To, vision_feature_dim)

        # low-dim encoding (linear)
        low_dim_features = []
        for t in range(To):
            low_dim_t = torch.cat([nobs[key][:,t,:] for key in self.low_dim_keys], dim=-1)
            low_dim_feature_t = self.low_dim_encoder(low_dim_t)
            low_dim_features.append(low_dim_feature_t.reshape(B, -1))
        low_dim_features = torch.stack(low_dim_features, dim=1)  # (B, To, low_dim_feature_dim)
        # low_dim is kept as condition tokens, but excluded from modality-attention.
        
        # Force encoding
        if self.force_encoder is not None:
            force_features = []
            combined_wrench_data = []
            for key in self.wrench_keys:
                combined_wrench_data.append(wrench_nobs[key]) # (B, To=1, wrench_axis, wrench_hist)
            wrench_total = torch.cat(combined_wrench_data, dim=-2) # (B, To=1, num_wrench_component, wrench_hist)
            force_feature = self.force_encoder(wrench_total.reshape(-1, *wrench_total.shape[-2:])) # (B, 1, feature_dim)
            force_features.append(force_feature.reshape(B, -1)) # (B, feature_dim)
            attention_features.append(force_feature) # (B, 1, feature_dim)
        # attention_features: image tokens + force token. low_dim bypasses this attention.

        
        # fuse mode; attention score볼거면 없애는게 좋을수도
        if self.fuse_mode == 'modality-attention':
            if len(attention_features) > 0:
                in_embeds = torch.cat(attention_features, dim=1) # (B, image_tokens + force_tokens, feature_dim)
                if self.position_encoding == 'learnable':
                    if self.position_embedding.device != in_embeds.device:
                        self.position_embedding = self.position_embedding.to(in_embeds.device)
                    in_embeds = in_embeds + self.position_embedding
                out_embeds = self.transformer_encoder(in_embeds)
                
                token_sizes = [x.shape[1] for x in attention_features]
                attended_modalities = list(torch.split(out_embeds, token_sizes, dim=1))
                attended_vision = attended_modalities[:len(self.rgb_keys)]
                attended_low_dim = low_dim_features
                attended_force = attended_modalities[len(self.rgb_keys)] if self.force_encoder is not None else None
            else:
                attended_vision = []
                attended_low_dim = low_dim_features
                attended_force = None

            condition_features = attended_vision + [attended_low_dim]
            if attended_force is not None:
                condition_features.append(attended_force)
            token_num = sum(x.shape[1] for x in condition_features)
            nobs_features = torch.cat(condition_features, dim=1) # (B, token_num, feature_dim)
            assert nobs_features.shape[1] == token_num
            
        # elif self.fuse_mode == 'concat':
        #     nobs_features = torch.cat(vision_features + [low_dim_features.reshape(B, -1)] + force_features, dim=-1)
        
        assert nobs_features.shape[-1] == Do, f"Expected obs feature dim {Do}, got {nobs_features.shape[-1]}"


        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            cond = nobs_features
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape) # 전부 False

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)


        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        self.input_pertub = 0.1
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)


        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask # 전체 action 모두 loss

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

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
