from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from omegaconf import DictConfig

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.bae_transformer_for_diffusion_force_adaln_vector import (
    TransformerForDiffusionVectorCond,
)
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.force.force_encoder import CausalConvForceEncoder, GRUForceEncoder
from diffusion_policy.model.vision.attention_pool_2d import AttentionPool2d
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionTransformerHybridImageWrenchEncoderDiTPolicy(BaseImagePolicy):
    """
    Force DiT policy.
    image / wrench는 token으로 만들어 modality-attention에 넣고,
    low_dim(pose, quat)은 token화하지 않고 flatten해서 AdaLN condition vector에 concat한다.
    """

    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            obs_encoder: DictConfig,
            num_inference_steps=None,
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=512,
            p_drop_emb=0.0,
            p_drop_attn=0.1,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            pose_repr: dict = {},
            **kwargs):
        super().__init__()

        # DiT vector condition은 obs_as_cond=True에서만 사용한다.
        if not obs_as_cond:
            raise NotImplementedError(
                "This policy expects obs_as_cond=True with vector AdaLN conditioning."
            )
        if n_cond_layers != 0:
            raise NotImplementedError(
                "n_cond_layers is not used for vector AdaLN conditioning."
            )

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

        rgb_keys = []
        low_dim_keys = []
        wrench_keys = []
        key_shape_map = dict()
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')

            # obs relative시에 quat -> rot6d
            if self.obs_pose_repr == 'relative' and 'quat' in key:
                shape = (6,)
            key_shape_map[key] = shape

            if obs_type == 'rgb':
                obs_config['rgb'].append(key)
                rgb_keys.append(key)
                self.num_image += 1
                image_resolution = shape[1:] # (H, W)
            elif obs_type == 'low_dim':
                obs_config['low_dim'].append(key)
                low_dim_keys.append(key)
                self.num_low_dim += 1
                self.num_low_dim_component += shape[0]
            elif obs_type == 'wrench':
                obs_config['wrench'].append(key)
                wrench_keys.append(key)
                self.num_wrench += 1
                self.num_wrench_component += shape[0]
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        print("========= Initialized Observation ==========")
        print(f"Pose representation: obs {self.obs_pose_repr}, action {self.action_pose_repr}")
        print(f"Observation config: {obs_config}")

        vc = obs_encoder.vision_encoder_cfg
        fc = obs_encoder.force_encoder_cfg

        ### Vision Encoder
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
        if len(rgb_keys) > 0:
            randomcrop = T.RandomCrop(size=int(image_resolution[0] * vc.transforms.randomcrop.ratio))
            centercrop = T.CenterCrop(size=int(image_resolution[0] * vc.transforms.randomcrop.ratio))
            resize = T.Resize(size=image_resolution[0], antialias=True)
            colorjitter = T.ColorJitter(
                brightness=vc.transforms.colorjitter.brightness,
                contrast=vc.transforms.colorjitter.contrast,
                saturation=vc.transforms.colorjitter.saturation,
                hue=vc.transforms.colorjitter.hue,
            )
            grayscale = T.RandomGrayscale(p=vc.transforms.colorjitter.grayscale)
            self.transform_train = T.Compose([randomcrop, resize, colorjitter, grayscale])
            self.transform_eval = T.Compose([centercrop, resize])

        ### Vision Feature Adapter
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

            if len(rgb_keys) > 0 and vc.use_group_norm and not vc.pretrained:
                vision_encoder = replace_submodules(
                    root_module=vision_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=((x.num_features // 16)
                                    if (x.num_features % 16 == 0)
                                    else (x.num_features // 8)),
                        num_channels=x.num_features
                    )
                )

            if len(rgb_keys) > 0 and vc.feature_aggregation == 'attention_pool_2d':
                feature_map_shape = [x // vc.downsample_ratio for x in image_resolution]
                self.attention_pool_2d = AttentionPool2d(
                    spacial_dim=feature_map_shape[0],
                    embed_dim=vision_feature_dim,
                    num_heads=vision_feature_dim // 64,
                    output_dim=vision_feature_dim
                )

        # ViT
        elif vc.model_name.startswith('vit'):
            vision_feature_dim = 768
        else:
            raise ValueError(f"Unsupported vision model {vc.model_name}")

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
            if force_feature_dim != vision_feature_dim:
                raise ValueError(
                    f"force feature dim {force_feature_dim} must match "
                    f"vision feature dim {vision_feature_dim} for modality-attention."
                )

        # fuse mode
        # image + force token만 modality-attention을 수행하고,
        # low_dim은 token으로 만들지 않고 AdaLN condition vector에 바로 concat한다.
        if obs_encoder.fuse_mode != 'modality-attention':
            raise NotImplementedError("Only modality-attention fuse_mode is supported.")

        self.transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=vision_feature_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.0,
        )
        n_force_tokens = 1 if len(wrench_keys) > 0 else 0
        n_attention_features = len(rgb_keys) * n_obs_steps + n_force_tokens
        self.position_embedding = None
        if obs_encoder.position_encoding == 'learnable':
            self.position_embedding = torch.nn.Parameter(
                torch.randn(n_attention_features, vision_feature_dim))
        self.modality_attention_token_count = n_attention_features
        self.visual_force_projection = nn.Linear(
            vision_feature_dim * n_attention_features,
            n_emb
        )

        # Diffusion Model 시작 ========================================
        # image/force attention 결과: (B, n_emb)
        # low_dim concat 결과: (B, To * low_dim_dim)
        # 최종 AdaLN condition: (B, n_emb + To * low_dim_dim)
        low_dim_vector_dim = self.num_low_dim_component * n_obs_steps
        cond_dim = n_emb + low_dim_vector_dim

        # Transformer DiT
        model = TransformerForDiffusionVectorCond(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
        )

        self.obs_encoder = obs_encoder
        self.vision_model_name = vc.model_name
        self.vision_encoder = vision_encoder
        self.force_encoder = force_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.vision_feature_dim = vision_feature_dim
        self.force_feature_dim = force_feature_dim
        self.low_dim_vector_dim = low_dim_vector_dim
        self.cond_dim = cond_dim
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
        print(f"Modality attention tokens: {self.modality_attention_token_count}")
        print(f"Lowdim concat dim: {self.low_dim_vector_dim}, AdaLN cond dim: {self.cond_dim}")

    def _preprocess_images(self, obs_dict, train: bool):
        # image crop, resize, colorjitter
        obs_dict = dict(obs_dict)
        if len(self.rgb_keys) == 0:
            return obs_dict
        transform = self.transform_train if train else self.transform_eval
        for key in self.rgb_keys:
            image = obs_dict[key]
            flat = image.reshape(-1, *image.shape[2:])
            flat = transform(flat)
            obs_dict[key] = flat.reshape(*image.shape)
        return obs_dict

    def _encode_visual_force_lowdim_condition(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        nobs -> AdaLN condition vector.
        image / force: token으로 만들어 modality-attention 수행
        low_dim: token화하지 않고 obs step 전체를 flatten해서 concat
        """
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps
        attention_features = []

        # Image encoding
        for key in self.rgb_keys:
            image = nobs[key][:, :To, ...].reshape(-1, *nobs[key].shape[2:])
            assert image.shape[1:] == self.key_shape_map[key]
            raw_vision_feature = self.vision_encoder(image) # (B*To, C, H, W) or (B*To, token, C)
            if self.vision_model_name.startswith('resnet'):
                if self.feature_aggregation == 'attention_pool_2d':
                    vision_feature = self.attention_pool_2d(raw_vision_feature) # (B*To, vision_feature_dim)
                else:
                    raise NotImplementedError(self.feature_aggregation)
            else:
                vision_feature = raw_vision_feature[:, 0, :] # CLS token
            attention_features.append(vision_feature.reshape(B, To, -1)) # (B, To, vision_feature_dim)

        # Force encoding
        if self.force_encoder is not None:
            wrench_inputs = [nobs[key] for key in self.wrench_keys]
            wrench_total = torch.cat(wrench_inputs, dim=-2) # (B, To=1, wrench_axis, wrench_hist)
            force_feature = self.force_encoder(
                wrench_total.reshape(-1, *wrench_total.shape[-2:])
            ) # (B, 1, feature_dim)
            attention_features.append(force_feature.reshape(B, -1, self.vision_feature_dim)) # (B, 1, vision_feature_dim)

        # fuse mode: image token + force token만 modality-attention
        in_embeds = torch.cat(attention_features, dim=1) # (B, image_tokens + force_tokens, feature_dim)
        if self.position_embedding is not None:
            if self.position_embedding.device != in_embeds.device:
                self.position_embedding = self.position_embedding.to(in_embeds.device)
            in_embeds = in_embeds + self.position_embedding
        attended = self.transformer_encoder(in_embeds) # (B, image_tokens + force_tokens, feature_dim)
        visual_force_cond = self.visual_force_projection(attended.reshape(B, -1)) # (B, n_emb)

        # low-dim concat (pose, quat). token으로 만들지 않는다.
        low_dim_inputs = []
        for key in self.low_dim_keys:
            low_dim_inputs.append(nobs[key][:, :To, ...].reshape(B, -1)) # (B, To * low_dim_component)
        if len(low_dim_inputs) > 0:
            low_dim_cond = torch.cat(low_dim_inputs, dim=-1) # (B, To * low_dim_dim)
        else:
            low_dim_cond = visual_force_cond.new_zeros((B, 0))

        cond = torch.cat([visual_force_cond, low_dim_cond], dim=-1) # (B, cond_dim)
        assert cond.shape[-1] == self.cond_dim
        return cond

    # ========= inference ============
    def conditional_sample(
            self,
            condition_data,
            condition_mask,
            cond=None,
            generator=None,
            **kwargs):
        model = self.model # Transformer DiT model
        scheduler = self.noise_scheduler

        # 랜덤 trajectory 생성
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values; scheduler.timestep 생성
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous sample: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include observation keys
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict

        # image crop, resize
        obs_dict = self._preprocess_images(obs_dict, train=False)

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build AdaLN condition vector
        cond = self._encode_visual_force_lowdim_condition(nobs) # (B, cond_dim)
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling; Denoising후 trajectory
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(nsample[..., :Da])

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        return {
            'action': action,
            'action_pred': action_pred
        }

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float,
            obs_encoder_weight_decay: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=weight_decay)

        modules = [self.transformer_encoder, self.visual_force_projection]
        if self.force_encoder is not None:
            modules.append(self.force_encoder)
        if hasattr(self, 'attention_pool_2d'):
            modules.append(self.attention_pool_2d)

        obs_params = []
        for module in modules:
            obs_params.extend([p for p in module.parameters() if p.requires_grad])
        if self.position_embedding is not None and self.position_embedding.requires_grad:
            obs_params.append(self.position_embedding)

        optim_groups.append({
            "params": obs_params,
            "weight_decay": obs_encoder_weight_decay
        })
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
        obs = self._preprocess_images(batch['obs'], train=True)
        nobs = self.normalizer.normalize(obs)
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions

        # build AdaLN condition vector
        cond = self._encode_visual_force_lowdim_condition(nobs) # (B, cond_dim)
        if self.pred_action_steps_only:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]

        # generate inpainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the trajectory
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # input perturbation by adding additional noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + 0.1 * torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

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
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()

    def forward(self, batch):
        return self.compute_loss(batch)
