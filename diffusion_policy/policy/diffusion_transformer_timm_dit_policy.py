from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.bae_transformer_for_diffusion_force_adaln import (
    TransformerForDiffusion,
)
from diffusion_policy.model.vision.transformer_obs_encoder import TransformerObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DiffusionTransformerTimmDiTPolicy(BaseImagePolicy):
    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TransformerObsEncoder,
            horizon=None,
            n_action_steps=None,
            n_obs_steps=None,
            num_inference_steps=None,
            input_pertub=0.1,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=8,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.1,
            causal_attn=False,
            time_as_cond=True,
            obs_as_cond=True,
            # parameters passed to scheduler.step
            **kwargs):
        super().__init__()

        if not obs_as_cond:
            raise NotImplementedError(
                "DiffusionTransformerTimmDiTPolicy currently expects obs_as_cond=True."
            )

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        if horizon is not None:
            assert horizon == action_horizon
        if n_obs_steps is None:
            obs_horizons = [
                attr['horizon'] for attr in shape_meta['obs'].values()
                if 'horizon' in attr
            ]
            assert len(obs_horizons) > 0
            n_obs_steps = max(obs_horizons)
        if n_action_steps is None:
            n_action_steps = action_horizon - n_obs_steps + 1

        obs_shape = obs_encoder.output_shape()
        obs_tokens = obs_shape[-2]
        obs_feature_dim = obs_shape[-1]

        self.obs_cond_proj = nn.Identity()
        if obs_feature_dim != n_emb:
            self.obs_cond_proj = nn.Linear(obs_feature_dim, n_emb)

        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=action_horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=n_emb,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
            n_cond_tokens=obs_tokens,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = action_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_tokens = self.obs_encoder(obs_dict)
        obs_tokens = self.obs_cond_proj(obs_tokens)
        assert obs_tokens.shape[-1] == self.model.n_emb
        return obs_tokens

    # ========= inference ============
    def conditional_sample(
            self,
            condition_data,
            condition_mask,
            cond=None,
            generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        obs_tokens = self._encode_obs(nobs)

        cond_data = torch.zeros(
            size=(B, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            **self.kwargs)

        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

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

        backbone_params = list()
        other_obs_params = list(self.obs_cond_proj.parameters())
        for key, value in self.obs_encoder.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)

        optim_groups.append({
            "params": backbone_params,
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr
        })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": obs_encoder_weight_decay
        })

        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions

        obs_tokens = self._encode_obs(nobs)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],), device=trajectory.device
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)

        pred = self.model(
            noisy_trajectory,
            timesteps,
            cond=obs_tokens
        )

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
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
