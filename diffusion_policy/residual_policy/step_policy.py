from typing import Dict, Iterable
import pathlib

import dill
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.residual_policy.force_encoder_util import get_wrench_keys, make_force_encoder


def _mlp(input_dim, hidden_dims: Iterable[int], output_dim, dropout=0.0):
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def load_policy_from_workspace_ckpt(ckpt_path, use_ema=True):
    ckpt_path = pathlib.Path(ckpt_path).expanduser()
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key], strict=False)
    del payload
    policy.eval()
    return policy


class FastResidualStepPolicy(BaseImagePolicy):
    """One-step residual corrector.

    This is intentionally smaller than the diffusion-style residual policy:
        current image / proprio / force + base_action_rel -> residual_delta6
    """

    def __init__(
            self,
            shape_meta: dict,
            slow_ckpt_path: str,
            base_action_key: str = "base_action_rel",
            slow_use_ema: bool = True,
            hidden_dims=(512, 512, 256),
            dropout: float = 0.0,
            freeze_vision_encoder: bool = True,
            freeze_force_encoder: bool = True,
            train_force_encoder: bool = False,
            force_encoder_cfg=None,
        ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]

        if base_action_key not in shape_meta["obs"]:
            raise KeyError(f"shape_meta.obs must include '{base_action_key}'")
        self.base_action_key = base_action_key
        self.base_action_dim = shape_meta["obs"][base_action_key]["shape"][0]

        slow_policy = load_policy_from_workspace_ckpt(
            slow_ckpt_path,
            use_ema=slow_use_ema,
        )
        slow_policy.eval()
        slow_policy.requires_grad_(False)

        self.slow_policy = slow_policy
        self.vision_encoder = getattr(slow_policy, "vision_encoder", None)
        self.force_encoder = getattr(slow_policy, "force_encoder", None)
        self.attention_pool_2d = getattr(slow_policy, "attention_pool_2d", None)
        self.train_force_encoder = bool(train_force_encoder)

        if self.vision_encoder is None:
            raise AttributeError("Slow policy must expose vision_encoder")
        if freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
        shape_wrench_keys = get_wrench_keys(shape_meta)
        if force_encoder_cfg is not None and len(shape_wrench_keys) > 0:
            self.force_encoder, self.force_feature_dim = make_force_encoder(shape_meta, force_encoder_cfg)
            self.wrench_keys = shape_wrench_keys
        else:
            self.wrench_keys = [
                key for key in getattr(slow_policy, "wrench_keys", [])
                if key in shape_meta["obs"]
            ]
            self.force_feature_dim = getattr(slow_policy, "force_feature_dim", 0)

        if self.force_encoder is not None and freeze_force_encoder and not self.train_force_encoder:
            self.force_encoder.requires_grad_(False)

        self.rgb_keys = [
            key for key in getattr(slow_policy, "rgb_keys", [])
            if key in shape_meta["obs"]
        ]
        self.low_dim_keys = [
            key for key in getattr(slow_policy, "low_dim_keys", [])
            if key in shape_meta["obs"] and key != base_action_key
        ]
        if len(self.rgb_keys) == 0:
            raise ValueError("Step residual policy needs at least one rgb key")

        self.vision_model_name = getattr(slow_policy, "vision_model_name", "")
        self.vision_feature_dim = getattr(slow_policy, "vision_feature_dim", None)
        if self.vision_feature_dim is None:
            raise AttributeError("Slow policy must expose vision_feature_dim")
        low_dim = 0
        for key in self.low_dim_keys:
            low_dim += shape_meta["obs"][key]["shape"][0]
        force_dim = self.force_feature_dim if len(self.wrench_keys) > 0 else 0
        input_dim = (
            self.vision_feature_dim * len(self.rgb_keys)
            + force_dim
            + low_dim
            + self.base_action_dim
        )

        self.head = _mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            dropout=dropout,
        )
        self.normalizer = LinearNormalizer()
        self.shape_meta = shape_meta
        self.n_obs_steps = 1
        self.n_action_steps = 1
        self.horizon = 1

        print("Frozen slow policy params: %e" % sum(p.numel() for p in self.slow_policy.parameters()))
        print("Fast step residual head params: %e" % sum(p.numel() for p in self.head.parameters()))
        print(
            "Fast step inputs: rgb=%s, low_dim=%s, wrench=%s, base_action=%s"
            % (self.rgb_keys, self.low_dim_keys, self.wrench_keys, self.base_action_key)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.slow_policy.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
        if self.force_encoder is not None and not self.train_force_encoder:
            self.force_encoder.eval()
        return self

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _latest(self, value):
        if value.ndim >= 3:
            return value[:, -1]
        return value

    @torch.no_grad()
    def _encode_image(self, nobs):
        features = []
        for key in self.rgb_keys:
            img = self._latest(nobs[key])
            raw_feature = self.vision_encoder(img)
            if self.vision_model_name.startswith("resnet"):
                if self.attention_pool_2d is not None and raw_feature.ndim == 4:
                    feature = self.attention_pool_2d(raw_feature)
                elif raw_feature.ndim == 4:
                    feature = raw_feature.mean(dim=(-2, -1))
                else:
                    feature = raw_feature
            else:
                feature = raw_feature[:, 0, :] if raw_feature.ndim == 3 else raw_feature
            features.append(feature)
        return torch.cat(features, dim=-1)

    def _encode_wrench(self, nobs):
        if len(self.wrench_keys) == 0:
            return None
        if self.force_encoder is None:
            raise AttributeError("wrench_keys are configured, but slow policy has no force_encoder")
        wrench_total = torch.cat([nobs[key] for key in self.wrench_keys], dim=-2)
        if wrench_total.ndim == 4:
            wrench_total = wrench_total[:, -1]
        if self.train_force_encoder:
            return self.force_encoder(wrench_total).flatten(start_dim=1)
        with torch.no_grad():
            return self.force_encoder(wrench_total).flatten(start_dim=1)

    def _build_head_input(self, obs_dict: Dict[str, torch.Tensor]):
        encoder_obs = {
            key: obs_dict[key]
            for key in self.rgb_keys + self.low_dim_keys
            if key in obs_dict
        }
        if hasattr(self.slow_policy, "_apply_image_transform"):
            encoder_obs = self.slow_policy._apply_image_transform(
                encoder_obs,
                self.slow_policy.transform_eval,
            )
        slow_nobs = self.slow_policy.normalizer.normalize(encoder_obs)
        fast_nobs = self.normalizer.normalize(obs_dict)

        parts = [self._encode_image(slow_nobs)]
        if len(self.wrench_keys) > 0:
            parts.append(self._encode_wrench(fast_nobs))
        if len(self.low_dim_keys) > 0:
            low_dim = torch.cat([
                self._latest(slow_nobs[key])
                for key in self.low_dim_keys
            ], dim=-1)
            parts.append(low_dim)
        parts.append(self._latest(fast_nobs[self.base_action_key]))
        return torch.cat(parts, dim=-1)

    def forward(self, obs_dict: Dict[str, torch.Tensor]):
        head_input = self._build_head_input(obs_dict)
        return self.head(head_input)[:, None, :]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nresidual_pred = self.forward(obs_dict)
        residual_pred = self.normalizer["action"].unnormalize(nresidual_pred)
        return {
            "action": residual_pred,
            "action_pred": residual_pred,
        }

    def compute_loss(self, batch):
        nresidual_pred = self.forward(batch["obs"])
        action_target = batch["action"]
        if action_target.ndim == 2:
            action_target = action_target[:, None, :]
        if action_target.shape[1] > 1:
            action_target = action_target[:, -1:]
        nresidual_target = self.normalizer["action"].normalize(action_target)
        loss = F.mse_loss(nresidual_pred, nresidual_target)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite fast step residual loss: {loss.detach().item()}")
        return loss
