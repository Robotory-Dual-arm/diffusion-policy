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


class FastResidualContextStepPolicy(BaseImagePolicy):
    """Independent per-step residual MLP with a fixed chunk-start context.

    The first image/low-dim observation in the sampled window is encoded once and
    repeated for every fast step. Each step then receives only the latest force
    feature and that step's base/slow action.
    """

    def __init__(
            self,
            shape_meta: dict,
            slow_ckpt_path: str,
            base_action_key: str = "base_action_rel",
            slow_use_ema: bool = True,
            n_obs_steps: int = 16,
            hidden_dims=(512, 512, 256),
            dropout: float = 0.0,
            freeze_vision_encoder: bool = True,
            freeze_force_encoder: bool = True,
            train_force_encoder: bool = False,
            include_initial_low_dim: bool = True,
            include_initial_wrench: bool = False,
            include_step_low_dim: bool = False,
            force_encoder_cfg=None,
            step_encoding: str = "none",
        ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]

        if base_action_key not in shape_meta["obs"]:
            raise KeyError(f"shape_meta.obs must include '{base_action_key}'")
        self.base_action_key = base_action_key
        self.base_action_dim = shape_meta["obs"][base_action_key]["shape"][0]
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = 1
        self.horizon = 1
        self.include_initial_low_dim = bool(include_initial_low_dim)
        self.include_initial_wrench = bool(include_initial_wrench)
        self.include_step_low_dim = bool(include_step_low_dim)
        self.step_encoding = step_encoding

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

        if self.vision_encoder is None:
            raise AttributeError("Slow policy must expose vision_encoder")
        if freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
        self.rgb_keys = [
            key for key in getattr(slow_policy, "rgb_keys", [])
            if key in shape_meta["obs"]
        ]
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
        if self.force_encoder is not None:
            if train_force_encoder:
                self.force_encoder.requires_grad_(True)
            elif freeze_force_encoder:
                self.force_encoder.requires_grad_(False)
        self.low_dim_keys = [
            key for key in getattr(slow_policy, "low_dim_keys", [])
            if key in shape_meta["obs"] and key != base_action_key
        ]
        if len(self.rgb_keys) == 0:
            raise ValueError("Context-step residual policy needs at least one rgb key")

        self.vision_model_name = getattr(slow_policy, "vision_model_name", "")
        self.vision_feature_dim = getattr(slow_policy, "vision_feature_dim", None)
        if self.vision_feature_dim is None:
            raise AttributeError("Slow policy must expose vision_feature_dim")

        low_dim = sum(shape_meta["obs"][key]["shape"][0] for key in self.low_dim_keys)
        force_dim = self.force_feature_dim if len(self.wrench_keys) > 0 else 0
        step_encoding_dim = {"none": 0, "scalar": 1, "sin_cos": 2}[self.step_encoding]
        context_dim = self.vision_feature_dim * len(self.rgb_keys)
        if self.include_initial_low_dim:
            context_dim += low_dim
        if self.include_initial_wrench:
            context_dim += force_dim
        step_low_dim = low_dim if self.include_step_low_dim else 0
        step_dim = step_low_dim + force_dim + self.base_action_dim + step_encoding_dim
        input_dim = context_dim + step_dim

        self.head = _mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            dropout=dropout,
        )
        self.normalizer = LinearNormalizer()
        self.shape_meta = shape_meta
        self.train_force_encoder = bool(train_force_encoder)
        self.uses_fixed_context_sequence = True

        print("Frozen slow policy params: %e" % sum(p.numel() for p in self.slow_policy.parameters()))
        print("Fast context-step residual head params: %e" % sum(p.numel() for p in self.head.parameters()))
        print(
            "Fast context-step inputs: fixed rgb=%s first, fixed low_dim=%s, fixed wrench=%s, per-step low_dim=%s + wrench=%s + base_action=%s + step_encoding=%s, n_obs_steps=%d"
            % (
                self.rgb_keys,
                self.low_dim_keys if self.include_initial_low_dim else [],
                self.wrench_keys if self.include_initial_wrench else [],
                self.low_dim_keys if self.include_step_low_dim else [],
                self.wrench_keys,
                self.base_action_key,
                self.step_encoding,
                self.n_obs_steps,
            )
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

    def _pool_image_feature(self, raw_feature):
        if self.vision_model_name.startswith("resnet"):
            if self.attention_pool_2d is not None and raw_feature.ndim == 4:
                return self.attention_pool_2d(raw_feature)
            if raw_feature.ndim == 4:
                return raw_feature.mean(dim=(-2, -1))
            return raw_feature
        return raw_feature[:, 0, :] if raw_feature.ndim == 3 else raw_feature

    @torch.no_grad()
    def _encode_initial_image(self, nobs):
        features = []
        for key in self.rgb_keys:
            img = nobs[key]
            if img.ndim == 5:
                img = img[:, 0]
            feature = self._pool_image_feature(self.vision_encoder(img))
            features.append(feature)
        return torch.cat(features, dim=-1)

    def _encode_wrench_sequence(self, nobs):
        if len(self.wrench_keys) == 0:
            return None
        if self.force_encoder is None:
            raise AttributeError("wrench_keys are configured, but slow policy has no force_encoder")

        wrench_total = torch.cat([nobs[key] for key in self.wrench_keys], dim=-2)
        if wrench_total.ndim == 3:
            wrench_total = wrench_total[:, None]
        b, t = wrench_total.shape[:2]
        flat_wrench = wrench_total.reshape(b * t, *wrench_total.shape[2:])
        if self.train_force_encoder:
            flat_feature = self.force_encoder(flat_wrench)
        else:
            with torch.no_grad():
                flat_feature = self.force_encoder(flat_wrench)
        return flat_feature.flatten(start_dim=1).reshape(b, t, -1)

    def _step_encoding(self, batch_size, steps, device, dtype):
        if self.step_encoding == "none":
            return None
        phase = torch.arange(steps, device=device, dtype=dtype)
        phase = phase / max(self.n_obs_steps - 1, 1)
        phase = phase[None, :, None].expand(batch_size, -1, -1)
        if self.step_encoding == "scalar":
            return phase
        if self.step_encoding == "sin_cos":
            angle = phase * (2.0 * torch.pi)
            return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        raise ValueError(f"Unsupported step_encoding: {self.step_encoding}")

    def _build_head_input(self, obs_dict: Dict[str, torch.Tensor]):
        encoder_keys = list(self.rgb_keys)
        if self.include_initial_low_dim:
            encoder_keys = encoder_keys + self.low_dim_keys
        encoder_obs = {
            key: obs_dict[key]
            for key in encoder_keys
            if key in obs_dict
        }
        if hasattr(self.slow_policy, "_apply_image_transform"):
            encoder_obs = self.slow_policy._apply_image_transform(
                encoder_obs,
                self.slow_policy.transform_eval,
            )
        slow_nobs = self.slow_policy.normalizer.normalize(encoder_obs)
        fast_nobs = self.normalizer.normalize(obs_dict)

        force_seq = self._encode_wrench_sequence(fast_nobs)
        base_action = fast_nobs[self.base_action_key]
        if base_action.ndim == 2:
            base_action = base_action[:, None]
        b, t = base_action.shape[:2]

        context_parts = [self._encode_initial_image(slow_nobs)]
        if self.include_initial_low_dim and len(self.low_dim_keys) > 0:
            context_parts.append(torch.cat([
                slow_nobs[key][:, 0] if slow_nobs[key].ndim >= 3 else slow_nobs[key]
                for key in self.low_dim_keys
            ], dim=-1))
        if self.include_initial_wrench and force_seq is not None:
            context_parts.append(force_seq[:, 0])
        context = torch.cat(context_parts, dim=-1)
        context = context[:, None].expand(-1, t, -1)

        step_parts = []
        if self.include_step_low_dim and len(self.low_dim_keys) > 0:
            step_low_dim = []
            for key in self.low_dim_keys:
                value = fast_nobs[key]
                if value.ndim == 2:
                    value = value[:, None]
                if value.shape[1] != t:
                    value = value[:, -t:]
                step_low_dim.append(value)
            step_parts.append(torch.cat(step_low_dim, dim=-1))
        if force_seq is not None:
            if force_seq.shape[1] != t:
                force_seq = force_seq[:, -t:]
            step_parts.append(force_seq)
        step_parts.append(base_action)
        step_feature = self._step_encoding(b, t, base_action.device, base_action.dtype)
        if step_feature is not None:
            step_parts.append(step_feature)
        step_input = torch.cat(step_parts, dim=-1)
        return torch.cat([context, step_input], dim=-1)

    def forward(self, obs_dict: Dict[str, torch.Tensor]):
        head_input = self._build_head_input(obs_dict)
        b, t = head_input.shape[:2]
        pred = self.head(head_input.reshape(b * t, -1)).reshape(b, t, self.action_dim)
        return pred

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nresidual_pred = self.forward(obs_dict)
        residual_pred = self.normalizer["action"].unnormalize(nresidual_pred)
        residual_last = residual_pred[:, -1:]
        return {
            "action": residual_last,
            "action_pred": residual_last,
            "action_sequence": residual_pred,
        }

    def compute_loss(self, batch):
        nresidual_pred = self.forward(batch["obs"])
        action_target = batch["action"]
        if action_target.ndim == 2:
            action_target = action_target[:, None, :]
        if action_target.shape[1] != nresidual_pred.shape[1]:
            min_t = min(action_target.shape[1], nresidual_pred.shape[1])
            action_target = action_target[:, -min_t:]
            nresidual_pred = nresidual_pred[:, -min_t:]
        nresidual_target = self.normalizer["action"].normalize(action_target)
        loss = F.mse_loss(nresidual_pred, nresidual_target)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite fast context-step residual loss: {loss.detach().item()}")
        return loss
