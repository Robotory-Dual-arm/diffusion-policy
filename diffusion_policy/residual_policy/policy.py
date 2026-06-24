from typing import Dict, Iterable
import pathlib

import dill
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


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


class FastResidualImagePolicy(BaseImagePolicy):
    """Fast residual MLP policy with a frozen slow policy encoder.

    Input:
        obs image / wrench / proprio keys required by the slow policy
        obs[slow_action_key], usually a slow relative pose waypoint

    Output:
        residual delta action, e.g. dpos(3) + drotvec(3)
    """

    def __init__(
            self,
            shape_meta: dict,
            slow_ckpt_path: str,
            horizon: int = 1,
            n_action_steps: int = 1,
            n_obs_steps: int = 1,
            slow_action_key: str = "slow_action_rel",
            slow_use_ema: bool = True,
            hidden_dims=(512, 512, 256),
            dropout: float = 0.0,
            include_obs_feature: bool = True,
            include_slow_action: bool = True,
        ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        if slow_action_key not in shape_meta["obs"]:
            raise KeyError(f"shape_meta.obs must include '{slow_action_key}'")
        slow_action_dim = shape_meta["obs"][slow_action_key]["shape"][0]

        slow_policy = load_policy_from_workspace_ckpt(
            slow_ckpt_path,
            use_ema=slow_use_ema,
        )
        slow_policy.eval()
        slow_policy.requires_grad_(False)

        obs_feature_dim = getattr(slow_policy, "obs_feature_dim", None)
        if obs_feature_dim is None:
            raise AttributeError("Slow policy must expose obs_feature_dim")

        input_dim = 0
        if include_obs_feature:
            input_dim += obs_feature_dim
        if include_slow_action:
            input_dim += slow_action_dim
        if input_dim == 0:
            raise ValueError("At least one fast input feature must be enabled")

        self.slow_policy = slow_policy
        self.head = _mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=horizon * action_dim,
            dropout=dropout,
        )
        self.normalizer = LinearNormalizer()
        self.shape_meta = shape_meta
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.slow_action_key = slow_action_key
        self.action_dim = action_dim
        self.slow_action_dim = slow_action_dim
        self.obs_feature_dim = obs_feature_dim
        self.include_obs_feature = include_obs_feature
        self.include_slow_action = include_slow_action

        print("Slow policy params: %e" % sum(p.numel() for p in self.slow_policy.parameters()))
        print("Fast residual head params: %e" % sum(p.numel() for p in self.head.parameters()))

    def train(self, mode: bool = True):
        super().train(mode)
        self.slow_policy.eval()
        return self

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _filter_slow_obs(self, obs_dict):
        slow = self.slow_policy
        keys = []
        if hasattr(slow, "rgb_keys"):
            keys.extend(slow.rgb_keys)
        if hasattr(slow, "low_dim_keys"):
            keys.extend(slow.low_dim_keys)
        if hasattr(slow, "wrench_keys"):
            keys.extend(slow.wrench_keys)
        if not keys:
            keys = [k for k in obs_dict.keys() if k in slow.normalizer.params_dict]
        return {key: obs_dict[key] for key in keys if key in obs_dict}

    @torch.no_grad()
    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor]):
        slow = self.slow_policy
        slow_obs = self._filter_slow_obs(obs_dict)

        if hasattr(slow, "_apply_image_transform"):
            slow_obs = slow._apply_image_transform(slow_obs, slow.transform_eval)

        nobs = slow.normalizer.normalize(slow_obs)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        to = slow.n_obs_steps

        if hasattr(slow, "obs_encoder"):
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :to, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = slow.obs_encoder(this_nobs)
            return nobs_features.reshape(batch_size, -1)

        wrench_nobs = {}
        for key in getattr(slow, "wrench_keys", []):
            wrench_nobs[key] = nobs.pop(key)

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :to, ...].reshape(-1, *x.shape[2:]),
        )

        modality_features = []
        vision_features = []
        for key in slow.rgb_keys:
            img = this_nobs[key]
            raw_vision_feature = slow.vision_encoder(img)
            if slow.vision_model_name.startswith("resnet"):
                vision_feature = slow.attention_pool_2d(raw_vision_feature)
            else:
                vision_feature = raw_vision_feature[:, 0, :]
            vision_features.append(vision_feature.reshape(batch_size, -1))
            modality_features.append(vision_feature.reshape(batch_size, to, -1))

        if len(slow.low_dim_keys) > 0:
            low_dim_features = []
            for t in range(to):
                low_dim_t = torch.cat([nobs[key][:, t, :] for key in slow.low_dim_keys], dim=-1)
                low_dim_features.append(low_dim_t.reshape(batch_size, -1))
            low_dim_features = torch.stack(low_dim_features, dim=1)
        else:
            dtype = value.dtype
            device = value.device
            low_dim_features = torch.empty(batch_size, to, 0, device=device, dtype=dtype)

        force_features, force_modality_features = slow._encode_wrench(wrench_nobs, batch_size)
        modality_features.extend(force_modality_features)

        if slow.fuse_mode == "modality-attention":
            in_embeds = torch.cat(modality_features, dim=1)
            if slow.position_encoding == "learnable":
                pos_emb = slow.position_embedding.to(
                    device=in_embeds.device,
                    dtype=in_embeds.dtype,
                )
                in_embeds = in_embeds + pos_emb.unsqueeze(0)
            out_embeds = slow.transformer_encoder(in_embeds)
            projected_embeds = slow.linear_projection(out_embeds.flatten(start_dim=1))
            nobs_features = torch.cat(
                [projected_embeds, low_dim_features.reshape(batch_size, -1)],
                dim=-1,
            )
        elif slow.fuse_mode == "concat":
            nobs_features = torch.cat(
                vision_features + [low_dim_features.reshape(batch_size, -1)] + force_features,
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported slow fuse mode: {slow.fuse_mode}")
        return nobs_features

    def _build_head_input(self, obs_dict):
        parts = []
        if self.include_obs_feature:
            obs_feature = self._encode_obs_features(obs_dict)
            parts.append(obs_feature)
        if self.include_slow_action:
            nobs = self.normalizer.normalize(obs_dict)
            slow_action = nobs[self.slow_action_key]
            if slow_action.ndim == 3:
                slow_action = slow_action[:, -1, :]
            parts.append(slow_action)
        return torch.cat(parts, dim=-1)

    def forward(self, obs_dict):
        head_input = self._build_head_input(obs_dict)
        pred = self.head(head_input)
        return pred.reshape(-1, self.horizon, self.action_dim)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nresidual_pred = self.forward(obs_dict)
        residual_pred = self.normalizer["action"].unnormalize(nresidual_pred)
        action = residual_pred[:, :self.n_action_steps]
        return {
            "action": action,
            "action_pred": residual_pred,
        }

    def compute_loss(self, batch):
        nresidual_pred = self.forward(batch["obs"])
        action_target = batch["action"]
        if action_target.shape[1] < self.horizon:
            raise ValueError(
                f"Action target horizon {action_target.shape[1]} is shorter than policy horizon {self.horizon}"
            )
        if action_target.shape[1] > self.horizon:
            action_target = action_target[:, -self.horizon:]
        nresidual_target = self.normalizer["action"].normalize(action_target)
        loss = F.mse_loss(nresidual_pred, nresidual_target)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite fast residual loss: {loss.detach().item()}")
        return loss
