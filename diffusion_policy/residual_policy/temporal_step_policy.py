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


class FastResidualTemporalPolicy(BaseImagePolicy):
    """Temporal one-step residual corrector.

    The chunk-start slow-policy condition initializes the GRU hidden state. Each
    recurrent step then receives only force history features and that step's
    slow/base action, then predicts one residual_delta6.
    """

    def __init__(
            self,
            shape_meta: dict,
            slow_ckpt_path: str,
            base_action_key: str = "base_action_rel",
            slow_use_ema: bool = True,
            n_obs_steps: int = 4,
            rnn_hidden_dim: int = 256,
            rnn_num_layers: int = 1,
            hidden_dims=(512, 256),
            dropout: float = 0.0,
            freeze_vision_encoder: bool = True,
            freeze_force_encoder: bool = True,
            train_force_encoder: bool = False,
            include_initial_wrench: bool = True,
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
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = 1
        self.horizon = 1
        self.include_initial_wrench = bool(include_initial_wrench)

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
            raise ValueError("Temporal residual policy needs at least one rgb key")

        self.vision_model_name = getattr(slow_policy, "vision_model_name", "")
        self.vision_feature_dim = getattr(slow_policy, "vision_feature_dim", None)
        if self.vision_feature_dim is None:
            raise AttributeError("Slow policy must expose vision_feature_dim")
        low_dim = sum(shape_meta["obs"][key]["shape"][0] for key in self.low_dim_keys)
        force_dim = self.force_feature_dim if len(self.wrench_keys) > 0 else 0
        rnn_input_dim = force_dim + self.base_action_dim
        if rnn_input_dim <= 0:
            raise ValueError("Temporal residual policy needs force or base-action inputs")

        initial_context_dim = self.vision_feature_dim * len(self.rgb_keys) + low_dim
        if self.include_initial_wrench:
            initial_context_dim += force_dim
        self.initial_hidden = _mlp(
            input_dim=initial_context_dim,
            hidden_dims=(256,),
            output_dim=rnn_hidden_dim * rnn_num_layers,
            dropout=dropout,
        )

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )
        self.head = _mlp(
            input_dim=rnn_hidden_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            dropout=dropout,
        )
        self.normalizer = LinearNormalizer()
        self.shape_meta = shape_meta
        self.train_force_encoder = bool(train_force_encoder)

        print("Frozen slow policy params: %e" % sum(p.numel() for p in self.slow_policy.parameters()))
        print("Fast temporal initial hidden params: %e" % sum(p.numel() for p in self.initial_hidden.parameters()))
        print("Fast temporal GRU params: %e" % sum(p.numel() for p in self.rnn.parameters()))
        print("Fast temporal head params: %e" % sum(p.numel() for p in self.head.parameters()))
        print(
            "Fast temporal inputs: h0 from rgb=%s first + low_dim=%s first + wrench=%s first; recurrent wrench=%s + base_action=%s, n_obs_steps=%d"
            % (
                self.rgb_keys,
                self.low_dim_keys,
                self.wrench_keys if self.include_initial_wrench else [],
                self.wrench_keys,
                self.base_action_key,
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

    def _latest(self, value):
        if value.ndim >= 3:
            return value[:, -1]
        return value

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
            else:
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

    def _build_sequence_inputs(self, obs_dict: Dict[str, torch.Tensor], need_initial_hidden=True):
        encoder_keys = []
        if need_initial_hidden:
            encoder_keys = self.rgb_keys + self.low_dim_keys
        encoder_obs = {
            key: obs_dict[key]
            for key in encoder_keys
            if key in obs_dict
        }
        if need_initial_hidden and hasattr(self.slow_policy, "_apply_image_transform"):
            encoder_obs = self.slow_policy._apply_image_transform(
                encoder_obs,
                self.slow_policy.transform_eval,
            )
        slow_nobs = self.slow_policy.normalizer.normalize(encoder_obs)
        fast_nobs = self.normalizer.normalize(obs_dict)

        force_seq = self._encode_wrench_sequence(fast_nobs)

        initial_hidden = None
        if need_initial_hidden:
            initial_parts = [self._encode_initial_image(slow_nobs)]
            if len(self.low_dim_keys) > 0:
                initial_parts.append(torch.cat([
                    slow_nobs[key][:, 0] if slow_nobs[key].ndim >= 3 else slow_nobs[key]
                    for key in self.low_dim_keys
                ], dim=-1))
            if self.include_initial_wrench and force_seq is not None:
                initial_parts.append(force_seq[:, 0])
            initial_context = torch.cat(initial_parts, dim=-1)
            b = initial_context.shape[0]
            initial_hidden = self.initial_hidden(initial_context)
            initial_hidden = initial_hidden.reshape(b, self.rnn.num_layers, self.rnn.hidden_size)
            initial_hidden = initial_hidden.transpose(0, 1).contiguous()

        seq_parts = []
        if force_seq is not None:
            seq_parts.append(force_seq)
        seq_parts.append(fast_nobs[self.base_action_key])
        temporal_input = torch.cat(seq_parts, dim=-1)
        return initial_hidden, temporal_input

    def forward(self, obs_dict: Dict[str, torch.Tensor], hidden=None, return_hidden=False):
        initial_hidden, temporal_input = self._build_sequence_inputs(
            obs_dict,
            need_initial_hidden=hidden is None,
        )
        if hidden is None:
            hidden = initial_hidden
        rnn_out, hidden = self.rnn(temporal_input, hidden)
        head_input = rnn_out
        b, t = head_input.shape[:2]
        pred = self.head(head_input.reshape(b * t, -1)).reshape(b, t, self.action_dim)
        if return_hidden:
            return pred, hidden
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

    def predict_step(self, obs_dict: Dict[str, torch.Tensor], hidden=None):
        nresidual_pred, hidden = self.forward(obs_dict, hidden=hidden, return_hidden=True)
        residual_pred = self.normalizer["action"].unnormalize(nresidual_pred[:, -1:])
        return {
            "action": residual_pred,
            "action_pred": residual_pred,
            "hidden": hidden,
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
            raise FloatingPointError(f"Non-finite fast temporal residual loss: {loss.detach().item()}")
        return loss
