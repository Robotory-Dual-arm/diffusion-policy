import copy
import logging

import torch
import torch.nn as nn

from diffusion_policy.model.force.force_encoder import CausalConvForceEncoder, GRUForceEncoder
from diffusion_policy.model.vision.transformer_obs_encoder import TransformerObsEncoder


logger = logging.getLogger(__name__)


class TransformerObsWrenchEncoder(TransformerObsEncoder):
    """
    TransformerObsEncoder extension that keeps the existing rgb token behavior,
    projects all low-dim observations into one token per observation step, and
    appends wrench-history tokens encoded by a force encoder.

    Wrench observations are expected as (B, T, C, H), where C is force/wrench
    channels and H is the force history length.
    """

    def __init__(
            self,
            shape_meta: dict,
            force_encoder_cfg: dict,
            *args,
            **kwargs):
        self.full_shape_meta = copy.deepcopy(shape_meta)
        obs_shape_meta = shape_meta['obs']
        concat_low_dim_keys = sorted([
            key for key, attr in obs_shape_meta.items()
            if attr.get('type', 'low_dim') == 'low_dim'
        ])
        self.low_dim_key_shape_map = {
            key: tuple(obs_shape_meta[key]['shape'])
            for key in concat_low_dim_keys
        }
        self.low_dim_key_horizon_map = {
            key: obs_shape_meta[key].get('horizon', 1)
            for key in concat_low_dim_keys
        }
        self.wrench_keys = sorted([
            key for key, attr in obs_shape_meta.items()
            if attr.get('type', 'low_dim') == 'wrench'
        ])
        self.wrench_key_shape_map = {
            key: tuple(obs_shape_meta[key]['shape'])
            for key in self.wrench_keys
        }
        self.wrench_key_horizon_map = {
            key: obs_shape_meta[key].get('horizon', 1)
            for key in self.wrench_keys
        }

        base_shape_meta = copy.deepcopy(shape_meta)
        base_shape_meta['obs'] = {
            key: attr for key, attr in obs_shape_meta.items()
            if attr.get('type', 'low_dim') == 'rgb'
        }
        super().__init__(shape_meta=base_shape_meta, *args, **kwargs)
        self.concat_low_dim_keys = concat_low_dim_keys

        self.low_dim_projection = nn.Identity()
        if len(self.concat_low_dim_keys) > 0:
            low_dim = sum(
                int(torch.tensor(self.low_dim_key_shape_map[key]).prod().item())
                for key in self.concat_low_dim_keys
            )
            if low_dim != self.n_emb:
                self.low_dim_projection = nn.Linear(low_dim, self.n_emb)

        self.force_encoder = None
        self.force_feature_dim = 0
        if len(self.wrench_keys) > 0:
            input_dim = sum(self.wrench_key_shape_map[key][0] for key in self.wrench_keys)
            force_feature_dim = force_encoder_cfg.get('feature_dim', self.n_emb)
            model_name = force_encoder_cfg.get('model_name', 'causalconv')
            if model_name == 'causalconv':
                self.force_encoder = CausalConvForceEncoder(
                    input_dim=input_dim,
                    feature_dim=force_feature_dim
                )
            elif model_name == 'gru':
                self.force_encoder = GRUForceEncoder(
                    input_dim=input_dim,
                    feature_dim=force_feature_dim
                )
            else:
                raise ValueError(f"Unsupported force encoder: {model_name}")
            self.force_feature_dim = force_feature_dim
            self.force_projection = nn.Identity()
            if force_feature_dim != self.n_emb:
                self.force_projection = nn.Linear(force_feature_dim, self.n_emb)
        else:
            self.force_projection = nn.Identity()

        logger.info(
            "low_dim keys: %s, wrench keys: %s, force encoder params: %e",
            self.concat_low_dim_keys,
            self.wrench_keys,
            0.0 if self.force_encoder is None
            else sum(p.numel() for p in self.force_encoder.parameters())
        )

    def _encode_low_dim(self, obs_dict):
        if len(self.concat_low_dim_keys) == 0:
            return None

        batch_size = next(iter(obs_dict.values())).shape[0]
        low_dim_inputs = []
        horizon = None
        for key in self.concat_low_dim_keys:
            data = obs_dict[key]
            assert data.shape[0] == batch_size
            assert tuple(data.shape[2:]) == self.low_dim_key_shape_map[key]
            if horizon is None:
                horizon = data.shape[1]
            else:
                assert data.shape[1] == horizon, "All low_dim keys must share T"
            low_dim_inputs.append(data.reshape(batch_size, data.shape[1], -1))

        low_dim_total = torch.cat(low_dim_inputs, dim=-1)
        low_dim_tokens = self.low_dim_projection(low_dim_total)
        assert low_dim_tokens.shape[-1] == self.n_emb
        return low_dim_tokens

    def _encode_wrench(self, obs_dict):
        if len(self.wrench_keys) == 0:
            return None
        assert self.force_encoder is not None

        batch_size = next(iter(obs_dict.values())).shape[0]
        wrench_inputs = []
        horizon = None
        history = None
        for key in self.wrench_keys:
            wrench = obs_dict[key]
            assert wrench.ndim == 4, f"Expected {key} as (B,T,C,H), got {wrench.shape}"
            assert wrench.shape[0] == batch_size
            assert tuple(wrench.shape[2:]) == self.wrench_key_shape_map[key]
            if horizon is None:
                horizon = wrench.shape[1]
                history = wrench.shape[-1]
            else:
                assert wrench.shape[1] == horizon, "All wrench keys must share T"
                assert wrench.shape[-1] == history, "All wrench keys must share history length"
            wrench_inputs.append(wrench)

        wrench_total = torch.cat(wrench_inputs, dim=-2)
        force_feature = self.force_encoder(
            wrench_total.reshape(-1, *wrench_total.shape[-2:])
        )
        assert force_feature.ndim == 3
        force_feature = self.force_projection(force_feature)
        assert force_feature.shape[-1] == self.n_emb
        return force_feature.reshape(batch_size, -1, self.n_emb)

    def forward(self, obs_dict):
        embeddings = []
        if len(self.rgb_keys) > 0:
            embeddings.append(super().forward(obs_dict))
        low_dim_tokens = self._encode_low_dim(obs_dict)
        if low_dim_tokens is not None:
            embeddings.append(low_dim_tokens)
        wrench_tokens = self._encode_wrench(obs_dict)
        if wrench_tokens is not None:
            embeddings.append(wrench_tokens)
        return torch.cat(embeddings, dim=1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.full_shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            horizon = attr.get('horizon', 1)
            example_obs_dict[key] = torch.zeros(
                (1, horizon) + shape,
                dtype=self.dtype,
                device=self.device
            )
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1
        return example_output.shape
