"""Feed-forward actor and twin critics for residual TD3.

The actor deliberately has no recurrent state and no previous-residual input.
It consumes the latest structured robot observation and a 16-D nominal action:
relative EE pose9 followed by the slow policy's 7-D hand target.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.residual_rl.normalizer import StructuredAffineNormalizer


DEFAULT_LOW_DIM_KEYS = ("robot_pose_R", "robot_quat_R", "hand_pose_R")
DEFAULT_LOW_DIM_DIMS = (3, 4, 7)


def _mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    layer_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _latest(value: torch.Tensor, event_ndim: int, key: str) -> torch.Tensor:
    """Accept ``[B, event...]`` or ``[B, T, event...]`` and return latest."""
    if value.ndim == event_ndim + 1:
        return value
    if value.ndim == event_ndim + 2:
        return value[:, -1]
    raise ValueError(
        f"{key} must have a batch dimension and optional time dimension; "
        f"got shape {tuple(value.shape)}"
    )


def _as_float_image(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.to(dtype=torch.float32).div_(255.0)
    if not image.is_floating_point():
        image = image.to(dtype=torch.float32)
    return image


class ConvImageEncoder(nn.Module):
    """Compact default image encoder; production can inject the BC encoder."""

    def __init__(self, input_channels: int = 3, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.device.type == "cpu":
            # torch 2.9 oneDNN convolution backward can emit NaN gradients for
            # these small residual encoders on CPU. Build the graph with native
            # kernels; CUDA behavior is unchanged.
            with torch.backends.mkldnn.flags(enabled=False):
                return self.network(image)
        return self.network(image)


class WrenchHistoryEncoder(nn.Module):
    """Default encoder for one ``[6, 32]`` force/torque history window."""

    def __init__(self, input_channels: int = 6, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )

    def forward(self, wrench: torch.Tensor) -> torch.Tensor:
        if wrench.device.type == "cpu":
            with torch.backends.mkldnn.flags(enabled=False):
                return self.network(wrench)
        return self.network(wrench)


class StructuredObservationEncoder(nn.Module):
    """Encode image, current proprioception, and wrist wrench history.

    Existing BC modules can be reused by passing ``image_encoder``,
    ``image_pool`` (for example the slow policy's attention pool), and
    ``wrench_encoder``. Feature dimensions are explicit so configuration errors
    fail at the encoder boundary instead of inside a critic head.
    """

    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        image_feature_dim: int = 128,
        image_pool: Optional[nn.Module] = None,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        wrench_encoder: Optional[nn.Module] = None,
        wrench_feature_dim: int = 64,
        image_key: str = "image0",
        wrench_key: str = "wrench_wrist_R",
        low_dim_keys: Sequence[str] = DEFAULT_LOW_DIM_KEYS,
        low_dim_dims: Sequence[int] = DEFAULT_LOW_DIM_DIMS,
        freeze_image_encoder: bool = False,
        freeze_wrench_encoder: bool = False,
        normalizer: Optional[StructuredAffineNormalizer] = None,
    ):
        super().__init__()
        if len(low_dim_keys) != len(low_dim_dims):
            raise ValueError("low_dim_keys and low_dim_dims must have the same length")
        self.image_encoder = image_encoder or ConvImageEncoder(output_dim=image_feature_dim)
        self.image_pool = image_pool
        self.image_transform = image_transform
        self.wrench_encoder = wrench_encoder or WrenchHistoryEncoder(output_dim=wrench_feature_dim)
        self.image_feature_dim = int(image_feature_dim)
        self.wrench_feature_dim = int(wrench_feature_dim)
        self.image_key = image_key
        self.wrench_key = wrench_key
        self.low_dim_keys = tuple(low_dim_keys)
        self.low_dim_dims = tuple(int(dim) for dim in low_dim_dims)
        self.normalizer = normalizer or StructuredAffineNormalizer()
        self.freeze_image_encoder = bool(freeze_image_encoder)
        self.freeze_wrench_encoder = bool(freeze_wrench_encoder)
        self.output_dim = (
            self.image_feature_dim
            + self.wrench_feature_dim
            + sum(self.low_dim_dims)
        )
        self.set_encoder_freeze(
            image=self.freeze_image_encoder,
            wrench=self.freeze_wrench_encoder,
        )

    def set_encoder_freeze(
        self,
        image: Optional[bool] = None,
        wrench: Optional[bool] = None,
    ) -> None:
        if image is not None:
            self.freeze_image_encoder = bool(image)
            self.image_encoder.requires_grad_(not self.freeze_image_encoder)
            if self.image_pool is not None:
                self.image_pool.requires_grad_(not self.freeze_image_encoder)
        if wrench is not None:
            self.freeze_wrench_encoder = bool(wrench)
            self.wrench_encoder.requires_grad_(not self.freeze_wrench_encoder)
        self._enforce_frozen_eval()

    def _enforce_frozen_eval(self) -> None:
        if self.freeze_image_encoder:
            self.image_encoder.eval()
            if self.image_pool is not None:
                self.image_pool.eval()
        if self.freeze_wrench_encoder:
            self.wrench_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._enforce_frozen_eval()
        return self

    def _pool_image(self, feature: torch.Tensor) -> torch.Tensor:
        if self.image_pool is not None:
            feature = self.image_pool(feature)
        elif feature.ndim == 4:
            feature = F.adaptive_avg_pool2d(feature, 1).flatten(start_dim=1)
        elif feature.ndim == 3:
            feature = feature[:, 0]
        elif feature.ndim != 2:
            feature = feature.flatten(start_dim=1)
        return feature.flatten(start_dim=1)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = (self.image_key, self.wrench_key) + self.low_dim_keys
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(f"Missing structured observation keys: {missing}")

        image = _latest(observation[self.image_key], event_ndim=3, key=self.image_key)
        image = _as_float_image(image)
        if self.image_transform is not None:
            image = self.image_transform(image)
        image = self.normalizer.normalize_field(self.image_key, image)
        image_feature = self._pool_image(self.image_encoder(image))
        if image_feature.shape[-1] != self.image_feature_dim:
            raise ValueError(
                f"Image encoder produced {image_feature.shape[-1]} features, "
                f"expected {self.image_feature_dim}"
            )

        low_dim_features = []
        for key, expected_dim in zip(self.low_dim_keys, self.low_dim_dims):
            value = _latest(observation[key], event_ndim=1, key=key)
            value = value.to(dtype=torch.float32)
            value = self.normalizer.normalize_field(key, value)
            if value.shape[-1] != expected_dim:
                raise ValueError(
                    f"{key} has dimension {value.shape[-1]}, expected {expected_dim}"
                )
            low_dim_features.append(value)

        wrench = _latest(observation[self.wrench_key], event_ndim=2, key=self.wrench_key)
        wrench = wrench.to(dtype=torch.float32)
        wrench = self.normalizer.normalize_field(self.wrench_key, wrench)
        wrench_feature = self.wrench_encoder(wrench).flatten(start_dim=1)
        if wrench_feature.shape[-1] != self.wrench_feature_dim:
            raise ValueError(
                f"Wrench encoder produced {wrench_feature.shape[-1]} features, "
                f"expected {self.wrench_feature_dim}"
            )

        batch_sizes = {
            image_feature.shape[0],
            wrench_feature.shape[0],
            *(value.shape[0] for value in low_dim_features),
        }
        if len(batch_sizes) != 1:
            raise ValueError(f"Observation batch sizes do not match: {batch_sizes}")
        return torch.cat([image_feature, *low_dim_features, wrench_feature], dim=-1)


class ResidualActor(nn.Module):
    """Deterministic actor producing a bounded physical residual delta6."""

    def __init__(
        self,
        state_encoder: StructuredObservationEncoder,
        residual_min: Sequence[float],
        residual_max: Sequence[float],
        base_action_dim: int = 16,
        residual_dim: int = 6,
        hidden_dims: Sequence[int] = (512, 256),
        squash_output: bool = True,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.base_action_dim = int(base_action_dim)
        self.residual_dim = int(residual_dim)
        if self.base_action_dim != 16:
            raise ValueError(
                "Residual RL base_action must be 16-D: relative EE pose9 + hand7"
            )
        minimum = torch.as_tensor(residual_min, dtype=torch.float32).flatten()
        maximum = torch.as_tensor(residual_max, dtype=torch.float32).flatten()
        if minimum.shape != (self.residual_dim,) or maximum.shape != (self.residual_dim,):
            raise ValueError(
                f"Residual bounds must each have shape ({self.residual_dim},)"
            )
        if not torch.all(minimum < maximum):
            raise ValueError("Every residual_min entry must be less than residual_max")
        self.register_buffer("residual_min", minimum)
        self.register_buffer("residual_max", maximum)
        self.squash_output = bool(squash_output)
        self.head = _mlp(
            input_dim=self.state_encoder.output_dim + self.base_action_dim,
            hidden_dims=hidden_dims,
            output_dim=self.residual_dim,
        )

    def _base(self, base_action: torch.Tensor) -> torch.Tensor:
        base_action = _latest(base_action, event_ndim=1, key="base_action")
        base_action = base_action.to(dtype=torch.float32)
        if base_action.shape[-1] != self.base_action_dim:
            raise ValueError(
                f"base_action has dimension {base_action.shape[-1]}, "
                f"expected {self.base_action_dim}"
            )
        return self.state_encoder.normalizer.normalize_field("base_action", base_action)

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.maximum(
            torch.minimum(action, self.residual_max),
            self.residual_min,
        )

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        state = self.state_encoder(observation)
        base = self._base(base_action)
        if state.shape[0] != base.shape[0]:
            raise ValueError("state and base_action batch sizes do not match")
        prediction = self.head(torch.cat([state, base], dim=-1))
        if self.squash_output:
            prediction = torch.tanh(prediction)
        if self.state_encoder.normalizer.has_field("residual"):
            prediction = self.state_encoder.normalizer.unnormalize_field(
                "residual",
                prediction,
            )
        elif self.squash_output:
            # Without BC residual statistics, parameterize directly inside the
            # explicit physical safety interval instead of relying on a
            # gradient-killing hard clamp of an arbitrary [-1, 1] output.
            center = 0.5 * (self.residual_min + self.residual_max)
            half_range = 0.5 * (self.residual_max - self.residual_min)
            prediction = center + half_range * prediction
        return self.clip_action(prediction)


class TwinQCritic(nn.Module):
    """Twin Q heads over ``state, base_action16, executed_residual6``."""

    def __init__(
        self,
        state_encoder: StructuredObservationEncoder,
        base_action_dim: int = 16,
        residual_dim: int = 6,
        hidden_dims: Sequence[int] = (512, 256),
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.base_action_dim = int(base_action_dim)
        self.residual_dim = int(residual_dim)
        if self.base_action_dim != 16:
            raise ValueError(
                "Residual RL critic base_action must be 16-D: relative EE pose9 + hand7"
            )
        input_dim = self.state_encoder.output_dim + self.base_action_dim + self.residual_dim
        self.q1_head = _mlp(input_dim, hidden_dims, 1)
        self.q2_head = _mlp(input_dim, hidden_dims, 1)

    def _features(
        self,
        observation: Mapping[str, torch.Tensor],
        base_action: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        state = self.state_encoder(observation)
        base_action = _latest(base_action, event_ndim=1, key="base_action")
        residual = _latest(residual, event_ndim=1, key="action")
        base_action = base_action.to(dtype=torch.float32)
        residual = residual.to(dtype=torch.float32)
        if base_action.shape[-1] != self.base_action_dim:
            raise ValueError(
                f"base_action has dimension {base_action.shape[-1]}, "
                f"expected {self.base_action_dim}"
            )
        if residual.shape[-1] != self.residual_dim:
            raise ValueError(
                f"residual has dimension {residual.shape[-1]}, "
                f"expected {self.residual_dim}"
            )
        base_action = self.state_encoder.normalizer.normalize_field(
            "base_action", base_action
        )
        residual = self.state_encoder.normalizer.normalize_field(
            "residual", residual
        )
        if not (state.shape[0] == base_action.shape[0] == residual.shape[0]):
            raise ValueError("critic input batch sizes do not match")
        return torch.cat([state, base_action, residual], dim=-1)

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        base_action: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(observation, base_action, residual)
        return self.q1_head(features), self.q2_head(features)

    def q1(
        self,
        observation: Mapping[str, torch.Tensor],
        base_action: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        return self.q1_head(self._features(observation, base_action, residual))
