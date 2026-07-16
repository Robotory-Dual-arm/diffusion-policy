"""Self-describing BC/TD3 checkpoints and inference-only actor loading."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diffusion_policy.residual_rl.models import (
    ResidualActor,
    StructuredObservationEncoder,
    TwinQCritic,
)
from diffusion_policy.residual_rl.normalizer import StructuredAffineNormalizer


CHECKPOINT_FORMAT = "diffusion_policy.residual_rl"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class ResidualModelConfig:
    image_feature_dim: int = 128
    wrench_feature_dim: int = 64
    actor_hidden_dims: tuple[int, ...] = (512, 256)
    critic_hidden_dims: tuple[int, ...] = (512, 256)
    base_action_dim: int = 16
    residual_dim: int = 6
    residual_min: tuple[float, ...] = ()
    residual_max: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.image_feature_dim <= 0 or self.wrench_feature_dim <= 0:
            raise ValueError("Encoder feature dimensions must be positive")
        if self.base_action_dim != 16 or self.residual_dim != 6:
            raise ValueError("This runtime requires base_action_dim=16 and residual_dim=6")
        if len(self.residual_min) != 6 or len(self.residual_max) != 6:
            raise ValueError("residual_min and residual_max must be explicitly set to 6 values")
        minimum = np.asarray(self.residual_min, dtype=np.float32)
        maximum = np.asarray(self.residual_max, dtype=np.float32)
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("Residual model bounds contain NaN or Inf")
        if np.any(minimum >= maximum):
            raise ValueError("Every residual_min entry must be smaller than residual_max")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResidualModelConfig":
        data = dict(value)
        for key in (
            "actor_hidden_dims",
            "critic_hidden_dims",
            "residual_min",
            "residual_max",
        ):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


def normalizer_spec(
    normalizer: StructuredAffineNormalizer,
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for key, field in normalizer.fields.items():
        result[key] = {
            "scale": field.scale.detach().cpu().reshape(-1).tolist(),
            "offset": field.offset.detach().cpu().reshape(-1).tolist(),
        }
    return result


def build_normalizer(
    spec: Mapping[str, Mapping[str, Sequence[float]]],
) -> StructuredAffineNormalizer:
    return StructuredAffineNormalizer(spec)


def build_actor(
    config: ResidualModelConfig,
    normalizer: StructuredAffineNormalizer,
    *,
    freeze_image_encoder: bool,
    freeze_wrench_encoder: bool = False,
) -> ResidualActor:
    encoder = StructuredObservationEncoder(
        image_feature_dim=config.image_feature_dim,
        wrench_feature_dim=config.wrench_feature_dim,
        freeze_image_encoder=freeze_image_encoder,
        freeze_wrench_encoder=freeze_wrench_encoder,
        normalizer=normalizer,
    )
    return ResidualActor(
        state_encoder=encoder,
        base_action_dim=config.base_action_dim,
        residual_dim=config.residual_dim,
        hidden_dims=config.actor_hidden_dims,
        residual_min=config.residual_min,
        residual_max=config.residual_max,
    )


def build_critics_from_actor(
    actor: ResidualActor,
    config: ResidualModelConfig,
    *,
    freeze_image_encoder: bool = True,
    freeze_wrench_encoder: bool = False,
) -> TwinQCritic:
    encoder = copy.deepcopy(actor.state_encoder)
    encoder.set_encoder_freeze(
        image=freeze_image_encoder,
        wrench=freeze_wrench_encoder,
    )
    return TwinQCritic(
        state_encoder=encoder,
        base_action_dim=config.base_action_dim,
        residual_dim=config.residual_dim,
        hidden_dims=config.critic_hidden_dims,
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def save_bc_checkpoint(
    path: str | Path,
    *,
    actor: ResidualActor,
    model_config: ResidualModelConfig,
    step: int,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    statistics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    path = Path(path).expanduser().resolve()
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "kind": "bc",
        "model_config": asdict(model_config),
        "normalizer": normalizer_spec(actor.state_encoder.normalizer),
        "actor": actor.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "statistics": dict(statistics or {}),
        "metadata": dict(metadata or {}),
    }
    _atomic_torch_save(payload, path)
    return str(path)


def save_td3_checkpoint(
    path: str | Path,
    *,
    td3,
    model_config: ResidualModelConfig,
    metadata: Mapping[str, Any] | None = None,
    replay_state: Mapping[str, Any] | None = None,
) -> str:
    path = Path(path).expanduser().resolve()
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "kind": "td3",
        "model_config": asdict(model_config),
        "normalizer": normalizer_spec(td3.actor.state_encoder.normalizer),
        "td3": td3.checkpoint_state(extra=metadata),
        "replay": None if replay_state is None else dict(replay_state),
        "metadata": dict(metadata or {}),
    }
    _atomic_torch_save(payload, path)
    return str(path)


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location=map_location)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a residual_rl checkpoint: {path}")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported residual_rl checkpoint version in {path}")
    if payload.get("kind") not in {"bc", "td3"}:
        raise ValueError(f"Unknown residual_rl checkpoint kind in {path}")
    return payload


def load_bc_actor(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    freeze_image_encoder: bool = False,
    freeze_wrench_encoder: bool = False,
) -> tuple[ResidualActor, ResidualModelConfig, dict[str, Any]]:
    payload = load_checkpoint_payload(path, map_location="cpu")
    if payload["kind"] != "bc":
        raise ValueError(f"Expected BC checkpoint, got {payload['kind']!r}")
    config = ResidualModelConfig.from_mapping(payload["model_config"])
    actor = build_actor(
        config,
        build_normalizer(payload["normalizer"]),
        freeze_image_encoder=freeze_image_encoder,
        freeze_wrench_encoder=freeze_wrench_encoder,
    )
    actor.load_state_dict(payload["actor"], strict=True)
    actor.to(torch.device(device))
    return actor, config, payload


class TorchResidualPolicyRunner:
    """Numpy-facing, optimizer-free policy object used by collect/evaluate."""

    observation_shapes = {
        "image0": (3, 224, 224),
        "robot_pose_R": (3,),
        "robot_quat_R": (4,),
        "hand_pose_R": (7,),
        "wrench_wrist_R": (6, 32),
    }
    shape_meta = {
        "obs": {
            "image0": {"shape": [3, 224, 224], "type": "rgb"},
            "robot_pose_R": {"shape": [3], "type": "low_dim"},
            "robot_quat_R": {"shape": [4], "type": "low_dim"},
            "hand_pose_R": {"shape": [7], "type": "low_dim"},
            "wrench_wrist_R": {"shape": [6, 32], "type": "wrench"},
        },
        "action": {"shape": [6]},
    }

    def __init__(
        self,
        actor: ResidualActor,
        *,
        device: str | torch.device,
        checkpoint_id: str,
        exploration_seed: int | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        checkpoint_kind: str | None = None,
    ):
        self.actor = actor.to(torch.device(device)).eval()
        self.actor.requires_grad_(False)
        self.device = torch.device(device)
        self.checkpoint_id = str(checkpoint_id)
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.checkpoint_kind = checkpoint_kind
        self.rng = np.random.default_rng(exploration_seed)

    def _observation_tensors(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, expected_shape in self.observation_shapes.items():
            if key not in observation:
                raise KeyError(f"Actor observation missing {key!r}")
            value = np.asarray(observation[key])
            if key == "image0":
                if value.ndim == 4:
                    value = value[-1]
                if value.ndim == 3 and value.shape[-1] == 3:
                    value = np.moveaxis(value, -1, 0)
            # Runtime may provide a short time dimension; residual actor uses latest.
            if tuple(value.shape) != expected_shape:
                if value.ndim == len(expected_shape) + 1 and tuple(value.shape[1:]) == expected_shape:
                    value = value[-1]
                else:
                    raise ValueError(
                        f"{key} has shape {value.shape}, expected {expected_shape}"
                    )
            tensor = torch.from_numpy(np.array(value, copy=True)).unsqueeze(0)
            result[key] = tensor.to(device=self.device)
        return result

    @torch.no_grad()
    def predict(
        self,
        obs_dict_np: Mapping[str, np.ndarray],
        base_action16: np.ndarray,
        exploration_std: float | Sequence[float] = 0.0,
    ) -> np.ndarray:
        obs = self._observation_tensors(obs_dict_np)
        base = torch.as_tensor(
            base_action16,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 16)
        action = self.actor(obs, base)[0].detach().cpu().numpy()
        std = np.asarray(exploration_std, dtype=np.float32)
        if np.any(std < 0) or std.size not in (1, 6):
            raise ValueError("exploration_std must be non-negative scalar or 6-vector")
        if np.any(std > 0):
            action = action + self.rng.normal(size=6).astype(np.float32) * std
        minimum = self.actor.residual_min.detach().cpu().numpy()
        maximum = self.actor.residual_max.detach().cpu().numpy()
        action = np.clip(action, minimum, maximum).astype(np.float32)
        if not np.isfinite(action).all():
            raise FloatingPointError("Residual actor produced NaN or Inf")
        return action


def load_actor_for_inference(
    path: str | Path,
    *,
    device: str | torch.device = "cuda",
    exploration_seed: int | None = None,
) -> TorchResidualPolicyRunner:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    config = ResidualModelConfig.from_mapping(payload["model_config"])
    actor = build_actor(
        config,
        build_normalizer(payload["normalizer"]),
        freeze_image_encoder=True,
        freeze_wrench_encoder=True,
    )
    actor_state = payload["actor"] if payload["kind"] == "bc" else payload["td3"]["actor"]
    actor.load_state_dict(actor_state, strict=True)
    return TorchResidualPolicyRunner(
        actor,
        device=device,
        checkpoint_id=str(checkpoint_path),
        exploration_seed=exploration_seed,
        checkpoint_metadata=payload.get("metadata", {}),
        checkpoint_kind=str(payload["kind"]),
    )


__all__ = [
    "ResidualModelConfig",
    "TorchResidualPolicyRunner",
    "build_actor",
    "build_critics_from_actor",
    "load_actor_for_inference",
    "load_bc_actor",
    "load_checkpoint_payload",
    "save_bc_checkpoint",
    "save_td3_checkpoint",
]
