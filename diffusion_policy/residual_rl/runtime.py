"""Stateless collect/evaluate runtime for real-robot residual RL.

One slow/base inference produces an absolute action16 chunk.  Several fast
ticks consume that chunk, but every fast tick fetches a fresh observation and
recomputes the selected base EE target relative to the *current* pose.  The
fast actor is feed-forward: this runtime never supplies a previous residual,
GRU hidden state, or a fixed image context.

There is deliberately no optimizer in this module.  An actor checkpoint is
loaded before an episode, identified by ``checkpoint_id``, and guarded against
replacement until the episode is complete.
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.real_world.real_inference_util import (
    get_abs_action_from_relative,
    get_real_obs_dict,
    get_real_relative_obs_dict,
)
from diffusion_policy.residual_rl.base_policy import (
    FrozenBasePolicy,
    numpy_obs_to_batch,
    predict_base_action_chunk,
)
from diffusion_policy.residual_rl.pose import (
    abs_pose9_to_relative_pose9,
    apply_residual_action_to_pose9,
    current_obs_to_pose9,
)
from diffusion_policy.residual_rl.safety import (
    ResidualRobotSafety,
    SafetyLimits,
    SafetyViolation,
)


CANONICAL_OBSERVATION_KEYS = (
    "image0",
    "robot_pose_R",
    "robot_quat_R",
    "hand_pose_R",
    "wrench_wrist_R",
)
CANONICAL_OBSERVATION_SHAPES = {
    "image0": (3, 224, 224),
    "robot_pose_R": (3,),
    "robot_quat_R": (4,),
    "hand_pose_R": (7,),
    "wrench_wrist_R": (6, 32),
}


@runtime_checkable
class BasePolicyRunner(Protocol):
    """Frozen slow policy interface returning absolute action16 chunks."""

    @property
    def checkpoint_id(self) -> str:
        ...

    def predict_chunk(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        ...


@runtime_checkable
class ResidualPolicyRunner(Protocol):
    """Stateless fast actor interface used by the robot runtime."""

    @property
    def checkpoint_id(self) -> str:
        ...

    def predict(
        self,
        observation: Mapping[str, np.ndarray],
        base_action16: np.ndarray,
        *,
        exploration_std: float = 0.0,
    ) -> np.ndarray:
        ...


@runtime_checkable
class RobotEnvironment(Protocol):
    """Subset of the existing real environment required here."""

    def get_obs(self) -> dict[str, np.ndarray]:
        ...

    def exec_actions(self, actions: np.ndarray, timestamps: np.ndarray):
        ...

    def start_episode(self, start_time: float | None = None):
        ...

    def end_episode(self):
        ...


def _plain_config(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def _select_config(cfg, path: str, default=None):
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, path, default=default)
    current = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _ensure_wrench_time_dimension(
    observation: dict[str, np.ndarray],
    shape_meta: Mapping,
) -> dict[str, np.ndarray]:
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") != "wrench" or key not in observation:
            continue
        expected = tuple(attr["shape"])
        if tuple(observation[key].shape) == expected:
            observation[key] = observation[key][None]
    return observation


class FrozenBasePolicyRunner:
    """Adapt :class:`FrozenBasePolicy` to physical real-robot observations."""

    def __init__(
        self,
        loaded: FrozenBasePolicy,
        *,
        shape_meta: Mapping | None = None,
        observation_pose_representation: str | None = None,
        action_pose_representation: str | None = None,
        device: str | torch.device | None = None,
    ):
        self.loaded = loaded
        self.policy = loaded.policy
        if shape_meta is None:
            shape_meta = _select_config(loaded.cfg, "task.shape_meta")
        if shape_meta is None:
            raise ValueError("Base checkpoint has no task.shape_meta")
        self.shape_meta = _plain_config(shape_meta)
        self.observation_pose_representation = observation_pose_representation or str(
            getattr(
                self.policy,
                "obs_pose_repr",
                _select_config(loaded.cfg, "task.pose_repr.obs_pose_repr", "abs"),
            )
        )
        self.action_pose_representation = action_pose_representation or str(
            getattr(self.policy, "action_pose_repr", "abs")
        )
        if device is None:
            parameter = next(self.policy.parameters(), None)
            device = parameter.device if parameter is not None else "cpu"
        self.device = torch.device(device)
        self.policy.eval()
        self.policy.requires_grad_(False)

    @property
    def checkpoint_id(self) -> str:
        return f"{self.loaded.checkpoint_path}:{self.loaded.state_key}"

    def _policy_observation(
        self,
        environment_observation: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if self.observation_pose_representation == "relative":
            result = get_real_relative_obs_dict(
                env_obs=dict(environment_observation),
                shape_meta=self.shape_meta,
            )
        elif self.observation_pose_representation == "abs":
            result = get_real_obs_dict(
                env_obs=dict(environment_observation),
                shape_meta=self.shape_meta,
            )
        else:
            raise ValueError(
                "Unsupported base observation pose representation: "
                f"{self.observation_pose_representation}"
            )
        return _ensure_wrench_time_dimension(result, self.shape_meta)

    @torch.inference_mode()
    def predict_chunk(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        policy_observation = self._policy_observation(observation)
        tensor_observation = numpy_obs_to_batch(
            policy_observation,
            device=self.device,
        )
        chunk = predict_base_action_chunk(self.loaded, tensor_observation)
        chunk_np = chunk[0].detach().to("cpu").numpy().astype(np.float32)
        if self.action_pose_representation == "relative":
            chunk_np = get_abs_action_from_relative(
                action=chunk_np,
                env_obs=dict(observation),
            ).astype(np.float32)
        elif self.action_pose_representation != "abs":
            raise ValueError(
                "Base runtime needs abs or relative actions, got "
                f"{self.action_pose_representation!r}"
            )
        if chunk_np.ndim != 2 or chunk_np.shape[1] != 16:
            raise ValueError(
                f"Base policy must return an absolute [H,16] chunk, got {chunk_np.shape}"
            )
        if not np.all(np.isfinite(chunk_np)):
            raise SafetyViolation("Base policy produced NaN or Inf")
        return chunk_np


def _latest_actor_observation(
    environment_observation: Mapping[str, np.ndarray],
    shape_meta: Mapping,
) -> dict[str, np.ndarray]:
    """Preprocess only the newest fast observation; no fixed context/history."""

    processed = get_real_obs_dict(
        env_obs=dict(environment_observation),
        shape_meta=shape_meta,
    )
    latest: dict[str, np.ndarray] = {}
    for key, value in processed.items():
        attr = shape_meta["obs"][key]
        value = np.asarray(value)
        if attr.get("type", "low_dim") == "wrench":
            expected = tuple(attr["shape"])
            if tuple(value.shape) == expected:
                latest[key] = value
            else:
                latest[key] = value[-1]
        else:
            # The new actor is deliberately feed-forward.  Keep no temporal
            # dimension after selecting the freshest environment sample;
            # ``numpy_obs_to_batch`` below adds only the batch dimension.
            latest[key] = value[-1]
    return latest


class TorchResidualPolicyRunner:
    """Feed-forward PyTorch actor adapter with residual-only exploration."""

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        shape_meta: Mapping,
        checkpoint_id: str,
        device: str | torch.device,
        exploration_seed: int | None = None,
    ):
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required for episode immutability")
        self.actor = actor
        self.shape_meta = _plain_config(shape_meta)
        self._checkpoint_id = str(checkpoint_id)
        self.device = torch.device(device)
        self.generator = np.random.default_rng(exploration_seed)
        self.actor.to(self.device)
        self.actor.eval()
        self.actor.requires_grad_(False)

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

    @torch.inference_mode()
    def predict(
        self,
        observation: Mapping[str, np.ndarray],
        base_action16: np.ndarray,
        *,
        exploration_std: float = 0.0,
    ) -> np.ndarray:
        actor_observation = _latest_actor_observation(observation, self.shape_meta)
        tensor_observation = numpy_obs_to_batch(
            actor_observation,
            device=self.device,
        )
        base = torch.as_tensor(
            base_action16,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 16)
        result = self.actor(tensor_observation, base)
        if isinstance(result, Mapping):
            result = result["action"]
        residual = result.detach().to("cpu").numpy()
        residual = np.asarray(residual, dtype=np.float32).reshape(-1, 6)[-1]
        exploration_std = float(exploration_std)
        if exploration_std < 0.0 or not np.isfinite(exploration_std):
            raise ValueError("exploration_std must be finite and >= 0")
        if exploration_std > 0.0:
            residual = residual + self.generator.normal(
                loc=0.0,
                scale=exploration_std,
                size=(6,),
            ).astype(np.float32)
        return residual.astype(np.float32)


@dataclass(frozen=True)
class RuntimeConfig:
    frequency_hz: float = 10.0
    fast_steps_per_slow_inference: int = 6
    slow_action_start_index: int = 1
    command_latency_s: float = 0.01
    episode_start_delay_s: float = 1.0
    max_episode_duration_s: float = 60.0
    max_commanded_steps: int | None = None
    exploration_std: float = 0.0
    exploration_mode: str = "none"
    exploration_step_start: int = 0
    exploration_seed: int | None = None
    resfit_learning_starts: int = 10_000
    resfit_random_noise_scale: float = 0.2
    resfit_stddev: float = 0.025
    arm: str = "R"
    sidecar_observation_keys: tuple[str, ...] = CANONICAL_OBSERVATION_KEYS

    def __post_init__(self) -> None:
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and > 0")
        if self.fast_steps_per_slow_inference <= 0:
            raise ValueError("fast_steps_per_slow_inference must be > 0")
        if self.slow_action_start_index < 0:
            raise ValueError("slow_action_start_index must be >= 0")
        if not np.isfinite(self.command_latency_s) or self.command_latency_s < 0.0:
            raise ValueError("command_latency_s must be finite and >= 0")
        if not np.isfinite(self.episode_start_delay_s) or self.episode_start_delay_s < 0.0:
            raise ValueError("episode_start_delay_s must be finite and >= 0")
        if not np.isfinite(self.max_episode_duration_s) or self.max_episode_duration_s <= 0.0:
            raise ValueError("max_episode_duration_s must be finite and > 0")
        if self.max_commanded_steps is not None and self.max_commanded_steps <= 0:
            raise ValueError("max_commanded_steps must be > 0 when configured")
        if not np.isfinite(self.exploration_std) or self.exploration_std < 0.0:
            raise ValueError("exploration_std must be finite and >= 0")
        if self.exploration_mode not in ("none", "gaussian", "resfit"):
            raise ValueError("exploration_mode must be none, gaussian, or resfit")
        if self.exploration_mode == "gaussian" and self.exploration_std <= 0.0:
            raise ValueError("gaussian exploration requires exploration_std > 0")
        if self.exploration_step_start < 0:
            raise ValueError("exploration_step_start must be >= 0")
        if self.resfit_learning_starts < 0:
            raise ValueError("resfit_learning_starts must be >= 0")
        for name in ("resfit_random_noise_scale", "resfit_stddev"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.arm not in ("L", "R"):
            raise ValueError("arm must be 'L' or 'R'")
        if tuple(self.sidecar_observation_keys) != CANONICAL_OBSERVATION_KEYS:
            raise ValueError(
                "The first runtime version uses the canonical observation schema: "
                f"{CANONICAL_OBSERVATION_KEYS}"
            )


def build_base_condition16(
    current_observation: Mapping[str, np.ndarray],
    slow_absolute_action16: np.ndarray,
    *,
    arm: str = "R",
) -> np.ndarray:
    """Return current-pose-relative EE pose9 plus the slow hand target7."""

    target = np.asarray(slow_absolute_action16, dtype=np.float32)
    if target.shape != (16,) or not np.all(np.isfinite(target)):
        raise SafetyViolation(f"Slow target must be finite action16, got {target.shape}")
    current_pose9 = current_obs_to_pose9(current_observation, arm=arm)
    relative_pose9 = abs_pose9_to_relative_pose9(current_pose9, target[:9])
    condition = np.concatenate((relative_pose9, target[9:16]), axis=-1)
    if condition.shape != (16,) or not np.all(np.isfinite(condition)):
        raise SafetyViolation("Constructed base action16 contains NaN or Inf")
    return condition.astype(np.float32)


def compose_residual_command16(
    slow_absolute_action16: np.ndarray,
    residual_delta6: np.ndarray,
) -> np.ndarray:
    """Right-compose EE residual and preserve the slow hand target7."""

    result = apply_residual_action_to_pose9(
        slow_absolute_action16,
        residual_delta6,
    )
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (16,):
        raise SafetyViolation(f"Composed robot command must be action16, got {result.shape}")
    return result


def _observation_timestamp(observation: Mapping[str, np.ndarray]) -> float:
    value = np.asarray(observation["timestamp"], dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value[-1]):
        raise SafetyViolation("Observation timestamp is missing or non-finite")
    return float(value[-1])


def _guard_safety_call(context: str, function: Callable, *args, **kwargs):
    """Turn inference/robot API failures into one fail-closed episode abort."""

    try:
        return function(*args, **kwargs)
    except SafetyViolation:
        raise
    except Exception as error:
        raise SafetyViolation(
            f"{context} failed ({type(error).__name__}): {error}"
        ) from error


def _snapshot_observation(
    observation: Mapping[str, np.ndarray],
    keys: Sequence[str],
) -> dict[str, np.ndarray]:
    missing = [key for key in keys if key not in observation]
    if missing:
        raise SafetyViolation(f"Cannot record observation; missing keys: {missing}")
    snapshot: dict[str, np.ndarray] = {}
    for key in keys:
        expected = CANONICAL_OBSERVATION_SHAPES[key]
        value = np.asarray(observation[key])
        if key == "image0":
            if value.ndim == 4:
                value = value[-1]
            if value.ndim == 3 and value.shape[-1] == 3:
                value = np.moveaxis(value, -1, 0)
        elif value.ndim == len(expected) + 1:
            value = value[-1]
        if tuple(value.shape) != expected:
            raise SafetyViolation(
                f"Canonical observation {key!r} has shape {value.shape}, "
                f"expected {expected}"
            )
        if not np.all(np.isfinite(value)):
            raise SafetyViolation(f"Canonical observation {key!r} contains NaN or Inf")
        snapshot[key] = np.array(value, copy=True)
    return snapshot


def _image_to_uint8(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype == np.uint8:
        return image
    if not np.all(np.isfinite(image)):
        raise ValueError("Cannot serialize an image containing NaN/Inf")
    image = image.astype(np.float32)
    if image.size > 0 and float(np.max(image)) <= 1.0 and float(np.min(image)) >= 0.0:
        image = image * 255.0
    return np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)


@dataclass
class CanonicalEpisode:
    """Only transitions whose robot command was accepted for scheduling."""

    observation_keys: tuple[str, ...] = CANONICAL_OBSERVATION_KEYS
    observations: list[dict[str, np.ndarray]] = field(default_factory=list)
    base_actions: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    command_timestamps: list[float] = field(default_factory=list)
    observation_timestamps: list[float] = field(default_factory=list)

    def append_commanded_transition(
        self,
        *,
        observation: Mapping[str, np.ndarray],
        base_action16: np.ndarray,
        executed_residual6: np.ndarray,
        command_timestamp: float,
    ) -> None:
        self.observations.append(_snapshot_observation(observation, self.observation_keys))
        self.base_actions.append(np.asarray(base_action16, dtype=np.float32).copy())
        self.actions.append(np.asarray(executed_residual6, dtype=np.float32).copy())
        self.command_timestamps.append(float(command_timestamp))
        self.observation_timestamps.append(_observation_timestamp(observation))

    def finalize(
        self,
        *,
        success: int,
        terminal_observation: Mapping[str, np.ndarray] | None,
        terminal_base_action16: np.ndarray | None,
    ) -> dict:
        if success not in (0, 1):
            raise ValueError(f"success must be 0 or 1, got {success}")
        count = len(self.actions)
        if count == 0:
            raise ValueError("Cannot finalize an episode with no commanded transitions")

        terminal_snapshot = self.observations[-1]
        if terminal_observation is not None:
            try:
                terminal_snapshot = _snapshot_observation(
                    terminal_observation,
                    self.observation_keys,
                )
            except SafetyViolation:
                terminal_snapshot = self.observations[-1]
        terminal_base = self.base_actions[-1]
        if terminal_base_action16 is not None:
            candidate = np.asarray(terminal_base_action16, dtype=np.float32)
            if candidate.shape == (16,) and np.all(np.isfinite(candidate)):
                terminal_base = candidate

        next_observations = self.observations[1:] + [terminal_snapshot]
        next_base_actions = self.base_actions[1:] + [terminal_base]
        reward = np.zeros((count,), dtype=np.float32)
        reward[-1] = float(success)
        done = np.zeros((count,), dtype=np.bool_)
        done[-1] = True
        return {
            "obs": {
                key: np.stack([obs[key] for obs in self.observations], axis=0)
                for key in self.observation_keys
            },
            "next_obs": {
                key: np.stack([obs[key] for obs in next_observations], axis=0)
                for key in self.observation_keys
            },
            "base_action": np.stack(self.base_actions, axis=0).astype(np.float32),
            "next_base_action": np.stack(next_base_actions, axis=0).astype(np.float32),
            "action": np.stack(self.actions, axis=0).astype(np.float32),
            "reward": reward,
            "done": done,
            "command_timestamp": np.asarray(self.command_timestamps, dtype=np.float64),
            "observation_timestamp": np.asarray(
                self.observation_timestamps,
                dtype=np.float64,
            ),
        }


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


class EpisodeSidecarWriter:
    """Atomically write one canonical commanded episode per HDF5 file."""

    def __init__(self, output_directory: str | Path):
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write(self, episode: Mapping, metadata: Mapping) -> Path:
        unique = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        path = self.output_directory / f"episode_{unique}.hdf5"
        temporary = path.with_suffix(".hdf5.tmp")
        with h5py.File(temporary, "w") as output:
            output.attrs["schema"] = "residual_rl_commanded_episode_v1"
            output.attrs["metadata_json"] = json.dumps(
                dict(metadata),
                sort_keys=True,
                default=_json_default,
            )
            for group_name in ("obs", "next_obs"):
                group = output.create_group(group_name)
                for key, value in episode[group_name].items():
                    value = np.asarray(value)
                    if key.startswith("image"):
                        value = _image_to_uint8(value)
                    group.create_dataset(key, data=value, compression="gzip")
            for key in (
                "base_action",
                "next_base_action",
                "action",
                "reward",
                "done",
                "command_timestamp",
                "observation_timestamp",
            ):
                output.create_dataset(key, data=np.asarray(episode[key]), compression="gzip")
            output.flush()
        os.replace(temporary, path)
        return path


def prompt_success_label(prompt: str = "Episode success? [1/0]: ") -> int:
    """Block after command production stops until a valid human label arrives."""

    while True:
        answer = input(prompt).strip()
        if answer in ("0", "1"):
            return int(answer)
        print("Please enter exactly 1 for success or 0 for failure.")


@dataclass(frozen=True)
class EpisodeResult:
    success: int
    commanded_steps: int
    termination_reason: str
    actor_checkpoint_id: str
    base_checkpoint_id: str
    sidecar_path: Path | None
    safety_violation: str | None = None


class SlowFastResidualRuntime:
    """Execute frozen slow chunks plus a stateless fast residual actor."""

    def __init__(
        self,
        *,
        environment: RobotEnvironment,
        base_policy: BasePolicyRunner,
        residual_policy: ResidualPolicyRunner,
        safety: ResidualRobotSafety,
        config: RuntimeConfig,
        safe_stop: Callable[[str], None] | None = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not isinstance(safety.limits, SafetyLimits):
            raise TypeError("A validated, explicit SafetyLimits object is required")
        if not base_policy.checkpoint_id:
            raise ValueError("base_policy.checkpoint_id cannot be empty")
        if not residual_policy.checkpoint_id:
            raise ValueError("residual_policy.checkpoint_id cannot be empty")
        self.environment = environment
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.safety = safety
        self.config = config
        self.safe_stop = safe_stop
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self.sleep = sleep
        self._exploration_step = int(config.exploration_step_start)
        self._exploration_rng = np.random.default_rng(config.exploration_seed)

    def _resfit_exploration(self, residual: np.ndarray) -> np.ndarray:
        """Apply normalized ResFiT noise around the non-zero BC residual."""

        minimum = np.asarray(self.safety.limits.residual_min, dtype=np.float32)
        maximum = np.asarray(self.safety.limits.residual_max, dtype=np.float32)
        half_range = 0.5 * (maximum - minimum)
        if self._exploration_step < self.config.resfit_learning_starts:
            normalized_noise = self._exploration_rng.uniform(
                low=-self.config.resfit_random_noise_scale,
                high=self.config.resfit_random_noise_scale,
                size=(6,),
            )
        else:
            normalized_noise = self._exploration_rng.normal(
                loc=0.0,
                scale=self.config.resfit_stddev,
                size=(6,),
            )
        return (
            np.asarray(residual, dtype=np.float32)
            + normalized_noise.astype(np.float32) * half_range
        )

    def _assert_checkpoint_unchanged(self, episode_checkpoint_id: str) -> None:
        if self.residual_policy.checkpoint_id != episode_checkpoint_id:
            raise SafetyViolation(
                "Residual actor checkpoint changed during an episode: "
                f"{episode_checkpoint_id!r} -> {self.residual_policy.checkpoint_id!r}"
            )

    def _request_safe_stop(self, reason: str) -> None:
        if self.safe_stop is not None:
            self.safe_stop(reason)

    def _episode_limit_reached(self, start_monotonic: float, commanded_steps: int) -> str | None:
        if self.monotonic_clock() - start_monotonic >= self.config.max_episode_duration_s:
            return "max_duration"
        if (
            self.config.max_commanded_steps is not None
            and commanded_steps >= self.config.max_commanded_steps
        ):
            return "max_commanded_steps"
        return None

    def run_episode(
        self,
        *,
        label_success: Callable[[], int] = prompt_success_label,
        sidecar_writer: EpisodeSidecarWriter | None = None,
        stop_requested: Callable[[], bool] | None = None,
        exploration_std: float | None = None,
    ) -> EpisodeResult:
        """Run one episode with no policy update or checkpoint reload.

        ``stop_requested`` is polled between fast ticks and can be connected to
        a UI/key handler.  A safety violation aborts before the unsafe command,
        optionally calls ``safe_stop``, and still allows the human to label and
        retain all earlier commanded transitions.
        """

        legacy_exploration_override = exploration_std is not None
        if legacy_exploration_override:
            exploration_std = float(exploration_std)
            if exploration_std < 0.0 or not np.isfinite(exploration_std):
                raise ValueError("exploration_std must be finite and >= 0")
        elif self.config.exploration_mode == "gaussian":
            exploration_std = float(self.config.exploration_std)
        else:
            exploration_std = 0.0

        actor_checkpoint_id = str(self.residual_policy.checkpoint_id)
        base_checkpoint_id = str(self.base_policy.checkpoint_id)
        episode = CanonicalEpisode(self.config.sidecar_observation_keys)
        period = 1.0 / self.config.frequency_hz
        termination_reason = "unknown"
        safety_violation: str | None = None
        terminal_observation: Mapping[str, np.ndarray] | None = None
        terminal_base_action: np.ndarray | None = None
        last_slow_target: np.ndarray | None = None
        started = False

        slow_targets: np.ndarray | None = None
        slow_cursor = 0

        try:
            start_wall = self.wall_clock() + self.config.episode_start_delay_s
            _guard_safety_call(
                "Start robot episode",
                self.environment.start_episode,
                start_wall,
            )
            started = True
            delay = start_wall - self.wall_clock()
            if delay > 0.0:
                self.sleep(delay)
            start_monotonic = self.monotonic_clock()
            next_tick_monotonic = start_monotonic

            while True:
                limit_reason = self._episode_limit_reached(
                    start_monotonic,
                    len(episode.actions),
                )
                if limit_reason is not None:
                    termination_reason = limit_reason
                    break
                if stop_requested is not None and stop_requested():
                    termination_reason = "operator_stop"
                    break

                self._assert_checkpoint_unchanged(actor_checkpoint_id)
                if slow_targets is None or slow_cursor >= len(slow_targets):
                    slow_observation = _guard_safety_call(
                        "Read slow-policy observation",
                        self.environment.get_obs,
                    )
                    self.safety.check_observation(
                        slow_observation,
                        now=self.wall_clock(),
                    )
                    slow_chunk = _guard_safety_call(
                        "Frozen base-policy inference",
                        self.base_policy.predict_chunk,
                        slow_observation,
                    )
                    start = self.config.slow_action_start_index
                    stop = start + self.config.fast_steps_per_slow_inference
                    if slow_chunk.ndim != 2 or slow_chunk.shape[1] != 16:
                        raise SafetyViolation(
                            f"Base policy returned invalid action chunk {slow_chunk.shape}"
                        )
                    if stop > len(slow_chunk):
                        raise SafetyViolation(
                            "Base action chunk is too short for the requested fast steps: "
                            f"need [{start}:{stop}], got H={len(slow_chunk)}"
                        )
                    slow_targets = np.asarray(slow_chunk[start:stop], dtype=np.float32)
                    slow_cursor = 0

                slow_target = slow_targets[slow_cursor]
                latest_observation = _guard_safety_call(
                    "Read fast-policy observation",
                    self.environment.get_obs,
                )
                self.safety.check_observation(
                    latest_observation,
                    now=self.wall_clock(),
                )
                base_action16 = _guard_safety_call(
                    "Build current-relative base condition",
                    build_base_condition16,
                    latest_observation,
                    slow_target,
                    arm=self.config.arm,
                )
                base_action16 = self.safety.check_base_condition(base_action16)
                raw_residual = _guard_safety_call(
                    "Residual actor inference",
                    self.residual_policy.predict,
                    latest_observation,
                    base_action16,
                    exploration_std=exploration_std,
                )
                if (
                    not legacy_exploration_override
                    and self.config.exploration_mode == "resfit"
                ):
                    raw_residual = self._resfit_exploration(raw_residual)
                executed_residual = self.safety.clip_residual(raw_residual)
                final_command = _guard_safety_call(
                    "Compose residual robot command",
                    compose_residual_command16,
                    slow_target,
                    executed_residual,
                )
                final_command = self.safety.check_robot_command(final_command)

                # Inference can be slow or stall.  Revalidate the exact state
                # and checkpoint immediately before scheduling, rather than
                # relying only on the checks made before actor inference.
                self._assert_checkpoint_unchanged(actor_checkpoint_id)
                self.safety.check_observation(
                    latest_observation,
                    now=self.wall_clock(),
                )
                post_inference_limit = self._episode_limit_reached(
                    start_monotonic,
                    len(episode.actions),
                )
                if post_inference_limit is not None:
                    termination_reason = post_inference_limit
                    break

                now = self.wall_clock()
                action_timestamp = max(
                    now + self.config.command_latency_s,
                    _observation_timestamp(latest_observation) + period,
                )
                # Keep the existing environment's strict ``target_time > now``
                # assertion true even when command_latency_s is explicitly zero.
                if action_timestamp <= now:
                    action_timestamp = np.nextafter(now, np.inf)
                accepted = _guard_safety_call(
                    "Schedule robot command",
                    self.environment.exec_actions,
                    actions=final_command[None],
                    timestamps=np.asarray([action_timestamp], dtype=np.float64),
                )
                if accepted is not False:
                    episode.append_commanded_transition(
                        observation=latest_observation,
                        base_action16=base_action16,
                        executed_residual6=executed_residual,
                        command_timestamp=action_timestamp,
                    )
                    terminal_observation = latest_observation
                    terminal_base_action = base_action16
                    last_slow_target = slow_target
                    slow_cursor += 1
                    self._exploration_step += 1

                next_tick_monotonic = max(
                    next_tick_monotonic + period,
                    self.monotonic_clock(),
                )
                wait = next_tick_monotonic - self.monotonic_clock()
                if wait > 0.0:
                    self.sleep(wait)

        except KeyboardInterrupt:
            termination_reason = "keyboard_interrupt"
        except SafetyViolation as error:
            termination_reason = "safety_violation"
            safety_violation = str(error)
            self._request_safe_stop(safety_violation)
        finally:
            # Command production has stopped before recording is ended or the
            # blocking human success prompt is shown.
            if len(episode.actions) > 0:
                try:
                    candidate = self.environment.get_obs()
                    terminal_observation = candidate
                    if last_slow_target is not None:
                        terminal_base_action = build_base_condition16(
                            candidate,
                            last_slow_target,
                            arm=self.config.arm,
                        )
                except Exception:
                    # Terminal transitions are done=True; keep the last valid
                    # commanded state if a final observation cannot be read.
                    pass
            if started:
                try:
                    self.environment.end_episode()
                except Exception as error:
                    if safety_violation is None:
                        termination_reason = "environment_end_error"
                        safety_violation = (
                            "End robot episode failed "
                            f"({type(error).__name__}): {error}"
                        )
                        self._request_safe_stop(safety_violation)

        success = int(label_success())
        if success not in (0, 1):
            raise ValueError(f"Human episode label must be 0 or 1, got {success}")

        sidecar_path: Path | None = None
        if len(episode.actions) > 0 and sidecar_writer is not None:
            finalized = episode.finalize(
                success=success,
                terminal_observation=terminal_observation,
                terminal_base_action16=terminal_base_action,
            )
            metadata = {
                "success": success,
                "terminal_reward_only": True,
                "termination_reason": termination_reason,
                "safety_violation": safety_violation,
                "actor_checkpoint_id": actor_checkpoint_id,
                "base_checkpoint_id": base_checkpoint_id,
                "runtime_config": asdict(self.config),
                "safety_limits": asdict(self.safety.limits),
                "commanded_steps": len(episode.actions),
                "exploration_mode": (
                    "gaussian_override"
                    if legacy_exploration_override
                    else self.config.exploration_mode
                ),
                "exploration_std": exploration_std,
                "exploration_step_end": self._exploration_step,
            }
            sidecar_path = sidecar_writer.write(finalized, metadata)

        return EpisodeResult(
            success=success,
            commanded_steps=len(episode.actions),
            termination_reason=termination_reason,
            actor_checkpoint_id=actor_checkpoint_id,
            base_checkpoint_id=base_checkpoint_id,
            sidecar_path=sidecar_path,
            safety_violation=safety_violation,
        )


def merge_real_environment_shape_meta(
    base_shape_meta: Mapping,
    actor_shape_meta: Mapping,
) -> dict:
    """Union slow/fast observation requirements and force action16 commands."""

    base = _plain_config(base_shape_meta)
    actor = _plain_config(actor_shape_meta)
    merged = {"obs": copy.deepcopy(base["obs"]), "action": {"shape": [16]}}
    for key, attr in actor["obs"].items():
        if key in merged["obs"]:
            existing = merged["obs"][key]
            if tuple(existing["shape"]) != tuple(attr["shape"]):
                raise ValueError(f"Conflicting shape metadata for observation {key!r}")
            if existing.get("type", "low_dim") != attr.get("type", "low_dim"):
                raise ValueError(f"Conflicting type metadata for observation {key!r}")
            # Keep the base horizon/history metadata. The fast actor selects
            # only the latest element from the environment-provided history.
            continue
        merged["obs"][key] = copy.deepcopy(attr)
    return merged


__all__ = [
    "BasePolicyRunner",
    "CANONICAL_OBSERVATION_KEYS",
    "CANONICAL_OBSERVATION_SHAPES",
    "CanonicalEpisode",
    "EpisodeResult",
    "EpisodeSidecarWriter",
    "FrozenBasePolicyRunner",
    "ResidualPolicyRunner",
    "RobotEnvironment",
    "RuntimeConfig",
    "SlowFastResidualRuntime",
    "TorchResidualPolicyRunner",
    "build_base_condition16",
    "compose_residual_command16",
    "merge_real_environment_shape_meta",
    "prompt_success_label",
]
