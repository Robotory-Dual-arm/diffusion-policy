"""Fail-closed safety checks for real-robot residual execution.

The repository does not currently define authoritative residual or wrist
force/torque limits.  This module therefore has no permissive numeric defaults:
all limits must be supplied explicitly before a runtime can send a command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class SafetyConfigurationError(ValueError):
    """Raised before execution when a required safety limit is missing."""


class SafetyViolation(RuntimeError):
    """Raised for one unsafe observation, policy output, or robot command."""


def _required_positive(name: str, value: float | None) -> float:
    if value is None:
        raise SafetyConfigurationError(
            f"{name} must be explicitly configured; no repository default exists"
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise SafetyConfigurationError(f"{name} must be finite and > 0, got {value}")
    return value


def _required_residual_bound(
    name: str,
    value: Sequence[float] | np.ndarray | None,
) -> tuple[float, ...]:
    if value is None:
        raise SafetyConfigurationError(
            f"{name} must be explicitly configured; no repository default exists"
        )
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (6,):
        raise SafetyConfigurationError(f"{name} must contain 6 values, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise SafetyConfigurationError(f"{name} contains NaN or Inf")
    return tuple(float(x) for x in array)


def latest_wrench6(value: np.ndarray | Sequence[float]) -> np.ndarray:
    """Extract the newest ``[fx, fy, fz, tx, ty, tz]`` sample.

    Supported real-environment layouts are ``[6]``, ``[T, 6]``, ``[6, H]``,
    and ``[T, 6, H]``.  Ambiguous or unsupported layouts fail closed.
    """

    wrench = np.asarray(value, dtype=np.float64)
    if wrench.ndim == 1 and wrench.shape == (6,):
        result = wrench
    elif wrench.ndim == 2:
        if wrench.shape[0] == 6 and wrench.shape[1] != 6:
            result = wrench[:, -1]
        elif wrench.shape[1] == 6 and wrench.shape[0] != 6:
            result = wrench[-1]
        elif wrench.shape == (6, 6):
            raise SafetyViolation("Ambiguous wrist wrench shape (6, 6)")
        else:
            raise SafetyViolation(f"Unsupported wrist wrench shape {wrench.shape}")
    elif wrench.ndim == 3 and wrench.shape[-2] == 6:
        result = wrench[-1, :, -1]
    else:
        raise SafetyViolation(f"Unsupported wrist wrench shape {wrench.shape}")
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise SafetyViolation("Latest wrist wrench is missing or contains NaN/Inf")
    return result.astype(np.float32)


@dataclass(frozen=True)
class SafetyLimits:
    """Explicit limits required by :class:`ResidualRobotSafety`.

    Residual limits are component-wise in the existing delta6 convention:
    local-frame XYZ translation in meters followed by a rotation vector in
    radians.  Force and torque limits are Euclidean norms in N and Nm.
    """

    residual_min: Sequence[float] | np.ndarray | None = None
    residual_max: Sequence[float] | np.ndarray | None = None
    max_force_norm_n: float | None = None
    max_torque_norm_nm: float | None = None
    max_observation_age_s: float | None = None
    wrench_key: str = "wrench_wrist_R"
    timestamp_key: str = "timestamp"
    robot_timestamp_key: str = "robot_receive_timestamp"
    reject_all_zero_wrench_history: bool = True
    required_observation_keys: tuple[str, ...] = (
        "image0",
        "robot_pose_R",
        "robot_quat_R",
        "hand_pose_R",
        "wrench_wrist_R",
    )

    def __post_init__(self) -> None:
        minimum = _required_residual_bound("residual_min", self.residual_min)
        maximum = _required_residual_bound("residual_max", self.residual_max)
        if np.any(np.asarray(minimum) >= np.asarray(maximum)):
            raise SafetyConfigurationError(
                "Every residual_min component must be less than residual_max"
            )
        object.__setattr__(self, "residual_min", minimum)
        object.__setattr__(self, "residual_max", maximum)
        object.__setattr__(
            self,
            "max_force_norm_n",
            _required_positive("max_force_norm_n", self.max_force_norm_n),
        )
        object.__setattr__(
            self,
            "max_torque_norm_nm",
            _required_positive("max_torque_norm_nm", self.max_torque_norm_nm),
        )
        object.__setattr__(
            self,
            "max_observation_age_s",
            _required_positive("max_observation_age_s", self.max_observation_age_s),
        )
        if not self.wrench_key or not self.timestamp_key or not self.robot_timestamp_key:
            raise SafetyConfigurationError(
                "wrench_key, timestamp_key, and robot_timestamp_key cannot be empty"
            )
        if self.wrench_key not in self.required_observation_keys:
            raise SafetyConfigurationError(
                f"required_observation_keys must contain {self.wrench_key!r}"
            )


class ResidualRobotSafety:
    """Validate every fast tick and clip residuals to explicit bounds."""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self._residual_min = np.asarray(limits.residual_min, dtype=np.float32)
        self._residual_max = np.asarray(limits.residual_max, dtype=np.float32)

    @staticmethod
    def require_finite(name: str, value, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
        array = np.asarray(value)
        if shape is not None and array.shape != shape:
            raise SafetyViolation(f"{name} must have shape {shape}, got {array.shape}")
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise SafetyViolation(f"{name} is empty or contains NaN/Inf")
        return array

    def check_observation(self, observation: Mapping[str, np.ndarray], *, now: float) -> float:
        missing = [
            key for key in self.limits.required_observation_keys if key not in observation
        ]
        if missing:
            raise SafetyViolation(f"Observation is missing required keys: {missing}")
        ages = []
        for timestamp_key, source_name in (
            (self.limits.timestamp_key, "camera observation"),
            (self.limits.robot_timestamp_key, "robot observation"),
        ):
            if timestamp_key not in observation:
                raise SafetyViolation(
                    f"Observation has no {timestamp_key!r} freshness timestamp"
                )
            timestamps = self.require_finite(
                timestamp_key,
                observation[timestamp_key],
            ).reshape(-1)
            age = float(now) - float(timestamps[-1])
            if age > float(self.limits.max_observation_age_s):
                raise SafetyViolation(
                    f"{source_name.capitalize()} age {age:.6f}s exceeds "
                    f"{self.limits.max_observation_age_s:.6f}s"
                )
            ages.append(age)

        for key in self.limits.required_observation_keys:
            self.require_finite(key, observation[key])

        wrench_history = np.asarray(observation[self.limits.wrench_key])
        if (
            self.limits.reject_all_zero_wrench_history
            and not np.any(wrench_history != 0.0)
        ):
            raise SafetyViolation(
                "Wrist wrench history is entirely zero; the F/T sensor may be "
                "missing or not calibrated"
            )
        wrench = latest_wrench6(wrench_history)
        force_norm = float(np.linalg.norm(wrench[:3]))
        torque_norm = float(np.linalg.norm(wrench[3:]))
        if force_norm > float(self.limits.max_force_norm_n):
            raise SafetyViolation(
                f"Force norm {force_norm:.6f}N exceeds "
                f"{self.limits.max_force_norm_n:.6f}N"
            )
        if torque_norm > float(self.limits.max_torque_norm_nm):
            raise SafetyViolation(
                f"Torque norm {torque_norm:.6f}Nm exceeds "
                f"{self.limits.max_torque_norm_nm:.6f}Nm"
            )
        return max(ages)

    def clip_residual(self, residual) -> np.ndarray:
        residual = self.require_finite(
            "residual action",
            residual,
            shape=(6,),
        ).astype(np.float32, copy=False)
        clipped = np.clip(residual, self._residual_min, self._residual_max)
        if not np.all(np.isfinite(clipped)):
            raise SafetyViolation("Clipped residual contains NaN or Inf")
        return clipped.astype(np.float32, copy=False)

    def check_base_condition(self, base_action) -> np.ndarray:
        return self.require_finite(
            "base action condition",
            base_action,
            shape=(16,),
        ).astype(np.float32, copy=False)

    def check_robot_command(self, command) -> np.ndarray:
        return self.require_finite(
            "robot action command",
            command,
            shape=(16,),
        ).astype(np.float64, copy=False)


__all__ = [
    "ResidualRobotSafety",
    "SafetyConfigurationError",
    "SafetyLimits",
    "SafetyViolation",
    "latest_wrench6",
]
