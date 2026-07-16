"""Streaming statistics and normalizer fitting for residual BC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from diffusion_policy.residual_rl.normalizer import StructuredAffineNormalizer


def symmetric_residual_bounds(
    residual_statistics: Mapping[str, np.ndarray],
    *,
    scale: float,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-centred per-axis bounds from observed signed delta6 data."""

    scale = float(scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("residual bound scale must be finite and >= 1")
    observed_min = np.asarray(residual_statistics["min"], dtype=np.float32).reshape(-1)
    observed_max = np.asarray(residual_statistics["max"], dtype=np.float32).reshape(-1)
    if observed_min.shape != (6,) or observed_max.shape != (6,):
        raise ValueError("Residual statistics must contain delta6 min/max")
    max_abs = np.maximum(np.abs(observed_min), np.abs(observed_max))
    if not np.isfinite(max_abs).all() or np.any(max_abs <= eps):
        raise ValueError(
            "Cannot derive a physical residual interval from a zero/non-finite axis"
        )
    limit = (max_abs * scale).astype(np.float32)
    return -limit, limit


@dataclass
class _Moments:
    count: int = 0
    total: np.ndarray | None = None
    total_square: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def update(self, value: torch.Tensor) -> None:
        array = value.detach().cpu().numpy().astype(np.float64, copy=False)
        array = array.reshape(array.shape[0], -1)
        batch_total = array.sum(axis=0)
        batch_square = np.square(array).sum(axis=0)
        batch_min = array.min(axis=0)
        batch_max = array.max(axis=0)
        if self.total is None:
            self.total = batch_total
            self.total_square = batch_square
            self.minimum = batch_min
            self.maximum = batch_max
        else:
            self.total += batch_total
            self.total_square += batch_square
            self.minimum = np.minimum(self.minimum, batch_min)
            self.maximum = np.maximum(self.maximum, batch_max)
        self.count += int(array.shape[0])

    def result(self, eps: float) -> dict[str, np.ndarray]:
        if self.count <= 0 or self.total is None:
            raise RuntimeError("Cannot compute statistics from zero samples")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), eps)
        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
        }


def episode_split_indices(
    episode_lengths: Sequence[int],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes, never individual transitions, into train/val."""

    lengths = np.asarray(episode_lengths, dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths <= 0):
        raise ValueError("episode_lengths must be a non-empty positive vector")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    episode_ids = np.arange(len(lengths))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)
    n_val = int(round(len(lengths) * val_ratio))
    if val_ratio > 0 and n_val == 0 and len(lengths) > 1:
        n_val = 1
    val_episodes = set(episode_ids[:n_val].tolist())
    starts = np.concatenate(([0], np.cumsum(lengths[:-1])))
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    for episode, (start, length) in enumerate(zip(starts, lengths)):
        target = val_indices if episode in val_episodes else train_indices
        target.append(np.arange(start, start + length, dtype=np.int64))
    train = np.concatenate(train_indices) if train_indices else np.empty(0, dtype=np.int64)
    val = np.concatenate(val_indices) if val_indices else np.empty(0, dtype=np.int64)
    return train, val


def fit_residual_normalizer(
    dataset: Dataset,
    *,
    indices: Iterable[int] | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    eps: float = 1e-6,
    residual_min: Sequence[float] | None = None,
    residual_max: Sequence[float] | None = None,
    residual_bound_scale: float | None = None,
) -> tuple[StructuredAffineNormalizer, dict[str, dict[str, np.ndarray]]]:
    """Fit state/base standardization and residual range normalization.

    Residuals use an affine map into ``[-1, 1]`` based on explicit bounds, an
    optional zero-centred scaled demonstration envelope, or (legacy fallback)
    the raw data range. Other numeric inputs use mean/std standardization.
    Images remain in ``[0, 1]`` and are omitted.
    """

    source: Dataset
    if indices is None:
        source = dataset
    else:
        source = Subset(dataset, [int(index) for index in indices])
    loader = DataLoader(
        source,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
    )
    moment_keys = (
        "robot_pose_R",
        "robot_quat_R",
        "hand_pose_R",
        "wrench_wrist_R",
        "base_action",
        "residual",
    )
    moments = {key: _Moments() for key in moment_keys}
    for batch in tqdm(loader, desc="Fitting residual normalizer"):
        for key in moment_keys[:-2]:
            moments[key].update(batch["obs"][key])
        moments["base_action"].update(batch["base_action"])
        moments["residual"].update(batch["action"])

    statistics = {key: value.result(eps) for key, value in moments.items()}
    fields: dict[str, Mapping[str, np.ndarray]] = {}
    for key in moment_keys[:-1]:
        stats = statistics[key]
        fields[key] = {
            "scale": 1.0 / stats["std"],
            "offset": -stats["mean"] / stats["std"],
        }

    residual_stats = statistics["residual"]
    if (residual_min is None) != (residual_max is None):
        raise ValueError("residual_min and residual_max must be supplied together")
    if residual_min is not None and residual_bound_scale is not None:
        raise ValueError(
            "Use either explicit residual bounds or residual_bound_scale, not both"
        )
    if residual_bound_scale is not None:
        normalization_min, normalization_max = symmetric_residual_bounds(
            residual_stats,
            scale=residual_bound_scale,
            eps=eps,
        )
    elif residual_min is None:
        normalization_min = residual_stats["min"]
        normalization_max = residual_stats["max"]
    else:
        normalization_min = np.asarray(residual_min, dtype=np.float32).reshape(-1)
        normalization_max = np.asarray(residual_max, dtype=np.float32).reshape(-1)
        if normalization_min.shape != (6,) or normalization_max.shape != (6,):
            raise ValueError("Explicit residual bounds must each contain 6 values")
        if not np.isfinite(normalization_min).all() or not np.isfinite(normalization_max).all():
            raise ValueError("Explicit residual bounds contain NaN or Inf")
        if np.any(normalization_min >= normalization_max):
            raise ValueError("Every residual_min entry must be below residual_max")
        if np.any(residual_stats["min"] < normalization_min) or np.any(
            residual_stats["max"] > normalization_max
        ):
            raise ValueError(
                "Demonstration residuals exceed the configured physical residual bounds"
            )

    residual_range = normalization_max - normalization_min
    residual_range = np.maximum(residual_range, eps)
    fields["residual"] = {
        "scale": 2.0 / residual_range,
        "offset": -(normalization_max + normalization_min) / residual_range,
    }
    return StructuredAffineNormalizer(fields), statistics


def statistics_to_jsonable(
    statistics: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, list[float]]]:
    return {
        key: {
            stat_name: np.asarray(value).reshape(-1).astype(float).tolist()
            for stat_name, value in field.items()
        }
        for key, field in statistics.items()
    }


__all__ = [
    "episode_split_indices",
    "fit_residual_normalizer",
    "statistics_to_jsonable",
    "symmetric_residual_bounds",
]
