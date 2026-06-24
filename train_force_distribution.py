#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import random
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset


DEFAULT_HDF5 = "/home/baetae/Downloads/common_data_height.hdf5"
DEFAULT_OUTPUT_DIR = "force_distribution_height/outputs"
DEFAULT_POLICY_CHECKPOINT = (
    "/home/baetae/diffusion-policy/data/outputs/2026.05.19/"
    "21.39.57_train_diffusion_unet_hybrid_bbbae_dualarm_erase_board_wrench_encoder/"
    "checkpoints/epoch=0900-train_loss=0.000.ckpt"
)
WRENCH_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
DEFAULT_DATALOADER_PREFETCH_FACTOR = 4
DEFAULT_DATALOADER_PERSISTENT_WORKERS = True


@dataclass
class SampleIndex:
    hdf5_path: str
    demo_names: list[str]
    demo_idx: np.ndarray
    frame_idx: np.ndarray
    phase: np.ndarray
    target_mean: np.ndarray
    target_var: np.ndarray
    tcp: np.ndarray | None
    image_key: str
    wrench_key: str
    force_dims: tuple[int, ...]
    window_sec: float
    wrench_zero_samples: int
    wrench_ema_alpha: float
    label_mode: str = "window"
    image_force_align: str = "center-window"
    phase_bins: int = 0


def demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        tail = name.split("_")[-1]
        if tail.isdigit():
            return int(tail), name
    return 10**9, name


def parse_dims(text: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        parts = text.replace(",", " ").split()
        dims = tuple(int(part) for part in parts)
    else:
        dims = tuple(int(part) for part in text)
    if not dims:
        raise ValueError("At least one force dimension is required.")
    bad = [dim for dim in dims if dim < 0 or dim >= 6]
    if bad:
        raise ValueError(f"Force dimensions must be in [0, 5], got {bad}.")
    return dims


def dim_names(force_dims: Iterable[int]) -> list[str]:
    return [WRENCH_NAMES[int(dim)] for dim in force_dims]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_threads(torch_threads: int, torch_interop_threads: int) -> None:
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    if torch_interop_threads > 0:
        try:
            torch.set_num_interop_threads(torch_interop_threads)
        except RuntimeError as exc:
            print(f"WARNING: could not set torch interop threads: {exc}")


def preprocess_wrench_series(
    raw_wrench: np.ndarray,
    force_dims: tuple[int, ...],
    wrench_ts: np.ndarray,
    zero_wrench_start_sec: float,
    wrench_zero_samples: int,
    wrench_ema_alpha: float,
) -> np.ndarray:
    wrench = np.asarray(raw_wrench, dtype=np.float32)[:, force_dims]
    baseline = np.zeros(len(force_dims), dtype=np.float32)

    if wrench_zero_samples > 0:
        count = min(int(wrench_zero_samples), len(wrench))
        if count > 0:
            baseline = wrench[:count].mean(axis=0)
    elif zero_wrench_start_sec > 0:
        cutoff_time = wrench_ts[0] + zero_wrench_start_sec
        cutoff = int(np.searchsorted(wrench_ts, cutoff_time, side="right"))
        cutoff = max(cutoff, 1)
        baseline = wrench[:cutoff].mean(axis=0)

    wrench = wrench - baseline

    if wrench_ema_alpha > 0:
        alpha = float(wrench_ema_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"--wrench-ema-alpha must be in (0, 1], got {alpha}.")
        filtered = np.empty_like(wrench)
        filtered[0] = wrench[0]
        for i in range(1, len(wrench)):
            filtered[i] = alpha * wrench[i] + (1.0 - alpha) * filtered[i - 1]
        wrench = filtered

    return wrench.astype(np.float32, copy=False)


def wrench_interval_for_image(
    robot_ts: np.ndarray,
    wrench_ts: np.ndarray,
    frame: int,
    window_sec: float,
    image_force_align: str,
) -> tuple[int, int]:
    center = float(robot_ts[int(frame)])
    if image_force_align == "nearest":
        nearest = int(np.searchsorted(wrench_ts, center, side="left"))
        if nearest >= len(wrench_ts):
            nearest = len(wrench_ts) - 1
        elif nearest > 0:
            prev_dist = abs(float(wrench_ts[nearest - 1]) - center)
            next_dist = abs(float(wrench_ts[nearest]) - center)
            if prev_dist <= next_dist:
                nearest -= 1
        return nearest, nearest + 1
    if image_force_align == "center-window":
        half_window = window_sec / 2.0
        lo_t = center - half_window
        hi_t = center + half_window
    elif image_force_align == "image-to-next":
        lo_t = center
        if frame + 1 < len(robot_ts):
            hi_t = float(robot_ts[int(frame) + 1])
        else:
            hi_t = center + window_sec
    elif image_force_align == "prev-to-image":
        hi_t = center
        if frame > 0:
            lo_t = float(robot_ts[int(frame) - 1])
        else:
            lo_t = center - window_sec
    else:
        raise ValueError(
            "--image-force-align must be one of center-window, "
            "image-to-next, prev-to-image, nearest."
        )

    lo = int(np.searchsorted(wrench_ts, lo_t, side="left"))
    hi = int(np.searchsorted(wrench_ts, hi_t, side="right"))
    if hi <= lo:
        nearest = int(np.searchsorted(wrench_ts, center, side="left"))
        lo = max(0, nearest - 1)
        hi = min(len(wrench_ts), nearest + 1)
    return lo, hi


def build_sample_index(
    hdf5_path: str,
    image_key: str,
    wrench_key: str,
    force_dims: tuple[int, ...],
    window_sec: float,
    target_std_floor: float,
    min_window_samples: int,
    zero_wrench_start_sec: float,
    wrench_zero_samples: int,
    wrench_ema_alpha: float,
    image_force_align: str,
    max_demos: int | None,
    limit_frames_per_demo: int | None,
) -> SampleIndex:
    hdf5_path = str(Path(hdf5_path).expanduser())
    demo_idx: list[int] = []
    frame_idx: list[int] = []
    phase_values: list[float] = []
    target_mean: list[np.ndarray] = []
    target_var: list[np.ndarray] = []
    kept_demos: list[str] = []
    half_window = window_sec / 2.0
    min_var = float(target_std_floor) ** 2

    with h5py.File(hdf5_path, "r") as h5:
        if "data" not in h5:
            raise KeyError("Expected a top-level /data group in the HDF5 file.")
        demos = sorted(h5["data"].keys(), key=demo_sort_key)
        if max_demos is not None:
            demos = demos[:max_demos]

        for demo_name in demos:
            obs = h5["data"][demo_name]["observations"]
            required = [image_key, wrench_key, "timestamp_robot", "timestamp_wrench"]
            missing = [key for key in required if key not in obs]
            if missing:
                raise KeyError(f"{demo_name} is missing observation keys: {missing}")

            image_count = int(obs[image_key].shape[0])
            robot_ts = np.asarray(obs["timestamp_robot"][:], dtype=np.float64)
            wrench_ts = np.asarray(obs["timestamp_wrench"][:], dtype=np.float64)
            if image_count != len(robot_ts):
                raise ValueError(
                    f"{demo_name}: {image_key} has {image_count} frames but "
                    f"timestamp_robot has {len(robot_ts)} entries."
                )

            if limit_frames_per_demo is None or limit_frames_per_demo >= image_count:
                selected_frames = np.arange(image_count, dtype=np.int64)
            else:
                selected_frames = np.unique(
                    np.linspace(
                        0,
                        image_count - 1,
                        int(limit_frames_per_demo),
                        dtype=np.int64,
                    )
                )

            wrench_series = preprocess_wrench_series(
                raw_wrench=obs[wrench_key][:],
                force_dims=force_dims,
                wrench_ts=wrench_ts,
                zero_wrench_start_sec=zero_wrench_start_sec,
                wrench_zero_samples=wrench_zero_samples,
                wrench_ema_alpha=wrench_ema_alpha,
            )

            current_demo_idx = len(kept_demos)
            frame_denom = max(image_count - 1, 1)
            added = 0
            for frame in selected_frames:
                lo, hi = wrench_interval_for_image(
                    robot_ts=robot_ts,
                    wrench_ts=wrench_ts,
                    frame=int(frame),
                    window_sec=window_sec,
                    image_force_align=image_force_align,
                )
                if hi - lo < min_window_samples:
                    continue

                wrench = wrench_series[lo:hi]
                mu = wrench.mean(axis=0)
                var = wrench.var(axis=0)
                var = np.maximum(var, min_var)

                demo_idx.append(current_demo_idx)
                frame_idx.append(int(frame))
                phase_values.append(float(frame) / float(frame_denom))
                target_mean.append(mu.astype(np.float32))
                target_var.append(var.astype(np.float32))
                added += 1

            if added > 0:
                kept_demos.append(demo_name)
            else:
                # Remove samples referencing this demo if none were added.
                demo_idx = [idx for idx in demo_idx if idx != current_demo_idx]

    if not target_mean:
        raise RuntimeError("No training samples were built. Check timestamps and keys.")

    return SampleIndex(
        hdf5_path=hdf5_path,
        demo_names=kept_demos,
        demo_idx=np.asarray(demo_idx, dtype=np.int64),
        frame_idx=np.asarray(frame_idx, dtype=np.int64),
        phase=np.asarray(phase_values, dtype=np.float32),
        target_mean=np.stack(target_mean).astype(np.float32),
        target_var=np.stack(target_var).astype(np.float32),
        tcp=None,
        image_key=image_key,
        wrench_key=wrench_key,
        force_dims=force_dims,
        window_sec=window_sec,
        wrench_zero_samples=wrench_zero_samples,
        wrench_ema_alpha=wrench_ema_alpha,
        label_mode="window",
        image_force_align=image_force_align,
        phase_bins=0,
    )


def build_single_force_sample_index(
    hdf5_path: str,
    image_key: str,
    wrench_key: str,
    force_dims: tuple[int, ...],
    window_sec: float,
    min_window_samples: int,
    zero_wrench_start_sec: float,
    wrench_zero_samples: int,
    wrench_ema_alpha: float,
    image_force_align: str,
    max_demos: int | None,
    limit_frames_per_demo: int | None,
) -> SampleIndex:
    hdf5_path = str(Path(hdf5_path).expanduser())
    demo_idx: list[int] = []
    frame_idx: list[int] = []
    phase_values: list[float] = []
    target_mean: list[np.ndarray] = []
    target_var: list[np.ndarray] = []
    kept_demos: list[str] = []

    with h5py.File(hdf5_path, "r") as h5:
        if "data" not in h5:
            raise KeyError("Expected a top-level /data group in the HDF5 file.")
        demos = sorted(h5["data"].keys(), key=demo_sort_key)
        if max_demos is not None:
            demos = demos[:max_demos]

        for demo_name in demos:
            obs = h5["data"][demo_name]["observations"]
            required = [image_key, wrench_key, "timestamp_robot", "timestamp_wrench"]
            missing = [key for key in required if key not in obs]
            if missing:
                raise KeyError(f"{demo_name} is missing observation keys: {missing}")

            image_count = int(obs[image_key].shape[0])
            robot_ts = np.asarray(obs["timestamp_robot"][:], dtype=np.float64)
            wrench_ts = np.asarray(obs["timestamp_wrench"][:], dtype=np.float64)
            if image_count != len(robot_ts):
                raise ValueError(
                    f"{demo_name}: {image_key} has {image_count} frames but "
                    f"timestamp_robot has {len(robot_ts)} entries."
                )

            if limit_frames_per_demo is None or limit_frames_per_demo >= image_count:
                selected_frames = np.arange(image_count, dtype=np.int64)
            else:
                selected_frames = np.unique(
                    np.linspace(
                        0,
                        image_count - 1,
                        int(limit_frames_per_demo),
                        dtype=np.int64,
                    )
                )

            wrench_series = preprocess_wrench_series(
                raw_wrench=obs[wrench_key][:],
                force_dims=force_dims,
                wrench_ts=wrench_ts,
                zero_wrench_start_sec=zero_wrench_start_sec,
                wrench_zero_samples=wrench_zero_samples,
                wrench_ema_alpha=wrench_ema_alpha,
            )

            current_demo_idx = len(kept_demos)
            frame_denom = max(image_count - 1, 1)
            added = 0
            for frame in selected_frames:
                lo, hi = wrench_interval_for_image(
                    robot_ts=robot_ts,
                    wrench_ts=wrench_ts,
                    frame=int(frame),
                    window_sec=window_sec,
                    image_force_align=image_force_align,
                )
                if hi - lo < min_window_samples:
                    continue

                force = wrench_series[lo:hi].mean(axis=0).astype(np.float32)
                demo_idx.append(current_demo_idx)
                frame_idx.append(int(frame))
                phase_values.append(float(frame) / float(frame_denom))
                target_mean.append(force)
                target_var.append(np.zeros(len(force_dims), dtype=np.float32))
                added += 1

            if added > 0:
                kept_demos.append(demo_name)
            else:
                demo_idx = [idx for idx in demo_idx if idx != current_demo_idx]

    if not target_mean:
        raise RuntimeError("No training samples were built. Check timestamps and keys.")

    return SampleIndex(
        hdf5_path=hdf5_path,
        demo_names=kept_demos,
        demo_idx=np.asarray(demo_idx, dtype=np.int64),
        frame_idx=np.asarray(frame_idx, dtype=np.int64),
        phase=np.asarray(phase_values, dtype=np.float32),
        target_mean=np.stack(target_mean).astype(np.float32),
        target_var=np.stack(target_var).astype(np.float32),
        tcp=None,
        image_key=image_key,
        wrench_key=wrench_key,
        force_dims=force_dims,
        window_sec=window_sec,
        wrench_zero_samples=wrench_zero_samples,
        wrench_ema_alpha=wrench_ema_alpha,
        label_mode="sample",
        image_force_align=image_force_align,
        phase_bins=0,
    )


def build_phase_sample_index(
    hdf5_path: str,
    image_key: str,
    wrench_key: str,
    force_dims: tuple[int, ...],
    window_sec: float,
    target_std_floor: float,
    min_window_samples: int,
    zero_wrench_start_sec: float,
    wrench_zero_samples: int,
    wrench_ema_alpha: float,
    image_force_align: str,
    phase_bins: int,
    phase_source: str,
    max_demos: int | None,
    limit_frames_per_demo: int | None,
) -> SampleIndex:
    if phase_bins < 2:
        raise ValueError("--phase-bins must be at least 2 for phase label mode.")
    if phase_source not in {"frame", "time"}:
        raise ValueError("--phase-source must be frame or time.")

    hdf5_path = str(Path(hdf5_path).expanduser())
    min_var = float(target_std_floor) ** 2
    records: list[tuple[int, int, float, np.ndarray]] = []
    kept_demos: list[str] = []

    with h5py.File(hdf5_path, "r") as h5:
        if "data" not in h5:
            raise KeyError("Expected a top-level /data group in the HDF5 file.")
        demos = sorted(h5["data"].keys(), key=demo_sort_key)
        if max_demos is not None:
            demos = demos[:max_demos]

        for demo_name in demos:
            obs = h5["data"][demo_name]["observations"]
            required = [image_key, wrench_key, "timestamp_robot", "timestamp_wrench"]
            missing = [key for key in required if key not in obs]
            if missing:
                raise KeyError(f"{demo_name} is missing observation keys: {missing}")

            image_count = int(obs[image_key].shape[0])
            robot_ts = np.asarray(obs["timestamp_robot"][:], dtype=np.float64)
            wrench_ts = np.asarray(obs["timestamp_wrench"][:], dtype=np.float64)
            if image_count != len(robot_ts):
                raise ValueError(
                    f"{demo_name}: {image_key} has {image_count} frames but "
                    f"timestamp_robot has {len(robot_ts)} entries."
                )

            if limit_frames_per_demo is None or limit_frames_per_demo >= image_count:
                selected_frames = np.arange(image_count, dtype=np.int64)
            else:
                selected_frames = np.unique(
                    np.linspace(
                        0,
                        image_count - 1,
                        int(limit_frames_per_demo),
                        dtype=np.int64,
                    )
                )

            wrench_series = preprocess_wrench_series(
                raw_wrench=obs[wrench_key][:],
                force_dims=force_dims,
                wrench_ts=wrench_ts,
                zero_wrench_start_sec=zero_wrench_start_sec,
                wrench_zero_samples=wrench_zero_samples,
                wrench_ema_alpha=wrench_ema_alpha,
            )

            current_demo_idx = len(kept_demos)
            added = 0
            time_denom = max(float(robot_ts[-1] - robot_ts[0]), 1e-9)
            frame_denom = max(image_count - 1, 1)
            for frame in selected_frames:
                frame = int(frame)
                lo, hi = wrench_interval_for_image(
                    robot_ts=robot_ts,
                    wrench_ts=wrench_ts,
                    frame=frame,
                    window_sec=window_sec,
                    image_force_align=image_force_align,
                )
                if hi - lo < min_window_samples:
                    continue

                if phase_source == "time":
                    phase = float((robot_ts[frame] - robot_ts[0]) / time_denom)
                else:
                    phase = float(frame / frame_denom)
                phase = min(max(phase, 0.0), 1.0)
                mu = wrench_series[lo:hi].mean(axis=0).astype(np.float32)
                records.append((current_demo_idx, frame, phase, mu))
                added += 1

            if added > 0:
                kept_demos.append(demo_name)
            else:
                records = [record for record in records if record[0] != current_demo_idx]

    if not records:
        raise RuntimeError("No training samples were built. Check timestamps and keys.")

    bin_values: list[list[np.ndarray]] = [[] for _ in range(phase_bins)]
    record_bins = []
    for _, _, phase, mu in records:
        bin_idx = min(int(phase * phase_bins), phase_bins - 1)
        record_bins.append(bin_idx)
        bin_values[bin_idx].append(mu)

    global_values = np.stack([record[3] for record in records]).astype(np.float32)
    global_mean = global_values.mean(axis=0)
    global_var = np.maximum(global_values.var(axis=0), min_var)
    bin_mean = np.zeros((phase_bins, len(force_dims)), dtype=np.float32)
    bin_var = np.zeros((phase_bins, len(force_dims)), dtype=np.float32)
    for bin_idx, values in enumerate(bin_values):
        if len(values) == 0:
            bin_mean[bin_idx] = global_mean
            bin_var[bin_idx] = global_var
            continue
        stacked = np.stack(values).astype(np.float32)
        bin_mean[bin_idx] = stacked.mean(axis=0)
        bin_var[bin_idx] = np.maximum(stacked.var(axis=0), min_var)

    demo_idx = []
    frame_idx = []
    phase_values = []
    target_mean = []
    target_var = []
    for record, bin_idx in zip(records, record_bins):
        demo_idx.append(record[0])
        frame_idx.append(record[1])
        phase_values.append(record[2])
        target_mean.append(bin_mean[bin_idx])
        target_var.append(bin_var[bin_idx])

    return SampleIndex(
        hdf5_path=hdf5_path,
        demo_names=kept_demos,
        demo_idx=np.asarray(demo_idx, dtype=np.int64),
        frame_idx=np.asarray(frame_idx, dtype=np.int64),
        phase=np.asarray(phase_values, dtype=np.float32),
        target_mean=np.stack(target_mean).astype(np.float32),
        target_var=np.stack(target_var).astype(np.float32),
        tcp=None,
        image_key=image_key,
        wrench_key=wrench_key,
        force_dims=force_dims,
        window_sec=window_sec,
        wrench_zero_samples=wrench_zero_samples,
        wrench_ema_alpha=wrench_ema_alpha,
        label_mode="phase",
        image_force_align=image_force_align,
        phase_bins=phase_bins,
    )


def build_index_from_args(args: argparse.Namespace) -> SampleIndex:
    force_dims = parse_dims(args.force_dims)
    label_mode = getattr(args, "label_mode", "window")
    common_kwargs = dict(
        hdf5_path=args.hdf5,
        image_key=args.image_key,
        wrench_key=args.wrench_key,
        force_dims=force_dims,
        window_sec=args.window_sec,
        target_std_floor=args.target_std_floor,
        min_window_samples=args.min_window_samples,
        zero_wrench_start_sec=args.zero_wrench_start_sec,
        wrench_zero_samples=args.wrench_zero_samples,
        wrench_ema_alpha=args.wrench_ema_alpha,
        image_force_align=args.image_force_align,
        max_demos=args.max_demos,
        limit_frames_per_demo=args.limit_frames_per_demo,
    )
    if label_mode == "window":
        index = build_sample_index(**common_kwargs)
        return maybe_attach_tcp_from_args(index, args)
    if label_mode == "sample":
        sample_kwargs = common_kwargs.copy()
        sample_kwargs.pop("target_std_floor")
        index = build_single_force_sample_index(**sample_kwargs)
        return maybe_attach_tcp_from_args(index, args)
    if label_mode == "phase":
        index = build_phase_sample_index(
            **common_kwargs,
            phase_bins=args.phase_bins,
            phase_source=args.phase_source,
        )
        return maybe_attach_tcp_from_args(index, args)
    raise ValueError("--label-mode must be window, sample, or phase.")


_TCP_ROBOT = None


def get_tcp_robot():
    global _TCP_ROBOT
    if _TCP_ROBOT is None:
        import roboticstoolbox as rtb

        urdf_path = Path(__file__).resolve().parent / "m0609.white.urdf"
        _TCP_ROBOT = rtb.ERobot.URDF(str(urdf_path))
    return _TCP_ROBOT


def joint_to_tcp_features(joints: np.ndarray, tcp_pose_dim: int = 6) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    batched = joints.ndim > 1
    robot = get_tcp_robot()
    tcp = robot.fkine(joints)
    pos = np.asarray(tcp.t, dtype=np.float32)
    if tcp_pose_dim == 3:
        features = pos.astype(np.float32)
        return np.atleast_2d(features) if batched else features
    if tcp_pose_dim == 6:
        rotmat = np.asarray(tcp.R, dtype=np.float32)
        if rotmat.ndim == 2:
            rot6d = np.concatenate([rotmat[:, 0], rotmat[:, 1]], axis=0)
        else:
            rot6d = np.concatenate([rotmat[:, :, 0], rotmat[:, :, 1]], axis=1)
        features = np.concatenate([pos, rot6d], axis=-1).astype(np.float32)
        return np.atleast_2d(features) if batched else features
    raise ValueError("tcp_pose_dim must be 3 or 6.")


def joint_to_tcp_rotmats(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    batched = joints.ndim > 1
    robot = get_tcp_robot()
    tcp = robot.fkine(joints)
    rotmat = np.asarray(tcp.R, dtype=np.float32)
    if rotmat.ndim == 2:
        rotmat = rotmat[None, :, :]
    return rotmat if batched else rotmat[0]


def rotate_force_local_to_world(force: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32)
    rotmat = np.asarray(rotmat, dtype=np.float32)
    if force.ndim == 1:
        return (rotmat @ force).astype(np.float32)
    return np.einsum("nij,nj->ni", rotmat, force).astype(np.float32)


SQRT2_OVER_2 = float(np.sqrt(2.0) / 2.0)
RIGHT_BASE_TO_WORLD_ROT = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, -SQRT2_OVER_2, -SQRT2_OVER_2],
        [0.0, SQRT2_OVER_2, -SQRT2_OVER_2],
    ],
    dtype=np.float32,
)


def rotate_right_base_force_to_world(force: np.ndarray) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32)
    if force.ndim == 1:
        return (RIGHT_BASE_TO_WORLD_ROT @ force).astype(np.float32)
    return (RIGHT_BASE_TO_WORLD_ROT @ force.T).T.astype(np.float32)


def rotate_force_to_world(
    force: np.ndarray,
    tcp_rotmat_base: np.ndarray | None,
    wrench_frame: str,
) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32)
    if wrench_frame == "base":
        force_base = force
    elif wrench_frame == "tcp":
        if tcp_rotmat_base is None:
            raise ValueError("tcp_rotmat_base is required for wrench_frame='tcp'.")
        force_base = rotate_force_local_to_world(force, tcp_rotmat_base)
    elif wrench_frame == "tcp-inverse":
        if tcp_rotmat_base is None:
            raise ValueError("tcp_rotmat_base is required for wrench_frame='tcp-inverse'.")
        force_base = np.einsum("nji,nj->ni", tcp_rotmat_base, force).astype(np.float32)
    else:
        raise ValueError("--wrench-frame must be base, tcp, or tcp-inverse.")
    return rotate_right_base_force_to_world(force_base)


def tcp_feature_size(tcp_pose_dim: int = 6) -> int:
    if tcp_pose_dim == 3:
        return 3
    if tcp_pose_dim == 6:
        return 9
    raise ValueError("tcp_pose_dim must be 3 or 6.")


def tcp_history_feature_size(
    tcp_pose_dim: int = 6,
    tcp_history_steps: int = 1,
    tcp_input_mode: str = "absolute",
) -> int:
    steps = max(1, int(tcp_history_steps))
    base = tcp_feature_size(tcp_pose_dim)
    if tcp_input_mode == "absolute":
        return base * steps
    if tcp_input_mode == "delta":
        if steps < 2:
            raise ValueError("--tcp-input-mode delta requires --tcp-history-steps >= 2.")
        return base * (steps - 1)
    if tcp_input_mode == "current-delta-xyz":
        if steps < 2:
            raise ValueError(
                "--tcp-input-mode current-delta-xyz requires --tcp-history-steps >= 2."
            )
        return base + 3
    raise ValueError("--tcp-input-mode must be absolute, delta, or current-delta-xyz.")


def tcp_history_frames(
    frames: np.ndarray,
    tcp_history_steps: int = 1,
    tcp_history_stride: int = 1,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.int64)
    steps = max(1, int(tcp_history_steps))
    stride = max(1, int(tcp_history_stride))
    offsets = np.arange(steps - 1, -1, -1, dtype=np.int64) * stride
    return np.maximum(frames[:, None] - offsets[None, :], 0)


def joints_to_tcp_history_features(
    joints: np.ndarray,
    tcp_pose_dim: int = 6,
    tcp_history_steps: int = 1,
    tcp_input_mode: str = "absolute",
) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    steps = max(1, int(tcp_history_steps))
    if joints.ndim != 3:
        raise ValueError(
            "joints_to_tcp_history_features expects shape "
            "(batch, tcp_history_steps, joint_dim)."
        )
    if joints.shape[1] != steps:
        raise ValueError(f"Expected {steps} TCP history steps, got {joints.shape[1]}.")
    batch = joints.shape[0]
    tcp = joint_to_tcp_features(
        joints.reshape(batch * steps, joints.shape[-1]),
        tcp_pose_dim=tcp_pose_dim,
    )
    base = tcp_feature_size(tcp_pose_dim)
    tcp = tcp.reshape(batch, steps, base)
    if tcp_input_mode == "absolute":
        return tcp.reshape(batch, steps * base).astype(np.float32)
    if tcp_input_mode == "delta":
        if steps < 2:
            raise ValueError("--tcp-input-mode delta requires --tcp-history-steps >= 2.")
        delta = tcp[:, 1:, :] - tcp[:, :-1, :]
        return delta.reshape(batch, (steps - 1) * base).astype(np.float32)
    if tcp_input_mode == "current-delta-xyz":
        if steps < 2:
            raise ValueError(
                "--tcp-input-mode current-delta-xyz requires --tcp-history-steps >= 2."
            )
        current = tcp[:, -1, :]
        delta_xyz = tcp[:, -1, :3] - tcp[:, 0, :3]
        return np.concatenate([current, delta_xyz], axis=-1).astype(np.float32)
    raise ValueError("--tcp-input-mode must be absolute, delta, or current-delta-xyz.")


def read_hdf5_rows_unordered(dataset, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    flat = indices.reshape(-1)
    unique, inverse = np.unique(flat, return_inverse=True)
    values = np.asarray(dataset[unique])
    return values[inverse].reshape(*indices.shape, *values.shape[1:])


def attach_tcp_to_index(
    index: SampleIndex,
    joint_key: str,
    tcp_pose_dim: int = 6,
    tcp_history_steps: int = 1,
    tcp_history_stride: int = 1,
    tcp_input_mode: str = "absolute",
) -> SampleIndex:
    tcp = np.zeros(
        (
            len(index.frame_idx),
            tcp_history_feature_size(
                tcp_pose_dim,
                tcp_history_steps,
                tcp_input_mode=tcp_input_mode,
            ),
        ),
        dtype=np.float32,
    )
    with h5py.File(index.hdf5_path, "r") as h5:
        for demo_id, demo_name in enumerate(tqdm(index.demo_names, desc="tcp", dynamic_ncols=True)):
            sample_ids = np.nonzero(index.demo_idx == demo_id)[0]
            if len(sample_ids) == 0:
                continue
            obs = h5["data"][demo_name]["observations"]
            if joint_key not in obs:
                raise KeyError(f"{demo_name} is missing observation key {joint_key!r}.")
            frames = index.frame_idx[sample_ids].astype(np.int64)
            history_frames = tcp_history_frames(
                frames,
                tcp_history_steps=tcp_history_steps,
                tcp_history_stride=tcp_history_stride,
            )
            joints = read_hdf5_rows_unordered(
                obs[joint_key],
                history_frames,
            ).astype(np.float64)
            tcp[sample_ids] = joints_to_tcp_history_features(
                joints,
                tcp_pose_dim=tcp_pose_dim,
                tcp_history_steps=tcp_history_steps,
                tcp_input_mode=tcp_input_mode,
            )
    index.tcp = tcp
    return index


def maybe_attach_tcp_from_args(index: SampleIndex, args: argparse.Namespace) -> SampleIndex:
    if not getattr(args, "use_tcp", False):
        return index
    if args.tcp_history_steps < 1:
        raise ValueError("--tcp-history-steps must be >= 1.")
    if args.tcp_history_stride < 1:
        raise ValueError("--tcp-history-stride must be >= 1.")
    return attach_tcp_to_index(
        index=index,
        joint_key=args.tcp_joint_key,
        tcp_pose_dim=6,
        tcp_history_steps=args.tcp_history_steps,
        tcp_history_stride=args.tcp_history_stride,
        tcp_input_mode=args.tcp_input_mode,
    )


def split_by_demo(
    index: SampleIndex,
    val_ratio: float,
    val_demo_count: int | None,
    val_demos: str | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    demo_ids = np.arange(len(index.demo_names), dtype=np.int64)
    if val_demos:
        requested = {
            text if text.startswith("demo_") else f"demo_{text}"
            for text in val_demos.replace(",", " ").split()
            if text
        }
        name_to_id = {name: i for i, name in enumerate(index.demo_names)}
        missing = sorted(requested - set(name_to_id))
        if missing:
            raise ValueError(f"--val-demos not found in dataset: {missing}")
        val_demo_ids = np.asarray(
            [name_to_id[name] for name in sorted(requested, key=demo_sort_key)],
            dtype=np.int64,
        )
    elif val_demo_count is not None and val_demo_count > 0:
        if len(demo_ids) < 2:
            val_demo_ids = np.asarray([], dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            shuffled = demo_ids.copy()
            rng.shuffle(shuffled)
            count = min(int(val_demo_count), len(demo_ids) - 1)
            val_demo_ids = shuffled[:count]
    elif val_ratio <= 0 or len(demo_ids) < 2:
        val_demo_ids = np.asarray([], dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        shuffled = demo_ids.copy()
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(demo_ids) * val_ratio)))
        val_count = min(val_count, len(demo_ids) - 1)
        val_demo_ids = shuffled[:val_count]

    val_set = set(int(idx) for idx in val_demo_ids)
    val_mask = np.asarray([int(idx) in val_set for idx in index.demo_idx], dtype=bool)
    train_sample_ids = np.nonzero(~val_mask)[0]
    val_sample_ids = np.nonzero(val_mask)[0]

    train_demo_names = [
        name for i, name in enumerate(index.demo_names) if i not in val_set
    ]
    val_demo_names = [
        name for i, name in enumerate(index.demo_names) if i in val_set
    ]
    return train_sample_ids, val_sample_ids, train_demo_names, val_demo_names


def resize_center_crop_image(
    image: np.ndarray,
    image_size: int,
    bgr_to_rgb: bool,
) -> np.ndarray:
    if image_size <= 0:
        if bgr_to_rgb:
            image = image[..., ::-1]
        return np.ascontiguousarray(image)

    ih, iw = image.shape[:2]
    ow = oh = int(image_size)
    interp_method = cv2.INTER_AREA

    if (iw / ih) >= (ow / oh):
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR

    image = cv2.resize(image, (rw, rh), interpolation=interp_method)
    h0 = (rh - oh) // 2
    w0 = (rw - ow) // 2
    image = image[h0 : h0 + oh, w0 : w0 + ow]
    if bgr_to_rgb:
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def gaussian_blur_tensor_cv2(tensor: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        return tensor
    kernel_size = max(3, int(math.ceil(float(radius) * 6.0)) | 1)
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=float(radius),
        sigmaY=float(radius),
        borderType=cv2.BORDER_REFLECT_101,
    )
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).to(dtype=tensor.dtype)


def preprocess_image(
    image: np.ndarray,
    image_size: int,
    augment: bool,
    crop_ratio: float,
    bgr_to_rgb: bool,
    color_jitter: bool,
    grayscale_p: float,
    post_blur_radius: float = 0.0,
    lowpass_size: int = 0,
) -> torch.Tensor:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = resize_center_crop_image(
        image=image,
        image_size=image_size,
        bgr_to_rgb=bgr_to_rgb,
    )
    image = image.transpose(2, 0, 1)
    tensor = torch.from_numpy(image).float().div(255.0)

    if image_size > 0 and crop_ratio > 0 and crop_ratio < 1:
        crop_size = max(1, int(image_size * crop_ratio))
        if augment:
            tensor = T.RandomCrop(size=crop_size)(tensor)
        else:
            tensor = T.CenterCrop(size=crop_size)(tensor)
        tensor = T.Resize(size=image_size, antialias=True)(tensor)

    if augment and color_jitter:
        tensor = T.ColorJitter(
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            hue=0.08,
        )(tensor)
        tensor = T.RandomGrayscale(p=grayscale_p)(tensor)

    if lowpass_size > 0 and image_size > 0:
        small = max(2, int(lowpass_size))
        tensor = T.Resize(size=small, antialias=True)(tensor)
        tensor = T.Resize(size=image_size, antialias=True)(tensor)

    if post_blur_radius > 0:
        tensor = gaussian_blur_tensor_cv2(tensor, float(post_blur_radius))

    return tensor.mul(2.0).sub(1.0)


class Hdf5ForceDataset(Dataset):
    def __init__(
        self,
        index: SampleIndex,
        sample_ids: np.ndarray,
        force_mean: np.ndarray,
        force_std: np.ndarray,
        tcp_mean: np.ndarray | None,
        tcp_std: np.ndarray | None,
        image_size: int,
        augment: bool,
        image_crop_ratio: float,
        bgr_to_rgb: bool,
        color_jitter: bool,
        grayscale_p: float,
        post_blur_radius: float,
        lowpass_size: int,
        return_phase: bool,
    ) -> None:
        self.index = index
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.force_mean = force_mean.astype(np.float32)
        self.force_std = force_std.astype(np.float32)
        self.tcp_mean = None if tcp_mean is None else tcp_mean.astype(np.float32)
        self.tcp_std = None if tcp_std is None else tcp_std.astype(np.float32)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.image_crop_ratio = float(image_crop_ratio)
        self.bgr_to_rgb = bool(bgr_to_rgb)
        self.color_jitter = bool(color_jitter)
        self.grayscale_p = float(grayscale_p)
        self.post_blur_radius = float(post_blur_radius)
        self.lowpass_size = int(lowpass_size)
        self.return_phase = bool(return_phase)
        self._h5: h5py.File | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.index.hdf5_path, "r")
        return self._h5

    def __getitem__(self, item: int):
        sample_id = int(self.sample_ids[item])
        demo_name = self.index.demo_names[int(self.index.demo_idx[sample_id])]
        frame = int(self.index.frame_idx[sample_id])
        image = np.asarray(
            self.h5["data"][demo_name]["observations"][self.index.image_key][frame]
        )
        image_tensor = preprocess_image(
            image=image,
            image_size=self.image_size,
            augment=self.augment,
            crop_ratio=self.image_crop_ratio,
            bgr_to_rgb=self.bgr_to_rgb,
            color_jitter=self.color_jitter,
            grayscale_p=self.grayscale_p,
            post_blur_radius=self.post_blur_radius,
            lowpass_size=self.lowpass_size,
        )

        mean = (self.index.target_mean[sample_id] - self.force_mean) / self.force_std
        var = self.index.target_var[sample_id] / (self.force_std**2)
        phase_tensor = torch.tensor(float(self.index.phase[sample_id]), dtype=torch.float32)
        if self.index.tcp is None:
            result = (
                image_tensor,
                torch.from_numpy(mean.astype(np.float32)),
                torch.from_numpy(var.astype(np.float32)),
            )
        else:
            result = (
                image_tensor,
                torch.from_numpy(
                    ((self.index.tcp[sample_id] - self.tcp_mean) / self.tcp_std).astype(
                        np.float32
                    )
                ),
                torch.from_numpy(mean.astype(np.float32)),
                torch.from_numpy(var.astype(np.float32)),
            )
        if self.return_phase:
            return (*result, phase_tensor)
        return result


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ForceDistributionCNN(nn.Module):
    def __init__(
        self,
        output_dim: int,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, kernel=5, stride=2),
            ConvBlock(32, 64, kernel=3, stride=2),
            ConvBlock(64, 128, kernel=3, stride=2),
            ConvBlock(128, 256, kernel=3, stride=2),
            ConvBlock(256, 256, kernel=3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, output_dim * 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        tcp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.encoder(x))
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        return mean, logvar


class FrozenPolicyVisionForceModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        policy_checkpoint: str,
        mlp_hidden: int = 128,
        mlp_dropout: float = 0.3,
        head_mode: str = "direct",
        bottleneck_dim: int = 8,
        template_count: int = 4,
        residual_scale: float = 0.25,
        feature_noise_std: float = 0.0,
        feature_dropout: float = 0.0,
        tcp_dim: int = 0,
        tcp_noise_std: float = 0.0,
        tcp_dropout: float = 0.0,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.feature_noise_std = float(feature_noise_std)
        self.tcp_dim = int(tcp_dim)
        self.tcp_noise_std = float(tcp_noise_std)
        self.output_dim = int(output_dim)
        self.head_mode = str(head_mode)
        self.residual_scale = float(residual_scale)
        self.policy_checkpoint = str(policy_checkpoint)
        self.vision_encoder, self.attention_pool_2d, feature_dim = (
            self._load_policy_vision_encoder(self.policy_checkpoint)
        )
        self.vision_encoder.eval()
        self.attention_pool_2d.eval()
        self.vision_encoder.requires_grad_(False)
        self.attention_pool_2d.requires_grad_(False)

        hidden = int(mlp_hidden)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_dropout = nn.Dropout(p=float(feature_dropout))
        self.tcp_dropout = nn.Dropout(p=float(tcp_dropout))
        head_input_dim = feature_dim + self.tcp_dim
        dropout = float(mlp_dropout)
        if self.head_mode == "direct":
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden, output_dim * 2),
            )
        elif self.head_mode == "bottleneck":
            bottleneck = max(1, int(bottleneck_dim))
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden, bottleneck),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(bottleneck, output_dim * 2),
            )
        elif self.head_mode == "mixture-template":
            count = max(1, int(template_count))
            self.template_count = count
            self.template_mean = nn.Parameter(torch.zeros(count, output_dim))
            self.template_logvar = nn.Parameter(torch.zeros(count, output_dim))
            self.shared_head = nn.Sequential(
                nn.Linear(head_input_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            self.gate_head = nn.Linear(hidden, count)
            self.residual_head = nn.Linear(hidden, output_dim)
            self.logvar_residual_head = nn.Linear(hidden, output_dim)
        else:
            raise ValueError(
                "--head-mode must be one of direct, bottleneck, mixture-template; "
                f"got {self.head_mode!r}."
            )

    @staticmethod
    def _load_policy_vision_encoder(
        policy_checkpoint: str,
    ) -> tuple[nn.Module, nn.Module, int]:
        import timm
        from diffusion_policy.common.pytorch_util import replace_submodules
        from diffusion_policy.policy.bae_diffusion_unet_hybrid_image_wrench_encoder_policy import (
            AttentionPool2d,
        )

        ckpt = torch.load(policy_checkpoint, map_location="cpu")
        cfg = ckpt["cfg"]
        state = ckpt["state_dicts"]["model"]
        vc = cfg.policy.obs_encoder.vision_encoder_cfg
        image_resolution = cfg.task.image_shape[1:]

        vision_encoder = timm.create_model(
            model_name=vc.model_name,
            pretrained=False,
            global_pool=vc.global_pool,
            num_classes=0,
        )
        if not vc.model_name.startswith("resnet"):
            raise ValueError(
                f"Frozen policy encoder currently supports ResNet, got {vc.model_name}."
            )
        if int(vc.downsample_ratio) == 32:
            vision_encoder = nn.Sequential(*list(vision_encoder.children())[:-2])
            feature_dim = 512
        elif int(vc.downsample_ratio) == 16:
            vision_encoder = nn.Sequential(*list(vision_encoder.children())[:-3])
            feature_dim = 256
        else:
            raise ValueError(f"Unsupported downsample ratio: {vc.downsample_ratio}")

        if bool(vc.use_group_norm) and not bool(vc.pretrained):
            vision_encoder = replace_submodules(
                root_module=vision_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(
                        (x.num_features // 16)
                        if (x.num_features % 16 == 0)
                        else (x.num_features // 8)
                    ),
                    num_channels=x.num_features,
                ),
            )

        feature_map_shape = [
            math.ceil(int(x) / int(vc.downsample_ratio)) for x in image_resolution
        ]
        if vc.feature_aggregation != "attention_pool_2d":
            raise ValueError(
                "Frozen policy encoder currently expects attention_pool_2d, "
                f"got {vc.feature_aggregation}."
            )
        attention_pool_2d = AttentionPool2d(
            spacial_dim=feature_map_shape[0],
            embed_dim=feature_dim,
            num_heads=feature_dim // 64,
            output_dim=feature_dim,
        )

        vision_state = {
            key.removeprefix("vision_encoder."): value
            for key, value in state.items()
            if key.startswith("vision_encoder.")
        }
        pool_state = {
            key.removeprefix("attention_pool_2d."): value
            for key, value in state.items()
            if key.startswith("attention_pool_2d.")
        }
        vision_encoder.load_state_dict(vision_state, strict=True)
        attention_pool_2d.load_state_dict(pool_state, strict=True)
        return vision_encoder, attention_pool_2d, feature_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        self.attention_pool_2d.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        tcp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            raw_feature = self.vision_encoder(x)
            feature = self.attention_pool_2d(raw_feature)
        feature = self.feature_norm(feature)
        if self.training:
            if self.feature_noise_std > 0:
                feature = feature + torch.randn_like(feature) * self.feature_noise_std
            feature = self.feature_dropout(feature)
        if self.tcp_dim > 0:
            if tcp is None:
                raise ValueError("This model was configured with TCP input but tcp is None.")
            if self.training:
                if self.tcp_noise_std > 0:
                    tcp = tcp + torch.randn_like(tcp) * self.tcp_noise_std
                tcp = self.tcp_dropout(tcp)
            feature = torch.cat([feature, tcp], dim=-1)
        if self.head_mode in ("direct", "bottleneck"):
            out = self.head(feature)
            mean, logvar = out.chunk(2, dim=-1)
        else:
            hidden = self.shared_head(feature)
            gate = torch.softmax(self.gate_head(hidden), dim=-1)
            base_mean = gate @ self.template_mean
            base_logvar = gate @ self.template_logvar
            residual = torch.tanh(self.residual_head(hidden)) * self.residual_scale
            logvar_residual = torch.tanh(self.logvar_residual_head(hidden)) * self.residual_scale
            mean = base_mean + residual
            logvar = base_logvar + logvar_residual
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        return mean, logvar


def build_force_model_from_kwargs(model_kwargs: dict) -> nn.Module:
    backend = model_kwargs.get("encoder_backend", "cnn")
    if backend == "cnn":
        kwargs = dict(model_kwargs)
        kwargs.pop("encoder_backend", None)
        kwargs.pop("use_tcp", None)
        kwargs.pop("tcp_joint_key", None)
        kwargs.pop("tcp_pose_dim", None)
        kwargs.pop("tcp_rotation_rep", None)
        kwargs.pop("tcp_history_steps", None)
        kwargs.pop("tcp_history_stride", None)
        kwargs.pop("tcp_input_mode", None)
        return ForceDistributionCNN(**kwargs)
    if backend == "policy-frozen":
        kwargs = dict(model_kwargs)
        kwargs.pop("encoder_backend", None)
        kwargs.pop("use_tcp", None)
        kwargs.pop("tcp_joint_key", None)
        kwargs.pop("tcp_pose_dim", None)
        kwargs.pop("tcp_rotation_rep", None)
        kwargs.pop("tcp_history_steps", None)
        kwargs.pop("tcp_history_stride", None)
        kwargs.pop("tcp_input_mode", None)
        return FrozenPolicyVisionForceModel(**kwargs)
    raise ValueError(f"Unsupported encoder backend: {backend}")


def build_force_model_from_args(args: argparse.Namespace, output_dim: int) -> nn.Module:
    backend = getattr(args, "encoder_backend", "cnn")
    kwargs = {
        "encoder_backend": backend,
        "output_dim": output_dim,
        "logvar_min": args.logvar_min,
        "logvar_max": args.logvar_max,
    }
    if backend == "policy-frozen":
        kwargs.update(
            {
                "policy_checkpoint": args.policy_checkpoint,
                "mlp_hidden": args.mlp_hidden,
                "mlp_dropout": args.mlp_dropout,
                "head_mode": args.head_mode,
                "bottleneck_dim": args.bottleneck_dim,
                "template_count": args.template_count,
                "residual_scale": args.residual_scale,
                "feature_noise_std": args.feature_noise_std,
                "feature_dropout": args.feature_dropout,
                "tcp_dim": tcp_history_feature_size(
                    tcp_history_steps=args.tcp_history_steps,
                    tcp_input_mode=args.tcp_input_mode,
                ) if args.use_tcp else 0,
                "tcp_noise_std": args.tcp_noise_std if args.use_tcp else 0.0,
                "tcp_dropout": args.tcp_dropout if args.use_tcp else 0.0,
            }
        )
    return build_force_model_from_kwargs(kwargs)


def model_checkpoint_kwargs(
    args: argparse.Namespace,
    output_dim: int,
) -> dict:
    backend = getattr(args, "encoder_backend", "cnn")
    kwargs = {
        "encoder_backend": backend,
        "output_dim": output_dim,
        "logvar_min": args.logvar_min,
        "logvar_max": args.logvar_max,
    }
    if backend == "policy-frozen":
        kwargs.update(
            {
                "policy_checkpoint": args.policy_checkpoint,
                "mlp_hidden": args.mlp_hidden,
                "mlp_dropout": args.mlp_dropout,
                "head_mode": args.head_mode,
                "bottleneck_dim": args.bottleneck_dim,
                "template_count": args.template_count,
                "residual_scale": args.residual_scale,
                "feature_noise_std": args.feature_noise_std,
                "feature_dropout": args.feature_dropout,
                "tcp_dim": tcp_history_feature_size(
                    tcp_history_steps=args.tcp_history_steps,
                    tcp_input_mode=args.tcp_input_mode,
                ) if args.use_tcp else 0,
                "tcp_noise_std": args.tcp_noise_std if args.use_tcp else 0.0,
                "tcp_dropout": args.tcp_dropout if args.use_tcp else 0.0,
            }
        )
    kwargs["use_tcp"] = bool(getattr(args, "use_tcp", False))
    if getattr(args, "use_tcp", False):
        kwargs["tcp_joint_key"] = args.tcp_joint_key
        kwargs["tcp_pose_dim"] = 6
        kwargs["tcp_rotation_rep"] = "rotation_6d"
        kwargs["tcp_history_steps"] = args.tcp_history_steps
        kwargs["tcp_history_stride"] = args.tcp_history_stride
        kwargs["tcp_input_mode"] = args.tcp_input_mode
    return kwargs


def gaussian_target_nll(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target_mean: torch.Tensor,
    target_var: torch.Tensor,
) -> torch.Tensor:
    pred_var = torch.exp(pred_logvar)
    loss = 0.5 * (
        math.log(2.0 * math.pi)
        + pred_logvar
        + (target_var + (target_mean - pred_mean).pow(2)) / pred_var
    )
    return loss.mean()


def phase_consistency_loss(
    pred_mean: torch.Tensor,
    phase: torch.Tensor | None,
    bins: int,
) -> torch.Tensor:
    if phase is None or bins <= 1 or pred_mean.shape[0] < 2:
        return pred_mean.new_zeros(())
    phase = phase.view(-1).clamp(0.0, 1.0)
    bin_idx = torch.clamp((phase * bins).long(), max=bins - 1)
    losses = []
    for idx in torch.unique(bin_idx):
        mask = bin_idx == idx
        if int(mask.sum().item()) < 2:
            continue
        values = pred_mean[mask]
        losses.append(((values - values.mean(dim=0, keepdim=True)) ** 2).mean())
    if not losses:
        return pred_mean.new_zeros(())
    return torch.stack(losses).mean()


def phase_smoothness_loss(
    pred_mean: torch.Tensor,
    phase: torch.Tensor | None,
) -> torch.Tensor:
    if phase is None or pred_mean.shape[0] < 2:
        return pred_mean.new_zeros(())
    order = torch.argsort(phase.view(-1))
    values = pred_mean[order]
    return ((values[1:] - values[:-1]) ** 2).mean()


def unpack_batch(batch, device: torch.device):
    phase = None
    if len(batch) == 3:
        images, target_mean, target_var = batch
        tcp = None
    elif len(batch) == 4:
        if batch[3].ndim == 1:
            images, target_mean, target_var, phase = batch
            tcp = None
            phase = phase.to(device, non_blocking=True)
        else:
            images, tcp, target_mean, target_var = batch
            tcp = tcp.to(device, non_blocking=True)
    elif len(batch) == 5:
        images, tcp, target_mean, target_var, phase = batch
        tcp = tcp.to(device, non_blocking=True)
    else:
        raise ValueError(f"Expected batch of length 3, 4, or 5, got {len(batch)}.")
    images = images.to(device, non_blocking=True)
    target_mean = target_mean.to(device, non_blocking=True)
    target_var = target_var.to(device, non_blocking=True)
    if phase is not None:
        phase = phase.to(device, non_blocking=True)
    return images, tcp, target_mean, target_var, phase


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA was requested but no CUDA GPU is available; using CPU.")
        return torch.device("cpu")
    return device


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    force_std: torch.Tensor,
    use_amp: bool,
    log_interval: int,
    grad_clip: float,
    epoch: int,
    consistency_loss_weight: float,
    consistency_bins: int,
    smoothness_loss_weight: float,
) -> dict[str, float]:
    model.train()
    totals = init_metric_totals()
    total_batches = len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"train {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_idx, batch in enumerate(progress, start=1):
        images, tcp, target_mean, target_var, phase = unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred_mean, pred_logvar = model(images, tcp)
            loss = gaussian_target_nll(
                pred_mean,
                pred_logvar,
                target_mean,
                target_var,
            )
            if consistency_loss_weight > 0:
                loss = loss + float(consistency_loss_weight) * phase_consistency_loss(
                    pred_mean,
                    phase,
                    int(consistency_bins),
                )
            if smoothness_loss_weight > 0:
                loss = loss + float(smoothness_loss_weight) * phase_smoothness_loss(
                    pred_mean,
                    phase,
                )

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        update_metric_totals(
            totals,
            loss.detach(),
            pred_mean.detach(),
            pred_logvar.detach(),
            target_mean,
            target_var,
            force_std,
        )
        if log_interval > 0 and (
            batch_idx % log_interval == 0 or batch_idx == total_batches
        ):
            progress.set_postfix(loss=f"{loss.item():.4f}")
    return finalize_metrics(totals)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    force_std: torch.Tensor,
    use_amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    totals = init_metric_totals()
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"val {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress:
        images, tcp, target_mean, target_var, _phase = unpack_batch(batch, device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred_mean, pred_logvar = model(images, tcp)
            loss = gaussian_target_nll(
                pred_mean,
                pred_logvar,
                target_mean,
                target_var,
            )
        update_metric_totals(
            totals,
            loss,
            pred_mean,
            pred_logvar,
            target_mean,
            target_var,
            force_std,
        )
    return finalize_metrics(totals)


def init_metric_totals() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "sqerr_sum": 0.0,
        "element_count": 0.0,
        "pred_std_sum": 0.0,
        "target_std_sum": 0.0,
        "within_1std": 0.0,
        "within_2std": 0.0,
    }


def update_metric_totals(
    totals: dict[str, float],
    loss: torch.Tensor,
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target_mean: torch.Tensor,
    target_var: torch.Tensor,
    force_std: torch.Tensor,
) -> None:
    batch = int(target_mean.shape[0])
    dims = int(target_mean.shape[1])
    pred_std = torch.exp(0.5 * pred_logvar) * force_std
    target_std = torch.sqrt(target_var) * force_std
    residual = torch.abs(pred_mean - target_mean) * force_std

    totals["loss_sum"] += float(loss.item()) * batch
    totals["sample_count"] += batch
    totals["sqerr_sum"] += float((residual.pow(2)).sum().item())
    totals["element_count"] += batch * dims
    totals["pred_std_sum"] += float(pred_std.sum().item())
    totals["target_std_sum"] += float(target_std.sum().item())
    totals["within_1std"] += float((residual <= pred_std).sum().item())
    totals["within_2std"] += float((residual <= 2.0 * pred_std).sum().item())


def finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    samples = max(totals["sample_count"], 1.0)
    elements = max(totals["element_count"], 1.0)
    return {
        "loss": totals["loss_sum"] / samples,
        "rmse": math.sqrt(totals["sqerr_sum"] / elements),
        "pred_std": totals["pred_std_sum"] / elements,
        "target_std": totals["target_std_sum"] / elements,
        "within_1std": totals["within_1std"] / elements,
        "within_2std": totals["within_2std"] / elements,
    }


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} loss={metrics['loss']:.4f} "
        f"rmse={metrics['rmse']:.4f} "
        f"pred_std={metrics['pred_std']:.4f} "
        f"target_std={metrics['target_std']:.4f} "
        f"in1={metrics['within_1std']:.3f} "
        f"in2={metrics['within_2std']:.3f}"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    index: SampleIndex,
    force_mean: np.ndarray,
    force_std: np.ndarray,
    tcp_mean: np.ndarray | None,
    tcp_std: np.ndarray | None,
    train_demos: list[str],
    val_demos: list[str],
    epoch: int,
    best_val_loss: float,
    metrics: dict[str, float],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_kwargs": model_checkpoint_kwargs(args, output_dim=len(index.force_dims)),
        "force_mean": force_mean.astype(np.float32),
        "force_std": force_std.astype(np.float32),
        "tcp_mean": None if tcp_mean is None else tcp_mean.astype(np.float32),
        "tcp_std": None if tcp_std is None else tcp_std.astype(np.float32),
        "target_names": dim_names(index.force_dims),
        "hdf5_path": index.hdf5_path,
        "image_key": index.image_key,
        "wrench_key": index.wrench_key,
        "force_dims": index.force_dims,
        "window_sec": index.window_sec,
        "wrench_zero_samples": index.wrench_zero_samples,
        "wrench_ema_alpha": index.wrench_ema_alpha,
        "label_mode": index.label_mode,
        "image_force_align": index.image_force_align,
        "phase_bins": index.phase_bins,
        "phase_source": getattr(args, "phase_source", "frame"),
        "image_size": args.image_size,
        "image_crop_ratio": args.image_crop_ratio,
        "bgr_to_rgb": args.bgr_to_rgb,
        "color_jitter": args.color_jitter,
        "grayscale_p": args.grayscale_p,
        "post_blur_radius": args.post_blur_radius,
        "lowpass_size": args.lowpass_size,
        "consistency_loss_weight": args.consistency_loss_weight,
        "consistency_bins": args.consistency_bins,
        "smoothness_loss_weight": args.smoothness_loss_weight,
        "tcp_history_steps": args.tcp_history_steps,
        "tcp_history_stride": args.tcp_history_stride,
        "tcp_input_mode": args.tcp_input_mode,
        "train_demos": train_demos,
        "val_demos": val_demos,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "args": serializable_args(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def serializable_args(args: argparse.Namespace) -> dict:
    clean = {}
    for key, value in vars(args).items():
        if key == "func":
            continue
        if isinstance(value, Path):
            clean[key] = str(value)
        else:
            clean[key] = value
    return clean


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    index: SampleIndex,
    force_mean: np.ndarray,
    force_std: np.ndarray,
    tcp_mean: np.ndarray | None,
    tcp_std: np.ndarray | None,
    train_demos: list[str],
    val_demos: list[str],
) -> None:
    payload = {
        "hdf5_path": index.hdf5_path,
        "image_key": index.image_key,
        "wrench_key": index.wrench_key,
        "force_dims": list(index.force_dims),
        "target_names": dim_names(index.force_dims),
        "window_sec": index.window_sec,
        "wrench_zero_samples": index.wrench_zero_samples,
        "wrench_ema_alpha": index.wrench_ema_alpha,
        "label_mode": index.label_mode,
        "image_force_align": index.image_force_align,
        "phase_bins": index.phase_bins,
        "phase_source": getattr(args, "phase_source", "frame"),
        "sample_count": int(len(index.frame_idx)),
        "demo_count": int(len(index.demo_names)),
        "train_demo_count": len(train_demos),
        "val_demo_count": len(val_demos),
        "train_demos": train_demos,
        "val_demos": val_demos,
        "force_mean": force_mean.tolist(),
        "force_std": force_std.tolist(),
        "tcp_mean": None if tcp_mean is None else tcp_mean.tolist(),
        "tcp_std": None if tcp_std is None else tcp_std.tolist(),
        "image_size": args.image_size,
        "image_crop_ratio": args.image_crop_ratio,
        "bgr_to_rgb": args.bgr_to_rgb,
        "color_jitter": args.color_jitter,
        "grayscale_p": args.grayscale_p,
        "post_blur_radius": args.post_blur_radius,
        "lowpass_size": args.lowpass_size,
        "consistency_loss_weight": args.consistency_loss_weight,
        "consistency_bins": args.consistency_bins,
        "smoothness_loss_weight": args.smoothness_loss_weight,
        "tcp_history_steps": args.tcp_history_steps,
        "tcp_history_stride": args.tcp_history_stride,
        "tcp_input_mode": args.tcp_input_mode,
        "args": serializable_args(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_experiment_summary(
    path: Path,
    args: argparse.Namespace,
    index: SampleIndex,
    force_mean: np.ndarray,
    force_std: np.ndarray,
    tcp_mean: np.ndarray | None,
    tcp_std: np.ndarray | None,
    train_demos: list[str],
    val_demos: list[str],
) -> None:
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    target_lines = [
        f"  {name}: mean={mean:.6f}, std={std:.6f}"
        for name, mean, std in zip(dim_names(index.force_dims), force_mean, force_std)
    ]
    tcp_lines = ["TCP input: disabled"]
    if tcp_mean is not None and tcp_std is not None:
        tcp_lines = [
            "TCP input: enabled",
            f"  joint_key: {args.tcp_joint_key}",
            "  pose_mode: xyz + rotation_6d",
            f"  input_mode: {args.tcp_input_mode}",
            f"  history_steps: {args.tcp_history_steps}",
            f"  history_stride: {args.tcp_history_stride}",
            f"  feature_dim: {len(tcp_mean)}",
            f"  noise_std: {args.tcp_noise_std}",
            f"  dropout: {args.tcp_dropout}",
            f"  mean: {np.array2string(tcp_mean, precision=6, separator=', ')}",
            f"  std: {np.array2string(tcp_std, precision=6, separator=', ')}",
        ]

    lines = [
        "Force Distribution Experiment Summary",
        "=" * 37,
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"note: {getattr(args, 'experiment_note', '') or '(none)'}",
        "",
        "Command",
        "-" * 7,
        command,
        "",
        "Dataset",
        "-" * 7,
        f"hdf5_path: {index.hdf5_path}",
        f"image_key: {index.image_key}",
        f"wrench_key: {index.wrench_key}",
        f"demos: {len(index.demo_names)}",
        f"samples: {len(index.frame_idx)}",
        f"train_demos ({len(train_demos)}): {', '.join(train_demos)}",
        f"val_demos ({len(val_demos)}): {', '.join(val_demos) if val_demos else '(none)'}",
        "",
        "Label",
        "-" * 5,
        f"label_mode: {index.label_mode}",
        f"image_force_align: {index.image_force_align}",
        f"force_dims: {list(index.force_dims)} ({', '.join(dim_names(index.force_dims))})",
        f"window_sec: {index.window_sec}",
        f"wrench_zero_samples: {index.wrench_zero_samples}",
        f"wrench_ema_alpha: {index.wrench_ema_alpha}",
        "",
        "Model",
        "-" * 5,
        f"encoder_backend: {args.encoder_backend}",
        f"policy_checkpoint: {args.policy_checkpoint if args.encoder_backend == 'policy-frozen' else '(unused)'}",
        f"head_mode: {args.head_mode}",
        f"mlp_hidden: {args.mlp_hidden}",
        f"mlp_dropout: {args.mlp_dropout}",
        f"bottleneck_dim: {args.bottleneck_dim}",
        f"template_count: {args.template_count}",
        f"residual_scale: {args.residual_scale}",
        f"feature_noise_std: {args.feature_noise_std}",
        f"feature_dropout: {args.feature_dropout}",
        f"post_blur_radius: {args.post_blur_radius}",
        f"lowpass_size: {args.lowpass_size}",
        f"consistency_loss_weight: {args.consistency_loss_weight}",
        f"consistency_bins: {args.consistency_bins}",
        f"smoothness_loss_weight: {args.smoothness_loss_weight}",
        "",
        "TCP",
        "-" * 3,
        *tcp_lines,
        "",
        "Training",
        "-" * 8,
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"num_workers: {args.num_workers}",
        f"prefetch_factor: {DEFAULT_DATALOADER_PREFETCH_FACTOR if args.num_workers > 0 else '(unused)'}",
        f"persistent_workers: {DEFAULT_DATALOADER_PERSISTENT_WORKERS if args.num_workers > 0 else '(unused)'}",
        f"lr: {args.lr}",
        f"weight_decay: {args.weight_decay}",
        f"device: {args.device}",
        f"amp: {args.amp}",
        f"val_every: {args.val_every}",
        f"save_every: {args.save_every}",
        "",
        "Normalization",
        "-" * 13,
        *target_lines,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def tensor_to_uint8_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    image = image.add(1.0).mul(127.5).round().byte().numpy()
    return np.ascontiguousarray(image.transpose(1, 2, 0))


def save_test_images(
    output_dir: Path,
    index: SampleIndex,
    sample_ids: np.ndarray,
    count: int,
    image_size: int,
    image_crop_ratio: float,
    bgr_to_rgb: bool,
    post_blur_radius: float,
    lowpass_size: int,
) -> None:
    if count <= 0 or len(sample_ids) == 0:
        return

    export_dir = output_dir / "test_images"
    export_dir.mkdir(parents=True, exist_ok=True)
    selected = np.asarray(sample_ids, dtype=np.int64)
    if len(selected) > count:
        selected = selected[
            np.linspace(0, len(selected) - 1, int(count), dtype=np.int64)
        ]

    names = dim_names(index.force_dims)
    manifest = []
    with h5py.File(index.hdf5_path, "r") as h5:
        for out_idx, sample_id in enumerate(selected):
            sample_id = int(sample_id)
            demo_name = index.demo_names[int(index.demo_idx[sample_id])]
            frame = int(index.frame_idx[sample_id])
            image = np.asarray(
                h5["data"][demo_name]["observations"][index.image_key][frame]
            )
            image_tensor = preprocess_image(
                image=image,
                image_size=image_size,
                augment=False,
                crop_ratio=image_crop_ratio,
                bgr_to_rgb=bgr_to_rgb,
                color_jitter=False,
                grayscale_p=0.0,
                post_blur_radius=post_blur_radius,
                lowpass_size=lowpass_size,
            )
            filename = f"{out_idx:03d}_{demo_name}_frame_{frame:06d}.png"
            Image.fromarray(tensor_to_uint8_rgb(image_tensor)).save(export_dir / filename)

            target_mean = index.target_mean[sample_id].astype(float)
            target_std = np.sqrt(index.target_var[sample_id]).astype(float)
            tcp = None if index.tcp is None else index.tcp[sample_id].astype(float)
            manifest.append(
                {
                    "file": filename,
                    "demo": demo_name,
                    "frame": frame,
                    "sample_id": sample_id,
                    "target_names": names,
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "tcp": None if tcp is None else tcp.tolist(),
                }
            )

    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"saved_test_images={len(manifest)} dir={export_dir}")


def print_index_summary(index: SampleIndex) -> None:
    names = dim_names(index.force_dims)
    means = index.target_mean
    window_std = np.sqrt(index.target_var)
    print(f"HDF5: {index.hdf5_path}")
    print(f"demos={len(index.demo_names)} samples={len(index.frame_idx)}")
    print(f"image_key={index.image_key} wrench_key={index.wrench_key}")
    print(f"force_dims={list(index.force_dims)} names={names}")
    print(
        f"label_mode={index.label_mode} image_force_align={index.image_force_align} "
        f"window_sec={index.window_sec} phase_bins={index.phase_bins}"
    )
    print(
        f"wrench_zero_samples={index.wrench_zero_samples} "
        f"wrench_ema_alpha={index.wrench_ema_alpha}"
    )
    for i, name in enumerate(names):
        print(
            f"{name}: target_mean mean={means[:, i].mean():.4f} "
            f"std={means[:, i].std():.4f} min={means[:, i].min():.4f} "
            f"max={means[:, i].max():.4f} "
            f"window_std_mean={window_std[:, i].mean():.4f}"
        )


def resolve_hdf5_path(args: argparse.Namespace, default: str | None) -> str | None:
    hdf5_path = getattr(args, "hdf5", default)
    data_path = getattr(args, "data_path", None)
    if data_path is None:
        return hdf5_path
    if hdf5_path not in (None, default):
        raise ValueError("Pass only one dataset path: positional data path or --hdf5/--data.")
    return data_path


def train_command(args: argparse.Namespace) -> None:
    configure_torch_threads(args.torch_threads, args.torch_interop_threads)
    seed_everything(args.seed)
    args.hdf5 = resolve_hdf5_path(args, DEFAULT_HDF5)
    force_dims = parse_dims(args.force_dims)
    index = build_index_from_args(args)
    print_index_summary(index)

    train_ids, val_ids, train_demos, val_demos = split_by_demo(
        index,
        val_ratio=args.val_ratio,
        val_demo_count=args.val_demo_count,
        val_demos=args.val_demos,
        seed=args.seed,
    )
    if len(train_ids) == 0:
        raise RuntimeError("Train split is empty.")
    force_mean = index.target_mean[train_ids].mean(axis=0)
    force_std = index.target_mean[train_ids].std(axis=0)
    force_std = np.maximum(force_std, 1e-6).astype(np.float32)
    force_mean = force_mean.astype(np.float32)
    if index.tcp is not None:
        tcp_mean = index.tcp[train_ids].mean(axis=0).astype(np.float32)
        tcp_std = index.tcp[train_ids].std(axis=0)
        tcp_std = np.maximum(tcp_std, 1e-6).astype(np.float32)
    else:
        tcp_mean = None
        tcp_std = None

    print(f"train_samples={len(train_ids)} val_samples={len(val_ids)}")
    print(f"train_demos={len(train_demos)} val_demos={len(val_demos)}")
    if val_demos:
        preview = ", ".join(val_demos[:20])
        suffix = " ..." if len(val_demos) > 20 else ""
        print(f"heldout_val_demos={preview}{suffix}")
    for name, mean, std in zip(dim_names(force_dims), force_mean, force_std):
        print(f"normalize {name}: mean={mean:.4f} std={std:.4f}")
    if tcp_mean is not None and tcp_std is not None:
        print(f"tcp_mean={tcp_mean}")
        print(f"tcp_std={tcp_std}")

    output_root = Path(args.output_dir)
    output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(output_dir)
    print(f"output_dir={output_dir}")
    write_run_metadata(
        output_dir / "run_metadata.json",
        args,
        index,
        force_mean,
        force_std,
        tcp_mean,
        tcp_std,
        train_demos,
        val_demos,
    )
    write_experiment_summary(
        output_dir / "experiment_summary.txt",
        args,
        index,
        force_mean,
        force_std,
        tcp_mean,
        tcp_std,
        train_demos,
        val_demos,
    )
    test_image_ids = val_ids if len(val_ids) > 0 else train_ids
    save_test_images(
        output_dir=output_dir,
        index=index,
        sample_ids=test_image_ids,
        count=args.save_test_images,
        image_size=args.image_size,
        image_crop_ratio=args.image_crop_ratio,
        bgr_to_rgb=args.bgr_to_rgb,
        post_blur_radius=args.post_blur_radius,
        lowpass_size=args.lowpass_size,
    )

    return_phase = (
        args.consistency_loss_weight > 0
        or args.smoothness_loss_weight > 0
    )
    train_dataset = Hdf5ForceDataset(
        index=index,
        sample_ids=train_ids,
        force_mean=force_mean,
        force_std=force_std,
        tcp_mean=tcp_mean,
        tcp_std=tcp_std,
        image_size=args.image_size,
        augment=args.augment,
        image_crop_ratio=args.image_crop_ratio,
        bgr_to_rgb=args.bgr_to_rgb,
        color_jitter=args.color_jitter,
        grayscale_p=args.grayscale_p,
        post_blur_radius=args.post_blur_radius,
        lowpass_size=args.lowpass_size,
        return_phase=return_phase,
    )
    val_dataset = Hdf5ForceDataset(
        index=index,
        sample_ids=val_ids,
        force_mean=force_mean,
        force_std=force_std,
        tcp_mean=tcp_mean,
        tcp_std=tcp_std,
        image_size=args.image_size,
        augment=False,
        image_crop_ratio=args.image_crop_ratio,
        bgr_to_rgb=args.bgr_to_rgb,
        color_jitter=args.color_jitter,
        grayscale_p=args.grayscale_p,
        post_blur_radius=args.post_blur_radius,
        lowpass_size=args.lowpass_size,
        return_phase=False,
    )

    device = make_device(args.device)
    if device.type == "cpu" and not args.enable_mkldnn:
        torch.backends.mkldnn.enabled = False
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = DEFAULT_DATALOADER_PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = DEFAULT_DATALOADER_PREFETCH_FACTOR
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    use_amp = bool(args.amp and device.type == "cuda")
    model = build_force_model_from_args(args, output_dim=len(force_dims)).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(
        f"encoder_backend={args.encoder_backend} "
        f"params_total={total_params:.6e} params_trainable={trainable_params:.6e}"
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(use_amp)
    force_std_t = torch.as_tensor(force_std, device=device).view(1, -1)

    print(f"device={device} amp={use_amp}")
    print(
        f"torch_threads={torch.get_num_threads()} "
        f"torch_interop_threads={torch.get_num_interop_threads()} "
        f"mkldnn={torch.backends.mkldnn.enabled}"
    )
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            force_std=force_std_t,
            use_amp=use_amp,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
            epoch=epoch,
            consistency_loss_weight=args.consistency_loss_weight,
            consistency_bins=args.consistency_bins,
            smoothness_loss_weight=args.smoothness_loss_weight,
        )

        run_val = (
            val_loader is not None
            and args.val_every > 0
            and epoch % args.val_every == 0
        )
        val_metrics = None
        if run_val:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                force_std=force_std_t,
                use_amp=use_amp,
                epoch=epoch,
            )

        message = f"epoch {epoch:03d} {format_metrics('train', train_metrics)}"
        if val_metrics is not None:
            message += f" {format_metrics('val', val_metrics)}"
        print(message)

        checkpoint_metrics = val_metrics if val_metrics is not None else train_metrics

        save_checkpoint(
            output_dir / "last.pt",
            model,
            args,
            index,
            force_mean,
            force_std,
            tcp_mean,
            tcp_std,
            train_demos,
            val_demos,
            epoch,
            best_val_loss,
            checkpoint_metrics,
        )

        if val_metrics is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model,
                args,
                index,
                force_mean,
                force_std,
                tcp_mean,
                tcp_std,
                train_demos,
                val_demos,
                epoch,
                best_val_loss,
                val_metrics,
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                args,
                index,
                force_mean,
                force_std,
                tcp_mean,
                tcp_std,
                train_demos,
                val_demos,
                epoch,
                best_val_loss,
                checkpoint_metrics,
            )

    if best_epoch >= 0:
        print(f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}")
        print(f"saved: {output_dir / 'best.pt'}")
    else:
        print("best_epoch=None best_val_loss=None (validation did not run)")


def inspect_command(args: argparse.Namespace) -> None:
    args.hdf5 = resolve_hdf5_path(args, DEFAULT_HDF5)
    index = build_index_from_args(args)
    print_index_summary(index)


def parse_hdf5_frame(frame_spec: str) -> tuple[str, int]:
    if ":" not in frame_spec:
        raise ValueError("--hdf5-frame must look like demo_0:123 or 0:123.")
    demo_text, frame_text = frame_spec.split(":", 1)
    return demo_text, int(frame_text)


def load_predict_image(args: argparse.Namespace, checkpoint: dict) -> np.ndarray:
    if args.image is not None:
        return np.asarray(Image.open(args.image).convert("RGB"))

    if args.hdf5_frame is None:
        raise ValueError("Pass either --image or --hdf5-frame for prediction.")

    demo_text, frame = parse_hdf5_frame(args.hdf5_frame)
    hdf5_path = resolve_hdf5_path(args, None) or checkpoint["hdf5_path"]
    image_key = args.image_key or checkpoint["image_key"]
    with h5py.File(hdf5_path, "r") as h5:
        demo_name = resolve_demo_name(h5, demo_text)
        return np.asarray(h5["data"][demo_name]["observations"][image_key][frame])


def load_image_tensor_for_prediction(
    image: np.ndarray,
    checkpoint: dict,
    args: argparse.Namespace,
    preprocessed: bool,
) -> torch.Tensor:
    if preprocessed:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return tensor.float().div(255.0).mul(2.0).sub(1.0)

    image_size = int(args.image_size or checkpoint["image_size"])
    if args.bgr_to_rgb is not None:
        bgr_to_rgb = bool(args.bgr_to_rgb)
    else:
        bgr_to_rgb = False if getattr(args, "image", None) is not None else bool(
            checkpoint.get("bgr_to_rgb", True)
        )
    return preprocess_image(
        image=image,
        image_size=image_size,
        augment=False,
        crop_ratio=float(checkpoint.get("image_crop_ratio", 0.95)),
        bgr_to_rgb=bgr_to_rgb,
        color_jitter=False,
        grayscale_p=0.0,
        post_blur_radius=float(checkpoint.get("post_blur_radius", 0.0)),
        lowpass_size=int(checkpoint.get("lowpass_size", 0)),
    )


def checkpoint_uses_tcp(checkpoint: dict) -> bool:
    kwargs = checkpoint.get("model_kwargs", {})
    return bool(kwargs.get("use_tcp", False) or int(kwargs.get("tcp_dim", 0)) > 0)


def raw_tcp_for_frame(
    hdf5_path: str,
    demo_name: str,
    frame: int,
    checkpoint: dict,
) -> np.ndarray:
    kwargs = checkpoint.get("model_kwargs", {})
    tcp_pose_dim = int(kwargs.get("tcp_pose_dim", 6))
    joint_key = kwargs.get("tcp_joint_key", "joint_R")
    tcp_history_steps = int(kwargs.get("tcp_history_steps", 1))
    tcp_history_stride = int(kwargs.get("tcp_history_stride", 1))
    tcp_input_mode = kwargs.get("tcp_input_mode", "absolute")
    frames = tcp_history_frames(
        np.asarray([int(frame)], dtype=np.int64),
        tcp_history_steps=tcp_history_steps,
        tcp_history_stride=tcp_history_stride,
    )
    with h5py.File(hdf5_path, "r") as h5:
        joints = read_hdf5_rows_unordered(
            h5["data"][demo_name]["observations"][joint_key],
            frames,
        ).astype(np.float64)
    return joints_to_tcp_history_features(
        joints,
        tcp_pose_dim=tcp_pose_dim,
        tcp_history_steps=tcp_history_steps,
        tcp_input_mode=tcp_input_mode,
    )[0]


def raw_tcp_for_frames_from_obs(
    obs,
    frames: np.ndarray,
    checkpoint: dict,
) -> np.ndarray:
    kwargs = checkpoint.get("model_kwargs", {})
    tcp_pose_dim = int(kwargs.get("tcp_pose_dim", 6))
    joint_key = kwargs.get("tcp_joint_key", "joint_R")
    tcp_history_steps = int(kwargs.get("tcp_history_steps", 1))
    tcp_history_stride = int(kwargs.get("tcp_history_stride", 1))
    tcp_input_mode = kwargs.get("tcp_input_mode", "absolute")
    history_frames = tcp_history_frames(
        frames,
        tcp_history_steps=tcp_history_steps,
        tcp_history_stride=tcp_history_stride,
    )
    joints = read_hdf5_rows_unordered(obs[joint_key], history_frames).astype(np.float64)
    return joints_to_tcp_history_features(
        joints,
        tcp_pose_dim=tcp_pose_dim,
        tcp_history_steps=tcp_history_steps,
        tcp_input_mode=tcp_input_mode,
    )


def normalize_tcp_for_prediction(
    raw_tcp: np.ndarray,
    checkpoint: dict,
    device: torch.device,
) -> torch.Tensor:
    if checkpoint.get("tcp_mean") is None or checkpoint.get("tcp_std") is None:
        raise RuntimeError("Checkpoint requires TCP input but has no tcp_mean/tcp_std.")
    tcp_mean = np.asarray(checkpoint["tcp_mean"], dtype=np.float32)
    tcp_std = np.asarray(checkpoint["tcp_std"], dtype=np.float32)
    tcp = (np.asarray(raw_tcp, dtype=np.float32) - tcp_mean) / tcp_std
    return torch.from_numpy(tcp).float().to(device)


@torch.no_grad()
def predict_tensor(
    model: nn.Module,
    checkpoint: dict,
    image_tensor: torch.Tensor,
    device: torch.device,
    tcp_tensor: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = image_tensor.unsqueeze(0).to(device)
    tcp = None if tcp_tensor is None else tcp_tensor.view(1, -1).to(device)
    force_mean = torch.as_tensor(
        np.asarray(checkpoint["force_mean"], dtype=np.float32),
        device=device,
    ).view(1, -1)
    force_std = torch.as_tensor(
        np.asarray(checkpoint["force_std"], dtype=np.float32),
        device=device,
    ).view(1, -1)

    pred_mean_norm, pred_logvar = model(x, tcp)
    pred_mean = pred_mean_norm * force_std + force_mean
    pred_std = torch.exp(0.5 * pred_logvar) * force_std
    pred_var = pred_std.pow(2)
    return (
        pred_mean.squeeze(0).cpu().numpy(),
        pred_std.squeeze(0).cpu().numpy(),
        pred_var.squeeze(0).cpu().numpy(),
    )


def load_model_for_prediction(
    checkpoint_path: str,
    device_arg: str,
) -> tuple[torch.device, dict, nn.Module]:
    device = make_device(device_arg)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_force_model_from_kwargs(checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return device, checkpoint, model


def predict_command(args: argparse.Namespace) -> None:
    device, checkpoint, model = load_model_for_prediction(args.checkpoint, args.device)
    image = load_predict_image(args, checkpoint)
    image_tensor = load_image_tensor_for_prediction(
        image=image,
        checkpoint=checkpoint,
        args=args,
        preprocessed=False,
    )
    tcp_tensor = None
    if checkpoint_uses_tcp(checkpoint):
        if args.hdf5_frame is None:
            raise RuntimeError(
                "This checkpoint uses TCP input. Use --hdf5-frame demo:frame so "
                "the matching joint_R can be converted to actual TCP."
            )
        hdf5_path = resolve_hdf5_path(args, None) or checkpoint["hdf5_path"]
        demo_text, frame_text = parse_hdf5_frame(args.hdf5_frame)
        with h5py.File(hdf5_path, "r") as h5:
            demo_name = resolve_demo_name(h5, demo_text)
        raw_tcp = raw_tcp_for_frame(
            hdf5_path=hdf5_path,
            demo_name=demo_name,
            frame=int(frame_text),
            checkpoint=checkpoint,
        )
        tcp_tensor = normalize_tcp_for_prediction(raw_tcp, checkpoint, device)

    mean_np, std_np, var_np = predict_tensor(
        model,
        checkpoint,
        image_tensor,
        device,
        tcp_tensor=tcp_tensor,
    )
    names = checkpoint.get("target_names", dim_names(checkpoint["force_dims"]))
    print(f"checkpoint: {args.checkpoint}")
    print(f"target_names: {names}")
    for name, mean, std, var in zip(names, mean_np, std_np, var_np):
        print(f"{name}: mean={mean:.5f} std={std:.5f} var={var:.5f}")


def gaussian_pdf(xs: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    z = (xs - float(mean)) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def load_test_manifest(image_dir: Path) -> dict[str, dict]:
    manifest_path = image_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["file"]: entry for entry in data}


def plot_force_distribution(
    image: np.ndarray,
    names: list[str],
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    out_path: Path,
    gt_mean: list[float] | None = None,
    gt_std: list[float] | None = None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:cyan"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].axis("off")

    values = list(pred_mean) + list(pred_mean - 4 * pred_std) + list(pred_mean + 4 * pred_std)
    if gt_mean is not None and gt_std is not None:
        gt_mean_np = np.asarray(gt_mean, dtype=np.float32)
        gt_std_np = np.asarray(gt_std, dtype=np.float32)
        values += list(gt_mean_np) + list(gt_mean_np - 4 * gt_std_np) + list(gt_mean_np + 4 * gt_std_np)
    xmin = float(np.nanmin(values))
    xmax = float(np.nanmax(values))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-6:
        xmin, xmax = -1.0, 1.0
    pad = 0.1 * (xmax - xmin)
    xs = np.linspace(xmin - pad, xmax + pad, 400)

    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        axes[1].plot(
            xs,
            gaussian_pdf(xs, pred_mean[i], pred_std[i]),
            color=color,
            linewidth=1.0,
            label=f"pred {name}: {pred_mean[i]:.2f}±{pred_std[i]:.2f}",
        )
        if gt_mean is not None:
            axes[1].axvline(
                float(gt_mean[i]),
                color=color,
                linewidth=1.3,
                linestyle="--",
                alpha=0.95,
                label=f"GT {name}: {gt_mean[i]:.2f}",
            )
    axes[1].set_xlabel("force")
    axes[1].set_ylabel("density")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_test_images_command(args: argparse.Namespace) -> None:
    device, checkpoint, model = load_model_for_prediction(args.checkpoint, args.device)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_test_manifest(image_dir)
    names = checkpoint.get("target_names", dim_names(checkpoint["force_dims"]))

    image_paths = sorted(
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
    )
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}.")

    rows = []
    for image_path in tqdm(image_paths, desc="plot", dynamic_ncols=True):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        image_tensor = load_image_tensor_for_prediction(
            image=image,
            checkpoint=checkpoint,
            args=args,
            preprocessed=not args.raw_images,
        )
        entry = manifest.get(image_path.name)
        tcp_tensor = None
        if checkpoint_uses_tcp(checkpoint):
            if entry is None or "demo" not in entry or "frame" not in entry:
                raise RuntimeError(
                    "This checkpoint uses TCP input, but the test image has no "
                    "manifest demo/frame entry. Use test_images saved by this "
                    "script, or use plot-demo-timeseries/predict --hdf5-frame."
                )
            raw_tcp = raw_tcp_for_frame(
                hdf5_path=checkpoint["hdf5_path"],
                demo_name=str(entry["demo"]),
                frame=int(entry["frame"]),
                checkpoint=checkpoint,
            )
            tcp_tensor = normalize_tcp_for_prediction(raw_tcp, checkpoint, device)
        pred_mean, pred_std, pred_var = predict_tensor(
            model=model,
            checkpoint=checkpoint,
            image_tensor=image_tensor,
            device=device,
            tcp_tensor=tcp_tensor,
        )
        gt_mean = entry.get("target_mean") if entry is not None else None
        gt_std = entry.get("target_std") if entry is not None else None
        out_path = output_dir / f"{image_path.stem}_force_plot.png"
        plot_force_distribution(
            image=image,
            names=names,
            pred_mean=pred_mean,
            pred_std=pred_std,
            out_path=out_path,
            gt_mean=gt_mean,
            gt_std=gt_std,
        )
        rows.append(
            {
                "image": str(image_path),
                "plot": str(out_path),
                "pred_mean": pred_mean.tolist(),
                "pred_std": pred_std.tolist(),
                "pred_var": pred_var.tolist(),
                "target_mean": gt_mean,
                "target_std": gt_std,
            }
        )

    (output_dir / "predictions.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(f"saved plots: {output_dir}")


def checkpoint_value(checkpoint: dict, key: str, default):
    if key in checkpoint:
        return checkpoint[key]
    return checkpoint.get("args", {}).get(key, default)


def build_index_from_checkpoint(
    checkpoint: dict,
    hdf5_path: str,
    image_key: str,
    wrench_key: str,
) -> SampleIndex:
    label_mode = checkpoint_value(checkpoint, "label_mode", "window")
    common_kwargs = dict(
        hdf5_path=hdf5_path,
        image_key=image_key,
        wrench_key=wrench_key,
        force_dims=parse_dims(checkpoint_value(checkpoint, "force_dims", "0,1,2")),
        window_sec=float(checkpoint_value(checkpoint, "window_sec", 0.05)),
        target_std_floor=float(checkpoint_value(checkpoint, "target_std_floor", 0.05)),
        min_window_samples=int(checkpoint_value(checkpoint, "min_window_samples", 1)),
        zero_wrench_start_sec=float(
            checkpoint_value(checkpoint, "zero_wrench_start_sec", 0.0)
        ),
        wrench_zero_samples=int(checkpoint_value(checkpoint, "wrench_zero_samples", 10)),
        wrench_ema_alpha=float(checkpoint_value(checkpoint, "wrench_ema_alpha", 0.03)),
        image_force_align=checkpoint_value(
            checkpoint,
            "image_force_align",
            "center-window",
        ),
        max_demos=None,
        limit_frames_per_demo=None,
    )
    if label_mode == "phase":
        return build_phase_sample_index(
            **common_kwargs,
            phase_bins=int(checkpoint_value(checkpoint, "phase_bins", 100)),
            phase_source=checkpoint_value(checkpoint, "phase_source", "frame"),
        )
    if label_mode == "sample":
        sample_kwargs = common_kwargs.copy()
        sample_kwargs.pop("target_std_floor")
        return build_single_force_sample_index(**sample_kwargs)
    return build_sample_index(**common_kwargs)


def resolve_demo_name(h5: h5py.File, demo: str) -> str:
    if "data" not in h5:
        raise KeyError("Expected a top-level /data group in the HDF5 file.")

    candidates = [demo]
    if demo.isdigit():
        candidates.append(f"demo_{demo}")
    for candidate in candidates:
        if candidate in h5["data"]:
            return candidate

    available = sorted(h5["data"].keys(), key=demo_sort_key)
    preview = ", ".join(available[:10])
    raise KeyError(f"Demo {demo!r} not found. Available demos include: {preview}")


@torch.no_grad()
def predict_image_batch(
    model: nn.Module,
    checkpoint: dict,
    image_tensors: list[torch.Tensor],
    device: torch.device,
    tcp_tensors: list[torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.stack(image_tensors, dim=0).to(device)
    tcp = None
    if tcp_tensors is not None:
        tcp = torch.stack(tcp_tensors, dim=0).to(device)
    force_mean = torch.as_tensor(
        np.asarray(checkpoint["force_mean"], dtype=np.float32),
        device=device,
    ).view(1, -1)
    force_std = torch.as_tensor(
        np.asarray(checkpoint["force_std"], dtype=np.float32),
        device=device,
    ).view(1, -1)

    pred_mean_norm, pred_logvar = model(x, tcp)
    pred_mean = pred_mean_norm * force_std + force_mean
    pred_std = torch.exp(0.5 * pred_logvar) * force_std
    return pred_mean.cpu().numpy(), pred_std.cpu().numpy()


def plot_demo_timeseries(
    times: np.ndarray,
    frames: np.ndarray,
    names: list[str],
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target_mean: np.ndarray,
    out_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(names)
    fig_height = max(2.4 * count, 4.0)
    fig, axes = plt.subplots(
        count,
        1,
        figsize=(12, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    if count == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.set_facecolor("white")
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.plot(times, target_mean[:, i], color="black", linewidth=1.8, label="actual")
        ax.fill_between(
            times,
            pred_mean[:, i] - pred_std[:, i],
            pred_mean[:, i] + pred_std[:, i],
            color="tab:red",
            alpha=0.18,
            linewidth=0,
            label="pred ±1 std",
        )
        ax.plot(
            times,
            pred_mean[:, i],
            color="tab:red",
            linewidth=1.5,
            linestyle="--",
            label="pred mean",
        )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("time from first selected image (sec)")
    title = f"frames {int(frames[0])}..{int(frames[-1])} ({len(frames)} images)"
    fig.suptitle(title)
    fig.savefig(out_path, dpi=140, facecolor="white", edgecolor="white")
    plt.close(fig)


def plot_demo_timeseries_command(args: argparse.Namespace) -> None:
    device, checkpoint, model = load_model_for_prediction(args.checkpoint, args.device)
    hdf5_path = resolve_hdf5_path(args, None) or checkpoint["hdf5_path"]
    image_key = args.image_key or checkpoint_value(checkpoint, "image_key", "image_R")
    wrench_key = args.wrench_key or checkpoint_value(
        checkpoint,
        "wrench_key",
        "wrench_wrist_R",
    )
    force_dims = parse_dims(checkpoint_value(checkpoint, "force_dims", "0,1,2"))
    index = build_index_from_checkpoint(
        checkpoint=checkpoint,
        hdf5_path=hdf5_path,
        image_key=image_key,
        wrench_key=wrench_key,
    )

    with h5py.File(hdf5_path, "r") as h5:
        demo_name = resolve_demo_name(h5, args.demo)

    demo_ids = {
        name: i
        for i, name in enumerate(index.demo_names)
    }
    if demo_name not in demo_ids:
        raise RuntimeError(
            f"{demo_name} has no valid samples after timestamp/window filtering."
        )

    sample_ids = np.nonzero(index.demo_idx == demo_ids[demo_name])[0]
    order = np.argsort(index.frame_idx[sample_ids])
    sample_ids = sample_ids[order]
    if args.stride > 1:
        sample_ids = sample_ids[:: args.stride]
    if args.max_frames is not None:
        sample_ids = sample_ids[: args.max_frames]
    if len(sample_ids) == 0:
        raise RuntimeError(f"No samples selected for {demo_name}.")

    frames = index.frame_idx[sample_ids].astype(np.int64)
    target_mean = index.target_mean[sample_ids].astype(np.float32)
    image_size = int(args.image_size or checkpoint_value(checkpoint, "image_size", 224))
    crop_ratio = float(checkpoint_value(checkpoint, "image_crop_ratio", 0.95))
    bgr_to_rgb = (
        bool(args.bgr_to_rgb)
        if args.bgr_to_rgb is not None
        else bool(checkpoint_value(checkpoint, "bgr_to_rgb", True))
    )

    pred_chunks = []
    use_tcp = checkpoint_uses_tcp(checkpoint)
    with h5py.File(hdf5_path, "r") as h5:
        obs = h5["data"][demo_name]["observations"]
        robot_ts = np.asarray(obs["timestamp_robot"][:], dtype=np.float64)
        times = robot_ts[frames]
        times = times - times[0]
        for start in tqdm(
            range(0, len(frames), args.batch_size),
            desc=f"predict {demo_name}",
            dynamic_ncols=True,
        ):
            batch_frames = frames[start : start + args.batch_size]
            tensors = []
            tcp_tensors = [] if use_tcp else None
            for frame in batch_frames:
                image = np.asarray(obs[image_key][int(frame)])
                tensors.append(
                    preprocess_image(
                        image=image,
                        image_size=image_size,
                        augment=False,
                        crop_ratio=crop_ratio,
                        bgr_to_rgb=bgr_to_rgb,
                        color_jitter=False,
                        grayscale_p=0.0,
                        post_blur_radius=float(checkpoint_value(checkpoint, "post_blur_radius", 0.0)),
                        lowpass_size=int(checkpoint_value(checkpoint, "lowpass_size", 0)),
                    )
                )
            if use_tcp:
                raw_tcp = raw_tcp_for_frames_from_obs(
                    obs=obs,
                    frames=batch_frames,
                    checkpoint=checkpoint,
                )
                tcp_tensors = [
                    normalize_tcp_for_prediction(row, checkpoint, device)
                    for row in raw_tcp
                ]
            pred_chunks.append(
                predict_image_batch(
                    model=model,
                    checkpoint=checkpoint,
                    image_tensors=tensors,
                    device=device,
                    tcp_tensors=tcp_tensors,
                )
            )

    pred_mean = np.concatenate([chunk[0] for chunk in pred_chunks], axis=0)
    pred_std = np.concatenate([chunk[1] for chunk in pred_chunks], axis=0)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).resolve().parent / "demo_timeseries"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{demo_name}_force_timeseries"
    plot_path = output_dir / f"{stem}.png"
    data_path = output_dir / f"{stem}.json"
    names = checkpoint.get("target_names", dim_names(force_dims))

    plot_demo_timeseries(
        times=times,
        frames=frames,
        names=names,
        pred_mean=pred_mean,
        pred_std=pred_std,
        target_mean=target_mean,
        out_path=plot_path,
    )
    rows = [
        {
            "time_sec": float(times[i]),
            "frame": int(frames[i]),
            "pred_mean": pred_mean[i].astype(float).tolist(),
            "pred_std": pred_std[i].astype(float).tolist(),
            "target_mean": target_mean[i].astype(float).tolist(),
        }
        for i in range(len(frames))
    ]
    payload = {
        "checkpoint": str(args.checkpoint),
        "hdf5_path": str(hdf5_path),
        "demo": demo_name,
        "target_names": names,
        "image_key": image_key,
        "wrench_key": wrench_key,
        "window_sec": float(index.window_sec),
        "stride": int(args.stride),
        "count": int(len(frames)),
        "rows": rows,
    }
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved plot: {plot_path}")
    print(f"saved data: {data_path}")


def plot_all_demo_forces(
    demo_rows: list[dict],
    names: list[str],
    out_path: Path,
    x_axis: str,
    show_mean: bool,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(names)
    fig_height = max(2.4 * count, 4.0)
    fig, axes = plt.subplots(
        count,
        1,
        figsize=(12, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    if count == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, names)):
        for row in demo_rows:
            ax.plot(
                row["x"],
                row["force"][:, i],
                linewidth=0.8,
                alpha=0.22,
                color="tab:red",
            )
        if show_mean:
            grid = np.linspace(0.0, 1.0, 300) if x_axis == "phase" else None
            if grid is not None:
                values = []
                for row in demo_rows:
                    if len(row["x"]) < 2:
                        continue
                    values.append(np.interp(grid, row["x"], row["force"][:, i]))
                if values:
                    stacked = np.stack(values)
                    mean = stacked.mean(axis=0)
                    std = stacked.std(axis=0)
                    ax.fill_between(
                        grid,
                        mean - std,
                        mean + std,
                        color="black",
                        alpha=0.10,
                        linewidth=0,
                        label="all-demo mean ±1 std",
                    )
                    ax.plot(
                        grid,
                        mean,
                        color="black",
                        linewidth=2.0,
                        label="all-demo mean",
                    )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("normalized demo progress" if x_axis == "phase" else "time (sec)")
    fig.suptitle(f"Actual force overlaid across {len(demo_rows)} demos")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_all_demo_forces_command(args: argparse.Namespace) -> None:
    args.hdf5 = resolve_hdf5_path(args, DEFAULT_HDF5)
    force_dims = parse_dims(args.force_dims)
    hdf5_path = str(Path(args.hdf5).expanduser())
    names = dim_names(force_dims)
    demo_rows = []
    min_samples = max(int(args.min_window_samples), 1)

    with h5py.File(hdf5_path, "r") as h5:
        if "data" not in h5:
            raise KeyError("Expected a top-level /data group in the HDF5 file.")
        demos = sorted(h5["data"].keys(), key=demo_sort_key)
        if args.max_demos is not None:
            demos = demos[: args.max_demos]

        for demo_name in tqdm(demos, desc="demos", dynamic_ncols=True):
            obs = h5["data"][demo_name]["observations"]
            required = [args.image_key, args.wrench_key, "timestamp_robot", "timestamp_wrench"]
            missing = [key for key in required if key not in obs]
            if missing:
                raise KeyError(f"{demo_name} is missing observation keys: {missing}")

            image_count = int(obs[args.image_key].shape[0])
            robot_ts = np.asarray(obs["timestamp_robot"][:], dtype=np.float64)
            wrench_ts = np.asarray(obs["timestamp_wrench"][:], dtype=np.float64)
            if image_count != len(robot_ts):
                raise ValueError(
                    f"{demo_name}: {args.image_key} has {image_count} frames but "
                    f"timestamp_robot has {len(robot_ts)} entries."
                )

            if args.limit_frames_per_demo is None or args.limit_frames_per_demo >= image_count:
                frames = np.arange(image_count, dtype=np.int64)
            else:
                frames = np.unique(
                    np.linspace(
                        0,
                        image_count - 1,
                        int(args.limit_frames_per_demo),
                        dtype=np.int64,
                    )
                )

            wrench_series = preprocess_wrench_series(
                raw_wrench=obs[args.wrench_key][:],
                force_dims=force_dims,
                wrench_ts=wrench_ts,
                zero_wrench_start_sec=args.zero_wrench_start_sec,
                wrench_zero_samples=args.wrench_zero_samples,
                wrench_ema_alpha=args.wrench_ema_alpha,
            )

            xs = []
            forces = []
            frame_ids = []
            for frame in frames:
                frame = int(frame)
                lo, hi = wrench_interval_for_image(
                    robot_ts=robot_ts,
                    wrench_ts=wrench_ts,
                    frame=frame,
                    window_sec=args.window_sec,
                    image_force_align=args.image_force_align,
                )
                if hi - lo < min_samples:
                    continue
                forces.append(wrench_series[lo:hi].mean(axis=0).astype(np.float32))
                frame_ids.append(frame)
                if args.x_axis == "time":
                    xs.append(float(robot_ts[frame] - robot_ts[0]))
                else:
                    xs.append(float(frame / max(image_count - 1, 1)))

            if len(forces) == 0:
                continue
            demo_rows.append(
                {
                    "demo": demo_name,
                    "x": np.asarray(xs, dtype=np.float32),
                    "frame": np.asarray(frame_ids, dtype=np.int64),
                    "force": np.stack(forces).astype(np.float32),
                }
            )

    if not demo_rows:
        raise RuntimeError("No demo force traces were built. Check keys and timestamps.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("force_distribution_height") / "force_overlays"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "all_demo_force_overlay.png"
    data_path = output_dir / "all_demo_force_overlay.json"

    plot_all_demo_forces(
        demo_rows=demo_rows,
        names=names,
        out_path=plot_path,
        x_axis=args.x_axis,
        show_mean=not args.no_mean,
    )
    payload = {
        "hdf5_path": hdf5_path,
        "target_names": names,
        "image_key": args.image_key,
        "wrench_key": args.wrench_key,
        "force_dims": list(force_dims),
        "image_force_align": args.image_force_align,
        "window_sec": args.window_sec,
        "wrench_zero_samples": args.wrench_zero_samples,
        "wrench_ema_alpha": args.wrench_ema_alpha,
        "x_axis": args.x_axis,
        "demo_count": len(demo_rows),
        "demos": [
            {
                "demo": row["demo"],
                "x": row["x"].astype(float).tolist(),
                "frame": row["frame"].astype(int).tolist(),
                "force": row["force"].astype(float).tolist(),
            }
            for row in demo_rows
        ],
    }
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved plot: {plot_path}")
    print(f"saved data: {data_path}")


def style_overlay_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.grid(True, alpha=0.25)


def interp_overlay_stats(
    rows: list[dict],
    key: str,
    dim: int,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    values = []
    for row in rows:
        x = row["x"]
        y = row[key][:, dim]
        if len(x) < 2:
            continue
        values.append(np.interp(grid, x, y))
    if not values:
        return None
    stacked = np.stack(values)
    return stacked.mean(axis=0), stacked.std(axis=0)


def plot_actual_pred_demo_overlays(
    demo_rows: list[dict],
    names: list[str],
    out_path: Path,
    x_axis: str,
    show_mean: bool,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(names),
        2,
        figsize=(16, max(2.7 * len(names), 6.5)),
        sharex=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    if len(names) == 1:
        axes = np.asarray([axes])

    grid = np.linspace(0.0, 1.0, 300) if x_axis == "phase" else None
    x_label = "normalized demo progress" if x_axis == "phase" else "time (sec)"

    for i, name in enumerate(names):
        actual_ax = axes[i, 0]
        pred_ax = axes[i, 1]
        for ax in (actual_ax, pred_ax):
            style_overlay_axis(ax)

        actual_values = np.concatenate([row["actual"][:, i] for row in demo_rows])
        pred_values = np.concatenate([row["pred"][:, i] for row in demo_rows])
        all_values = np.concatenate([actual_values, pred_values])
        ymin = float(np.nanmin(all_values))
        ymax = float(np.nanmax(all_values))
        if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
            ymin, ymax = -1.0, 1.0
        pad = 0.08 * (ymax - ymin)
        shared_ylim = (ymin - pad, ymax + pad)

        for row in demo_rows:
            actual_ax.plot(
                row["x"],
                row["actual"][:, i],
                linewidth=0.75,
                alpha=0.20,
                color="tab:red",
            )
            pred_ax.plot(
                row["x"],
                row["pred"][:, i],
                linewidth=0.75,
                alpha=0.20,
                color="tab:blue",
            )

        if show_mean and grid is not None:
            actual_stats = interp_overlay_stats(demo_rows, "actual", i, grid)
            pred_stats = interp_overlay_stats(demo_rows, "pred", i, grid)
            if actual_stats is not None:
                mean, _ = actual_stats
                actual_ax.plot(grid, mean, color="black", linewidth=2.0, label="actual mean")
            if pred_stats is not None:
                mean, _ = pred_stats
                pred_ax.plot(grid, mean, color="black", linewidth=2.0, label="pred mean")

        actual_ax.set_ylabel(name)
        actual_ax.set_ylim(shared_ylim)
        pred_ax.set_ylim(shared_ylim)
        if i == 0:
            actual_ax.set_title("Actual force overlaid")
            pred_ax.set_title("Predicted mean force overlaid")
        actual_ax.legend(loc="best", fontsize=8)
        pred_ax.legend(loc="best", fontsize=8)

    axes[-1, 0].set_xlabel(x_label)
    axes[-1, 1].set_xlabel(x_label)
    fig.suptitle(f"Actual vs predicted force overlays across {len(demo_rows)} demos")
    fig.savefig(out_path, dpi=140, facecolor="white", edgecolor="white")
    plt.close(fig)


def plot_all_demo_actual_pred_command(args: argparse.Namespace) -> None:
    device, checkpoint, model = load_model_for_prediction(args.checkpoint, args.device)
    hdf5_path = resolve_hdf5_path(args, None) or checkpoint["hdf5_path"]
    image_key = args.image_key or checkpoint_value(checkpoint, "image_key", "image_R")
    wrench_key = args.wrench_key or checkpoint_value(
        checkpoint,
        "wrench_key",
        "wrench_wrist_R",
    )
    force_dims = parse_dims(checkpoint_value(checkpoint, "force_dims", "0,1,2"))
    names = checkpoint.get("target_names", dim_names(force_dims))
    force_frame = str(getattr(args, "force_frame", "sensor"))
    wrench_frame = str(getattr(args, "wrench_frame", "base"))
    if force_frame == "world":
        if tuple(force_dims) != (0, 1, 2):
            raise ValueError("--force-frame world currently requires force dims 0,1,2.")
        names = [f"{name}_world" for name in names]
        if wrench_frame not in {"base", "tcp", "tcp-inverse"}:
            raise ValueError("--wrench-frame must be base, tcp, or tcp-inverse.")
    index = build_index_from_checkpoint(
        checkpoint=checkpoint,
        hdf5_path=hdf5_path,
        image_key=image_key,
        wrench_key=wrench_key,
    )

    image_size = int(args.image_size or checkpoint_value(checkpoint, "image_size", 224))
    crop_ratio = float(checkpoint_value(checkpoint, "image_crop_ratio", 0.95))
    bgr_to_rgb = (
        bool(args.bgr_to_rgb)
        if args.bgr_to_rgb is not None
        else bool(checkpoint_value(checkpoint, "bgr_to_rgb", True))
    )
    use_tcp = checkpoint_uses_tcp(checkpoint)

    demo_rows = []
    with h5py.File(hdf5_path, "r") as h5:
        for demo_id, demo_name in enumerate(tqdm(index.demo_names, desc="demos", dynamic_ncols=True)):
            if args.max_demos is not None and len(demo_rows) >= args.max_demos:
                break
            sample_ids = np.nonzero(index.demo_idx == demo_id)[0]
            if len(sample_ids) == 0:
                continue
            order = np.argsort(index.frame_idx[sample_ids])
            sample_ids = sample_ids[order]
            if args.stride > 1:
                sample_ids = sample_ids[:: args.stride]
            if args.limit_frames_per_demo is not None:
                sample_ids = sample_ids[: args.limit_frames_per_demo]
            if len(sample_ids) == 0:
                continue

            obs = h5["data"][demo_name]["observations"]
            frames = index.frame_idx[sample_ids].astype(np.int64)
            robot_ts = np.asarray(obs["timestamp_robot"][:], dtype=np.float64)
            image_count = int(obs[image_key].shape[0])
            if args.x_axis == "time":
                xs = robot_ts[frames] - robot_ts[frames[0]]
            else:
                xs = frames.astype(np.float32) / float(max(image_count - 1, 1))
            actual = index.target_mean[sample_ids].astype(np.float32)
            rotations = None
            if force_frame == "world":
                if wrench_frame != "base":
                    joint_key = checkpoint_value(checkpoint, "tcp_joint_key", "joint_R")
                    if joint_key not in obs:
                        raise KeyError(
                            f"{demo_name} is missing observation key {joint_key!r} "
                            "required for --force-frame world with TCP wrench frames."
                        )
                    joints = read_hdf5_rows_unordered(obs[joint_key], frames).astype(np.float64)
                    rotations = joint_to_tcp_rotmats(joints)
                actual = rotate_force_to_world(
                    actual,
                    tcp_rotmat_base=rotations,
                    wrench_frame=wrench_frame,
                )

            pred_chunks = []
            for start in range(0, len(frames), args.batch_size):
                batch_frames = frames[start : start + args.batch_size]
                tensors = []
                tcp_tensors = [] if use_tcp else None
                for frame in batch_frames:
                    image = np.asarray(obs[image_key][int(frame)])
                    tensors.append(
                        preprocess_image(
                            image=image,
                            image_size=image_size,
                            augment=False,
                            crop_ratio=crop_ratio,
                            bgr_to_rgb=bgr_to_rgb,
                            color_jitter=False,
                            grayscale_p=0.0,
                            post_blur_radius=float(checkpoint_value(checkpoint, "post_blur_radius", 0.0)),
                            lowpass_size=int(checkpoint_value(checkpoint, "lowpass_size", 0)),
                        )
                    )
                if use_tcp:
                    raw_tcp = raw_tcp_for_frames_from_obs(
                        obs=obs,
                        frames=batch_frames,
                        checkpoint=checkpoint,
                    )
                    tcp_tensors = [
                        normalize_tcp_for_prediction(row, checkpoint, device)
                        for row in raw_tcp
                    ]
                pred_mean, _ = predict_image_batch(
                    model=model,
                    checkpoint=checkpoint,
                    image_tensors=tensors,
                    device=device,
                    tcp_tensors=tcp_tensors,
                )
                pred_chunks.append(pred_mean.astype(np.float32))

            pred = np.concatenate(pred_chunks, axis=0)
            if force_frame == "world":
                pred = rotate_force_to_world(
                    pred,
                    tcp_rotmat_base=rotations,
                    wrench_frame=wrench_frame,
                )

            demo_rows.append(
                {
                    "demo": demo_name,
                    "x": np.asarray(xs, dtype=np.float32),
                    "frame": frames,
                    "actual": actual,
                    "pred": pred,
                }
            )

    if not demo_rows:
        raise RuntimeError("No demo traces were built.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).resolve().parent / "demo_overlay_actual_vs_pred"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "actual_vs_pred_overlay.png"
    data_path = output_dir / "actual_vs_pred_overlay.json"
    plot_actual_pred_demo_overlays(
        demo_rows=demo_rows,
        names=names,
        out_path=plot_path,
        x_axis=args.x_axis,
        show_mean=not args.no_mean,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "hdf5_path": str(hdf5_path),
        "target_names": names,
        "force_frame": force_frame,
        "wrench_frame": wrench_frame,
        "image_key": image_key,
        "wrench_key": wrench_key,
        "x_axis": args.x_axis,
        "demo_count": len(demo_rows),
        "demos": [
            {
                "demo": row["demo"],
                "x": row["x"].astype(float).tolist(),
                "frame": row["frame"].astype(int).tolist(),
                "actual": row["actual"].astype(float).tolist(),
                "pred": row["pred"].astype(float).tolist(),
            }
            for row in demo_rows
        ],
    }
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved plot: {plot_path}")
    print(f"saved data: {data_path}")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Optional positional HDF5 dataset path.",
    )
    parser.add_argument(
        "--hdf5",
        "--data",
        "--dataset",
        dest="hdf5",
        default=DEFAULT_HDF5,
        help="Input HDF5 dataset path.",
    )
    parser.add_argument("--image-key", default="image_R", help="Observation image key.")
    parser.add_argument(
        "--wrench-key",
        default="wrench_wrist_R",
        help="Observation wrench key.",
    )
    parser.add_argument(
        "--force-dims",
        default="0,1,2",
        help="Comma or space separated wrench dims. 0,1,2 are force xyz.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.05,
        help="Centered timestamp window length for wrench distribution labels.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("window", "sample", "phase"),
        default="window",
        help=(
            "window: label each image from its local wrench interval. "
            "sample: train mean/logvar from one force sample per image. "
            "phase: label each image from the cross-demo force distribution at "
            "the same normalized task phase."
        ),
    )
    parser.add_argument(
        "--image-force-align",
        choices=("center-window", "image-to-next", "prev-to-image", "nearest"),
        default="center-window",
        help=(
            "How to aggregate high-rate wrench samples for one image. "
            "image-to-next assigns wrench samples until the next image to the "
            "current image. nearest uses the closest single wrench sample."
        ),
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=100,
        help="Number of normalized task-progress bins for --label-mode phase.",
    )
    parser.add_argument(
        "--phase-source",
        choices=("frame", "time"),
        default="frame",
        help="Use normalized frame index or timestamp as phase for phase labels.",
    )
    parser.add_argument(
        "--target-std-floor",
        type=float,
        default=0.05,
        help="Minimum target std in physical units before normalization.",
    )
    parser.add_argument(
        "--min-window-samples",
        type=int,
        default=1,
        help="Skip image frames with fewer wrench samples in the time window.",
    )
    parser.add_argument(
        "--zero-wrench-start-sec",
        type=float,
        default=0.0,
        help=(
            "Legacy zero offset from the first N seconds. Used only when "
            "--wrench-zero-samples is 0. 0 disables it."
        ),
    )
    parser.add_argument(
        "--wrench-zero-samples",
        type=int,
        default=10,
        help="Subtract the mean of the first N wrench samples per demo. 0 disables it.",
    )
    parser.add_argument(
        "--wrench-ema-alpha",
        type=float,
        default=0.03,
        help="Apply EMA to zeroed wrench with this alpha. 0 disables EMA.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Use only the first N demos, useful for quick tests.",
    )
    parser.add_argument(
        "--limit-frames-per-demo",
        type=int,
        default=None,
        help="Use at most N evenly spaced image frames per demo.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an image-to-force-distribution model from HDF5 demos.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Train the model.")
    add_data_args(train)
    train.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    train.add_argument(
        "--experiment-note",
        default="",
        help="Short human-readable note saved to experiment_summary.txt.",
    )
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save epoch_XXXX.pt every N epochs. 0 disables periodic checkpoints.",
    )
    train.add_argument(
        "--save-test-images",
        type=int,
        default=20,
        help="Save N transformed held-out images and labels under output_dir/test_images.",
    )
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument(
        "--encoder-backend",
        choices=("cnn", "policy-frozen"),
        default="cnn",
        help="Use the small trainable CNN or a frozen vision encoder from a policy checkpoint.",
    )
    train.add_argument(
        "--policy-checkpoint",
        default=DEFAULT_POLICY_CHECKPOINT,
        help="Diffusion policy checkpoint used when --encoder-backend policy-frozen.",
    )
    train.add_argument(
        "--mlp-hidden",
        type=int,
        default=128,
        help="Hidden width for the force head when using --encoder-backend policy-frozen.",
    )
    train.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.3,
        help="Dropout for the force head when using --encoder-backend policy-frozen.",
    )
    train.add_argument(
        "--head-mode",
        choices=("direct", "bottleneck", "mixture-template"),
        default="direct",
        help=(
            "Force head structure for policy-frozen encoder. direct is the original "
            "MLP; bottleneck forces a small latent; mixture-template predicts from "
            "learned shared force templates plus a bounded residual."
        ),
    )
    train.add_argument(
        "--bottleneck-dim",
        type=int,
        default=8,
        help="Latent dimension used by --head-mode bottleneck.",
    )
    train.add_argument(
        "--template-count",
        type=int,
        default=4,
        help="Number of learned force templates used by --head-mode mixture-template.",
    )
    train.add_argument(
        "--residual-scale",
        type=float,
        default=0.25,
        help=(
            "Maximum normalized residual magnitude for --head-mode mixture-template. "
            "Smaller values force predictions to stay closer to shared templates."
        ),
    )
    train.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.0,
        help=(
            "Gaussian noise std added to normalized frozen policy features during "
            "training. Used with --encoder-backend policy-frozen."
        ),
    )
    train.add_argument(
        "--feature-dropout",
        type=float,
        default=0.0,
        help=(
            "Dropout directly on normalized frozen policy features during training. "
            "Used with --encoder-backend policy-frozen."
        ),
    )
    train.add_argument(
        "--tcp-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to normalized TCP features during training.",
    )
    train.add_argument(
        "--tcp-dropout",
        type=float,
        default=0.0,
        help="Dropout applied to normalized TCP features during training.",
    )
    train.add_argument(
        "--use-tcp",
        action="store_true",
        help="Append actual TCP pose from joint FK to the force MLP input.",
    )
    train.add_argument(
        "--tcp-joint-key",
        default="joint_R",
        help="Observation joint key used for FK when --use-tcp is enabled.",
    )
    train.add_argument(
        "--tcp-history-steps",
        type=int,
        default=1,
        help=(
            "Number of consecutive TCP poses appended to the MLP input, including "
            "the current image frame. 3 means [t-2, t-1, t]."
        ),
    )
    train.add_argument(
        "--tcp-history-stride",
        type=int,
        default=1,
        help="Frame stride between TCP history poses.",
    )
    train.add_argument(
        "--tcp-input-mode",
        choices=("absolute", "delta", "current-delta-xyz"),
        default="absolute",
        help=(
            "absolute flattens TCP history poses. delta uses consecutive TCP "
            "feature differences. current-delta-xyz uses tcp(t) plus "
            "xyz(t)-xyz(t-stride)."
        ),
    )
    train.add_argument(
        "--image-crop-ratio",
        type=float,
        default=0.95,
        help="UNet-style crop ratio after resize/center-crop. 0 disables crop.",
    )
    train.add_argument(
        "--bgr-to-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert common-data BGR images to RGB.",
    )
    train.add_argument(
        "--color-jitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply UNet-style ColorJitter during training augmentation.",
    )
    train.add_argument(
        "--grayscale-p",
        type=float,
        default=0.005,
        help="Random grayscale probability during training augmentation.",
    )
    train.add_argument(
        "--post-blur-radius",
        type=float,
        default=0.0,
        help=(
            "Gaussian blur radius applied after crop/color augmentation and before "
            "normalization. Saved in checkpoints and reused for prediction/plots."
        ),
    )
    train.add_argument(
        "--lowpass-size",
        type=int,
        default=0,
        help=(
            "Downsample image to this square size and upsample back before blur. "
            "0 disables it."
        ),
    )
    train.add_argument(
        "--consistency-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra loss that reduces prediction variance within normalized progress "
            "bins inside each batch. 0 disables it."
        ),
    )
    train.add_argument(
        "--consistency-bins",
        type=int,
        default=32,
        help="Number of normalized progress bins for --consistency-loss-weight.",
    )
    train.add_argument(
        "--smoothness-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra loss on adjacent predictions after sorting each batch by normalized "
            "progress. 0 disables it."
        ),
    )
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--val-ratio", type=float, default=0.1)
    train.add_argument(
        "--val-demo-count",
        type=int,
        default=None,
        help=(
            "Hold out exactly N whole demos for validation/test image export. "
            "Overrides --val-ratio when set."
        ),
    )
    train.add_argument(
        "--val-demos",
        default=None,
        help=(
            "Comma or space separated demo names/numbers to hold out, e.g. "
            "'demo_0,demo_5' or '0 5'. Overrides --val-demo-count."
        ),
    )
    train.add_argument(
        "--val-every",
        type=int,
        default=20,
        help="Run validation every N epochs. 0 disables validation.",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=16)
    train.add_argument("--device", default="auto")
    train.add_argument("--amp", action="store_true", help="Use CUDA AMP.")
    train.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply train image random crop/color jitter. Use --no-augment to disable.",
    )
    train.add_argument(
        "--torch-threads",
        type=int,
        default=8,
        help="Torch CPU compute threads. 0 keeps PyTorch default.",
    )
    train.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="Torch CPU interop threads. 0 keeps PyTorch default.",
    )
    train.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="Use MKL-DNN CPU kernels. Disabled by default to avoid NaN gradients.",
    )
    train.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print training progress every N batches. 0 disables batch logs.",
    )
    train.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Clip gradient norm. 0 disables clipping.",
    )
    train.add_argument("--logvar-min", type=float, default=-10.0)
    train.add_argument("--logvar-max", type=float, default=5.0)
    train.set_defaults(func=train_command)

    inspect_parser = subparsers.add_parser("inspect", help="Print dataset label stats.")
    add_data_args(inspect_parser)
    inspect_parser.set_defaults(func=inspect_command)

    predict = subparsers.add_parser("predict", help="Predict distribution for an image.")
    predict.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Optional HDF5 dataset path override for --hdf5-frame.",
    )
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", default=None, help="Standalone image file path.")
    predict.add_argument(
        "--hdf5-frame",
        default=None,
        help="Frame inside HDF5, e.g. demo_0:10 or 0:10.",
    )
    predict.add_argument(
        "--hdf5",
        "--data",
        "--dataset",
        dest="hdf5",
        default=None,
        help="Override HDF5 path.",
    )
    predict.add_argument("--image-key", default=None, help="Override HDF5 image key.")
    predict.add_argument("--image-size", type=int, default=None)
    predict.add_argument(
        "--bgr-to-rgb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint BGR-to-RGB setting for prediction.",
    )
    predict.add_argument("--device", default="auto")
    predict.set_defaults(func=predict_command)

    plot = subparsers.add_parser(
        "plot-test-images",
        help="Plot test images with predicted force distributions.",
    )
    plot.add_argument("--checkpoint", required=True)
    plot.add_argument("--image-dir", required=True)
    plot.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to IMAGE_DIR/plots.",
    )
    plot.add_argument("--device", default="auto")
    plot.add_argument("--image-size", type=int, default=None)
    plot.add_argument(
        "--bgr-to-rgb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Used only with --raw-images.",
    )
    plot.add_argument(
        "--raw-images",
        action="store_true",
        help="Apply checkpoint image preprocessing before plotting/predicting.",
    )
    plot.add_argument("--max-images", type=int, default=None)
    plot.set_defaults(func=plot_test_images_command)

    demo_plot = subparsers.add_parser(
        "plot-demo-timeseries",
        help="Plot predicted mean force and actual mean force over one demo.",
    )
    demo_plot.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Optional HDF5 dataset path override.",
    )
    demo_plot.add_argument("--checkpoint", required=True)
    demo_plot.add_argument(
        "--demo",
        required=True,
        help="Demo name or number, e.g. demo_0 or 0.",
    )
    demo_plot.add_argument(
        "--hdf5",
        "--data",
        "--dataset",
        dest="hdf5",
        default=None,
        help="Override HDF5 path.",
    )
    demo_plot.add_argument("--image-key", default=None, help="Override HDF5 image key.")
    demo_plot.add_argument("--wrench-key", default=None, help="Override HDF5 wrench key.")
    demo_plot.add_argument("--output-dir", default=None)
    demo_plot.add_argument("--device", default="auto")
    demo_plot.add_argument("--batch-size", type=int, default=128)
    demo_plot.add_argument("--image-size", type=int, default=None)
    demo_plot.add_argument(
        "--bgr-to-rgb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint BGR-to-RGB setting.",
    )
    demo_plot.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth image frame from the demo.",
    )
    demo_plot.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Plot only the first N selected image frames.",
    )
    demo_plot.set_defaults(func=plot_demo_timeseries_command)

    all_demo_plot = subparsers.add_parser(
        "plot-all-demo-forces",
        help="Overlay actual force traces from all demos.",
    )
    add_data_args(all_demo_plot)
    all_demo_plot.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the plot/data. Defaults to force_distribution_height/force_overlays.",
    )
    all_demo_plot.add_argument(
        "--x-axis",
        choices=("phase", "time"),
        default="phase",
        help="Use normalized demo progress or seconds from each demo start.",
    )
    all_demo_plot.add_argument(
        "--no-mean",
        action="store_true",
        help="Do not draw the all-demo mean overlay.",
    )
    all_demo_plot.set_defaults(func=plot_all_demo_forces_command)

    actual_pred_plot = subparsers.add_parser(
        "plot-all-demo-actual-pred",
        help="Overlay actual and predicted mean force traces from all demos in a 3x2 plot.",
    )
    actual_pred_plot.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Optional HDF5 dataset path override.",
    )
    actual_pred_plot.add_argument("--checkpoint", required=True)
    actual_pred_plot.add_argument(
        "--hdf5",
        "--data",
        "--dataset",
        dest="hdf5",
        default=None,
        help="Override HDF5 path.",
    )
    actual_pred_plot.add_argument("--image-key", default=None, help="Override HDF5 image key.")
    actual_pred_plot.add_argument("--wrench-key", default=None, help="Override HDF5 wrench key.")
    actual_pred_plot.add_argument("--output-dir", default=None)
    actual_pred_plot.add_argument("--device", default="auto")
    actual_pred_plot.add_argument("--batch-size", type=int, default=128)
    actual_pred_plot.add_argument("--image-size", type=int, default=None)
    actual_pred_plot.add_argument(
        "--bgr-to-rgb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint BGR-to-RGB setting.",
    )
    actual_pred_plot.add_argument(
        "--x-axis",
        choices=("phase", "time"),
        default="phase",
        help="Use normalized demo progress or seconds from each demo start.",
    )
    actual_pred_plot.add_argument(
        "--force-frame",
        choices=("sensor", "world"),
        default="sensor",
        help="Plot force in the original stored frame or rotate xyz force into world frame.",
    )
    actual_pred_plot.add_argument(
        "--wrench-frame",
        choices=("base", "tcp", "tcp-inverse"),
        default="base",
        help=(
            "Frame of stored wrench xyz before world conversion. base applies the "
            "right-base-to-world rotation only; tcp applies R_base_tcp first; "
            "tcp-inverse applies R_base_tcp.T first."
        ),
    )
    actual_pred_plot.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth image frame from each demo.",
    )
    actual_pred_plot.add_argument(
        "--limit-frames-per-demo",
        type=int,
        default=None,
        help="Use at most N selected frames from each demo.",
    )
    actual_pred_plot.add_argument("--max-demos", type=int, default=None)
    actual_pred_plot.add_argument(
        "--no-mean",
        action="store_true",
        help="Do not draw all-demo mean lines.",
    )
    actual_pred_plot.set_defaults(func=plot_all_demo_actual_pred_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    commands = {
        "train",
        "inspect",
        "predict",
        "plot-test-images",
        "plot-demo-timeseries",
        "plot-all-demo-forces",
        "plot-all-demo-actual-pred",
    }
    if not argv:
        argv = ["train"]
    elif argv[0] not in commands and argv[0] not in {"-h", "--help"}:
        argv = ["train"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
