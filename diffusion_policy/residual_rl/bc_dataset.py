"""Lazy paired-HDF5 dataset for residual behavior cloning.

Each sample is aligned as follows (the default target shift is one)::

    observation_t
    base_target_{t+1}
    residual(actual_target_{t+1} -> virtual_target_{t+1})

The base target may come from the actual-action file or an aligned slow-policy
prediction cache.  Residual supervision always comes from the paired actual
and virtual actions and only corrects the leading end-effector pose9.  A hand
action tail is owned by the base policy and never enters the residual target.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, Mapping, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.residual_rl.pose import (
    abs_pose9_to_relative_pose9,
    current_obs_to_pose9,
    delta6_from_base_to_target,
    pose_like_to_pose9,
)


PathLike = Union[str, os.PathLike]

DEFAULT_OBSERVATION_KEYS = {
    "image0": "obs/image0",
    "robot_pose_R": "obs/robot_pose_R",
    "robot_quat_R": "obs/robot_quat_R",
    "hand_pose_R": "obs/hand_pose_R",
    "wrench_wrist_R": "obs/wrench_wrist_R",
}


def _demo_sort_key(name: str) -> Tuple[int, object]:
    try:
        return (0, int(name.rsplit("_", 1)[-1]))
    except ValueError:
        return (1, name)


def _get_dataset(demo: h5py.Group, key: str) -> h5py.Dataset:
    node = demo
    clean_key = str(key).strip("/")
    if not clean_key:
        raise ValueError("HDF5 dataset key must not be empty")
    for part in clean_key.split("/"):
        if part not in node:
            raise KeyError(f"Missing HDF5 key '{key}' under '{demo.name}'")
        node = node[part]
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"HDF5 key '{key}' under '{demo.name}' is not a dataset")
    return node


def _as_float_tensor(value: np.ndarray) -> torch.Tensor:
    array = np.array(value, dtype=np.float32, copy=True)
    return torch.from_numpy(array)


class PairedResidualBCDataset(Dataset):
    """Memory-efficient, single-step residual BC dataset.

    Only demo names, lengths, and cumulative offsets are retained in memory.
    HDF5 files are opened lazily in each process, making the dataset safe to use
    with multi-worker ``DataLoader`` instances.

    Args:
        actual_path: HDF5 containing actual actions and observations.
        virtual_path: Aligned HDF5 containing virtual actions.
        base_action_source: ``"actual"`` (default), ``"virtual"``, or the path
            of an aligned slow-prediction HDF5.
        base_action_key: Dataset path relative to each demo for the absolute
            base action.  This can select a cache key such as
            ``"obs/slow_pred_target_abs"``.
        action_target_shift: Align ``obs_t`` with actual/virtual target
            ``t + shift``.  The default is one and trailing observations are
            excluded rather than padded.
        base_action_target_shift: Base target shift.  Defaults to
            ``action_target_shift``.
    """

    def __init__(
        self,
        actual_path: PathLike,
        virtual_path: PathLike,
        *,
        base_action_source: Union[str, os.PathLike] = "actual",
        base_action_key: str = "actions",
        actual_action_key: str = "actions",
        virtual_action_key: str = "actions",
        observation_keys: Optional[Mapping[str, str]] = None,
        action_target_shift: int = 1,
        base_action_target_shift: Optional[int] = None,
        normalize_uint8_image: bool = True,
        expected_action_dim: Optional[int] = 16,
    ) -> None:
        super().__init__()
        self.actual_path = pathlib.Path(actual_path).expanduser().resolve()
        self.virtual_path = pathlib.Path(virtual_path).expanduser().resolve()
        if not self.actual_path.is_file():
            raise FileNotFoundError(self.actual_path)
        if not self.virtual_path.is_file():
            raise FileNotFoundError(self.virtual_path)

        self.base_source_name, self.base_path = self._resolve_base_source(
            base_action_source
        )
        if not self.base_path.is_file():
            raise FileNotFoundError(self.base_path)

        self.base_action_key = str(base_action_key)
        self.actual_action_key = str(actual_action_key)
        self.virtual_action_key = str(virtual_action_key)
        self.observation_keys = dict(
            DEFAULT_OBSERVATION_KEYS
            if observation_keys is None
            else observation_keys
        )
        missing_obs_names = set(DEFAULT_OBSERVATION_KEYS) - set(
            self.observation_keys
        )
        if missing_obs_names:
            raise ValueError(
                "observation_keys is missing required logical keys: "
                f"{sorted(missing_obs_names)}"
            )

        self.action_target_shift = int(action_target_shift)
        if self.action_target_shift < 0:
            raise ValueError(
                f"action_target_shift must be non-negative, got {action_target_shift}"
            )
        if base_action_target_shift is None:
            base_action_target_shift = self.action_target_shift
        self.base_action_target_shift = int(base_action_target_shift)
        if self.base_action_target_shift < 0:
            raise ValueError(
                "base_action_target_shift must be non-negative, got "
                f"{base_action_target_shift}"
            )
        self.normalize_uint8_image = bool(normalize_uint8_image)
        self.expected_action_dim = (
            None if expected_action_dim is None else int(expected_action_dim)
        )

        self._handles: Dict[pathlib.Path, h5py.File] = {}
        self._owner_pid: Optional[int] = None
        (
            self.demo_names,
            self._source_episode_lengths,
            self._episode_lengths,
            self._episode_ends,
            self.action_dim,
            self.base_action_dim,
        ) = self._inspect_files()

    def _resolve_base_source(
        self, source: Union[str, os.PathLike]
    ) -> Tuple[str, pathlib.Path]:
        if isinstance(source, str):
            source_name = source.lower()
            if source_name == "actual":
                return source_name, self.actual_path
            if source_name == "virtual":
                return source_name, self.virtual_path
        path = pathlib.Path(source).expanduser().resolve()
        return "file", path

    def _inspect_files(self):
        unique_paths = {self.actual_path, self.virtual_path, self.base_path}
        files = {path: h5py.File(path, "r") for path in unique_paths}
        try:
            for path, file in files.items():
                if "data" not in file:
                    raise KeyError(f"HDF5 file has no 'data' group: {path}")

            actual_data = files[self.actual_path]["data"]
            virtual_data = files[self.virtual_path]["data"]
            base_data = files[self.base_path]["data"]
            demo_names = tuple(sorted(actual_data.keys(), key=_demo_sort_key))
            if not demo_names:
                raise ValueError(f"No demos found in {self.actual_path}")

            for label, data_group in (
                ("virtual", virtual_data),
                ("base", base_data),
            ):
                missing = [name for name in demo_names if name not in data_group]
                extra = [name for name in data_group if name not in actual_data]
                if missing or extra:
                    raise ValueError(
                        f"{label} demo set is not aligned with actual: "
                        f"missing={missing[:5]}, extra={extra[:5]}"
                    )

            source_lengths = []
            usable_lengths = []
            action_dim = None
            base_action_dim = None
            largest_shift = max(
                self.action_target_shift, self.base_action_target_shift
            )
            for demo_name in demo_names:
                actual_demo = actual_data[demo_name]
                virtual_demo = virtual_data[demo_name]
                base_demo = base_data[demo_name]
                actual_action = _get_dataset(
                    actual_demo, self.actual_action_key
                )
                virtual_action = _get_dataset(
                    virtual_demo, self.virtual_action_key
                )
                base_action = _get_dataset(base_demo, self.base_action_key)

                if actual_action.ndim != 2 or actual_action.shape[-1] < 9:
                    raise ValueError(
                        f"{actual_action.name} must have shape (T, D>=9), got "
                        f"{actual_action.shape}"
                    )
                if virtual_action.shape != actual_action.shape:
                    raise ValueError(
                        f"Action mismatch for {demo_name}: actual="
                        f"{actual_action.shape}, virtual={virtual_action.shape}"
                    )
                if base_action.ndim != 2 or base_action.shape[-1] < 9:
                    raise ValueError(
                        f"{base_action.name} must have shape (T, D>=9), got "
                        f"{base_action.shape}"
                    )
                episode_length = int(actual_action.shape[0])
                if int(base_action.shape[0]) != episode_length:
                    raise ValueError(
                        f"Base action length mismatch for {demo_name}: actual="
                        f"{episode_length}, base={base_action.shape[0]}"
                    )
                if self.expected_action_dim is not None:
                    if actual_action.shape[-1] != self.expected_action_dim:
                        raise ValueError(
                            f"{actual_action.name} expected action dim "
                            f"{self.expected_action_dim}, got {actual_action.shape[-1]}"
                        )
                    if base_action.shape[-1] != self.expected_action_dim:
                        raise ValueError(
                            f"{base_action.name} expected base action dim "
                            f"{self.expected_action_dim}, got {base_action.shape[-1]}"
                        )

                self._validate_observations(actual_demo, episode_length)
                usable_length = episode_length - largest_shift
                if usable_length <= 0:
                    raise ValueError(
                        f"{demo_name} length {episode_length} has no samples for "
                        f"target shift {largest_shift}"
                    )
                source_lengths.append(episode_length)
                usable_lengths.append(usable_length)
                action_dim = int(actual_action.shape[-1])
                base_action_dim = int(base_action.shape[-1])

            usable_lengths_array = np.asarray(usable_lengths, dtype=np.int64)
            return (
                demo_names,
                np.asarray(source_lengths, dtype=np.int64),
                usable_lengths_array,
                np.cumsum(usable_lengths_array, dtype=np.int64),
                action_dim,
                base_action_dim,
            )
        finally:
            for file in files.values():
                file.close()

    def _validate_observations(
        self, actual_demo: h5py.Group, episode_length: int
    ) -> None:
        datasets = {
            logical_name: _get_dataset(actual_demo, key)
            for logical_name, key in self.observation_keys.items()
        }
        for logical_name, dataset in datasets.items():
            if dataset.shape[0] != episode_length:
                raise ValueError(
                    f"Observation length mismatch for {dataset.name}: "
                    f"actions={episode_length}, observation={dataset.shape[0]}"
                )

        expected_shapes = {
            "robot_pose_R": (3,),
            "robot_quat_R": (4,),
            "hand_pose_R": (7,),
            "wrench_wrist_R": (6, 32),
        }
        for logical_name, expected_shape in expected_shapes.items():
            dataset = datasets[logical_name]
            if tuple(dataset.shape[1:]) != expected_shape:
                raise ValueError(
                    f"{dataset.name} expected per-step shape {expected_shape}, "
                    f"got {dataset.shape[1:]}"
                )
        image_shape = tuple(datasets["image0"].shape[1:])
        if len(image_shape) != 3 or (
            image_shape[0] != 3 and image_shape[-1] != 3
        ):
            raise ValueError(
                f"{datasets['image0'].name} must be CHW or HWC RGB, got "
                f"{image_shape}"
            )

    @property
    def episode_lengths(self) -> np.ndarray:
        """Number of usable shifted samples in each demo."""
        return self._episode_lengths.copy()

    @property
    def source_episode_lengths(self) -> np.ndarray:
        return self._source_episode_lengths.copy()

    def __len__(self) -> int:
        return int(self._episode_ends[-1])

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = int(
            np.searchsorted(self._episode_ends, index, side="right")
        )
        episode_start = (
            0 if episode_index == 0 else int(self._episode_ends[episode_index - 1])
        )
        return episode_index, int(index - episode_start)

    def _ensure_handles(self) -> None:
        pid = os.getpid()
        if self._owner_pid == pid and self._handles:
            return
        self.close()
        for path in {self.actual_path, self.virtual_path, self.base_path}:
            self._handles[path] = h5py.File(path, "r")
        self._owner_pid = pid

    def close(self) -> None:
        for file in self._handles.values():
            try:
                file.close()
            except Exception:
                pass
        self._handles = {}
        self._owner_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_owner_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _read_observation(
        self, actual_demo: h5py.Group, step_index: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
        obs_tensors: Dict[str, torch.Tensor] = {}
        obs_numpy: Dict[str, np.ndarray] = {}
        for logical_name, key in self.observation_keys.items():
            value = np.array(
                _get_dataset(actual_demo, key)[step_index], copy=True
            )
            if logical_name == "image0":
                if value.shape[-1] == 3:
                    value = np.moveaxis(value, -1, 0)
                if self.normalize_uint8_image and value.dtype == np.uint8:
                    value = value.astype(np.float32) / 255.0
                else:
                    value = value.astype(np.float32)
            else:
                value = value.astype(np.float32)
                obs_numpy[logical_name] = value
            obs_tensors[logical_name] = _as_float_tensor(value)
        return obs_tensors, obs_numpy

    def __getitem__(self, index: int):
        episode_index, step_index = self._locate(int(index))
        target_index = step_index + self.action_target_shift
        base_target_index = step_index + self.base_action_target_shift
        demo_name = self.demo_names[episode_index]
        self._ensure_handles()

        actual_demo = self._handles[self.actual_path]["data"][demo_name]
        virtual_demo = self._handles[self.virtual_path]["data"][demo_name]
        base_demo = self._handles[self.base_path]["data"][demo_name]

        obs_tensors, obs_numpy = self._read_observation(
            actual_demo, step_index
        )
        actual_action_abs = np.array(
            _get_dataset(actual_demo, self.actual_action_key)[target_index],
            dtype=np.float32,
            copy=True,
        )
        virtual_action_abs = np.array(
            _get_dataset(virtual_demo, self.virtual_action_key)[target_index],
            dtype=np.float32,
            copy=True,
        )
        base_action_abs = np.array(
            _get_dataset(base_demo, self.base_action_key)[base_target_index],
            dtype=np.float32,
            copy=True,
        )

        residual_delta6 = delta6_from_base_to_target(
            pose_like_to_pose9(actual_action_abs),
            pose_like_to_pose9(virtual_action_abs),
        )
        current_pose9 = current_obs_to_pose9(obs_numpy, arm="R")
        base_relative_pose9 = abs_pose9_to_relative_pose9(
            current_pose9, pose_like_to_pose9(base_action_abs)
        )
        if base_action_abs.shape[-1] > 9:
            base_action = np.concatenate(
                (base_relative_pose9, base_action_abs[9:]), axis=-1
            ).astype(np.float32)
        else:
            base_action = base_relative_pose9.astype(np.float32)

        return {
            "obs": obs_tensors,
            "base_action": _as_float_tensor(base_action),
            "base_action_abs": _as_float_tensor(base_action_abs),
            "action": _as_float_tensor(residual_delta6),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "step_index": torch.tensor(step_index, dtype=torch.int64),
            "target_index": torch.tensor(target_index, dtype=torch.int64),
        }


ResidualBCDataset = PairedResidualBCDataset


__all__ = [
    "DEFAULT_OBSERVATION_KEYS",
    "PairedResidualBCDataset",
    "ResidualBCDataset",
]
