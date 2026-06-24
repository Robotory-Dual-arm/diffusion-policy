from typing import Dict
import copy

import h5py
import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.common.pose_repr_util import (
    compute_hand_relative_pose,
    convert_pose_mat_rep,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    array_to_stats_for_wrench,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.pose_util import mat_to_pose10d, pose_to_mat
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.residual_policy.pose_util import (
    pose_like_to_pose9,
    residual_pose9_from_base_to_target,
)


register_codecs()


class FastResidualStepDataset(BaseImageDataset):
    """One-row-per-step dataset for fast residual learning.

    Each sample returns current observations, one base action, and one residual target:
        obs[t], obs[base_action_key][t] -> action[t]
    """

    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            base_action_key="base_action_rel",
            base_action_source_key="obs/actual_action_rel",
            action_key="obs/residual_delta6_gt_actual_to_virtual",
            n_obs_steps=1,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            pose_repr: dict = {},
        ):
        if use_cache:
            raise NotImplementedError("FastResidualStepDataset keeps conversion in memory; use_cache is not needed.")

        replay_buffer = _convert_step_hdf5_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            base_action_key=base_action_key,
            base_action_source_key=base_action_source_key,
            action_key=action_key,
        )

        rgb_keys = []
        lowdim_keys = []
        wrench_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            elif obs_type == "wrench":
                wrench_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        key_first_k = {key: n_obs_steps for key in rgb_keys + lowdim_keys + wrench_keys}
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=n_obs_steps,
            pad_before=n_obs_steps - 1,
            pad_after=0,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.wrench_keys = wrench_keys
        self.base_action_key = base_action_key
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.pose_repr = pose_repr
        self.obs_pose_repr = pose_repr.get("obs_pose_repr", "abs")

        self.use_left_arm = "robot_pose_L" in self.lowdim_keys
        self.use_right_arm = "robot_pose_R" in self.lowdim_keys
        self.use_left_hand = "hand_pose_L" in self.lowdim_keys
        self.use_right_hand = "hand_pose_R" in self.lowdim_keys

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        key_first_k = {
            key: self.n_obs_steps
            for key in self.rgb_keys + self.lowdim_keys + self.wrench_keys
        }
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.n_obs_steps,
            pad_before=self.n_obs_steps - 1,
            pad_after=0,
            episode_mask=~self.train_mask,
            key_first_k=key_first_k,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        data_cache = {key: [] for key in self.lowdim_keys + self.wrench_keys + ["action"]}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=0,
        )
        for batch in tqdm(dataloader, desc="iterating fast step residual dataset for normalization"):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            for key in self.wrench_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            data_cache["action"].append(copy.deepcopy(batch["action"]))
        self.sampler.ignore_rgb(False)

        wrench_history = {}
        for key in data_cache:
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            if key in self.lowdim_keys or key == "action":
                b, t, d = data_cache[key].shape
                data_cache[key] = data_cache[key].reshape(b * t, d)
            elif key in self.wrench_keys:
                b, t, c, h = data_cache[key].shape
                wrench_history[key] = h
                data_cache[key] = data_cache[key].reshape(b * t, c, h)

        if data_cache["action"].shape[-1] >= 9:
            normalizer["action"] = concatenate_normalizer([
                get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., :3])),
                get_identity_normalizer_from_stat(array_to_stats(data_cache["action"][..., 3:9])),
            ])
        else:
            normalizer["action"] = get_range_normalizer_from_stat(
                array_to_stats(data_cache["action"]))

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key == self.base_action_key and data_cache[key].shape[-1] >= 9:
                parts = [
                    get_range_normalizer_from_stat(array_to_stats(data_cache[key][..., :3])),
                    get_identity_normalizer_from_stat(array_to_stats(data_cache[key][..., 3:9])),
                ]
                if data_cache[key].shape[-1] > 9:
                    parts.append(
                        get_range_normalizer_from_stat(array_to_stats(data_cache[key][..., 9:])))
                this_normalizer = concatenate_normalizer(parts)
            elif key.endswith("quat") or "quat" in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("pos") or "pose" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif "wrench" in key or "force" in key or "torque" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        for key in self.wrench_keys:
            stat = array_to_stats_for_wrench(data_cache[key], history=wrench_history[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            if key not in data:
                continue
            image = data[key][t_slice]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            obs_dict[key] = np.moveaxis(image, -1, 1)
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        for key in self.wrench_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        self._apply_relative_obs_repr(obs_dict)
        action = data["action"].astype(np.float32)
        if action.ndim == 1:
            action = action[None]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }

    def _apply_relative_obs_repr(self, obs_dict):
        if self.obs_pose_repr != "relative":
            return

        if self.use_left_arm:
            obs_pose_mat_l = pose_to_mat(np.concatenate([
                obs_dict["robot_pose_L"],
                Rotation.from_quat(obs_dict["robot_quat_L"]).as_rotvec(),
            ], axis=-1))
            rel_mat_l = convert_pose_mat_rep(
                pose_mat=obs_pose_mat_l,
                base_pose_mat=obs_pose_mat_l[-1],
                pose_rep="relative",
                backward=False,
            )
            rel_l = mat_to_pose10d(rel_mat_l)
            obs_dict["robot_pose_L"] = rel_l[..., :3].astype(np.float32)
            obs_dict["robot_quat_L"] = rel_l[..., 3:].astype(np.float32)

        if self.use_right_arm:
            obs_pose_mat_r = pose_to_mat(np.concatenate([
                obs_dict["robot_pose_R"],
                Rotation.from_quat(obs_dict["robot_quat_R"]).as_rotvec(),
            ], axis=-1))
            rel_mat_r = convert_pose_mat_rep(
                pose_mat=obs_pose_mat_r,
                base_pose_mat=obs_pose_mat_r[-1],
                pose_rep="relative",
                backward=False,
            )
            rel_r = mat_to_pose10d(rel_mat_r)
            obs_dict["robot_pose_R"] = rel_r[..., :3].astype(np.float32)
            obs_dict["robot_quat_R"] = rel_r[..., 3:].astype(np.float32)

        if self.use_left_hand:
            obs_dict["hand_pose_L"] = compute_hand_relative_pose(
                pos=obs_dict["hand_pose_L"],
                base_pos=obs_dict["hand_pose_L"][-1],
            ).astype(np.float32)

        if self.use_right_hand:
            obs_dict["hand_pose_R"] = compute_hand_relative_pose(
                pos=obs_dict["hand_pose_R"],
                base_pos=obs_dict["hand_pose_R"][-1],
            ).astype(np.float32)


class FastResidualContextStepDataset(BaseImageDataset):
    """Memory-light dataset for fixed-context fast residual learning.

    Returns one context observation for rgb/proprio keys and a full temporal
    sequence for wrench/base-action/action:
        image/pose/quat[0:1], wrench[0:T], base_action[0:T] -> action[0:T]
    """

    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            base_action_key="base_action_rel",
            base_action_source_key="obs/actual_action_rel",
            action_key="obs/residual_delta6_gt_actual_to_virtual",
            n_obs_steps=16,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            pose_repr: dict = {},
        ):
        if use_cache:
            raise NotImplementedError("FastResidualContextStepDataset keeps conversion in memory; use_cache is not needed.")

        replay_buffer = _convert_step_hdf5_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            base_action_key=base_action_key,
            base_action_source_key=base_action_source_key,
            action_key=action_key,
        )

        rgb_keys = []
        lowdim_keys = []
        wrench_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            elif obs_type == "wrench":
                wrench_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=n_obs_steps,
            pad_before=n_obs_steps - 1,
            pad_after=0,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.wrench_keys = wrench_keys
        self.base_action_key = base_action_key
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.pose_repr = pose_repr
        self.obs_pose_repr = pose_repr.get("obs_pose_repr", "abs")

        self.context_lowdim_keys = [key for key in lowdim_keys if key != base_action_key]
        self.temporal_lowdim_keys = [key for key in lowdim_keys if key == base_action_key]
        self.use_left_arm = "robot_pose_L" in self.context_lowdim_keys
        self.use_right_arm = "robot_pose_R" in self.context_lowdim_keys
        self.use_left_hand = "hand_pose_L" in self.context_lowdim_keys
        self.use_right_hand = "hand_pose_R" in self.context_lowdim_keys

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.n_obs_steps,
            pad_before=self.n_obs_steps - 1,
            pad_after=0,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def _sample_temporal_key(self, key, sample_info):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sample_info
        input_arr = self.replay_buffer[key]
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        if (sample_start_idx > 0) or (sample_end_idx < self.n_obs_steps):
            data = np.zeros(
                shape=(self.n_obs_steps,) + input_arr.shape[1:],
                dtype=input_arr.dtype,
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < self.n_obs_steps:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
            return data
        return sample

    def _sample_context_key(self, key, sample_info):
        buffer_start_idx, _, _, _ = sample_info
        input_arr = self.replay_buffer[key]
        return input_arr[buffer_start_idx:buffer_start_idx + 1]

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        data_cache = {key: [] for key in self.lowdim_keys + self.wrench_keys + ["action"]}
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=0,
        )
        for batch in tqdm(dataloader, desc="iterating fast context residual dataset for normalization"):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            for key in self.wrench_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            data_cache["action"].append(copy.deepcopy(batch["action"]))

        wrench_history = {}
        for key in data_cache:
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            if key in self.lowdim_keys or key == "action":
                b, t, d = data_cache[key].shape
                data_cache[key] = data_cache[key].reshape(b * t, d)
            elif key in self.wrench_keys:
                b, t, c, h = data_cache[key].shape
                wrench_history[key] = h
                data_cache[key] = data_cache[key].reshape(b * t, c, h)

        if data_cache["action"].shape[-1] >= 9:
            normalizer["action"] = concatenate_normalizer([
                get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., :3])),
                get_identity_normalizer_from_stat(array_to_stats(data_cache["action"][..., 3:9])),
            ])
        else:
            normalizer["action"] = get_range_normalizer_from_stat(
                array_to_stats(data_cache["action"]))

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key == self.base_action_key and data_cache[key].shape[-1] >= 9:
                parts = [
                    get_range_normalizer_from_stat(array_to_stats(data_cache[key][..., :3])),
                    get_identity_normalizer_from_stat(array_to_stats(data_cache[key][..., 3:9])),
                ]
                if data_cache[key].shape[-1] > 9:
                    parts.append(
                        get_range_normalizer_from_stat(array_to_stats(data_cache[key][..., 9:])))
                this_normalizer = concatenate_normalizer(parts)
            elif key.endswith("quat") or "quat" in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("pos") or "pose" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif "wrench" in key or "force" in key or "torque" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        for key in self.wrench_keys:
            stat = array_to_stats_for_wrench(data_cache[key], history=wrench_history[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample_info = self.sampler.indices[idx]
        obs_dict = {}

        for key in self.rgb_keys:
            image = self._sample_context_key(key, sample_info)
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            obs_dict[key] = np.moveaxis(image, -1, 1)

        for key in self.context_lowdim_keys:
            obs_dict[key] = self._sample_context_key(key, sample_info).astype(np.float32)

        for key in self.temporal_lowdim_keys:
            obs_dict[key] = self._sample_temporal_key(key, sample_info).astype(np.float32)

        for key in self.wrench_keys:
            obs_dict[key] = self._sample_temporal_key(key, sample_info).astype(np.float32)

        self._apply_relative_obs_repr(obs_dict)
        action = self._sample_temporal_key("action", sample_info).astype(np.float32)
        if action.ndim == 1:
            action = action[None]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }

    def _apply_relative_obs_repr(self, obs_dict):
        if self.obs_pose_repr != "relative":
            return

        if self.use_left_arm:
            obs_pose_mat_l = pose_to_mat(np.concatenate([
                obs_dict["robot_pose_L"],
                Rotation.from_quat(obs_dict["robot_quat_L"]).as_rotvec(),
            ], axis=-1))
            rel_mat_l = convert_pose_mat_rep(
                pose_mat=obs_pose_mat_l,
                base_pose_mat=obs_pose_mat_l[-1],
                pose_rep="relative",
                backward=False,
            )
            rel_l = mat_to_pose10d(rel_mat_l)
            obs_dict["robot_pose_L"] = rel_l[..., :3].astype(np.float32)
            obs_dict["robot_quat_L"] = rel_l[..., 3:].astype(np.float32)

        if self.use_right_arm:
            obs_pose_mat_r = pose_to_mat(np.concatenate([
                obs_dict["robot_pose_R"],
                Rotation.from_quat(obs_dict["robot_quat_R"]).as_rotvec(),
            ], axis=-1))
            rel_mat_r = convert_pose_mat_rep(
                pose_mat=obs_pose_mat_r,
                base_pose_mat=obs_pose_mat_r[-1],
                pose_rep="relative",
                backward=False,
            )
            rel_r = mat_to_pose10d(rel_mat_r)
            obs_dict["robot_pose_R"] = rel_r[..., :3].astype(np.float32)
            obs_dict["robot_quat_R"] = rel_r[..., 3:].astype(np.float32)

        if self.use_left_hand:
            obs_dict["hand_pose_L"] = compute_hand_relative_pose(
                pos=obs_dict["hand_pose_L"],
                base_pos=obs_dict["hand_pose_L"][-1],
            ).astype(np.float32)

        if self.use_right_hand:
            obs_dict["hand_pose_R"] = compute_hand_relative_pose(
                pos=obs_dict["hand_pose_R"],
                base_pos=obs_dict["hand_pose_R"][-1],
            ).astype(np.float32)


def _sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def _read_demo_dataset(demo, key):
    if key == "obs/residual_pose9_gt_actual_to_virtual":
        obs = demo["obs"]
        actual = pose_like_to_pose9(np.asarray(obs["actual_target_abs"]))
        virtual = pose_like_to_pose9(np.asarray(obs["virtual_target_abs"]))
        return residual_pose9_from_base_to_target(actual, virtual)
    if key == "action":
        key = "actions"
    if "/" in key:
        group = demo
        for part in key.split("/"):
            group = group[part]
        return np.asarray(group)
    if key in demo:
        return np.asarray(demo[key])
    obs = demo.get("obs", None)
    if obs is not None and key in obs:
        return np.asarray(obs[key])
    raise KeyError(f"Could not find key '{key}' in demo '{demo.name}'")


def _convert_step_hdf5_to_replay(
        store,
        shape_meta,
        dataset_path,
        base_action_key,
        base_action_source_key,
        action_key):
    root = zarr.group(store=store)
    replay_buffer = ReplayBuffer.create_from_group(root)
    obs_meta = shape_meta["obs"]

    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        for demo_name in tqdm(_sorted_demo_keys(data_group), desc="Loading fast step hdf5"):
            demo = data_group[demo_name]
            episode = {}
            for key in obs_meta:
                if key == base_action_key:
                    episode[key] = _read_demo_dataset(demo, base_action_source_key)
                else:
                    episode[key] = _read_demo_dataset(demo, key)
            episode["action"] = _read_demo_dataset(demo, action_key)
            replay_buffer.add_episode(episode)
    return replay_buffer
