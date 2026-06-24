from typing import Dict
import copy
import os
import shutil

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
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


register_codecs()


class FastResidualReplayImageDataset(BaseImageDataset):
    """Dataset for stage-2 fast residual policy.

    Expected HDF5 layout:
        data/demo_i/actions
        data/demo_i/obs/<normal obs keys>
        data/demo_i/obs/<slow_action_key>

    The action target is a residual delta, usually dpos(3) + drotvec(3).
    """

    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            slow_action_key="slow_action_rel",
            action_key="actions",
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            pose_repr: dict = {},
        ):
        if use_cache:
            cache_zarr_path = dataset_path + ".fast_residual.zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on fast residual cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Fast residual cache does not exist. Creating.")
                        replay_buffer = _convert_fast_hdf5_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            action_key=action_key,
                        )
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise
                else:
                    print("Loading cached fast residual ReplayBuffer.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store,
                            store=zarr.MemoryStore(),
                        )
        else:
            replay_buffer = _convert_fast_hdf5_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
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

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys + wrench_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.wrench_keys = wrench_keys
        self.slow_action_key = slow_action_key
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.pose_repr = pose_repr
        self.obs_pose_repr = pose_repr.get("obs_pose_repr", "abs")

        self.use_left_arm = "robot_pose_L" in self.lowdim_keys
        self.use_right_arm = "robot_pose_R" in self.lowdim_keys
        self.use_left_hand = "hand_pose_L" in self.lowdim_keys
        self.use_right_hand = "hand_pose_R" in self.lowdim_keys

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        key_first_k = {}
        if self.n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys + self.wrench_keys:
                key_first_k[key] = self.n_obs_steps
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
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
            num_workers=16,
        )
        for batch in tqdm(dataloader, desc="iterating fast residual dataset for normalization"):
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

        normalizer["action"] = get_range_normalizer_from_stat(
            array_to_stats(data_cache["action"]))

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key == self.slow_action_key and data_cache[key].shape[-1] >= 9:
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
            obs_dict[key] = data[key][t_slice][-1:].astype(np.float32)
            del data[key]

        self._apply_relative_obs_repr(obs_dict)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data

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


def _convert_fast_hdf5_to_replay(store, shape_meta, dataset_path, action_key):
    root = zarr.group(store=store)
    replay_buffer = ReplayBuffer.create_from_group(root)
    obs_meta = shape_meta["obs"]

    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        for demo_name in tqdm(_sorted_demo_keys(data_group), desc="Loading fast residual hdf5"):
            demo = data_group[demo_name]
            episode = {}
            for key in obs_meta:
                episode[key] = _read_demo_dataset(demo, key)
            episode["action"] = _read_demo_dataset(demo, action_key)
            replay_buffer.add_episode(episode)
    return replay_buffer
