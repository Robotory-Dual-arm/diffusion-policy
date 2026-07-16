import pickle

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from diffusion_policy.residual_rl.bc_dataset import PairedResidualBCDataset
from diffusion_policy.residual_rl.pose import (
    abs_pose9_to_relative_pose9,
    apply_residual_action_to_pose9,
    current_obs_to_pose9,
    delta6_from_base_to_target,
    pose6_to_pose9,
    pose9_to_mat,
)


def _pose9(position, rotvec):
    pose6 = np.asarray([*position, *rotvec], dtype=np.float32)
    return pose6_to_pose9(pose6)


def _make_episode(length, demo_offset):
    actual_actions = []
    virtual_actions = []
    cached_actions = []
    expected_deltas = []
    robot_positions = []
    robot_quaternions = []
    for step in range(length):
        actual_pose = _pose9(
            [0.2 + 0.01 * step, -0.1 + 0.02 * step, 0.3],
            [0.01 * step, -0.02 * step, 0.03 * step],
        )
        hand = np.linspace(0.0, 0.6, 7, dtype=np.float32) + step
        actual = np.concatenate((actual_pose, hand)).astype(np.float32)
        delta = np.asarray(
            [
                0.001 * (step + 1),
                -0.002,
                0.003,
                0.01,
                -0.02 * (step + 1),
                0.005,
            ],
            dtype=np.float32,
        )
        virtual_pose = apply_residual_action_to_pose9(actual_pose, delta)
        # Deliberately change the virtual hand. The residual target must ignore it.
        virtual = np.concatenate((virtual_pose, hand + 100.0)).astype(np.float32)
        cache_delta = np.asarray(
            [-0.004, 0.002, 0.001, 0.0, 0.01, -0.01], dtype=np.float32
        )
        cached_pose = apply_residual_action_to_pose9(actual_pose, cache_delta)
        cached = np.concatenate((cached_pose, hand + 10.0)).astype(np.float32)

        obs_rotvec = np.asarray([0.0, 0.01 * step, -0.02 * step])
        robot_positions.append(
            np.asarray(
                [0.1 + 0.005 * step, -0.2, 0.25 + 0.001 * step],
                dtype=np.float32,
            )
        )
        robot_quaternions.append(
            Rotation.from_rotvec(obs_rotvec).as_quat().astype(np.float32)
        )
        actual_actions.append(actual)
        virtual_actions.append(virtual)
        cached_actions.append(cached)
        expected_deltas.append(delta)

    images = np.empty((length, 4, 5, 3), dtype=np.uint8)
    for step in range(length):
        images[step].fill(10 + demo_offset + step)
    wrench = np.arange(length * 6 * 32, dtype=np.float32).reshape(
        length, 6, 32
    )
    hand_obs = np.arange(length * 7, dtype=np.float32).reshape(length, 7)
    return {
        "actual": np.stack(actual_actions),
        "virtual": np.stack(virtual_actions),
        "cached": np.stack(cached_actions),
        "delta": np.stack(expected_deltas),
        "image0": images,
        "robot_pose_R": np.stack(robot_positions),
        "robot_quat_R": np.stack(robot_quaternions),
        "hand_pose_R": hand_obs,
        "wrench_wrist_R": wrench,
    }


def _write_files(tmp_path):
    actual_path = tmp_path / "actual.hdf5"
    virtual_path = tmp_path / "virtual.hdf5"
    cache_path = tmp_path / "slow_cache.hdf5"
    episodes = {
        "demo_0": _make_episode(4, 0),
        "demo_2": _make_episode(3, 20),
    }
    with h5py.File(actual_path, "w") as actual_file, h5py.File(
        virtual_path, "w"
    ) as virtual_file, h5py.File(cache_path, "w") as cache_file:
        actual_data = actual_file.create_group("data")
        virtual_data = virtual_file.create_group("data")
        cache_data = cache_file.create_group("data")
        for demo_name, episode in episodes.items():
            actual_demo = actual_data.create_group(demo_name)
            virtual_demo = virtual_data.create_group(demo_name)
            cache_demo = cache_data.create_group(demo_name)
            actual_demo.create_dataset("actions", data=episode["actual"])
            virtual_demo.create_dataset("actions", data=episode["virtual"])
            cache_demo.create_dataset("actions", data=episode["cached"])
            obs = actual_demo.create_group("obs")
            for key in (
                "image0",
                "robot_pose_R",
                "robot_quat_R",
                "hand_pose_R",
                "wrench_wrist_R",
            ):
                obs.create_dataset(key, data=episode[key])
    return actual_path, virtual_path, cache_path, episodes


def test_shifted_single_step_sample_shapes_and_targets(tmp_path):
    actual_path, virtual_path, _, episodes = _write_files(tmp_path)
    dataset = PairedResidualBCDataset(actual_path, virtual_path)

    assert len(dataset) == (4 - 1) + (3 - 1)
    np.testing.assert_array_equal(dataset.episode_lengths, [3, 2])
    assert dataset.action_dim == 16
    assert dataset.base_action_dim == 16

    sample = dataset[0]
    assert sample["obs"]["image0"].shape == (3, 4, 5)
    assert sample["obs"]["robot_pose_R"].shape == (3,)
    assert sample["obs"]["robot_quat_R"].shape == (4,)
    assert sample["obs"]["hand_pose_R"].shape == (7,)
    assert sample["obs"]["wrench_wrist_R"].shape == (6, 32)
    assert sample["base_action"].shape == (16,)
    assert sample["base_action_abs"].shape == (16,)
    assert sample["action"].shape == (6,)
    torch.testing.assert_close(
        sample["obs"]["image0"],
        torch.full((3, 4, 5), 10.0 / 255.0),
    )
    assert sample["step_index"].item() == 0
    assert sample["target_index"].item() == 1

    episode = episodes["demo_0"]
    expected_delta = delta6_from_base_to_target(
        episode["actual"][1, :9], episode["virtual"][1, :9]
    )
    torch.testing.assert_close(sample["action"], torch.from_numpy(expected_delta))
    # The deliberately different virtual hand must not affect residual delta6.
    torch.testing.assert_close(
        sample["action"], torch.from_numpy(episode["delta"][1])
    )
    torch.testing.assert_close(
        sample["base_action_abs"], torch.from_numpy(episode["actual"][1])
    )

    current_pose = current_obs_to_pose9(
        {
            "robot_pose_R": episode["robot_pose_R"][0],
            "robot_quat_R": episode["robot_quat_R"][0],
        }
    )
    expected_relative = abs_pose9_to_relative_pose9(
        current_pose, episode["actual"][1, :9]
    )
    np.testing.assert_allclose(
        pose9_to_mat(sample["base_action"][:9].numpy()),
        pose9_to_mat(expected_relative),
        atol=1e-6,
    )
    torch.testing.assert_close(
        sample["base_action"][9:],
        torch.from_numpy(episode["actual"][1, 9:]),
    )

    dataset.close()


def test_demo_boundary_does_not_pad_or_cross(tmp_path):
    actual_path, virtual_path, _, episodes = _write_files(tmp_path)
    dataset = PairedResidualBCDataset(actual_path, virtual_path)

    # demo_0 contributes indices 0,1,2; global index 3 starts demo_2 at step 0.
    sample = dataset[3]
    assert sample["episode_index"].item() == 1
    assert sample["step_index"].item() == 0
    assert sample["target_index"].item() == 1
    torch.testing.assert_close(
        sample["base_action_abs"],
        torch.from_numpy(episodes["demo_2"]["actual"][1]),
    )
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_optional_slow_cache_changes_only_base_condition(tmp_path):
    actual_path, virtual_path, cache_path, episodes = _write_files(tmp_path)
    actual_base_dataset = PairedResidualBCDataset(actual_path, virtual_path)
    cached_base_dataset = PairedResidualBCDataset(
        actual_path,
        virtual_path,
        base_action_source=cache_path,
        base_action_key="actions",
    )

    actual_sample = actual_base_dataset[1]
    cached_sample = cached_base_dataset[1]
    torch.testing.assert_close(actual_sample["action"], cached_sample["action"])
    torch.testing.assert_close(
        cached_sample["base_action_abs"],
        torch.from_numpy(episodes["demo_0"]["cached"][2]),
    )
    assert not torch.allclose(
        actual_sample["base_action"], cached_sample["base_action"]
    )


def test_hdf5_handles_are_lazy_and_pickle_safe(tmp_path):
    actual_path, virtual_path, _, _ = _write_files(tmp_path)
    dataset = PairedResidualBCDataset(actual_path, virtual_path)
    assert dataset._handles == {}

    expected = dataset[0]["action"]
    assert dataset._handles
    cloned = pickle.loads(pickle.dumps(dataset))
    assert cloned._handles == {}
    torch.testing.assert_close(cloned[0]["action"], expected)

    cloned.close()
    dataset.close()


def test_misaligned_demo_sets_fail_during_construction(tmp_path):
    actual_path, virtual_path, _, _ = _write_files(tmp_path)
    with h5py.File(virtual_path, "r+") as virtual_file:
        del virtual_file["data/demo_2"]

    with pytest.raises(ValueError, match="demo set is not aligned"):
        PairedResidualBCDataset(actual_path, virtual_path)

