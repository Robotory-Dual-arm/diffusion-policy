from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from diffusion_policy.residual_rl.offline_dataset import PairedResidualRLDataset
from diffusion_policy.residual_rl.checkpoint import (
    ResidualModelConfig,
    build_actor,
    save_bc_checkpoint,
)
from diffusion_policy.residual_rl.normalizer import StructuredAffineNormalizer
from diffusion_policy.residual_rl.runtime import EpisodeSidecarWriter
from diffusion_policy.residual_rl.train import train as train_td3
from diffusion_policy.residual_rl.pose import (
    apply_residual_action_to_pose9,
    pose6_to_pose9,
)


def _write_demo_files(root: Path):
    actual_path = root / "actual.hdf5"
    virtual_path = root / "virtual.hdf5"
    cache_path = root / "cache.hdf5"
    with h5py.File(actual_path, "w") as actual, h5py.File(
        virtual_path, "w"
    ) as virtual, h5py.File(cache_path, "w") as cache:
        actual_data = actual.create_group("data")
        virtual_data = virtual.create_group("data")
        cache_data = cache.create_group("data")
        for demo_index in range(2):
            name = f"demo_{demo_index}"
            actual_demo = actual_data.create_group(name)
            virtual_demo = virtual_data.create_group(name)
            cache_demo = cache_data.create_group(name)
            actions = []
            virtual_actions = []
            for step in range(4):
                pose = pose6_to_pose9(
                    np.asarray(
                        [0.1 * demo_index, 0.01 * step, 0.3, 0, 0, 0],
                        dtype=np.float32,
                    )
                )
                hand = np.full(7, step, dtype=np.float32)
                actions.append(np.concatenate((pose, hand)))
                delta = np.asarray([0.001, 0, 0, 0, 0, 0], dtype=np.float32)
                virtual_pose = apply_residual_action_to_pose9(pose, delta)
                virtual_actions.append(np.concatenate((virtual_pose, hand)))
            actions = np.asarray(actions, dtype=np.float32)
            actual_demo.create_dataset("actions", data=actions)
            virtual_demo.create_dataset(
                "actions", data=np.asarray(virtual_actions, dtype=np.float32)
            )
            cache_demo.create_dataset("actions", data=actions)
            observations = actual_demo.create_group("obs")
            observations.create_dataset(
                "image0",
                data=np.full(
                    (4, 224, 224, 3),
                    20 + 100 * demo_index,
                    dtype=np.uint8,
                ),
            )
            observations.create_dataset(
                "robot_pose_R",
                data=np.zeros((4, 3), dtype=np.float32),
            )
            observations.create_dataset(
                "robot_quat_R",
                data=np.tile(
                    np.asarray([0, 0, 0, 1], dtype=np.float32),
                    (4, 1),
                ),
            )
            observations.create_dataset(
                "hand_pose_R",
                data=np.zeros((4, 7), dtype=np.float32),
            )
            observations.create_dataset(
                "wrench_wrist_R",
                data=np.zeros((4, 6, 32), dtype=np.float32),
            )
    return actual_path, virtual_path, cache_path


class OfflineResidualDatasetTest(unittest.TestCase):
    def test_n_step_success_targets_and_demo_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            actual, virtual, cache = _write_demo_files(Path(temporary))
            dataset = PairedResidualRLDataset(
                actual,
                virtual,
                base_action_source=cache,
                n_step=2,
                gamma=0.9,
            )
            self.assertEqual(len(dataset), 6)
            first = dataset[0]
            self.assertEqual(float(first["done"]), 0.0)
            self.assertEqual(float(first["reward"]), 0.0)
            self.assertAlmostEqual(
                float(first["next_obs"]["image0"].mean()),
                20.0 / 255.0,
            )
            near_terminal = dataset[1]
            self.assertEqual(float(near_terminal["done"]), 1.0)
            self.assertAlmostEqual(float(near_terminal["reward"]), 0.9)
            terminal = dataset[2]
            self.assertEqual(float(terminal["reward"]), 1.0)

            second_demo = dataset[3]
            self.assertEqual(float(second_demo["done"]), 0.0)
            self.assertAlmostEqual(
                float(second_demo["next_obs"]["image0"].mean()),
                120.0 / 255.0,
                places=6,
            )
            dataset.close()

    def test_mixed_offline_online_batch_runs_through_td3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual, virtual, cache = _write_demo_files(root)
            config = ResidualModelConfig(
                residual_min=(-0.01,) * 6,
                residual_max=(0.01,) * 6,
            )
            actor = build_actor(
                config,
                StructuredAffineNormalizer(),
                freeze_image_encoder=False,
            )
            bc_path = root / "bc.pt"
            save_bc_checkpoint(
                bc_path,
                actor=actor,
                model_config=config,
                step=1,
                epoch=1,
                metadata={
                    "actual_dataset": str(actual),
                    "virtual_dataset": str(virtual),
                    "base_action_source": str(cache),
                    "base_action_key": "actions",
                    "target_shift": 1,
                },
            )

            length = 2
            observations = {
                "image0": np.zeros((length, 3, 224, 224), dtype=np.uint8),
                "robot_pose_R": np.zeros((length, 3), dtype=np.float32),
                "robot_quat_R": np.tile(
                    np.asarray([0, 0, 0, 1], dtype=np.float32),
                    (length, 1),
                ),
                "hand_pose_R": np.zeros((length, 7), dtype=np.float32),
                "wrench_wrist_R": np.zeros((length, 6, 32), dtype=np.float32),
            }
            episode = {
                "obs": observations,
                "next_obs": {key: value.copy() for key, value in observations.items()},
                "base_action": np.zeros((length, 16), dtype=np.float32),
                "next_base_action": np.zeros((length, 16), dtype=np.float32),
                "action": np.zeros((length, 6), dtype=np.float32),
                "reward": np.asarray([0, 1], dtype=np.float32),
                "done": np.asarray([0, 1], dtype=np.float32),
                "command_timestamp": np.arange(length, dtype=np.float64),
                "observation_timestamp": np.arange(length, dtype=np.float64),
            }
            episode_path = EpisodeSidecarWriter(root / "episodes").write(
                episode,
                {"success": 1},
            )
            output = root / "td3"
            args = argparse.Namespace(
                bc_checkpoint=str(bc_path),
                episodes=[str(episode_path)],
                output=str(output),
                updates=1,
                critic_warmup_updates=1,
                batch_size=4,
                offline_ratio=0.5,
                offline_actual_dataset=str(actual),
                offline_virtual_dataset=str(virtual),
                offline_base_predictions=str(cache),
                offline_num_workers=0,
                replay_capacity=8,
                gamma=0.99,
                n_step=3,
                tau=0.005,
                actor_lr=1e-5,
                critic_lr=1e-4,
                policy_delay=2,
                target_noise_fraction=0.1,
                target_noise_clip_fraction=0.2,
                lambda_bc=1.0,
                max_grad_norm=10.0,
                log_every=1,
                save_every=0,
                resume=None,
                seed=3,
                device="cpu",
                overwrite=False,
            )
            train_td3(args)
            self.assertTrue((output / "checkpoints/latest.pt").is_file())


if __name__ == "__main__":
    unittest.main()
