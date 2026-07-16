"""Checkpoint-to-inference round-trip tests using only stdlib unittest."""

from __future__ import annotations

import copy
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import h5py

from diffusion_policy.residual_rl.checkpoint import (
    ResidualModelConfig,
    build_actor,
    build_critics_from_actor,
    load_actor_for_inference,
    load_checkpoint_payload,
    save_bc_checkpoint,
    save_td3_checkpoint,
)
from diffusion_policy.residual_rl.collect import build_safety
from diffusion_policy.residual_rl.normalizer import StructuredAffineNormalizer
from diffusion_policy.residual_rl.pose import apply_residual_action_to_pose9
from diffusion_policy.residual_rl.runtime import (
    CANONICAL_OBSERVATION_SHAPES,
    EpisodeSidecarWriter,
)
from diffusion_policy.residual_rl.stats import (
    fit_residual_normalizer,
    symmetric_residual_bounds,
)
from diffusion_policy.residual_rl.td3 import ResidualTD3, TD3Config
from diffusion_policy.residual_rl.train import train as train_td3
from diffusion_policy.residual_rl.train_bc import train as train_bc


class CheckpointRoundTripTest(unittest.TestCase):
    def test_symmetric_demo_envelope_is_exact_and_rejects_invalid_scales(self):
        observed = {
            "min": np.asarray([-2, -1, -4, -1, -5, -2], dtype=np.float32),
            "max": np.asarray([1, 3, 2, 4, 1, 6], dtype=np.float32),
        }
        minimum, maximum = symmetric_residual_bounds(observed, scale=1.3)
        expected = np.asarray([2, 3, 4, 4, 5, 6], dtype=np.float32) * np.float32(1.3)
        np.testing.assert_allclose(maximum, expected)
        np.testing.assert_allclose(minimum, -expected)
        self.assertTrue(np.all(observed["min"] >= minimum))
        self.assertTrue(np.all(observed["max"] <= maximum))
        for invalid_scale in (0.9, float("nan")):
            with self.assertRaises(ValueError):
                symmetric_residual_bounds(observed, scale=invalid_scale)

    @staticmethod
    def _write_paired_dataset(root: Path) -> tuple[Path, Path]:
        actual_path = root / "actual.hdf5"
        virtual_path = root / "virtual.hdf5"
        identity6d = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        with h5py.File(actual_path, "w") as actual_file, h5py.File(
            virtual_path, "w"
        ) as virtual_file:
            actual_data = actual_file.create_group("data")
            virtual_data = virtual_file.create_group("data")
            for demo_index in range(2):
                length = 3
                actual_demo = actual_data.create_group(f"demo_{demo_index}")
                virtual_demo = virtual_data.create_group(f"demo_{demo_index}")
                actual_actions = []
                virtual_actions = []
                for step in range(length):
                    pose9 = np.concatenate(
                        (
                            np.asarray(
                                [0.1 + 0.01 * step, 0.0, 0.2],
                                dtype=np.float32,
                            ),
                            identity6d,
                        )
                    )
                    hand = np.linspace(0, 0.1, 7, dtype=np.float32)
                    actual = np.concatenate((pose9, hand))
                    sign = -1.0 if demo_index == 0 else 1.0
                    delta = np.asarray(
                        [
                            sign * 0.001 * (step + 1),
                            sign * 0.001,
                            sign * 0.0005,
                            sign * 0.002,
                            sign * 0.003,
                            sign * 0.001,
                        ],
                        dtype=np.float32,
                    )
                    virtual = apply_residual_action_to_pose9(actual, delta)
                    actual_actions.append(actual)
                    virtual_actions.append(virtual)
                actual_demo.create_dataset("actions", data=np.stack(actual_actions))
                virtual_demo.create_dataset("actions", data=np.stack(virtual_actions))
                obs = actual_demo.create_group("obs")
                obs.create_dataset(
                    "image0",
                    data=np.full(
                        (length, 16, 16, 3),
                        32 + demo_index,
                        dtype=np.uint8,
                    ),
                )
                obs.create_dataset(
                    "robot_pose_R",
                    data=np.asarray(actual_actions, dtype=np.float32)[:, :3],
                )
                obs.create_dataset(
                    "robot_quat_R",
                    data=np.tile(
                        np.asarray([0, 0, 0, 1], dtype=np.float32),
                        (length, 1),
                    ),
                )
                obs.create_dataset(
                    "hand_pose_R",
                    data=np.tile(
                        np.linspace(0, 0.1, 7, dtype=np.float32),
                        (length, 1),
                    ),
                )
                rng = np.random.default_rng(demo_index)
                obs.create_dataset(
                    "wrench_wrist_R",
                    data=rng.normal(size=(length, 6, 32)).astype(np.float32),
                )
        return actual_path, virtual_path

    def test_explicit_physical_bounds_parameterize_residual(self):
        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                sign = -1.0 if index == 0 else 1.0
                return {
                    "obs": {
                        "image0": torch.zeros(3, 8, 8),
                        "robot_pose_R": torch.full((3,), sign),
                        "robot_quat_R": torch.full((4,), sign),
                        "hand_pose_R": torch.full((7,), sign),
                        "wrench_wrist_R": torch.full((6, 32), sign),
                    },
                    "base_action": torch.full((16,), sign),
                    "action": torch.full((6,), 0.1 * sign),
                }

        minimum = [-0.5] * 6
        maximum = [0.5] * 6
        normalizer, _ = fit_residual_normalizer(
            TinyDataset(),
            batch_size=2,
            residual_min=minimum,
            residual_max=maximum,
        )
        normalized_min = normalizer.normalize_field(
            "residual", torch.tensor([minimum], dtype=torch.float32)
        )
        normalized_max = normalizer.normalize_field(
            "residual", torch.tensor([maximum], dtype=torch.float32)
        )
        torch.testing.assert_close(normalized_min, -torch.ones_like(normalized_min))
        torch.testing.assert_close(normalized_max, torch.ones_like(normalized_max))

    def _components(self):
        minimum = (-0.02, -0.02, -0.02, -0.1, -0.1, -0.1)
        maximum = tuple(-value for value in minimum)
        config = ResidualModelConfig(
            image_feature_dim=8,
            wrench_feature_dim=8,
            actor_hidden_dims=(16,),
            critic_hidden_dims=(16,),
            residual_min=minimum,
            residual_max=maximum,
        )
        fields = {
            "robot_pose_R": {"scale": np.ones(3), "offset": np.zeros(3)},
            "robot_quat_R": {"scale": np.ones(4), "offset": np.zeros(4)},
            "hand_pose_R": {"scale": np.ones(7), "offset": np.zeros(7)},
            "wrench_wrist_R": {
                "scale": np.ones(6 * 32),
                "offset": np.zeros(6 * 32),
            },
            "base_action": {"scale": np.ones(16), "offset": np.zeros(16)},
            "residual": {
                "scale": 2.0 / (np.asarray(maximum) - np.asarray(minimum)),
                "offset": np.zeros(6),
            },
        }
        actor = build_actor(
            config,
            StructuredAffineNormalizer(fields),
            freeze_image_encoder=False,
        )
        return actor, config

    @staticmethod
    def _numpy_input():
        rng = np.random.default_rng(12)
        observation = {
            "image0": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "robot_pose_R": rng.normal(size=3).astype(np.float32),
            "robot_quat_R": rng.normal(size=4).astype(np.float32),
            "hand_pose_R": rng.normal(size=7).astype(np.float32),
            "wrench_wrist_R": rng.normal(size=(6, 32)).astype(np.float32),
        }
        base = rng.normal(size=16).astype(np.float32)
        return observation, base

    def test_bc_and_td3_checkpoints_load_in_collect_runner(self):
        actor, config = self._components()
        observation, base = self._numpy_input()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bc_path = root / "bc.pt"
            save_bc_checkpoint(
                bc_path,
                actor=actor,
                model_config=config,
                step=3,
                epoch=1,
            )
            bc_runner = load_actor_for_inference(bc_path, device="cpu")
            safety = build_safety(
                argparse.Namespace(
                    residual_min=None,
                    residual_max=None,
                    max_force_norm_n=10.0,
                    max_torque_norm_nm=5.0,
                    max_observation_age_s=0.5,
                ),
                bc_runner,
            )
            np.testing.assert_allclose(
                safety.limits.residual_min,
                config.residual_min,
            )
            np.testing.assert_allclose(
                safety.limits.residual_max,
                config.residual_max,
            )
            bc_action = bc_runner.predict(observation, base)
            self.assertEqual(bc_action.shape, (6,))
            self.assertTrue(np.isfinite(bc_action).all())

            critics = build_critics_from_actor(
                actor,
                config,
                freeze_image_encoder=True,
            )
            td3 = ResidualTD3(
                actor=actor,
                critics=critics,
                config=TD3Config(),
                bc_prior=copy.deepcopy(actor),
                device="cpu",
            )
            td3_path = root / "td3.pt"
            save_td3_checkpoint(
                td3_path,
                td3=td3,
                model_config=config,
            )
            td3_runner = load_actor_for_inference(td3_path, device="cpu")
            td3_action = td3_runner.predict(observation, base)
            np.testing.assert_allclose(td3_action, bc_action, atol=1e-7)

    def test_saved_sidecar_runs_through_offline_td3_entrypoint(self):
        actor, config = self._components()
        length = 2
        rng = np.random.default_rng(5)
        observations = {
            "image0": rng.integers(
                0,
                256,
                (length,) + CANONICAL_OBSERVATION_SHAPES["image0"],
                dtype=np.uint8,
            ),
            "robot_pose_R": rng.normal(size=(length, 3)).astype(np.float32),
            "robot_quat_R": rng.normal(size=(length, 4)).astype(np.float32),
            "hand_pose_R": rng.normal(size=(length, 7)).astype(np.float32),
            "wrench_wrist_R": rng.normal(size=(length, 6, 32)).astype(np.float32),
        }
        episode = {
            "obs": observations,
            "next_obs": {key: value.copy() for key, value in observations.items()},
            "base_action": rng.normal(size=(length, 16)).astype(np.float32),
            "next_base_action": rng.normal(size=(length, 16)).astype(np.float32),
            "action": np.zeros((length, 6), dtype=np.float32),
            "reward": np.asarray([0.0, 1.0], dtype=np.float32),
            "done": np.asarray([False, True]),
            "command_timestamp": np.arange(length, dtype=np.float64),
            "observation_timestamp": np.arange(length, dtype=np.float64),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bc_path = root / "bc.pt"
            save_bc_checkpoint(
                bc_path,
                actor=actor,
                model_config=config,
                step=1,
                epoch=1,
            )
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
                batch_size=2,
                replay_capacity=8,
                gamma=0.99,
                tau=0.005,
                actor_lr=1e-5,
                critic_lr=1e-4,
                policy_delay=2,
                target_noise_fraction=0.1,
                target_noise_clip_fraction=0.2,
                lambda_bc=1.0,
                max_grad_norm=10.0,
                log_every=1,
                save_every=1,
                resume=None,
                seed=1,
                device="cpu",
                overwrite=False,
            )
            train_td3(args)
            runner = load_actor_for_inference(
                output / "checkpoints/latest.pt",
                device="cpu",
            )
            action = runner.predict(*self._numpy_input())
            self.assertEqual(action.shape, (6,))
            self.assertTrue(np.isfinite(action).all())

    def test_paired_data_runs_through_bc_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual_path, virtual_path = self._write_paired_dataset(root)
            output = root / "bc_run"
            args = argparse.Namespace(
                actual_dataset=str(actual_path),
                virtual_dataset=str(virtual_path),
                base_predictions=None,
                base_action_key="actions",
                output=str(output),
                target_shift=1,
                residual_min=None,
                residual_max=None,
                residual_bound_scale=1.3,
                epochs=1,
                batch_size=2,
                stats_batch_size=2,
                num_workers=0,
                val_ratio=0.5,
                learning_rate=1e-4,
                weight_decay=0.0,
                max_grad_norm=10.0,
                image_feature_dim=8,
                wrench_feature_dim=8,
                actor_hidden_dims=[16],
                critic_hidden_dims=[16],
                seed=3,
                device="cpu",
                overwrite=False,
            )
            train_bc(args)
            payload = load_checkpoint_payload(
                output / "checkpoints/best.pt",
                map_location="cpu",
            )
            minimum = np.asarray(payload["model_config"]["residual_min"])
            maximum = np.asarray(payload["model_config"]["residual_max"])
            np.testing.assert_allclose(minimum, -maximum)
            self.assertEqual(
                payload["metadata"]["residual_bound_source"],
                "symmetric_demonstration_max_abs",
            )
            runner = load_actor_for_inference(
                output / "checkpoints/best.pt",
                device="cpu",
            )
            action = runner.predict(*self._numpy_input())
            self.assertEqual(action.shape, (6,))
            self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
