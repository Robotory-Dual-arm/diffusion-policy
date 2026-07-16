from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

from diffusion_policy.residual_rl.episode_io import SIDECAR_SCHEMA
from diffusion_policy.residual_rl.data_identity import file_sha256
from diffusion_policy.residual_rl.interactive import (
    ReplayInventory,
    RolloutStopKeys,
    SessionPhase,
    SessionPhaseGuard,
    automatic_training_due,
    inspect_replay_directory,
    make_td3_namespace,
    parse_menu_command,
)
from diffusion_policy.residual_rl import interactive


class InteractiveWorkflowTest(unittest.TestCase):
    def test_menu_commands_are_deliberate(self):
        for value in ("", "r", "rollout"):
            self.assertEqual(parse_menu_command(value), "rollout")
        self.assertEqual(parse_menu_command(" T "), "train")
        self.assertEqual(parse_menu_command("q"), "quit")
        with self.assertRaises(ValueError):
            parse_menu_command("maybe")

    def test_training_requires_closed_collection_phase(self):
        guard = SessionPhaseGuard()
        self.assertIs(guard.phase, SessionPhase.IDLE)
        with guard.collecting():
            self.assertIs(guard.phase, SessionPhase.COLLECTING)
            with self.assertRaisesRegex(RuntimeError, "environment is closed"):
                with guard.training():
                    pass
        self.assertIs(guard.phase, SessionPhase.IDLE)
        with guard.training():
            self.assertIs(guard.phase, SessionPhase.TRAINING)
        self.assertIs(guard.phase, SessionPhase.IDLE)

    def test_automatic_trigger_needs_new_episodes_and_enough_replay(self):
        inventory = ReplayInventory(episodes=8, transitions=500, successes=3, failures=5)
        self.assertTrue(
            automatic_training_due(
                saved_since_training=5,
                train_after_episodes=5,
                inventory=inventory,
                minimum_transitions=256,
            )
        )
        self.assertFalse(
            automatic_training_due(
                saved_since_training=4,
                train_after_episodes=5,
                inventory=inventory,
                minimum_transitions=256,
            )
        )
        self.assertFalse(
            automatic_training_due(
                saved_since_training=5,
                train_after_episodes=5,
                inventory=ReplayInventory(8, 100, 3, 5),
                minimum_transitions=256,
            )
        )

    def test_inventory_counts_successes_and_failures_without_loading_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (length, success) in enumerate(((3, 1), (5, 0))):
                with h5py.File(root / f"episode_{index}.hdf5", "w") as output:
                    output.attrs["schema"] = SIDECAR_SCHEMA
                    output.attrs["metadata_json"] = f'{{"success": {success}}}'
                    output.create_dataset("action", data=np.zeros((length, 6)))
                    reward = np.zeros(length, dtype=np.float32)
                    reward[-1] = success
                    output.create_dataset("reward", data=reward)
            inventory = inspect_replay_directory(root)
        self.assertEqual(inventory, ReplayInventory(2, 8, 1, 1))

    def test_td3_adapter_saves_only_latest_inside_each_round(self):
        args = argparse.Namespace(
            td3_updates=100,
            critic_warmup_updates=10,
            batch_size=16,
            replay_capacity=100,
            gamma=0.99,
            tau=0.005,
            actor_lr=1e-6,
            critic_lr=1e-4,
            policy_delay=2,
            target_noise_fraction=0.1,
            target_noise_clip_fraction=0.2,
            lambda_bc=1.0,
            max_grad_norm=10.0,
            log_every=10,
            seed=42,
            training_device="cpu",
        )
        namespace = make_td3_namespace(
            args,
            bc_checkpoint=Path("/tmp/bc.pt"),
            resume_checkpoint=Path("/tmp/previous.pt"),
            sidecar_root=Path("/tmp/episodes"),
            round_output=Path("/tmp/round"),
        )
        self.assertEqual(namespace.save_every, 0)
        self.assertEqual(namespace.updates, 100)
        self.assertEqual(namespace.resume, "/tmp/previous.pt")
        self.assertFalse(namespace.overwrite)

        args.td3_updates = None
        args.utd_ratio = 4.0
        utd_namespace = make_td3_namespace(
            args,
            bc_checkpoint=Path("/tmp/bc.pt"),
            resume_checkpoint=None,
            sidecar_root=Path("/tmp/episodes"),
            round_output=Path("/tmp/round_utd"),
            new_online_transitions=25,
        )
        self.assertEqual(utd_namespace.updates, 100)

    def test_terminal_is_restored_before_reward_prompt(self):
        events = []

        class StopKeys:
            def restore(self):
                events.append("restore")

        def prompt(_message):
            events.append("prompt")
            return 1

        with mock.patch.object(interactive, "prompt_success_label", prompt):
            result = interactive._prompt_label_after_terminal_restore(StopKeys())
        self.assertEqual(result, 1)
        self.assertEqual(events, ["restore", "prompt"])

    def test_non_tty_rollout_is_rejected_without_a_stop_key(self):
        with self.assertRaisesRegex(RuntimeError, "without a stop key"):
            with RolloutStopKeys(stream=io.StringIO(), cv_stop=lambda: False):
                pass

    def test_concrete_controller_safety_stop_is_idempotent(self):
        calls = []
        environment = SimpleNamespace(
            robot=SimpleNamespace(
                stop=lambda *, wait: calls.append(wait),
            )
        )
        safe_stop = interactive.collect.make_robot_safe_stop(environment)
        safe_stop("force threshold")
        safe_stop("second request")
        self.assertEqual(calls, [False])

    def test_runtime_rejects_cache_schedule_or_base_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            base_checkpoint = Path(temporary) / "base.ckpt"
            base_checkpoint.write_bytes(b"frozen base")
            runner = SimpleNamespace(
                checkpoint_metadata={
                    "base_prediction_cache": {
                        "fast_steps_per_slow": 6,
                        "slow_action_start_index": 1,
                        "base_num_inference_steps": 16,
                        "base_state_key": "ema_model",
                        "base_checkpoint_sha256": file_sha256(base_checkpoint),
                    }
                }
            )
            args = SimpleNamespace(
                fast_steps_per_slow=6,
                slow_action_start_index=1,
                base_inference_steps=16,
                no_base_ema=False,
                base_checkpoint=str(base_checkpoint),
            )
            interactive.collect.verify_runtime_training_provenance(args, runner)
            args.fast_steps_per_slow = 5
            with self.assertRaisesRegex(ValueError, "differs from BC cache"):
                interactive.collect.verify_runtime_training_provenance(args, runner)

    def test_full_round_closes_environment_before_training_and_reloads_actor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = interactive.make_parser().parse_args(
                [
                    "--actor-checkpoint",
                    str(root / "bc.pt"),
                    "--base-checkpoint",
                    str(root / "base.ckpt"),
                    "--output",
                    str(root / "session"),
                    "--device",
                    "cpu",
                    "--training-device",
                    "cpu",
                    "--max-force-norm-n",
                    "10",
                    "--max-torque-norm-nm",
                    "5",
                    "--max-observation-age-s",
                    "0.5",
                    "--train-after-episodes",
                    "1",
                    "--min-replay-transitions",
                    "1",
                    "--batch-size",
                    "1",
                    "--replay-capacity",
                    "4",
                    "--td3-updates",
                    "1",
                    "--critic-warmup-updates",
                    "0",
                ]
            )
            events = []
            environment_active = False
            loaded_actors = []

            @contextmanager
            def environment_context(*_args, **_kwargs):
                nonlocal environment_active
                environment_active = True
                events.append("environment_enter")
                try:
                    yield SimpleNamespace(robot=SimpleNamespace(stop=lambda **_: None))
                finally:
                    environment_active = False
                    events.append("environment_exit")

            class StopKeys:
                def __enter__(self):
                    return lambda: False

                def __exit__(self, *_args):
                    return None

            class Runtime:
                def __init__(self, **_kwargs):
                    pass

                def run_episode(self, *, sidecar_writer, **_kwargs):
                    path = sidecar_writer.output_directory / "episode_fake.hdf5"
                    with h5py.File(path, "w") as output:
                        output.attrs["schema"] = SIDECAR_SCHEMA
                        output.attrs["metadata_json"] = '{"success": 1}'
                        output.create_dataset("action", data=np.zeros((1, 6)))
                        output.create_dataset(
                            "reward", data=np.ones(1, dtype=np.float32)
                        )
                    events.append("episode")
                    return SimpleNamespace(
                        sidecar_path=path,
                        safety_violation=None,
                    )

            def load_runners(current_args):
                loaded_actors.append(str(current_args.actor_checkpoint))
                return SimpleNamespace(), SimpleNamespace()

            def train(td3_args):
                self.assertFalse(environment_active)
                events.append("train")
                checkpoint = Path(td3_args.output) / "checkpoints" / "latest.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")

            with (
                mock.patch.object(
                    interactive,
                    "_checkpoint_role",
                    return_value=(root / "bc.pt", None),
                ),
                mock.patch.object(
                    interactive.collect,
                    "load_policy_runners",
                    side_effect=load_runners,
                ),
                mock.patch.object(
                    interactive.collect,
                    "build_safety",
                    return_value=SimpleNamespace(limits=object()),
                ),
                mock.patch.object(
                    interactive.collect,
                    "_verify_actor_and_runtime_bounds",
                ),
                mock.patch.object(
                    interactive.collect,
                    "build_runtime_config",
                    return_value=object(),
                ),
                mock.patch.object(
                    interactive.collect,
                    "real_environment",
                    side_effect=environment_context,
                ),
                mock.patch.object(interactive.collect, "warm_up_inference"),
                mock.patch.object(interactive, "SlowFastResidualRuntime", Runtime),
                mock.patch.object(interactive, "RolloutStopKeys", StopKeys),
                mock.patch.object(
                    interactive,
                    "_prompt_menu",
                    side_effect=["rollout", "quit"],
                ),
                mock.patch.object(interactive, "train_td3", side_effect=train),
                mock.patch.object(interactive, "_release_inference_models"),
            ):
                final_actor = interactive.run_interactive(args)

            self.assertEqual(
                events,
                [
                    "environment_enter",
                    "episode",
                    "environment_exit",
                    "train",
                    "environment_enter",
                    "environment_exit",
                ],
            )
            self.assertEqual(len(loaded_actors), 2)
            self.assertEqual(loaded_actors[0], str(root / "bc.pt"))
            self.assertEqual(loaded_actors[1], str(final_actor))


if __name__ == "__main__":
    unittest.main()
