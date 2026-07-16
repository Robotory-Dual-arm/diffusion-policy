from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from diffusion_policy.residual_rl.episode_io import load_episode
from diffusion_policy.residual_rl.runtime import (
    EpisodeSidecarWriter,
    RuntimeConfig,
    SlowFastResidualRuntime,
    merge_real_environment_shape_meta,
)
from diffusion_policy.residual_rl.safety import (
    ResidualRobotSafety,
    SafetyConfigurationError,
    SafetyLimits,
)


IDENTITY_ROT6D = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def now(self):
        return self.value

    def sleep(self, duration):
        self.value += max(0.0, float(duration))


class FakeEnvironment:
    def __init__(self, clock, accepted=None, force=0.001):
        self.clock = clock
        self.accepted = list(accepted or [])
        self.force = float(force)
        self.obs_calls = 0
        self.commands = []
        self.start_calls = []
        self.end_calls = 0

    def get_obs(self):
        self.obs_calls += 1
        latest_x = 0.01 * self.obs_calls
        pose = np.asarray([[latest_x - 0.01, 0, 0], [latest_x, 0, 0]], dtype=np.float32)
        quat = np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
        hand = np.zeros((2, 7), dtype=np.float32)
        wrench = np.zeros((6, 32), dtype=np.float32)
        wrench[0, -1] = self.force
        return {
            "image0": np.full((2, 224, 224, 3), 0.5, dtype=np.float32),
            "robot_pose_R": pose,
            "robot_quat_R": quat,
            "hand_pose_R": hand,
            "wrench_wrist_R": wrench,
            "timestamp": np.asarray(
                [self.clock.now() - 0.1, self.clock.now()], dtype=np.float64
            ),
            "robot_receive_timestamp": np.asarray(
                [self.clock.now()], dtype=np.float64
            ),
        }

    def exec_actions(self, actions, timestamps):
        accepted = self.accepted.pop(0) if self.accepted else True
        if accepted:
            self.commands.append((np.array(actions, copy=True), np.array(timestamps, copy=True)))
        return accepted

    def start_episode(self, start_time=None):
        self.start_calls.append(start_time)

    def end_episode(self):
        self.end_calls += 1


class FakeBasePolicy:
    checkpoint_id = "base-checkpoint"

    def __init__(self):
        self.calls = 0

    def predict_chunk(self, observation):
        self.calls += 1
        chunk = np.zeros((4, 16), dtype=np.float32)
        for index in range(4):
            chunk[index, :3] = [0.1 * (index + 1), 0, 0]
            chunk[index, 3:9] = IDENTITY_ROT6D
            chunk[index, 9:] = index + 1
        return chunk


class FakeResidualPolicy:
    def __init__(self, action=None):
        self.checkpoint_id = "actor-checkpoint"
        self.action = np.asarray(
            action if action is not None else [0.2, -0.2, 0, 0, 0, 0],
            dtype=np.float32,
        )
        self.calls = []

    def predict(self, observation, base_action16, *, exploration_std=0.0):
        self.calls.append(
            {
                "latest_x": float(np.asarray(observation["robot_pose_R"])[-1, 0]),
                "base_action": np.array(base_action16, copy=True),
                "exploration_std": float(exploration_std),
            }
        )
        return self.action.copy()


class SlowResidualPolicy(FakeResidualPolicy):
    def __init__(self, clock):
        super().__init__(action=np.zeros(6, dtype=np.float32))
        self.clock = clock

    def predict(self, observation, base_action16, *, exploration_std=0.0):
        result = super().predict(
            observation,
            base_action16,
            exploration_std=exploration_std,
        )
        self.clock.sleep(0.6)
        return result


class FailingSecondResidualPolicy(FakeResidualPolicy):
    def predict(self, observation, base_action16, *, exploration_std=0.0):
        if len(self.calls) == 1:
            raise FloatingPointError("synthetic actor NaN")
        return super().predict(
            observation,
            base_action16,
            exploration_std=exploration_std,
        )


def make_safety():
    return ResidualRobotSafety(
        SafetyLimits(
            residual_min=[-0.01] * 6,
            residual_max=[0.01] * 6,
            max_force_norm_n=10.0,
            max_torque_norm_nm=5.0,
            max_observation_age_s=0.5,
        )
    )


def make_runtime(environment, base, actor, clock, *, max_commanded_steps=2, safe_stop=None):
    return SlowFastResidualRuntime(
        environment=environment,
        base_policy=base,
        residual_policy=actor,
        safety=make_safety(),
        config=RuntimeConfig(
            frequency_hz=10.0,
            fast_steps_per_slow_inference=2,
            slow_action_start_index=0,
            command_latency_s=0.01,
            episode_start_delay_s=0.0,
            max_episode_duration_s=2.0,
            max_commanded_steps=max_commanded_steps,
        ),
        safe_stop=safe_stop,
        wall_clock=clock.now,
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )


def test_safety_limits_are_required_and_have_no_guessed_defaults():
    try:
        SafetyLimits()
    except SafetyConfigurationError:
        pass
    else:
        raise AssertionError("SafetyLimits must fail closed without explicit values")


def test_runtime_uses_one_slow_chunk_fresh_fast_obs_and_canonical_sidecar(tmp_path):
    clock = FakeClock()
    environment = FakeEnvironment(clock)
    base = FakeBasePolicy()
    actor = FakeResidualPolicy()
    writer = EpisodeSidecarWriter(tmp_path)
    runtime = make_runtime(environment, base, actor, clock)

    result = runtime.run_episode(
        label_success=lambda: 0,
        sidecar_writer=writer,
        exploration_std=0.0,
    )

    assert result.commanded_steps == 2
    assert result.termination_reason == "max_commanded_steps"
    assert base.calls == 1
    assert len(actor.calls) == 2
    assert actor.calls[0]["latest_x"] != actor.calls[1]["latest_x"]
    assert not np.array_equal(
        actor.calls[0]["base_action"], actor.calls[1]["base_action"]
    )
    assert len(environment.commands) == 2
    assert environment.end_calls == 1

    episode, metadata = load_episode(result.sidecar_path)
    assert metadata["success"] == 0
    assert episode["obs"]["image0"].shape == (2, 3, 224, 224)
    assert episode["obs"]["image0"].dtype == np.uint8
    assert episode["obs"]["robot_pose_R"].shape == (2, 3)
    assert episode["obs"]["wrench_wrist_R"].shape == (2, 6, 32)
    np.testing.assert_array_equal(episode["reward"], [0, 0])
    np.testing.assert_array_equal(episode["done"], [False, True])
    np.testing.assert_allclose(
        episode["action"],
        np.asarray([[0.01, -0.01, 0, 0, 0, 0]] * 2),
    )


def test_rejected_schedule_is_not_written_as_a_transition(tmp_path):
    clock = FakeClock()
    environment = FakeEnvironment(clock, accepted=[False, True])
    base = FakeBasePolicy()
    actor = FakeResidualPolicy(action=np.zeros(6, dtype=np.float32))
    runtime = make_runtime(
        environment,
        base,
        actor,
        clock,
        max_commanded_steps=1,
    )

    result = runtime.run_episode(
        label_success=lambda: 1,
        sidecar_writer=EpisodeSidecarWriter(tmp_path),
    )
    episode, _ = load_episode(result.sidecar_path)
    assert len(actor.calls) == 2
    assert len(environment.commands) == 1
    assert result.commanded_steps == 1
    assert episode["action"].shape == (1, 6)
    np.testing.assert_array_equal(episode["reward"], [1])
    np.testing.assert_array_equal(episode["done"], [True])


def test_resfit_exploration_uses_uniform_warmup_then_gaussian_phase(tmp_path):
    clock = FakeClock()
    environment = FakeEnvironment(clock)
    actor = FakeResidualPolicy(action=np.zeros(6, dtype=np.float32))
    runtime = SlowFastResidualRuntime(
        environment=environment,
        base_policy=FakeBasePolicy(),
        residual_policy=actor,
        safety=make_safety(),
        config=RuntimeConfig(
            frequency_hz=10.0,
            fast_steps_per_slow_inference=2,
            slow_action_start_index=0,
            command_latency_s=0.01,
            episode_start_delay_s=0.0,
            max_episode_duration_s=2.0,
            max_commanded_steps=2,
            exploration_mode="resfit",
            exploration_seed=7,
            resfit_learning_starts=1,
            resfit_random_noise_scale=0.2,
            resfit_stddev=0.0,
        ),
        wall_clock=clock.now,
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )
    result = runtime.run_episode(
        label_success=lambda: 0,
        sidecar_writer=EpisodeSidecarWriter(tmp_path),
    )
    episode, metadata = load_episode(result.sidecar_path)
    assert metadata["exploration_mode"] == "resfit"
    assert np.any(np.abs(episode["action"][0]) > 0.0)
    assert np.all(np.abs(episode["action"][0]) <= 0.002 + 1e-8)
    np.testing.assert_array_equal(episode["action"][1], np.zeros(6))


def test_force_violation_sends_no_command_and_requests_safe_stop():
    clock = FakeClock()
    environment = FakeEnvironment(clock, force=20.0)
    stop_reasons = []
    runtime = make_runtime(
        environment,
        FakeBasePolicy(),
        FakeResidualPolicy(),
        clock,
        safe_stop=stop_reasons.append,
    )

    result = runtime.run_episode(label_success=lambda: 0)
    assert result.termination_reason == "safety_violation"
    assert result.commanded_steps == 0
    assert environment.commands == []
    assert len(stop_reasons) == 1
    assert "Force norm" in stop_reasons[0]


def test_observation_is_rechecked_after_slow_actor_inference():
    clock = FakeClock()
    environment = FakeEnvironment(clock)
    runtime = make_runtime(
        environment,
        FakeBasePolicy(),
        SlowResidualPolicy(clock),
        clock,
    )
    result = runtime.run_episode(label_success=lambda: 0)
    assert result.termination_reason == "safety_violation"
    assert result.commanded_steps == 0
    assert environment.commands == []
    assert "observation age" in result.safety_violation.lower()


def test_actor_error_keeps_and_labels_earlier_commanded_transitions(tmp_path):
    clock = FakeClock()
    environment = FakeEnvironment(clock)
    runtime = make_runtime(
        environment,
        FakeBasePolicy(),
        FailingSecondResidualPolicy(action=np.zeros(6, dtype=np.float32)),
        clock,
    )
    result = runtime.run_episode(
        label_success=lambda: 0,
        sidecar_writer=EpisodeSidecarWriter(tmp_path),
    )
    episode, _ = load_episode(result.sidecar_path)
    assert result.termination_reason == "safety_violation"
    assert result.commanded_steps == 1
    assert "FloatingPointError" in result.safety_violation
    np.testing.assert_array_equal(episode["reward"], [0])
    np.testing.assert_array_equal(episode["done"], [True])


def test_shape_meta_merge_keeps_slow_history_metadata():
    base = {
        "obs": {
            "image0": {
                "shape": [3, 224, 224],
                "type": "rgb",
                "horizon": 2,
            },
            "robot_pose_R": {"shape": [3], "type": "low_dim", "horizon": 2},
        },
        "action": {"shape": [16]},
    }
    actor = {
        "obs": {
            "image0": {"shape": [3, 224, 224], "type": "rgb"},
            "robot_pose_R": {"shape": [3], "type": "low_dim"},
            "wrench_wrist_R": {"shape": [6, 32], "type": "wrench"},
        },
        "action": {"shape": [6]},
    }
    merged = merge_real_environment_shape_meta(base, actor)
    assert merged["action"]["shape"] == [16]
    assert merged["obs"]["image0"]["horizon"] == 2
    assert merged["obs"]["wrench_wrist_R"]["shape"] == [6, 32]


if __name__ == "__main__":
    test_safety_limits_are_required_and_have_no_guessed_defaults()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        test_runtime_uses_one_slow_chunk_fresh_fast_obs_and_canonical_sidecar(path)
    with tempfile.TemporaryDirectory() as directory:
        test_rejected_schedule_is_not_written_as_a_transition(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_resfit_exploration_uses_uniform_warmup_then_gaussian_phase(
            Path(directory)
        )
    test_force_violation_sends_no_command_and_requests_safe_stop()
    test_observation_is_rechecked_after_slow_actor_inference()
    with tempfile.TemporaryDirectory() as directory:
        test_actor_error_keeps_and_labels_earlier_commanded_transitions(
            Path(directory)
        )
    test_shape_meta_merge_keeps_slow_history_metadata()
