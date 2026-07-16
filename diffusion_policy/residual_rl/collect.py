"""Collect success and failure episodes with a frozen residual actor.

This command performs inference only.  It never constructs an optimizer and it
loads the actor once before the real environment is started, so an episode
cannot observe an in-place checkpoint swap.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.real_world.real_inference_util import get_real_obs_resolution
from diffusion_policy.residual_rl.base_policy import load_frozen_base_policy
from diffusion_policy.residual_rl.data_identity import file_sha256
from diffusion_policy.residual_rl.runtime import (
    EpisodeSidecarWriter,
    FrozenBasePolicyRunner,
    ResidualPolicyRunner,
    RuntimeConfig,
    SlowFastResidualRuntime,
    TorchResidualPolicyRunner,
    build_base_condition16,
    merge_real_environment_shape_meta,
    prompt_success_label,
)
from diffusion_policy.residual_rl.safety import ResidualRobotSafety, SafetyLimits


def _cfg_select(cfg, path: str, default=None):
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, path, default=default)
    current = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _actor_components(loaded):
    """Normalize the small checkpoint-loader surface to actor/meta/id."""

    if isinstance(loaded, tuple):
        if len(loaded) == 2:
            actor, metadata = loaded
        elif len(loaded) == 3:
            actor, metadata, checkpoint_id = loaded
            if isinstance(metadata, Mapping):
                metadata = dict(metadata)
                metadata.setdefault("checkpoint_id", checkpoint_id)
        else:
            raise TypeError("Unsupported actor checkpoint tuple")
    else:
        actor = getattr(loaded, "actor", getattr(loaded, "model", loaded))
        metadata = loaded

    shape_meta = getattr(metadata, "shape_meta", None)
    if shape_meta is None and isinstance(metadata, Mapping):
        shape_meta = metadata.get("shape_meta")
    if shape_meta is None:
        shape_meta = getattr(actor, "shape_meta", None)
    if shape_meta is None:
        cfg = getattr(metadata, "cfg", getattr(metadata, "config", None))
        if cfg is None and isinstance(metadata, Mapping):
            cfg = metadata.get("cfg", metadata.get("config"))
        shape_meta = _cfg_select(cfg, "shape_meta")
        if shape_meta is None:
            shape_meta = _cfg_select(cfg, "task.shape_meta")
    checkpoint_id = getattr(metadata, "checkpoint_id", None)
    if checkpoint_id is None and isinstance(metadata, Mapping):
        checkpoint_id = metadata.get("checkpoint_id")
    return actor, shape_meta, checkpoint_id


def load_residual_policy_runner(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    exploration_seed: int | None,
) -> ResidualPolicyRunner:
    """Load the inference-only actor through the isolated checkpoint helper."""

    from diffusion_policy.residual_rl.checkpoint import load_actor_for_inference

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    loaded = load_actor_for_inference(
        checkpoint_path,
        device=device,
        exploration_seed=exploration_seed,
    )
    if isinstance(loaded, ResidualPolicyRunner):
        return loaded
    actor, shape_meta, checkpoint_id = _actor_components(loaded)
    if not isinstance(actor, torch.nn.Module):
        raise TypeError(
            "load_actor_for_inference must return a ResidualPolicyRunner or "
            "a loaded torch actor"
        )
    if shape_meta is None:
        raise ValueError("Residual actor checkpoint has no shape_meta")
    if checkpoint_id is None:
        checkpoint_id = str(checkpoint_path)
    return TorchResidualPolicyRunner(
        actor,
        shape_meta=shape_meta,
        checkpoint_id=str(checkpoint_id),
        device=device,
        exploration_seed=exploration_seed,
    )


def _actor_shape_meta(runner: ResidualPolicyRunner):
    shape_meta = getattr(runner, "shape_meta", None)
    if shape_meta is None:
        raise ValueError(
            "ResidualPolicyRunner must expose shape_meta so the real environment "
            "can request the actor's latest observations"
        )
    return shape_meta


def _verify_actor_and_runtime_bounds(
    runner: ResidualPolicyRunner,
    safety_limits: SafetyLimits,
) -> None:
    actor = getattr(runner, "actor", None)
    if actor is None or not hasattr(actor, "residual_min"):
        return
    actor_min = np.asarray(actor.residual_min.detach().cpu(), dtype=np.float64)
    actor_max = np.asarray(actor.residual_max.detach().cpu(), dtype=np.float64)
    safety_min = np.asarray(safety_limits.residual_min, dtype=np.float64)
    safety_max = np.asarray(safety_limits.residual_max, dtype=np.float64)
    if np.any(np.maximum(actor_min, safety_min) >= np.minimum(actor_max, safety_max)):
        raise ValueError(
            "Actor and explicit robot residual bounds have an empty intersection"
        )
    if not np.allclose(actor_min, safety_min) or not np.allclose(actor_max, safety_max):
        print(
            "Actor and robot safety bounds differ; effective commands use their "
            "component-wise intersection."
        )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Existing real-env recording root")
    parser.add_argument(
        "--sidecar-output",
        default=None,
        help="Canonical RL episodes; defaults to OUTPUT/residual_rl_episodes",
    )
    parser.add_argument("--robot-ip", default="192.168.111.50")
    parser.add_argument(
        "--control-mode",
        choices=("impedance", "position"),
        default="impedance",
        help="The requested task uses impedance; position requires explicit opt-in",
    )
    parser.add_argument(
        "--camera-serial",
        action="append",
        default=None,
        help="Repeat for multiple cameras; defaults to the existing right camera",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--fast-steps-per-slow", type=int, default=6)
    parser.add_argument("--slow-action-start-index", type=int, default=1)
    parser.add_argument("--command-latency", type=float, default=0.01)
    parser.add_argument("--episode-start-delay", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--max-commanded-steps", type=int, default=None)
    parser.add_argument(
        "--base-inference-steps",
        type=int,
        default=16,
        help="Frozen base sampler steps; 16 matches the existing robot evaluator",
    )
    parser.add_argument("--no-base-ema", action="store_true")
    parser.add_argument("--init-joints", action="store_true")
    parser.add_argument("--exploration-seed", type=int, default=None)

    # Residual bounds are self-described by the BC/TD3 actor checkpoint. F/T
    # and freshness limits remain required because the repository has no
    # authoritative robot values for them.
    parser.add_argument(
        "--residual-min",
        type=float,
        nargs=6,
        default=None,
        help="Optional runtime override; default uses actor-checkpoint bounds",
    )
    parser.add_argument(
        "--residual-max",
        type=float,
        nargs=6,
        default=None,
        help="Optional runtime override; default uses actor-checkpoint bounds",
    )
    parser.add_argument("--max-force-norm-n", type=float, required=True)
    parser.add_argument("--max-torque-norm-nm", type=float, required=True)
    parser.add_argument("--max-observation-age-s", type=float, required=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--exploration-mode",
        choices=("resfit", "none", "gaussian"),
        default="resfit",
    )
    parser.add_argument(
        "--exploration-std",
        type=float,
        default=0.0,
        help="Physical scalar std used only by --exploration-mode gaussian",
    )
    parser.add_argument("--exploration-step-start", type=int, default=0)
    parser.add_argument("--resfit-learning-starts", type=int, default=10_000)
    parser.add_argument("--resfit-random-noise-scale", type=float, default=0.2)
    parser.add_argument("--resfit-stddev", type=float, default=0.025)
    return parser


def validate_common_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)


def _checkpoint_residual_bounds(
    runner: ResidualPolicyRunner,
) -> tuple[np.ndarray, np.ndarray]:
    actor = getattr(runner, "actor", None)
    if actor is None or not hasattr(actor, "residual_min"):
        raise ValueError(
            "Actor runner does not expose checkpoint residual bounds; pass both "
            "--residual-min and --residual-max"
        )
    minimum = np.asarray(actor.residual_min.detach().cpu(), dtype=np.float64)
    maximum = np.asarray(actor.residual_max.detach().cpu(), dtype=np.float64)
    return minimum, maximum


def build_safety(
    args: argparse.Namespace,
    residual_runner: ResidualPolicyRunner,
) -> ResidualRobotSafety:
    if (args.residual_min is None) != (args.residual_max is None):
        raise ValueError("--residual-min and --residual-max must be supplied together")
    if args.residual_min is None:
        residual_min, residual_max = _checkpoint_residual_bounds(residual_runner)
        print("Residual safety bounds: actor checkpoint")
    else:
        actor_min, actor_max = _checkpoint_residual_bounds(residual_runner)
        residual_min = np.maximum(
            actor_min,
            np.asarray(args.residual_min, dtype=np.float64),
        )
        residual_max = np.minimum(
            actor_max,
            np.asarray(args.residual_max, dtype=np.float64),
        )
        if np.any(residual_min >= residual_max):
            raise ValueError(
                "Actor and explicit robot residual bounds have an empty intersection"
            )
        print("Residual safety bounds: actor/runtime intersection")
    limits = SafetyLimits(
        residual_min=residual_min,
        residual_max=residual_max,
        max_force_norm_n=args.max_force_norm_n,
        max_torque_norm_nm=args.max_torque_norm_nm,
        max_observation_age_s=args.max_observation_age_s,
    )
    return ResidualRobotSafety(limits)


def build_runtime_config(
    args: argparse.Namespace,
    *,
    exploration_std: float = 0.0,
    exploration_mode: str = "none",
    exploration_step_start: int = 0,
) -> RuntimeConfig:
    return RuntimeConfig(
        frequency_hz=args.frequency,
        fast_steps_per_slow_inference=args.fast_steps_per_slow,
        slow_action_start_index=args.slow_action_start_index,
        command_latency_s=args.command_latency,
        episode_start_delay_s=args.episode_start_delay,
        max_episode_duration_s=args.max_duration,
        max_commanded_steps=args.max_commanded_steps,
        exploration_std=exploration_std,
        exploration_mode=exploration_mode,
        exploration_step_start=exploration_step_start,
        exploration_seed=args.exploration_seed,
        resfit_learning_starts=int(getattr(args, "resfit_learning_starts", 10_000)),
        resfit_random_noise_scale=float(
            getattr(args, "resfit_random_noise_scale", 0.2)
        ),
        resfit_stddev=float(getattr(args, "resfit_stddev", 0.025)),
    )


def load_policy_runners(args: argparse.Namespace):
    loaded_base = load_frozen_base_policy(
        args.base_checkpoint,
        device=args.device,
        use_ema=not args.no_base_ema,
        inference_steps=args.base_inference_steps,
    )
    base_runner = FrozenBasePolicyRunner(loaded_base, device=args.device)
    residual_runner = load_residual_policy_runner(
        args.actor_checkpoint,
        device=args.device,
        exploration_seed=args.exploration_seed,
    )
    verify_runtime_training_provenance(
        args,
        residual_runner,
        base_state_key=loaded_base.state_key,
    )
    return base_runner, residual_runner


def verify_runtime_training_provenance(
    args: argparse.Namespace,
    residual_runner: ResidualPolicyRunner,
    *,
    base_state_key: str | None = None,
) -> None:
    """Reject base/schedule settings that differ from fast-actor BC inputs."""

    metadata = getattr(residual_runner, "checkpoint_metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Residual actor checkpoint has no training metadata")
    cache_metadata = metadata.get("base_prediction_cache")
    if cache_metadata is None and isinstance(metadata.get("bc_metadata"), Mapping):
        cache_metadata = metadata["bc_metadata"].get("base_prediction_cache")
    if not isinstance(cache_metadata, Mapping):
        raise ValueError(
            "Residual actor checkpoint has no base prediction provenance; "
            "regenerate the cache and BC checkpoint with residual_rl"
        )
    expected_values = {
        "fast_steps_per_slow": int(args.fast_steps_per_slow),
        "slow_action_start_index": int(args.slow_action_start_index),
        "base_num_inference_steps": int(args.base_inference_steps),
        "base_state_key": (
            str(base_state_key)
            if base_state_key is not None
            else ("model" if args.no_base_ema else "ema_model")
        ),
    }
    for key, current in expected_values.items():
        trained = cache_metadata.get(key)
        if trained != current:
            raise ValueError(
                f"Runtime {key}={current!r} differs from BC cache value {trained!r}"
            )
    expected_hash = str(cache_metadata.get("base_checkpoint_sha256", ""))
    if not expected_hash:
        raise ValueError("BC cache metadata has no base checkpoint fingerprint")
    current_hash = file_sha256(args.base_checkpoint)
    if current_hash != expected_hash:
        raise ValueError(
            "Runtime base checkpoint content differs from the checkpoint used "
            "to create fast BC inputs"
        )


@contextmanager
def real_environment(
    args: argparse.Namespace,
    *,
    base_runner: FrozenBasePolicyRunner,
    residual_runner: ResidualPolicyRunner,
) -> Iterator:
    # Lazy import keeps offline tests and training independent from ROS2.
    # The existing controller exposes its selection as a module flag. Set it
    # before the multiprocessing controller is created so the forked process
    # uses the requested arm-control path. The same controller publishes the
    # seven-dimensional right-hand target independently of this flag.
    from diffusion_policy.real_world import (
        rightarm_hand_insert_plug_interpolation_controller as controller_module,
    )

    controller_module.USE_IMPEDANCE_CONTROLLER = args.control_mode == "impedance"
    if args.control_mode == "impedance" and mp.get_start_method() != "fork":
        raise RuntimeError(
            "The legacy controller's impedance selector is a process-global flag; "
            "this isolated launcher requires the Linux 'fork' start method"
        )
    from diffusion_policy.real_world.bae_real_env_rightarm_hand_insert_plug import (
        DualarmRealEnv,
    )

    class CommandReportingDualarmRealEnv(DualarmRealEnv):
        """Existing env command path plus an explicit scheduled/not-scheduled result.

        The original method returns ``None`` even when it discards an expired
        timestamp.  The residual sidecar must contain only commands actually
        queued for the controller, so this thin new-file adapter preserves the
        implementation while reporting that fact to the runtime.
        """

        def get_obs(self):
            observation = super().get_obs()
            robot_state = self.get_robot_state()
            if "robot_receive_timestamp" in robot_state:
                observation["robot_receive_timestamp"] = np.asarray(
                    robot_state["robot_receive_timestamp"],
                    dtype=np.float64,
                )
            return observation

        def exec_actions(self, actions, timestamps, stages=None):
            assert self.is_ready
            actions = np.asarray(actions)
            timestamps = np.asarray(timestamps)
            if stages is None:
                stages = np.zeros_like(timestamps, dtype=np.int64)
            else:
                stages = np.asarray(stages, dtype=np.int64)

            is_new = timestamps > time.time()
            new_actions = actions[is_new]
            new_timestamps = timestamps[is_new]
            new_stages = stages[is_new]
            for action, target_time in zip(new_actions, new_timestamps):
                self.robot.schedule_waypoint(
                    pose=action,
                    target_time=target_time,
                )
            if self.action_accumulator is not None:
                self.action_accumulator.put(new_actions, new_timestamps)
            if self.stage_accumulator is not None:
                self.stage_accumulator.put(new_stages, new_timestamps)
            return bool(len(new_actions))

    shape_meta = merge_real_environment_shape_meta(
        base_runner.shape_meta,
        _actor_shape_meta(residual_runner),
    )
    resolution = get_real_obs_resolution(base_runner.shape_meta)
    serials = args.camera_serial or ["126122270712"]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with SharedMemoryManager() as shared_memory_manager:
        with CommandReportingDualarmRealEnv(
            output_dir=str(output),
            robot_ip=args.robot_ip,
            frequency=args.frequency,
            camera_serial_numbers=serials,
            n_obs_steps=int(base_runner.policy.n_obs_steps),
            shape_meta=shape_meta,
            obs_image_resolution=resolution,
            obs_float32=True,
            init_joints=args.init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shared_memory_manager,
        ) as environment:
            import cv2

            cv2.setNumThreads(1)
            print("Robot control mode:", args.control_mode)
            yield environment


def operator_stop_requested() -> bool:
    """Match the existing robot evaluator: press ``s`` to end an episode."""

    import cv2

    return cv2.pollKey() == ord("s")


def make_robot_safe_stop(environment):
    """Return an idempotent callback that stops the controller process.

    This is used only for safety violations. Normal operator episode stops keep
    the impedance controller alive to hold its last target until the phase is
    closed.
    """

    stopped = False

    def safe_stop(reason: str) -> None:
        nonlocal stopped
        if stopped:
            return
        stopped = True
        print("SAFETY STOP: stopping robot controller:", reason)
        try:
            environment.robot.stop(wait=False)
        except Exception as error:
            print(
                "SAFETY STOP ERROR: controller stop request failed "
                f"({type(error).__name__}): {error}"
            )

    return safe_stop


def warm_up_inference(
    environment,
    *,
    base_runner: FrozenBasePolicyRunner,
    residual_runner: ResidualPolicyRunner,
    safety: ResidualRobotSafety,
    slow_action_start_index: int,
) -> None:
    """Warm CUDA/model paths without scheduling any robot command."""

    observation = environment.get_obs()
    safety.check_observation(observation, now=time.time())
    chunk = base_runner.predict_chunk(observation)
    if slow_action_start_index >= len(chunk):
        raise ValueError(
            "--slow-action-start-index exceeds the frozen base output chunk"
        )
    base_action = build_base_condition16(
        observation,
        chunk[slow_action_start_index],
    )
    residual = residual_runner.predict(
        observation,
        base_action,
        exploration_std=0.0,
    )
    safety.clip_residual(residual)
    print("Frozen base/actor inference warmup complete (no command sent).")


def run_collection(args: argparse.Namespace) -> list:
    validate_common_args(args)
    if args.exploration_mode == "gaussian":
        if not np.isfinite(args.exploration_std) or args.exploration_std <= 0.0:
            raise ValueError("gaussian exploration requires --exploration-std > 0")
        exploration_std = float(args.exploration_std)
    else:
        if args.exploration_std != 0.0:
            raise ValueError(
                "--exploration-std is only valid with --exploration-mode gaussian"
            )
        exploration_std = 0.0

    base_runner, residual_runner = load_policy_runners(args)
    safety = build_safety(args, residual_runner)
    _verify_actor_and_runtime_bounds(residual_runner, safety.limits)
    runtime_config = build_runtime_config(
        args,
        exploration_std=exploration_std,
        exploration_mode=args.exploration_mode,
        exploration_step_start=args.exploration_step_start,
    )
    sidecar_root = (
        Path(args.sidecar_output).expanduser().resolve()
        if args.sidecar_output is not None
        else Path(args.output).expanduser().resolve() / "residual_rl_episodes"
    )
    writer = EpisodeSidecarWriter(sidecar_root)

    print("Base checkpoint:", base_runner.checkpoint_id)
    print("Actor checkpoint:", residual_runner.checkpoint_id)
    print("Exploration mode:", args.exploration_mode)
    print("Exploration std:", exploration_std)
    print("Sidecar output:", sidecar_root)
    results = []
    with real_environment(
        args,
        base_runner=base_runner,
        residual_runner=residual_runner,
    ) as environment:
        warm_up_inference(
            environment,
            base_runner=base_runner,
            residual_runner=residual_runner,
            safety=safety,
            slow_action_start_index=args.slow_action_start_index,
        )
        runtime = SlowFastResidualRuntime(
            environment=environment,
            base_policy=base_runner,
            residual_policy=residual_runner,
            safety=safety,
            config=runtime_config,
            safe_stop=make_robot_safe_stop(environment),
        )
        for episode_index in range(args.episodes):
            print(f"Starting collect episode {episode_index + 1}/{args.episodes}")
            result = runtime.run_episode(
                label_success=lambda: prompt_success_label(
                    f"Episode {episode_index} success? [1/0]: "
                ),
                sidecar_writer=writer,
                stop_requested=operator_stop_requested,
            )
            results.append(result)
            print(result)
            if result.safety_violation is not None:
                print(
                    "Collection session aborted after safety stop; inspect and "
                    "reset the robot before restarting."
                )
                break
    return results


def main() -> None:
    run_collection(make_parser().parse_args())


if __name__ == "__main__":
    main()
