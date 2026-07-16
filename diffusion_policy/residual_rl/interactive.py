"""Interactive collect -> offline TD3 -> collect residual-RL session.

This is the operator-facing launcher for repeated real-robot improvement.  A
robot environment exists only during the collection phase.  When training is
requested (manually or by the configured episode threshold), that environment
is closed first, all inference models are released, TD3 runs offline, and the
new immutable actor checkpoint is loaded for the next collection phase.

Typical controls between episodes::

    Enter / r   start one rollout
    t           train now from every saved success and failure episode
    q           end the session

During a rollout press ``s``, ``q``, Space, or Enter in the terminal (or ``s`` in the
OpenCV window) to stop command production.  The launcher then asks for the
terminal success label, writes the episode atomically, and returns to the menu.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import select
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator

import h5py
import numpy as np
import torch

from diffusion_policy.residual_rl import collect
from diffusion_policy.residual_rl.checkpoint import load_checkpoint_payload
from diffusion_policy.residual_rl.episode_io import SIDECAR_SCHEMA
from diffusion_policy.residual_rl.runtime import (
    EpisodeSidecarWriter,
    SlowFastResidualRuntime,
    prompt_success_label,
)
from diffusion_policy.residual_rl.train import train as train_td3


class SessionPhase(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    TRAINING = "training"


class SessionPhaseGuard:
    """Fail closed if robot collection and optimizer training could overlap."""

    def __init__(self) -> None:
        self.phase = SessionPhase.IDLE

    @contextmanager
    def collecting(self) -> Iterator[None]:
        if self.phase is not SessionPhase.IDLE:
            raise RuntimeError(f"Cannot start collection while phase={self.phase.value}")
        self.phase = SessionPhase.COLLECTING
        try:
            yield
        finally:
            self.phase = SessionPhase.IDLE

    @contextmanager
    def training(self) -> Iterator[None]:
        if self.phase is not SessionPhase.IDLE:
            raise RuntimeError(
                "Offline TD3 cannot start until the robot environment is closed "
                f"(phase={self.phase.value})"
            )
        self.phase = SessionPhase.TRAINING
        try:
            yield
        finally:
            self.phase = SessionPhase.IDLE


@dataclass(frozen=True)
class ReplayInventory:
    episodes: int
    transitions: int
    successes: int
    failures: int


def inspect_replay_directory(path: str | Path) -> ReplayInventory:
    """Read only small HDF5 metadata/shapes for the operator status line."""

    directory = Path(path).expanduser().resolve()
    episodes = transitions = successes = failures = 0
    for episode_path in sorted(directory.glob("episode_*.hdf5")):
        with h5py.File(episode_path, "r") as source:
            if source.attrs.get("schema", "") != SIDECAR_SCHEMA:
                continue
            length = int(source["action"].shape[0])
            if length <= 0:
                continue
            metadata = json.loads(source.attrs.get("metadata_json", "{}"))
            success = int(metadata.get("success", int(source["reward"][-1])))
            if success not in (0, 1):
                raise ValueError(f"Invalid success label in {episode_path}: {success}")
            episodes += 1
            transitions += length
            successes += int(success == 1)
            failures += int(success == 0)
    return ReplayInventory(episodes, transitions, successes, failures)


def automatic_training_due(
    *,
    saved_since_training: int,
    train_after_episodes: int,
    inventory: ReplayInventory,
    minimum_transitions: int,
) -> bool:
    if train_after_episodes <= 0:
        raise ValueError("train_after_episodes must be > 0")
    if minimum_transitions <= 0:
        raise ValueError("minimum_transitions must be > 0")
    return (
        saved_since_training >= train_after_episodes
        and inventory.transitions >= minimum_transitions
    )


def parse_menu_command(value: str) -> str:
    command = value.strip().lower()
    if command in ("", "r", "rollout"):
        return "rollout"
    if command in ("t", "train"):
        return "train"
    if command in ("q", "quit", "exit"):
        return "quit"
    raise ValueError("Use Enter/r for rollout, t for train, or q for quit")


def _opencv_stop_requested() -> bool:
    try:
        import cv2

        return cv2.pollKey() == ord("s")
    except Exception:
        return False


class RolloutStopKeys:
    """Non-blocking terminal/OpenCV stop poller with restored terminal state."""

    STOP_KEYS = {"s", "S", "q", "Q", " ", "\n", "\r"}

    def __init__(self, stream=None, cv_stop: Callable[[], bool] = _opencv_stop_requested):
        self.stream = sys.stdin if stream is None else stream
        self.cv_stop = cv_stop
        self._fd: int | None = None
        self._attributes = None

    def __enter__(self) -> "RolloutStopKeys":
        try:
            fd = self.stream.fileno()
            if not self.stream.isatty():
                raise RuntimeError(
                    "Interactive robot rollout requires a TTY terminal so the "
                    "non-blocking safety stop key can be read. Run this command "
                    "directly in a terminal, not through redirected stdin."
                )
            self._fd = fd
            self._attributes = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except RuntimeError:
            raise
        except (AttributeError, OSError, termios.error):
            raise RuntimeError(
                "Could not put the operator terminal into non-blocking cbreak "
                "mode; refusing to start a robot rollout without a stop key"
            ) from None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.restore()

    def restore(self) -> None:
        """Restore canonical input before any blocking ``input()`` prompt."""

        if self._fd is not None and self._attributes is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._attributes)
        self._fd = None
        self._attributes = None

    def __call__(self) -> bool:
        if self.cv_stop():
            return True
        if self._fd is None:
            return False
        readable, _, _ = select.select([self.stream], [], [], 0.0)
        if not readable:
            return False
        return self.stream.read(1) in self.STOP_KEYS


def _release_inference_models() -> None:
    # Keep the large frozen diffusion base out of GPU memory while critics are
    # allocated.  The next collection phase reloads both immutable runners.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prompt_label_after_terminal_restore(stop_keys: RolloutStopKeys) -> int:
    # SlowFastResidualRuntime requests the terminal label before returning.
    # Explicit restoration here keeps ``input`` out of cbreak mode; __exit__ is
    # idempotent and remains the exception-safe fallback.
    stop_keys.restore()
    return prompt_success_label("Episode success? [1/0]: ")


def _checkpoint_role(
    actor_checkpoint: str | Path,
    bc_checkpoint: str | Path | None,
) -> tuple[Path, Path | None]:
    """Return frozen BC prior and optional TD3 optimizer resume checkpoint."""

    actor_path = Path(actor_checkpoint).expanduser().resolve()
    actor_payload = load_checkpoint_payload(actor_path, map_location="cpu")
    if actor_payload["kind"] == "bc":
        prior_path = actor_path
        if bc_checkpoint is not None:
            requested = Path(bc_checkpoint).expanduser().resolve()
            if requested != actor_path:
                raise ValueError(
                    "When --actor-checkpoint is BC, --bc-checkpoint must be the "
                    "same file. Start collection and training from one BC policy."
                )
        return prior_path, None

    if bc_checkpoint is None:
        raise ValueError(
            "A TD3 --actor-checkpoint requires --bc-checkpoint so the original "
            "frozen behavior-cloning prior is preserved"
        )
    prior_path = Path(bc_checkpoint).expanduser().resolve()
    prior_payload = load_checkpoint_payload(prior_path, map_location="cpu")
    if prior_payload["kind"] != "bc":
        raise ValueError("--bc-checkpoint must be an original BC checkpoint")
    if prior_payload["model_config"] != actor_payload["model_config"]:
        raise ValueError("BC prior and TD3 actor model configs differ")
    return prior_path, actor_path


def _next_round_directory(training_root: Path) -> tuple[int, Path]:
    training_root.mkdir(parents=True, exist_ok=True)
    used: set[int] = set()
    for path in training_root.glob("round_*"):
        if path.is_dir():
            try:
                used.add(int(path.name.removeprefix("round_")))
            except ValueError:
                pass
    index = max(used, default=0) + 1
    return index, training_root / f"round_{index:04d}"


def make_td3_namespace(
    args: argparse.Namespace,
    *,
    bc_checkpoint: Path,
    resume_checkpoint: Path | None,
    sidecar_root: Path,
    round_output: Path,
    new_online_transitions: int | None = None,
) -> argparse.Namespace:
    """Adapt interactive options to the existing offline TD3 entry point."""

    # No numbered intermediate checkpoint is needed inside an interactive
    # round. train.py always atomically writes checkpoints/latest.pt at the end.
    # This launcher defines one atomic handoff checkpoint per completed round.
    # train.py defines zero as "latest only", including resumed global steps.
    save_every = 0
    fixed_updates = getattr(args, "td3_updates", None)
    if fixed_updates is None:
        if new_online_transitions is None or int(new_online_transitions) <= 0:
            raise ValueError(
                "UTD training requires the number of newly collected online "
                "transitions to be positive"
            )
        updates = max(
            1,
            int(math.ceil(float(args.utd_ratio) * int(new_online_transitions))),
        )
    else:
        updates = int(fixed_updates)
    return argparse.Namespace(
        bc_checkpoint=str(bc_checkpoint),
        episodes=[str(sidecar_root)],
        output=str(round_output),
        updates=updates,
        critic_warmup_updates=int(args.critic_warmup_updates),
        batch_size=int(args.batch_size),
        offline_ratio=float(getattr(args, "offline_ratio", 0.5)),
        offline_actual_dataset=getattr(args, "offline_actual_dataset", None),
        offline_virtual_dataset=getattr(args, "offline_virtual_dataset", None),
        offline_base_predictions=getattr(args, "offline_base_predictions", None),
        offline_num_workers=int(getattr(args, "offline_num_workers", 2)),
        replay_capacity=int(args.replay_capacity),
        gamma=float(args.gamma),
        n_step=int(getattr(args, "n_step", 3)),
        tau=float(args.tau),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        policy_delay=int(args.policy_delay),
        target_noise_fraction=float(args.target_noise_fraction),
        target_noise_clip_fraction=float(args.target_noise_clip_fraction),
        lambda_bc=float(args.lambda_bc),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
        save_every=save_every,
        resume=None if resume_checkpoint is None else str(resume_checkpoint),
        seed=int(args.seed),
        device=args.training_device,
        overwrite=False,
    )


def _write_session_state(
    path: Path,
    *,
    actor_checkpoint: Path,
    bc_checkpoint: Path,
    completed_round: int,
    inventory: ReplayInventory,
) -> None:
    payload = {
        "actor_checkpoint": str(actor_checkpoint),
        "bc_checkpoint": str(bc_checkpoint),
        "completed_td3_round": completed_round,
        "replay": inventory.__dict__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _untrained_session_progress(
    state_path: Path,
    *,
    actor_checkpoint: Path,
    inventory: ReplayInventory,
    starting_from_bc: bool,
) -> tuple[int, int]:
    """Recover episode/transition counts collected after the last TD3 round."""

    trained_episodes = 0
    trained_transitions = 0
    if state_path.is_file():
        payload = json.loads(state_path.read_text())
        recorded_actor = Path(str(payload.get("actor_checkpoint", ""))).expanduser()
        if recorded_actor.resolve() == actor_checkpoint.resolve():
            replay = dict(payload.get("replay", {}))
            trained_episodes = int(replay.get("episodes", 0))
            trained_transitions = int(replay.get("transitions", 0))
    elif not starting_from_bc:
        # Without session provenance we cannot know whether old sidecars were
        # already consumed by an externally created TD3 checkpoint. Count only
        # transitions collected in this process unless a fixed update override
        # is explicitly requested.
        trained_episodes = inventory.episodes
        trained_transitions = inventory.transitions
    return (
        max(inventory.episodes - trained_episodes, 0),
        max(inventory.transitions - trained_transitions, 0),
    )


def _prompt_menu(inventory: ReplayInventory, actor: Path) -> str:
    print(
        "\nReplay: "
        f"{inventory.episodes} episodes / {inventory.transitions} transitions "
        f"(success {inventory.successes}, failure {inventory.failures})"
    )
    print("Actor:", actor)
    while True:
        try:
            return parse_menu_command(
                input("[Enter/r] rollout  [t] train now  [q] quit: ")
            )
        except ValueError as error:
            print(error)


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 0:
        raise ValueError("--episodes must be >= 0 (0 means unlimited)")
    if args.train_after_episodes <= 0:
        raise ValueError("--train-after-episodes must be > 0")
    if args.td3_updates is not None and args.td3_updates <= 0:
        raise ValueError("--td3-updates must be > 0 when supplied")
    if not np.isfinite(args.utd_ratio) or args.utd_ratio <= 0:
        raise ValueError("--utd-ratio must be finite and > 0")
    if args.n_step <= 0:
        raise ValueError("--n-step must be > 0")
    if not np.isfinite(args.offline_ratio) or not 0.0 < args.offline_ratio < 1.0:
        raise ValueError("Interactive mixed replay requires 0 < --offline-ratio < 1")
    if args.offline_num_workers < 0:
        raise ValueError("--offline-num-workers must be >= 0")
    if args.exploration_mode == "gaussian":
        if not np.isfinite(args.exploration_std) or args.exploration_std <= 0.0:
            raise ValueError(
                "--exploration-mode gaussian requires --exploration-std > 0"
            )
    elif args.exploration_std != 0.0:
        raise ValueError(
            "--exploration-std is only valid with --exploration-mode gaussian"
        )
    if args.resfit_learning_starts < 0:
        raise ValueError("--resfit-learning-starts must be >= 0")
    for name in ("resfit_random_noise_scale", "resfit_stddev"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if args.batch_size <= 0 or args.replay_capacity < args.batch_size:
        raise ValueError("Require 0 < batch-size <= replay-capacity")
    if args.min_replay_transitions is not None and args.min_replay_transitions <= 0:
        raise ValueError("--min-replay-transitions must be > 0")
    if args.critic_warmup_updates < 0:
        raise ValueError("--critic-warmup-updates must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")


def run_interactive(args: argparse.Namespace) -> Path:
    """Run until operator quit/max episodes and return the last actor path."""

    _validate_args(args)
    actor_checkpoint = Path(args.actor_checkpoint).expanduser().resolve()
    bc_checkpoint, resume_checkpoint = _checkpoint_role(
        actor_checkpoint,
        args.bc_checkpoint,
    )
    raw_output = Path(args.output).expanduser().resolve()
    sidecar_root = (
        Path(args.sidecar_output).expanduser().resolve()
        if args.sidecar_output is not None
        else raw_output / "residual_rl_episodes"
    )
    training_root = (
        Path(args.training_output).expanduser().resolve()
        if args.training_output is not None
        else raw_output / "residual_rl_training"
    )
    sidecar_root.mkdir(parents=True, exist_ok=True)
    writer = EpisodeSidecarWriter(sidecar_root)
    state_path = training_root / "session_state.json"
    phase_guard = SessionPhaseGuard()
    exploration_std = (
        float(args.exploration_std)
        if args.exploration_mode == "gaussian"
        else 0.0
    )
    minimum_transitions = (
        int(args.min_replay_transitions)
        if args.min_replay_transitions is not None
        else int(args.batch_size)
    )

    total_rollouts = 0
    initial_inventory = inspect_replay_directory(sidecar_root)
    saved_since_training, new_transitions_since_training = _untrained_session_progress(
        state_path,
        actor_checkpoint=actor_checkpoint,
        inventory=initial_inventory,
        starting_from_bc=resume_checkpoint is None,
    )
    quit_requested = False
    while not quit_requested:
        base_runner, residual_runner = collect.load_policy_runners(args)
        safety = collect.build_safety(args, residual_runner)
        collect._verify_actor_and_runtime_bounds(residual_runner, safety.limits)
        phase_inventory = inspect_replay_directory(sidecar_root)
        runtime_config = collect.build_runtime_config(
            args,
            exploration_std=exploration_std,
            exploration_mode=args.exploration_mode,
            exploration_step_start=phase_inventory.transitions,
        )
        train_requested = False

        # The phase guard is intentionally outside the environment context: if
        # either context is active, starting the optimizer is an invariant error.
        with phase_guard.collecting():
            with collect.real_environment(
                args,
                base_runner=base_runner,
                residual_runner=residual_runner,
            ) as environment:
                collect.warm_up_inference(
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
                    safe_stop=collect.make_robot_safe_stop(environment),
                )

                while True:
                    inventory = inspect_replay_directory(sidecar_root)
                    if automatic_training_due(
                        saved_since_training=saved_since_training,
                        train_after_episodes=args.train_after_episodes,
                        inventory=inventory,
                        minimum_transitions=minimum_transitions,
                    ):
                        print(
                            "Automatic TD3 trigger reached; closing the robot "
                            "environment before training."
                        )
                        train_requested = True
                        break
                    if args.episodes > 0 and total_rollouts >= args.episodes:
                        quit_requested = True
                        break

                    command = _prompt_menu(inventory, actor_checkpoint)
                    if command == "quit":
                        quit_requested = True
                        break
                    if command == "train":
                        if inventory.transitions < minimum_transitions:
                            print(
                                "Not enough replay for TD3: "
                                f"{inventory.transitions} < {minimum_transitions} "
                                "transitions. Collect another rollout."
                            )
                            continue
                        train_requested = True
                        break

                    print(
                        "Rollout running. Press s/q, Space, or Enter in this "
                        "terminal to stop (OpenCV s is a secondary fallback)."
                    )
                    with RolloutStopKeys() as stop_keys:
                        result = runtime.run_episode(
                            label_success=lambda: _prompt_label_after_terminal_restore(
                                stop_keys
                            ),
                            sidecar_writer=writer,
                            stop_requested=stop_keys,
                        )
                    total_rollouts += 1
                    if result.sidecar_path is not None:
                        saved_since_training += 1
                        new_transitions_since_training += int(
                            getattr(result, "commanded_steps", 0)
                        )
                    print("Episode result:", result)
                    if result.safety_violation is not None:
                        print(
                            "Interactive session aborted after safety stop. "
                            "Inspect and reset the robot before restarting."
                        )
                        quit_requested = True
                        break

        # Both contexts have exited here: controller/cameras/recording are
        # stopped before a critic, optimizer, or mutable actor is constructed.
        # A ``with ... as`` target remains referenced after context exit in
        # Python. Drop every environment/runtime/runner reference explicitly
        # before allocating TD3 critics and optimizers.
        del environment, runtime, runtime_config, safety
        del base_runner, residual_runner
        _release_inference_models()

        if not train_requested:
            break

        inventory = inspect_replay_directory(sidecar_root)
        if inventory.transitions < minimum_transitions:
            # Defensive recheck after closing hardware.
            print("Replay became too small for training; returning to collection.")
            continue
        round_index, round_output = _next_round_directory(training_root)
        update_basis = new_transitions_since_training
        if update_basis <= 0 and resume_checkpoint is None:
            # A first manual train can legitimately consume sidecars collected
            # by an earlier process before any TD3 checkpoint existed.
            update_basis = inventory.transitions
        if args.td3_updates is None and update_basis <= 0:
            print(
                "No newly collected transitions are available for UTD-based "
                "training. Collect a rollout or pass --td3-updates explicitly."
            )
            continue
        td3_args = make_td3_namespace(
            args,
            bc_checkpoint=bc_checkpoint,
            resume_checkpoint=resume_checkpoint,
            sidecar_root=sidecar_root,
            round_output=round_output,
            new_online_transitions=update_basis,
        )
        print(f"\n=== Offline TD3 round {round_index} ===")
        print("Robot environment: CLOSED")
        print(
            "Gradient updates:",
            td3_args.updates,
            "(fixed override)"
            if args.td3_updates is not None
            else f"(UTD {args.utd_ratio} × {update_basis} new transitions)",
        )
        with phase_guard.training():
            train_td3(td3_args)

        actor_checkpoint = round_output / "checkpoints" / "latest.pt"
        if not actor_checkpoint.is_file():
            raise FileNotFoundError(
                f"TD3 finished without its expected actor checkpoint: {actor_checkpoint}"
            )
        resume_checkpoint = actor_checkpoint
        args.actor_checkpoint = str(actor_checkpoint)
        saved_since_training = 0
        new_transitions_since_training = 0
        inventory = inspect_replay_directory(sidecar_root)
        _write_session_state(
            state_path,
            actor_checkpoint=actor_checkpoint,
            bc_checkpoint=bc_checkpoint,
            completed_round=round_index,
            inventory=inventory,
        )
        print("New actor ready for the next rollout:", actor_checkpoint)
        if args.episodes > 0 and total_rollouts >= args.episodes:
            quit_requested = True

    print("Interactive session ended. Last actor:", actor_checkpoint)
    return actor_checkpoint


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    collect.add_common_arguments(parser)
    parser.set_defaults(episodes=0)
    parser.add_argument(
        "--bc-checkpoint",
        default=None,
        help=(
            "Frozen BC prior. Omit when actor-checkpoint itself is BC; required "
            "when resuming from a TD3 actor."
        ),
    )
    parser.add_argument(
        "--training-output",
        default=None,
        help="Defaults to OUTPUT/residual_rl_training",
    )
    parser.add_argument(
        "--train-after-episodes",
        type=int,
        default=5,
        help="Automatically train after this many newly saved rollouts",
    )
    parser.add_argument(
        "--min-replay-transitions",
        type=int,
        default=None,
        help="Default is batch-size; auto/manual training waits for this many",
    )
    parser.add_argument(
        "--exploration-mode",
        choices=("resfit", "none", "gaussian"),
        default="resfit",
        help=(
            "resfit: uniform residual warmup then Gaussian actor noise; "
            "none: deterministic collection; gaussian: legacy physical std"
        ),
    )
    parser.add_argument(
        "--exploration-std",
        type=float,
        default=0.0,
        help="Physical scalar std used only by --exploration-mode gaussian",
    )
    parser.add_argument("--resfit-learning-starts", type=int, default=10_000)
    parser.add_argument("--resfit-random-noise-scale", type=float, default=0.2)
    parser.add_argument("--resfit-stddev", type=float, default=0.025)

    parser.add_argument(
        "--td3-updates",
        type=int,
        default=None,
        help="Fixed override; default derives updates from utd-ratio × new transitions",
    )
    parser.add_argument("--utd-ratio", type=float, default=4.0)
    parser.add_argument("--critic-warmup-updates", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--offline-ratio", type=float, default=0.5)
    parser.add_argument("--offline-actual-dataset", default=None)
    parser.add_argument("--offline-virtual-dataset", default=None)
    parser.add_argument("--offline-base-predictions", default=None)
    parser.add_argument("--offline-num-workers", type=int, default=2)
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=20_000,
        help="Maximum in-memory replay transitions (roughly 5.8 GiB at 20k)",
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-noise-fraction", type=float, default=0.1)
    parser.add_argument("--target-noise-clip-fraction", type=float, default=0.2)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-device",
        default="auto",
        help="TD3 device; collection still uses --device",
    )
    return parser


def main() -> None:
    run_interactive(make_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = [
    "ReplayInventory",
    "RolloutStopKeys",
    "SessionPhase",
    "SessionPhaseGuard",
    "automatic_training_due",
    "inspect_replay_directory",
    "make_parser",
    "make_td3_namespace",
    "parse_menu_command",
    "run_interactive",
]
