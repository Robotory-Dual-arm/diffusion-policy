"""Read canonical collect/evaluate episode sidecars for offline TD3 updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

from diffusion_policy.residual_rl.replay_buffer import CANONICAL_KEYS, ResidualReplayBuffer
from diffusion_policy.residual_rl.runtime import CANONICAL_OBSERVATION_SHAPES


SIDECAR_SCHEMA = "residual_rl_commanded_episode_v1"


def discover_episode_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("episode_*.hdf5")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError("No episode_*.hdf5 sidecars were found")
    return unique


def load_episode(path: str | Path) -> tuple[dict, dict]:
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as source:
        schema = source.attrs.get("schema", "")
        if schema != SIDECAR_SCHEMA:
            raise ValueError(f"Unsupported episode schema {schema!r}: {path}")
        metadata = json.loads(source.attrs.get("metadata_json", "{}"))
        episode = {
            "obs": {
                key: np.asarray(source["obs"][key])
                for key in CANONICAL_OBSERVATION_SHAPES
            },
            "next_obs": {
                key: np.asarray(source["next_obs"][key])
                for key in CANONICAL_OBSERVATION_SHAPES
            },
        }
        for key in ("base_action", "next_base_action", "action", "reward", "done"):
            episode[key] = np.asarray(source[key])

    missing = set(CANONICAL_KEYS) - set(episode)
    if missing:
        raise KeyError(f"{path} is missing canonical keys {sorted(missing)}")
    length = int(episode["action"].shape[0])
    if length <= 0:
        raise ValueError(f"Empty commanded episode: {path}")
    reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
    done = np.asarray(episode["done"], dtype=np.float32).reshape(-1)
    if reward.shape != (length,) or done.shape != (length,):
        raise ValueError(f"Invalid reward/done shape in {path}")
    if np.any(reward[:-1] != 0) or reward[-1] not in (0, 1):
        raise ValueError(f"Episode does not use terminal-only 0/1 reward: {path}")
    if np.any(done[:-1] != 0) or done[-1] != 1:
        raise ValueError(f"Episode has invalid terminal flags: {path}")
    metadata.setdefault("source_path", str(path))
    return episode, metadata


def make_replay_buffer(
    *,
    capacity: int,
    seed: int | None = None,
) -> ResidualReplayBuffer:
    return ResidualReplayBuffer(
        capacity=capacity,
        observation_shapes=CANONICAL_OBSERVATION_SHAPES,
        image_keys=("image0",),
        base_action_dim=16,
        action_dim=6,
        seed=seed,
    )


def make_n_step_episode(
    episode: Mapping,
    *,
    n_step: int,
    gamma: float,
) -> dict:
    """Convert one validated terminal-reward episode to n-step targets.

    The current action remains the actually executed residual at time ``t``.
    ``next_obs`` and ``next_base_action`` advance by up to ``n_step`` source
    transitions without crossing the episode boundary.  All raw sidecars are
    terminal, so a shortened final window has ``done=1`` and never bootstraps.
    Consequently the TD target can use one global ``gamma ** n_step`` factor
    for every non-terminal transformed transition.
    """

    n_step = int(n_step)
    gamma = float(gamma)
    if n_step <= 0:
        raise ValueError("n_step must be > 0")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")

    action = np.asarray(episode["action"])
    length = int(action.shape[0])
    if length <= 0:
        raise ValueError("episode must contain at least one transition")
    source_reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
    source_done = np.asarray(episode["done"], dtype=np.float32).reshape(-1)
    if source_reward.shape != (length,) or source_done.shape != (length,):
        raise ValueError("reward and done must each contain N scalars")
    if np.any(source_reward[:-1] != 0.0) or source_reward[-1] not in (0.0, 1.0):
        raise ValueError("n-step source episode must use terminal-only 0/1 reward")
    if np.any(source_done[:-1] != 0.0) or source_done[-1] != 1.0:
        raise ValueError("n-step source episode must end with done=1")

    reward = np.zeros(length, dtype=np.float32)
    done = np.zeros(length, dtype=np.float32)
    next_indices = np.empty(length, dtype=np.int64)
    for step in range(length):
        stop = min(step + n_step, length)
        offsets = np.arange(stop - step, dtype=np.float64)
        reward[step] = float(
            np.sum((gamma ** offsets) * source_reward[step:stop], dtype=np.float64)
        )
        done[step] = float(np.any(source_done[step:stop] > 0.0))
        next_indices[step] = stop - 1

    return {
        "obs": {
            key: np.asarray(value)
            for key, value in episode["obs"].items()
        },
        "next_obs": {
            key: np.asarray(value)[next_indices]
            for key, value in episode["next_obs"].items()
        },
        "base_action": np.asarray(episode["base_action"]),
        "next_base_action": np.asarray(episode["next_base_action"])[next_indices],
        "action": action,
        "reward": reward,
        "done": done,
    }


def load_into_replay(
    replay: ResidualReplayBuffer,
    paths: Iterable[str | Path],
    *,
    n_step: int = 1,
    gamma: float = 0.99,
) -> dict[str, int | float]:
    files = discover_episode_files(paths)
    transitions = 0
    successes = 0
    failures = 0
    for path in files:
        episode, metadata = load_episode(path)
        transformed = make_n_step_episode(
            episode,
            n_step=n_step,
            gamma=gamma,
        )
        transitions += replay.add_episode(
            transformed,
            terminal_reward_only=(int(n_step) == 1),
        )
        success = int(metadata.get("success", int(np.asarray(episode["reward"])[-1])))
        successes += int(success == 1)
        failures += int(success == 0)
    return {
        "episodes": len(files),
        "transitions": transitions,
        "successes": successes,
        "failures": failures,
        "n_step": int(n_step),
        "gamma": float(gamma),
    }


__all__ = [
    "SIDECAR_SCHEMA",
    "discover_episode_files",
    "load_episode",
    "load_into_replay",
    "make_n_step_episode",
    "make_replay_buffer",
]
