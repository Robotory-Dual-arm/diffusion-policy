"""In-memory replay buffer for separated real-robot collect/train phases."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch


CANONICAL_KEYS = (
    "obs",
    "next_obs",
    "base_action",
    "next_base_action",
    "action",
    "reward",
    "done",
)


class ResidualReplayBuffer:
    """Fixed-capacity ring buffer with a dict-based episode API.

    Canonical episodes contain:

    - ``obs`` and ``next_obs``: dictionaries of arrays with leading length N;
    - ``base_action`` and ``next_base_action``: ``[N, 16]``;
    - ``action``: the *executed* (already noise-added and clipped) residual
      ``[N, 6]``;
    - ``reward`` and ``done``: ``[N]`` or ``[N, 1]``.

    Image fields are stored as uint8 by default. Float images must be in [0, 1]
    and are quantized on insertion; samples are decoded to float32 [0, 1].
    """

    def __init__(
        self,
        capacity: int,
        observation_shapes: Mapping[str, Sequence[int]],
        image_keys: Sequence[str] = ("image0",),
        base_action_dim: int = 16,
        action_dim: int = 6,
        seed: Optional[int] = None,
    ):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if int(base_action_dim) != 16:
            raise ValueError("base_action_dim must be 16 (relative pose9 + hand7)")
        if int(action_dim) != 6:
            raise ValueError("action_dim must be 6 (residual delta6)")
        if not observation_shapes:
            raise ValueError("observation_shapes cannot be empty")

        self.capacity = capacity
        self.observation_shapes = {
            key: tuple(int(size) for size in shape)
            for key, shape in observation_shapes.items()
        }
        self.image_keys = tuple(image_keys)
        unknown_images = set(self.image_keys) - set(self.observation_shapes)
        if unknown_images:
            raise KeyError(f"image_keys missing from observation_shapes: {unknown_images}")
        self.base_action_dim = int(base_action_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self._size = 0
        self._position = 0

        self._obs = self._allocate_observation()
        self._next_obs = self._allocate_observation()
        self._base_action = np.empty((capacity, self.base_action_dim), dtype=np.float32)
        self._next_base_action = np.empty((capacity, self.base_action_dim), dtype=np.float32)
        self._action = np.empty((capacity, self.action_dim), dtype=np.float32)
        self._reward = np.empty((capacity, 1), dtype=np.float32)
        self._done = np.empty((capacity, 1), dtype=np.float32)

    def _allocate_observation(self) -> dict[str, np.ndarray]:
        return {
            key: np.empty(
                (self.capacity,) + shape,
                dtype=np.uint8 if key in self.image_keys else np.float32,
            )
            for key, shape in self.observation_shapes.items()
        }

    def __len__(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._position

    def _encode_image(self, value: Any, key: str) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype == np.uint8:
            return array
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or Inf")
        if array.size and (float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6):
            raise ValueError(f"Float image {key} must be in [0, 1]")
        return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _encode_observation(
        self,
        observation: Mapping[str, Any],
        leading_length: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        missing = set(self.observation_shapes) - set(observation)
        if missing:
            raise KeyError(f"Missing observation keys: {sorted(missing)}")
        encoded = {}
        for key, shape in self.observation_shapes.items():
            value = self._encode_image(observation[key], key) if key in self.image_keys else np.asarray(observation[key], dtype=np.float32)
            expected = shape if leading_length is None else (leading_length,) + shape
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{key} has shape {tuple(value.shape)}, expected {expected}"
                )
            if key not in self.image_keys and not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN or Inf")
            encoded[key] = value
        return encoded

    @staticmethod
    def _finite_array(value: Any, shape: tuple[int, ...], key: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if tuple(array.shape) != shape:
            raise ValueError(f"{key} has shape {tuple(array.shape)}, expected {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or Inf")
        return array

    def add(
        self,
        obs: Mapping[str, Any],
        next_obs: Mapping[str, Any],
        base_action: Any,
        next_base_action: Any,
        action: Any,
        reward: float,
        done: float,
    ) -> int:
        encoded_obs = self._encode_observation(obs)
        encoded_next_obs = self._encode_observation(next_obs)
        base_action = self._finite_array(
            base_action, (self.base_action_dim,), "base_action"
        )
        next_base_action = self._finite_array(
            next_base_action, (self.base_action_dim,), "next_base_action"
        )
        action = self._finite_array(action, (self.action_dim,), "action")
        reward = float(reward)
        done = float(done)
        if not np.isfinite(reward) or not np.isfinite(done):
            raise ValueError("reward and done must be finite")
        if not 0.0 <= done <= 1.0:
            raise ValueError("done must be in [0, 1]")

        index = self._position
        for key in self.observation_shapes:
            self._obs[key][index] = encoded_obs[key]
            self._next_obs[key][index] = encoded_next_obs[key]
        self._base_action[index] = base_action
        self._next_base_action[index] = next_base_action
        self._action[index] = action
        self._reward[index, 0] = reward
        self._done[index, 0] = done
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return index

    def add_episode(
        self,
        episode: Mapping[str, Any],
        *,
        terminal_reward_only: bool = True,
    ) -> int:
        """Add all transitions from one canonical episode.

        Raw robot sidecars use one terminal 0/1 reward and therefore keep
        ``terminal_reward_only=True``.  A preprocessed n-step episode has the
        same number of transitions, but its terminal success can appear in the
        final ``n`` return targets; loaders pass ``False`` only after validating
        and transforming the original terminal-only sidecar.
        """
        missing = set(CANONICAL_KEYS) - set(episode)
        if missing:
            raise KeyError(f"Missing canonical episode keys: {sorted(missing)}")
        base_action = np.asarray(episode["base_action"])
        if base_action.ndim != 2:
            raise ValueError("base_action must have shape [N, 16]")
        length = int(base_action.shape[0])
        if length <= 0:
            raise ValueError("episode must contain at least one transition")

        obs = self._encode_observation(episode["obs"], leading_length=length)
        next_obs = self._encode_observation(
            episode["next_obs"], leading_length=length
        )
        base_action = self._finite_array(
            base_action, (length, self.base_action_dim), "base_action"
        )
        next_base_action = self._finite_array(
            episode["next_base_action"],
            (length, self.base_action_dim),
            "next_base_action",
        )
        action = self._finite_array(
            episode["action"], (length, self.action_dim), "action"
        )
        reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
        done = np.asarray(episode["done"], dtype=np.float32).reshape(-1)
        if reward.shape != (length,) or done.shape != (length,):
            raise ValueError("reward and done must each contain N scalars")
        if not np.isfinite(reward).all() or not np.isfinite(done).all():
            raise ValueError("reward and done must be finite")
        if np.any((done < 0.0) | (done > 1.0)):
            raise ValueError("done must be in [0, 1]")
        if terminal_reward_only:
            if np.any(reward[:-1] != 0.0) or reward[-1] not in (0.0, 1.0):
                raise ValueError(
                    "Episode reward must be terminal-only: "
                    "[0, ..., success_0_or_1]"
                )
        elif np.any((reward < 0.0) | (reward > 1.0)):
            raise ValueError("Preprocessed n-step rewards must be in [0, 1]")
        if terminal_reward_only:
            if np.any(done[:-1] != 0.0) or done[-1] != 1.0:
                raise ValueError("Episode done must be [0, ..., 0, 1]")
        elif done[-1] != 1.0 or np.any(np.diff(done) < 0.0):
            raise ValueError(
                "Preprocessed n-step done flags must be a terminal suffix"
            )

        for step in range(length):
            self.add(
                obs={key: value[step] for key, value in obs.items()},
                next_obs={key: value[step] for key, value in next_obs.items()},
                base_action=base_action[step],
                next_base_action=next_base_action[step],
                action=action[step],
                reward=reward[step],
                done=done[step],
            )
        return length

    def add_labeled_episode(
        self,
        observations: Mapping[str, Any],
        base_actions: Any,
        actions: Any,
        success: int | bool,
    ) -> int:
        """Build terminal-only 0/1 rewards from T+1 states and base actions."""
        if success not in (0, 1, False, True):
            raise ValueError("success must be 0/1 or bool")
        actions = np.asarray(actions)
        length = len(actions)
        if length <= 0:
            raise ValueError("episode must contain at least one action")
        base_actions = np.asarray(base_actions)
        if base_actions.shape != (length + 1, self.base_action_dim):
            raise ValueError(
                f"base_actions must have shape ({length + 1}, {self.base_action_dim})"
            )
        encoded_observations = self._encode_observation(
            observations, leading_length=length + 1
        )
        reward = np.zeros(length, dtype=np.float32)
        reward[-1] = float(bool(success))
        done = np.zeros(length, dtype=np.float32)
        done[-1] = 1.0
        return self.add_episode({
            "obs": {key: value[:-1] for key, value in encoded_observations.items()},
            "next_obs": {key: value[1:] for key, value in encoded_observations.items()},
            "base_action": base_actions[:-1],
            "next_base_action": base_actions[1:],
            "action": actions,
            "reward": reward,
            "done": done,
        })

    def _chronological_indices(self) -> np.ndarray:
        if self._size < self.capacity:
            return np.arange(self._size, dtype=np.int64)
        return np.concatenate([
            np.arange(self._position, self.capacity, dtype=np.int64),
            np.arange(0, self._position, dtype=np.int64),
        ])

    def _torch_observation(
        self,
        arrays: Mapping[str, np.ndarray],
        indices: np.ndarray,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in arrays.items():
            tensor = torch.from_numpy(value[indices])
            if key in self.image_keys:
                tensor = tensor.to(dtype=torch.float32).div_(255.0)
            else:
                tensor = tensor.to(dtype=torch.float32)
            result[key] = tensor.to(device=device)
        return result

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer")
        indices = self.rng.integers(0, self._size, size=batch_size)
        # Before the first wrap, valid physical indices equal [0, size). Once
        # full, every physical index is valid and uniform sampling is unchanged.
        return {
            "obs": self._torch_observation(self._obs, indices, device),
            "next_obs": self._torch_observation(self._next_obs, indices, device),
            "base_action": torch.from_numpy(self._base_action[indices]).to(device),
            "next_base_action": torch.from_numpy(self._next_base_action[indices]).to(device),
            "action": torch.from_numpy(self._action[indices]).to(device),
            "reward": torch.from_numpy(self._reward[indices]).to(device),
            "done": torch.from_numpy(self._done[indices]).to(device),
        }

    def state_dict(self) -> dict[str, Any]:
        indices = self._chronological_indices()
        return {
            "version": 1,
            "capacity": self.capacity,
            "observation_shapes": copy.deepcopy(self.observation_shapes),
            "image_keys": self.image_keys,
            "base_action_dim": self.base_action_dim,
            "action_dim": self.action_dim,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "data": {
                "obs": {key: value[indices].copy() for key, value in self._obs.items()},
                "next_obs": {
                    key: value[indices].copy()
                    for key, value in self._next_obs.items()
                },
                "base_action": self._base_action[indices].copy(),
                "next_base_action": self._next_base_action[indices].copy(),
                "action": self._action[indices].copy(),
                "reward": self._reward[indices].copy(),
                "done": self._done[indices].copy(),
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError(f"Unsupported replay state version {state.get('version')}")
        expected = {
            "observation_shapes": self.observation_shapes,
            "image_keys": self.image_keys,
            "base_action_dim": self.base_action_dim,
            "action_dim": self.action_dim,
        }
        for key, expected_value in expected.items():
            loaded_value = state[key]
            if key == "observation_shapes":
                loaded_value = {
                    name: tuple(shape) for name, shape in loaded_value.items()
                }
            elif key == "image_keys":
                loaded_value = tuple(loaded_value)
            if loaded_value != expected_value:
                raise ValueError(
                    f"Replay state {key} mismatch: {loaded_value!r} != {expected_value!r}"
                )
        data = state["data"]
        length = int(np.asarray(data["action"]).shape[0])
        if length > self.capacity:
            # Loading into a smaller buffer keeps the newest transitions.
            start = length - self.capacity
        else:
            start = 0
        loaded_length = length - start
        self._size = loaded_length
        self._position = loaded_length % self.capacity
        for key in self.observation_shapes:
            self._obs[key][:loaded_length] = np.asarray(data["obs"][key])[start:]
            self._next_obs[key][:loaded_length] = np.asarray(data["next_obs"][key])[start:]
        self._base_action[:loaded_length] = np.asarray(data["base_action"])[start:]
        self._next_base_action[:loaded_length] = np.asarray(data["next_base_action"])[start:]
        self._action[:loaded_length] = np.asarray(data["action"])[start:]
        self._reward[:loaded_length] = np.asarray(data["reward"])[start:]
        self._done[:loaded_length] = np.asarray(data["done"])[start:]
        if "rng_state" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
