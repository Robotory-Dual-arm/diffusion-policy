"""Lazy successful-demonstration transitions for mixed residual TD3 replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.residual_rl.bc_dataset import PairedResidualBCDataset


class PairedResidualRLDataset(Dataset):
    """Expose the paired fast-BC data as successful n-step RL transitions.

    The wrapped BC sample already has the exact online conditioning:
    ``obs_t``, cached base target for that fast tick, and the demonstrated
    actual-to-virtual residual action.  This wrapper advances the bootstrap
    state within the same demonstration and assigns one terminal success
    reward.  HDF5 observations remain lazy, so a 20k-image demo buffer is not
    copied into the in-memory online replay.
    """

    def __init__(
        self,
        actual_path,
        virtual_path,
        *,
        base_action_source,
        base_action_key: str = "actions",
        action_target_shift: int = 1,
        base_action_target_shift: int | None = None,
        n_step: int = 3,
        gamma: float = 0.995,
    ) -> None:
        super().__init__()
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        if self.n_step <= 0:
            raise ValueError("n_step must be > 0")
        if not np.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        self.bc_dataset = PairedResidualBCDataset(
            actual_path,
            virtual_path,
            base_action_source=base_action_source,
            base_action_key=base_action_key,
            action_target_shift=action_target_shift,
            base_action_target_shift=base_action_target_shift,
        )
        self._episode_lengths = self.bc_dataset.episode_lengths.astype(np.int64)
        self._episode_starts = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(self._episode_lengths[:-1], dtype=np.int64),
            )
        )

    @property
    def demo_names(self) -> tuple[str, ...]:
        return self.bc_dataset.demo_names

    @property
    def episode_lengths(self) -> np.ndarray:
        return self._episode_lengths.copy()

    def __len__(self) -> int:
        return len(self.bc_dataset)

    def close(self) -> None:
        self.bc_dataset.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _rl_fields(sample: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        observation = {
            key: value
            for key, value in sample["obs"].items()
        }
        return observation, sample["base_action"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, step_index = self.bc_dataset._locate(int(index))
        episode_length = int(self._episode_lengths[episode_index])
        episode_start = int(self._episode_starts[episode_index])
        current = self.bc_dataset[index]

        bootstrap_step = step_index + self.n_step
        terminal = bootstrap_step >= episode_length
        if terminal:
            # The bootstrap term is masked by done. Reusing the last valid
            # current feature avoids fabricating a target action beyond the
            # paired demonstration while keeping every tensor finite.
            bootstrap = current
        else:
            bootstrap = self.bc_dataset[episode_start + bootstrap_step]

        observation, base_action = self._rl_fields(current)
        next_observation, next_base_action = self._rl_fields(bootstrap)
        reward = 0.0
        if terminal:
            terminal_distance = episode_length - 1 - step_index
            reward = self.gamma ** terminal_distance

        return {
            "obs": observation,
            "next_obs": next_observation,
            "base_action": base_action,
            "next_base_action": next_base_action,
            "action": current["action"],
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(float(terminal), dtype=torch.float32),
        }


__all__ = ["PairedResidualRLDataset"]
