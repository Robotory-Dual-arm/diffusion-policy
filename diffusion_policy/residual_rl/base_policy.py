"""Loading and inference helpers for the frozen slow/base policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig


@dataclass(frozen=True)
class FrozenBasePolicy:
    """A loaded base policy together with the checkpoint configuration."""

    policy: torch.nn.Module
    cfg: DictConfig
    checkpoint_path: Path
    state_key: str


def load_frozen_base_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
    inference_steps: int | None = None,
    n_action_steps: int | None = None,
    strict: bool = True,
) -> FrozenBasePolicy:
    """Load only the policy module from a diffusion-policy workspace checkpoint.

    The returned policy is in evaluation mode, has gradients disabled, and is
    never owned by an RL optimizer.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    with checkpoint_path.open("rb") as stream:
        payload = torch.load(
            stream,
            map_location="cpu",
            pickle_module=dill,
        )
    cfg = payload["cfg"]
    state_dicts = payload["state_dicts"]
    state_key = "ema_model" if use_ema and "ema_model" in state_dicts else "model"

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dicts[state_key], strict=strict)
    del payload

    if inference_steps is not None:
        if not hasattr(policy, "num_inference_steps"):
            raise AttributeError(
                f"{type(policy).__name__} has no num_inference_steps setting"
            )
        policy.num_inference_steps = int(inference_steps)

    if n_action_steps is None:
        horizon = int(getattr(policy, "horizon"))
        n_obs_steps = int(getattr(policy, "n_obs_steps"))
        n_action_steps = horizon - n_obs_steps + 1
    policy.n_action_steps = int(n_action_steps)
    policy.eval()
    policy.requires_grad_(False)
    policy.to(torch.device(device))

    return FrozenBasePolicy(
        policy=policy,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        state_key=state_key,
    )


def numpy_obs_to_batch(
    obs: Mapping[str, np.ndarray],
    *,
    device: str | torch.device,
    add_batch_dim: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert a physical-unit numpy observation dictionary to tensors."""

    result: dict[str, torch.Tensor] = {}
    device = torch.device(device)
    for key, value in obs.items():
        tensor = torch.from_numpy(np.asarray(value))
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(dtype=torch.float32).div_(255.0)
        elif not torch.is_floating_point(tensor):
            tensor = tensor.float()
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        result[key] = tensor.to(device=device, dtype=torch.float32)
    return result


@torch.no_grad()
def predict_base_action_chunk(
    loaded: FrozenBasePolicy,
    obs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Predict a physical-unit action chunk without changing base parameters."""

    policy = loaded.policy
    policy.eval()
    result = policy.predict_action(dict(obs))
    action = result["action"]
    if action.ndim != 3:
        raise RuntimeError(f"Expected base action [B,T,D], got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise FloatingPointError("Base policy produced NaN or Inf")
    return action
