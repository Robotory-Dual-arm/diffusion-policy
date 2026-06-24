from typing import Dict
import pathlib

import dill
import hydra
import numpy as np
import torch

from diffusion_policy.residual_policy.pose_util import (
    apply_residual_action_to_pose9,
    current_obs_to_pose9,
    relative_pose9_to_abs_pose9,
)


def load_policy_from_ckpt(ckpt_path, device, use_ema=True):
    ckpt_path = pathlib.Path(ckpt_path).expanduser()
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key], strict=False)
    policy.to(device)
    policy.eval()
    del payload
    return cfg, policy


def attach_slow_action_to_obs(
        obs_dict: Dict[str, torch.Tensor],
        slow_action_rel: torch.Tensor,
        slow_action_key="slow_action_rel",
        n_obs_steps=1):
    out = dict(obs_dict)
    if slow_action_rel.ndim == 2:
        slow_action_rel = slow_action_rel[:, None, :]
    if slow_action_rel.shape[1] != n_obs_steps:
        slow_action_rel = slow_action_rel[:, -1:, :].repeat(1, n_obs_steps, 1)
    out[slow_action_key] = slow_action_rel
    return out


@torch.no_grad()
def predict_slow_fast_residual_action(
        slow_policy,
        fast_policy,
        obs_dict: Dict[str, torch.Tensor],
        raw_env_obs: Dict[str, np.ndarray],
        arm="R",
        slow_action_index=0,
        slow_action_key=None,
        slow_action_pose_repr="relative"):
    """Run slow policy, fast residual policy, and compose final absolute pose9.

    Args:
        obs_dict: preprocessed torch obs used by the slow policy.
        raw_env_obs: raw numpy env obs with robot_pose_{arm}, robot_quat_{arm};
            only the latest pose is used as the relative-action base.

    Returns:
        dict with final absolute pose command and intermediate terms.
    """
    slow_result = slow_policy.predict_action(obs_dict)
    slow_action = slow_result["action"][:, slow_action_index]
    if slow_action_key is None:
        slow_action_key = getattr(fast_policy, "slow_action_key", "slow_action_rel")

    fast_obs = attach_slow_action_to_obs(
        obs_dict=obs_dict,
        slow_action_rel=slow_action,
        slow_action_key=slow_action_key,
        n_obs_steps=fast_policy.n_obs_steps,
    )
    fast_result = fast_policy.predict_action(fast_obs)
    residual_action = fast_result["action"][:, 0]

    slow_action_np = slow_action.detach().cpu().numpy()
    residual_np = residual_action.detach().cpu().numpy()
    base_pose9 = current_obs_to_pose9(raw_env_obs, arm=arm)

    slow_abs = []
    final_abs = []
    for i in range(slow_action_np.shape[0]):
        if slow_action_pose_repr == "relative":
            this_slow_abs = relative_pose9_to_abs_pose9(base_pose9, slow_action_np[i])
        elif slow_action_pose_repr == "abs":
            this_slow_abs = slow_action_np[i]
        else:
            raise ValueError(f"Unsupported slow_action_pose_repr: {slow_action_pose_repr}")
        this_final_abs = apply_residual_action_to_pose9(this_slow_abs, residual_np[i])
        slow_abs.append(this_slow_abs)
        final_abs.append(this_final_abs)

    slow_abs = np.stack(slow_abs, axis=0).astype(np.float32)
    final_abs = np.stack(final_abs, axis=0).astype(np.float32)

    return {
        "action": final_abs,
        "slow_action": slow_action_np,
        "slow_action_rel": slow_action_np,
        "slow_action_abs": slow_abs,
        "residual_action": residual_np,
        "residual_delta6": residual_np,
        "slow_result": slow_result,
        "fast_result": fast_result,
    }
