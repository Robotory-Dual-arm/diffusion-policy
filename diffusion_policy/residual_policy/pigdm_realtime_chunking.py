"""Runtime PiGDM-style realtime chunking for residual slow policies.

This module keeps the PiGDM logic local to ``diffusion_policy.residual_policy``.
It patches a loaded diffusion policy at inference time, so the checkpoint config
does not need to point at a special policy or scheduler class.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Dict, Optional

import numpy as np
import torch

from diffusion_policy.real_world.real_inference_util import (
    get_abs_action_from_relative,
    get_relative_action_from_abs,
)


@dataclass
class PiGDMRealtimeChunkingConfig:
    executed_steps: int = 6
    overlap_steps: Optional[int] = None
    hard_steps: int = 3
    guidance_scale: float = 5.0
    weight_mode: str = "exp_ramp"
    condition_start: int = 0


def _make_overlap_weights(
        horizon: int,
        overlap_steps: int,
        hard_steps: int,
        condition_start: int,
        mode: str,
        device,
        dtype):
    weights = torch.zeros(horizon, device=device, dtype=dtype)
    if overlap_steps <= 0 or condition_start >= horizon:
        return weights

    end = min(horizon, condition_start + overlap_steps)
    hard_end = min(end, condition_start + max(0, hard_steps))
    if hard_end > condition_start:
        weights[condition_start:hard_end] = 1.0

    taper_len = end - hard_end
    if taper_len > 0:
        denom = taper_len + 1
        taper = torch.arange(taper_len, 0, -1, device=device, dtype=dtype) / denom
        weights[hard_end:end] = taper

    if mode == "exp_ramp":
        weights = weights * torch.expm1(weights) / (np.e - 1.0)
    elif mode == "linear":
        pass
    elif mode == "uniform":
        weights = (weights > 0).to(dtype)
    else:
        raise ValueError(f"Unsupported PiGDM weight_mode: {mode}")
    return weights


def _build_overlap_target(prev_action, sample, config: PiGDMRealtimeChunkingConfig):
    """Build y and per-step weights for previous-chunk guidance."""
    if prev_action is None:
        return None, None
    if prev_action.ndim != 3:
        raise ValueError(f"prev_action must be B x T x D, got {prev_action.shape}")

    batch, horizon = sample.shape[:2]
    prev_action_dim = prev_action.shape[2]
    if prev_action.shape[0] != batch or prev_action_dim > sample.shape[2]:
        raise ValueError(
            "prev_action shape mismatch: "
            f"prev={tuple(prev_action.shape)}, sample={tuple(sample.shape)}"
        )

    executed_steps = max(0, int(config.executed_steps))
    condition_start = max(0, int(config.condition_start))
    available_prev = max(0, prev_action.shape[1] - executed_steps)
    available_dst = max(0, horizon - condition_start)
    overlap_steps = config.overlap_steps
    if overlap_steps is None:
        overlap_steps = min(available_prev, available_dst)
    else:
        overlap_steps = min(int(overlap_steps), available_prev, available_dst)

    if overlap_steps <= 0:
        return None, None

    target = torch.zeros(
        (batch, horizon, prev_action_dim),
        device=sample.device,
        dtype=sample.dtype,
    )
    src = prev_action[:, executed_steps:executed_steps + overlap_steps]
    target[:, condition_start:condition_start + overlap_steps] = src
    weights = _make_overlap_weights(
        horizon=horizon,
        overlap_steps=overlap_steps,
        hard_steps=int(config.hard_steps),
        condition_start=condition_start,
        mode=config.weight_mode,
        device=sample.device,
        dtype=sample.dtype,
    )
    return target, weights


def _call_diffusion_model(policy, trajectory, timestep, local_cond, global_cond, cond):
    if cond is not None or hasattr(policy, "obs_as_cond"):
        return policy.model(trajectory, timestep, cond)
    return policy.model(trajectory, timestep, local_cond=local_cond, global_cond=global_cond)


def _scheduler_alpha_prod_t(scheduler, timestep, device, dtype):
    if hasattr(scheduler, "alphas_cumprod"):
        alpha = scheduler.alphas_cumprod[timestep]
        return alpha.to(device=device, dtype=dtype)
    return torch.ones((), device=device, dtype=dtype)


class RealtimeChunkingPiGDM:
    """Attach PiGDM realtime chunk guidance to an already-loaded policy."""

    def __init__(self, policy, config: PiGDMRealtimeChunkingConfig):
        self.policy = policy
        self.config = config
        self.prev_action_for_guidance = None
        self.prev_unnormalized_action = None
        self.original_conditional_sample = None

    @property
    def action_pose_repr(self):
        return getattr(self.policy, "action_pose_repr", "abs")

    def install(self):
        if self.original_conditional_sample is None:
            self.original_conditional_sample = self.policy.conditional_sample

        def conditional_sample(
                policy_self,
                condition_data,
                condition_mask,
                local_cond=None,
                global_cond=None,
                cond=None,
                generator=None,
                **kwargs):
            return self._conditional_sample(
                policy_self,
                condition_data,
                condition_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                cond=cond,
                generator=generator,
                **kwargs,
            )

        self.policy.conditional_sample = MethodType(conditional_sample, self.policy)
        return self

    def reset(self):
        self.prev_action_for_guidance = None
        self.prev_unnormalized_action = None

    def _prepare_prev_action(self, abs_obs):
        if self.prev_unnormalized_action is None:
            self.prev_action_for_guidance = None
            return

        action_np = self.prev_unnormalized_action
        if self.action_pose_repr == "relative":
            action_np = get_relative_action_from_abs(action=action_np, env_obs=abs_obs)

        action = torch.as_tensor(
            action_np,
            device=self.policy.device,
            dtype=self.policy.dtype,
        )
        self.prev_action_for_guidance = self.policy.normalizer["action"].normalize(
            action
        ).unsqueeze(0)

    def _store_action_pred(self, action_pred, abs_obs):
        action_np = action_pred[0].detach().to("cpu").numpy()
        if self.action_pose_repr == "relative":
            self.prev_unnormalized_action = get_abs_action_from_relative(
                action=action_np,
                env_obs=abs_obs,
            )
        else:
            self.prev_unnormalized_action = action_np

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], abs_obs):
        self._prepare_prev_action(abs_obs)
        result = self.policy.predict_action(obs_dict)
        if "action_pred" in result:
            self._store_action_pred(result["action_pred"], abs_obs)
        return result

    def _conditional_sample(
            self,
            self_policy,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            cond=None,
            generator=None,
            **kwargs):
        wrapper = self
        scheduler = self_policy.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self_policy.num_inference_steps)

        for timestep in scheduler.timesteps:
            with torch.enable_grad():
                trajectory[condition_mask] = condition_data[condition_mask]
                trajectory.requires_grad_(True)

                model_output = _call_diffusion_model(
                    self_policy,
                    trajectory,
                    timestep,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    cond=cond,
                )
                step_out = scheduler.step(
                    model_output,
                    timestep,
                    trajectory,
                    generator=generator,
                    **kwargs,
                )
                next_trajectory = step_out.prev_sample

                target, weights = _build_overlap_target(
                    wrapper.prev_action_for_guidance,
                    trajectory,
                    wrapper.config,
                )
                if target is not None and wrapper.config.guidance_scale != 0:
                    pred_original = step_out.pred_original_sample
                    action_dim = target.shape[-1]
                    error = torch.zeros_like(pred_original)
                    error[..., :action_dim] = (
                        target - pred_original[..., :action_dim]
                    ) * weights.view(1, -1, 1)
                    guidance = torch.autograd.grad(
                        outputs=pred_original,
                        inputs=trajectory,
                        grad_outputs=error,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    alpha = _scheduler_alpha_prod_t(
                        scheduler,
                        timestep,
                        device=trajectory.device,
                        dtype=trajectory.dtype,
                    )
                    next_trajectory = (
                        next_trajectory
                        + torch.sqrt(alpha) * guidance * wrapper.config.guidance_scale
                    )

                trajectory = next_trajectory.detach()

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory


def make_realtime_chunking_pigdm(
        policy,
        executed_steps: int,
        overlap_steps: Optional[int] = None,
        hard_steps: int = 3,
        guidance_scale: float = 5.0,
        weight_mode: str = "exp_ramp",
        condition_start: int = 0):
    config = PiGDMRealtimeChunkingConfig(
        executed_steps=int(executed_steps),
        overlap_steps=overlap_steps,
        hard_steps=int(hard_steps),
        guidance_scale=float(guidance_scale),
        weight_mode=weight_mode,
        condition_start=int(condition_start),
    )
    return RealtimeChunkingPiGDM(policy, config).install()
