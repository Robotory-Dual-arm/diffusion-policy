"""TD3 with a frozen behavior-cloning prior for residual robot control."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from diffusion_policy.residual_rl.models import ResidualActor, TwinQCritic


ScalarOrVector = Union[float, Sequence[float]]


@dataclass
class TD3Config:
    gamma: float = 0.99
    n_step: int = 3
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    policy_delay: int = 2
    target_policy_noise: ScalarOrVector = 0.0
    target_noise_clip: ScalarOrVector = 0.0
    lambda_bc: float = 1.0
    max_grad_norm: Optional[float] = None

    def validate(self, residual_dim: int) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if int(self.n_step) <= 0:
            raise ValueError("n_step must be positive")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("actor_lr and critic_lr must be positive")
        if int(self.policy_delay) <= 0:
            raise ValueError("policy_delay must be positive")
        if self.lambda_bc < 0.0:
            raise ValueError("lambda_bc must be non-negative")
        for name in ("target_policy_noise", "target_noise_clip"):
            value = torch.as_tensor(getattr(self, name), dtype=torch.float32).flatten()
            if value.numel() not in (1, residual_dim):
                raise ValueError(f"{name} must be scalar or length {residual_dim}")
            if torch.any(value < 0):
                raise ValueError(f"{name} must be non-negative")


def _action_vector(
    value: ScalarOrVector,
    action: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=action.device, dtype=action.dtype).flatten()
    if tensor.numel() == 1:
        return tensor
    if tensor.numel() != action.shape[-1]:
        raise ValueError(f"{name} must be scalar or length {action.shape[-1]}")
    return tensor.reshape((1,) * (action.ndim - 1) + (-1,))


def _set_frozen(module: torch.nn.Module) -> None:
    module.eval()
    module.requires_grad_(False)


def _require_finite_gradients(module: torch.nn.Module, name: str) -> None:
    if any(
        parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    ):
        raise FloatingPointError(f"Non-finite TD3 {name} gradient")


@torch.no_grad()
def _polyak_update(
    source: torch.nn.Module,
    target: torch.nn.Module,
    tau: float,
) -> None:
    source_parameters = dict(source.named_parameters())
    for name, target_parameter in target.named_parameters():
        source_parameter = source_parameters[name]
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
    source_buffers = dict(source.named_buffers())
    for name, target_buffer in target.named_buffers():
        target_buffer.copy_(source_buffers[name])


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


class ResidualTD3:
    """Train only the fast residual actor; the base policy lives outside.

    ``actor`` and ``bc_prior`` share the same interface but must be independent
    module instances. If no prior is supplied, the initial actor is deep-copied
    before the first update. The prior and all target modules are permanently
    frozen.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        actor: ResidualActor,
        critics: TwinQCritic,
        config: Optional[TD3Config] = None,
        bc_prior: Optional[ResidualActor] = None,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.config = config or TD3Config()
        self.config.validate(actor.residual_dim)
        if actor.residual_dim != critics.residual_dim:
            raise ValueError("actor and critic residual dimensions differ")
        if actor.base_action_dim != critics.base_action_dim:
            raise ValueError("actor and critic base-action dimensions differ")

        self.actor = actor.to(self.device)
        self.critics = critics.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        self.bc_prior = copy.deepcopy(self.actor) if bc_prior is None else bc_prior
        self.bc_prior = self.bc_prior.to(self.device)
        if self.bc_prior.base_action_dim != self.actor.base_action_dim:
            raise ValueError("BC prior base-action dimension differs from actor")
        if self.bc_prior.residual_dim != self.actor.residual_dim:
            raise ValueError("BC prior residual dimension differs from actor")

        _set_frozen(self.target_actor)
        _set_frozen(self.target_critics)
        _set_frozen(self.bc_prior)
        self.actor_optimizer = torch.optim.Adam(
            [parameter for parameter in self.actor.parameters() if parameter.requires_grad],
            lr=self.config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in self.critics.parameters()
                if parameter.requires_grad
            ],
            lr=self.config.critic_lr,
        )
        self.total_updates = 0

    def _move_observation(
        self,
        observation: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(device=self.device, non_blocking=True)
            for key, value in observation.items()
        }

    def _prepare_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "obs",
            "next_obs",
            "base_action",
            "next_base_action",
            "action",
            "reward",
            "done",
        }
        missing = required - set(batch)
        if missing:
            raise KeyError(f"Missing TD3 batch keys: {sorted(missing)}")
        prepared = {
            "obs": self._move_observation(batch["obs"]),
            "next_obs": self._move_observation(batch["next_obs"]),
        }
        for key in ("base_action", "next_base_action", "action", "reward", "done"):
            value = batch[key].to(device=self.device, dtype=torch.float32, non_blocking=True)
            if key in ("reward", "done") and value.ndim == 1:
                value = value[:, None]
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"TD3 batch {key} contains NaN or Inf")
            prepared[key] = value
        if torch.any((prepared["done"] < 0.0) | (prepared["done"] > 1.0)):
            raise ValueError("done must be in [0, 1]")
        return prepared

    def _target_action(
        self,
        next_obs: Mapping[str, torch.Tensor],
        next_base_action: torch.Tensor,
    ) -> torch.Tensor:
        action = self.target_actor(next_obs, next_base_action)
        noise_scale = _action_vector(
            self.config.target_policy_noise,
            action,
            "target_policy_noise",
        )
        noise_limit = _action_vector(
            self.config.target_noise_clip,
            action,
            "target_noise_clip",
        )
        noise = torch.randn_like(action) * noise_scale
        noise = torch.maximum(torch.minimum(noise, noise_limit), -noise_limit)
        return self.target_actor.clip_action(action + noise)

    def update(
        self,
        batch: Mapping[str, Any],
        *,
        update_actor: bool = True,
    ) -> dict[str, float | bool]:
        batch = self._prepare_batch(batch)
        self.actor.train()
        self.critics.train()
        self.total_updates += 1

        with torch.no_grad():
            next_action = self._target_action(
                batch["next_obs"],
                batch["next_base_action"],
            )
            target_q1, target_q2 = self.target_critics(
                batch["next_obs"],
                batch["next_base_action"],
                next_action,
            )
            target_q = batch["reward"] + (
                self.config.gamma ** int(self.config.n_step)
                * (1.0 - batch["done"])
                * torch.minimum(target_q1, target_q2)
            )

        current_q1, current_q2 = self.critics(
            batch["obs"],
            batch["base_action"],
            batch["action"],
        )
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Non-finite TD3 critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        _require_finite_gradients(self.critics, "critic")
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critics.parameters(), self.config.max_grad_norm
            )
        self.critic_optimizer.step()

        metrics: dict[str, float | bool] = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "q1_mean": float(current_q1.detach().mean().cpu()),
            "q2_mean": float(current_q2.detach().mean().cpu()),
            "target_q_mean": float(target_q.detach().mean().cpu()),
            "actor_updated": False,
        }

        delayed_update = self.total_updates % int(self.config.policy_delay) == 0
        if delayed_update and update_actor:
            critic_requires_grad = [
                parameter.requires_grad for parameter in self.critics.parameters()
            ]
            try:
                self.critics.requires_grad_(False)
                actor_action = self.actor(batch["obs"], batch["base_action"])
                actor_rl_loss = -self.critics.q1(
                    batch["obs"], batch["base_action"], actor_action
                ).mean()
                with torch.no_grad():
                    prior_action = self.bc_prior(
                        batch["obs"], batch["base_action"]
                    )
                # Actor outputs are physical delta6 values (metres/radians).
                # Apply the same per-axis [-1, 1] residual transform used by
                # fast BC so lambda_bc has a stable meaning and rotation axes
                # do not dominate merely because their physical ranges differ.
                residual_normalizer = self.actor.state_encoder.normalizer
                normalized_actor_action = residual_normalizer.normalize_field(
                    "residual",
                    actor_action,
                )
                normalized_prior_action = residual_normalizer.normalize_field(
                    "residual",
                    prior_action,
                )
                bc_loss = F.mse_loss(
                    normalized_actor_action,
                    normalized_prior_action,
                )
                physical_bc_loss = F.mse_loss(actor_action, prior_action)
                actor_loss = actor_rl_loss + self.config.lambda_bc * bc_loss
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Non-finite TD3 actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                _require_finite_gradients(self.actor, "actor")
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.config.max_grad_norm
                    )
                self.actor_optimizer.step()
            finally:
                for parameter, requires_grad in zip(
                    self.critics.parameters(), critic_requires_grad
                ):
                    parameter.requires_grad_(requires_grad)

            _polyak_update(self.actor, self.target_actor, self.config.tau)
            metrics.update({
                "actor_updated": True,
                "actor_loss": float(actor_loss.detach().cpu()),
                "actor_rl_loss": float(actor_rl_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "physical_bc_loss": float(physical_bc_loss.detach().cpu()),
            })
        if delayed_update:
            # Keep the critic target tracking during the critic-only warmup.
            _polyak_update(self.critics, self.target_critics, self.config.tau)
        return metrics

    @torch.no_grad()
    def act(
        self,
        observation: Mapping[str, torch.Tensor],
        base_action: torch.Tensor,
        exploration_std: ScalarOrVector = 0.0,
    ) -> torch.Tensor:
        """Return a clipped residual; exploration is off when std is zero."""
        self.actor.eval()
        observation = self._move_observation(observation)
        base_action = base_action.to(device=self.device, dtype=torch.float32)
        action = self.actor(observation, base_action)
        std = _action_vector(exploration_std, action, "exploration_std")
        if torch.any(std < 0):
            raise ValueError("exploration_std must be non-negative")
        if torch.any(std > 0):
            action = action + torch.randn_like(action) * std
        action = self.actor.clip_action(action)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Actor produced NaN or Inf")
        return action

    def checkpoint_state(self, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "total_updates": self.total_updates,
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "bc_prior": self.bc_prior.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "extra": dict(extra or {}),
        }

    def load_checkpoint_state(
        self,
        state: Mapping[str, Any],
        strict: bool = True,
        load_optimizers: bool = True,
    ) -> Mapping[str, Any]:
        version = int(state.get("version", -1))
        if version != self.CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported TD3 checkpoint version {version}")
        self.actor.load_state_dict(state["actor"], strict=strict)
        self.target_actor.load_state_dict(state["target_actor"], strict=strict)
        self.critics.load_state_dict(state["critics"], strict=strict)
        self.target_critics.load_state_dict(state["target_critics"], strict=strict)
        self.bc_prior.load_state_dict(state["bc_prior"], strict=strict)
        self.total_updates = int(state["total_updates"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            _optimizer_to(self.actor_optimizer, self.device)
            _optimizer_to(self.critic_optimizer, self.device)
        _set_frozen(self.target_actor)
        _set_frozen(self.target_critics)
        _set_frozen(self.bc_prior)
        return state.get("extra", {})

    def save_checkpoint(
        self,
        path: str | Path,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(extra=extra), path)
        return str(path.absolute())

    def load_checkpoint(
        self,
        path: str | Path,
        strict: bool = True,
        load_optimizers: bool = True,
        map_location: Optional[torch.device | str] = None,
    ) -> Mapping[str, Any]:
        state = torch.load(
            Path(path),
            map_location=map_location or self.device,
        )
        return self.load_checkpoint_state(
            state,
            strict=strict,
            load_optimizers=load_optimizers,
        )
