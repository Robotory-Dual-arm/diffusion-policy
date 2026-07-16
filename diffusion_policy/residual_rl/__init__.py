"""Minimal feed-forward residual TD3 components.

The package is intentionally independent from the real-robot runtime.  Runtime
code only needs to produce the canonical transition dictionaries documented by
``ResidualReplayBuffer``.
"""

from diffusion_policy.residual_rl.models import (
    ResidualActor,
    StructuredObservationEncoder,
    TwinQCritic,
)
from diffusion_policy.residual_rl.normalizer import StructuredAffineNormalizer
from diffusion_policy.residual_rl.replay_buffer import ResidualReplayBuffer
from diffusion_policy.residual_rl.td3 import ResidualTD3, TD3Config

__all__ = [
    "ResidualActor",
    "ResidualReplayBuffer",
    "ResidualTD3",
    "StructuredAffineNormalizer",
    "StructuredObservationEncoder",
    "TD3Config",
    "TwinQCritic",
]
