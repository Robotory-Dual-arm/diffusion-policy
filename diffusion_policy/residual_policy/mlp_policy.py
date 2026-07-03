from diffusion_policy.residual_policy.context_step_policy import FastResidualContextStepPolicy


class FastResidualMLPPolicy(FastResidualContextStepPolicy):
    """Final residual MLP.

    Uses a fixed slow-context image and per-step low-dim, wrench, slow base
    action, and step encoding.
    """

    pass
