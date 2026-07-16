"""Evaluate one frozen residual actor with exploration strictly disabled."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusion_policy.residual_rl.collect import (
    _verify_actor_and_runtime_bounds,
    add_common_arguments,
    build_runtime_config,
    build_safety,
    load_policy_runners,
    make_robot_safe_stop,
    operator_stop_requested,
    real_environment,
    validate_common_args,
    warm_up_inference,
)
from diffusion_policy.residual_rl.runtime import (
    EpisodeSidecarWriter,
    SlowFastResidualRuntime,
    prompt_success_label,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser


def run_evaluation(args: argparse.Namespace) -> list:
    validate_common_args(args)
    base_runner, residual_runner = load_policy_runners(args)
    safety = build_safety(args, residual_runner)
    _verify_actor_and_runtime_bounds(residual_runner, safety.limits)
    runtime_config = build_runtime_config(args, exploration_std=0.0)
    sidecar_root = (
        Path(args.sidecar_output).expanduser().resolve()
        if args.sidecar_output is not None
        else Path(args.output).expanduser().resolve() / "residual_rl_evaluation"
    )
    writer = EpisodeSidecarWriter(sidecar_root)

    print("Base checkpoint:", base_runner.checkpoint_id)
    print("Actor checkpoint:", residual_runner.checkpoint_id)
    print("Exploration: disabled")
    print("Evaluation sidecars:", sidecar_root)
    results = []
    with real_environment(
        args,
        base_runner=base_runner,
        residual_runner=residual_runner,
    ) as environment:
        warm_up_inference(
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
            safe_stop=make_robot_safe_stop(environment),
        )
        for episode_index in range(args.episodes):
            print(f"Starting evaluation episode {episode_index + 1}/{args.episodes}")
            result = runtime.run_episode(
                label_success=lambda: prompt_success_label(
                    f"Evaluation episode {episode_index} success? [1/0]: "
                ),
                sidecar_writer=writer,
                stop_requested=operator_stop_requested,
                exploration_std=0.0,
            )
            results.append(result)
            print(result)
            if result.safety_violation is not None:
                print(
                    "Evaluation aborted after safety stop; inspect and reset the "
                    "robot before restarting."
                )
                break

    successes = sum(result.success for result in results)
    rate = successes / len(results)
    print(f"Evaluation success: {successes}/{len(results)} ({rate:.3f})")
    return results


def main() -> None:
    run_evaluation(make_parser().parse_args())


if __name__ == "__main__":
    main()
