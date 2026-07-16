"""Train residual TD3 offline while the real robot is stopped."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import h5py
from tqdm import trange
from torch.utils.data import DataLoader

from diffusion_policy.residual_rl.checkpoint import (
    build_critics_from_actor,
    load_bc_actor,
    load_checkpoint_payload,
    save_td3_checkpoint,
)
from diffusion_policy.residual_rl.episode_io import (
    discover_episode_files,
    load_into_replay,
    make_replay_buffer,
)
from diffusion_policy.residual_rl.offline_dataset import PairedResidualRLDataset
from diffusion_policy.residual_rl.td3 import ResidualTD3, TD3Config


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mixed_batch_sizes(batch_size: int, offline_ratio: float) -> tuple[int, int]:
    """Return ``(offline, online)`` counts with an exact total batch size."""

    batch_size = int(batch_size)
    offline_ratio = float(offline_ratio)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not np.isfinite(offline_ratio) or not 0.0 <= offline_ratio <= 1.0:
        raise ValueError("offline_ratio must be finite and in [0, 1]")
    offline = int(round(batch_size * offline_ratio))
    if 0.0 < offline_ratio < 1.0 and batch_size > 1:
        offline = min(max(offline, 1), batch_size - 1)
    online = batch_size - offline
    return offline, online


def _concat_batches(
    batches: list[dict],
    *,
    generator: torch.Generator,
) -> dict:
    """Concatenate offline/online batches and randomly interleave their rows."""

    if not batches:
        raise ValueError("At least one replay batch is required")
    merged = {
        "obs": {
            key: torch.cat([batch["obs"][key] for batch in batches], dim=0)
            for key in batches[0]["obs"]
        },
        "next_obs": {
            key: torch.cat([batch["next_obs"][key] for batch in batches], dim=0)
            for key in batches[0]["next_obs"]
        },
    }
    for key in ("base_action", "next_base_action", "action", "reward", "done"):
        values = []
        for batch in batches:
            value = batch[key]
            if key in ("reward", "done") and value.ndim == 1:
                value = value[:, None]
            values.append(value)
        merged[key] = torch.cat(values, dim=0)
    size = int(merged["action"].shape[0])
    permutation = torch.randperm(size, generator=generator)
    merged["obs"] = {
        key: value[permutation] for key, value in merged["obs"].items()
    }
    merged["next_obs"] = {
        key: value[permutation] for key, value in merged["next_obs"].items()
    }
    for key in ("base_action", "next_base_action", "action", "reward", "done"):
        merged[key] = merged[key][permutation]
    return merged


def _metadata_path(
    explicit: str | None,
    metadata: dict,
    key: str,
) -> Path:
    value = explicit if explicit is not None else metadata.get(key)
    if value is None:
        raise ValueError(
            f"Offline replay needs --offline-{key.replace('_', '-')} because "
            f"the BC checkpoint has no {key!r} metadata"
        )
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Offline replay {key} file does not exist: {path}. "
            "Pass the portable path explicitly on this computer."
        )
    return path


def _build_offline_dataset(args, bc_payload) -> PairedResidualRLDataset | None:
    if float(args.offline_ratio) <= 0.0:
        return None
    metadata = dict(bc_payload.get("metadata", {}))
    actual_path = _metadata_path(
        args.offline_actual_dataset,
        metadata,
        "actual_dataset",
    )
    virtual_path = _metadata_path(
        args.offline_virtual_dataset,
        metadata,
        "virtual_dataset",
    )
    base_value = (
        args.offline_base_predictions
        if args.offline_base_predictions is not None
        else metadata.get("base_action_source")
    )
    if base_value in (None, "actual", "virtual"):
        raise ValueError(
            "ResFiT-style offline replay requires the aligned frozen-base "
            "prediction cache. Pass --offline-base-predictions."
        )
    base_path = Path(str(base_value)).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Offline base prediction cache does not exist: {base_path}. "
            "Pass --offline-base-predictions for this computer."
        )
    target_shift = int(metadata.get("target_shift", 1))
    return PairedResidualRLDataset(
        actual_path,
        virtual_path,
        base_action_source=base_path,
        base_action_key=str(metadata.get("base_action_key", "actions")),
        action_target_shift=target_shift,
        base_action_target_shift=target_shift,
        n_step=int(args.n_step),
        gamma=float(args.gamma),
    )


def _next_offline_batch(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train(args: argparse.Namespace) -> None:
    # Keep programmatic callers/checkpoint tests from the initial residual_rl
    # version valid. The public CLI defaults below enable the new ResFiT-style
    # path; an older Namespace without these fields retains one-step online-only
    # behavior instead of silently inventing offline dataset paths.
    compatibility_defaults = {
        "n_step": 1,
        "offline_ratio": 0.0,
        "offline_actual_dataset": None,
        "offline_virtual_dataset": None,
        "offline_base_predictions": None,
        "offline_num_workers": 0,
    }
    for name, value in compatibility_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    _seed_everything(args.seed)
    device = _device(args.device)
    if device.type == "cpu":
        # Keep convolution backward on native kernels for the whole update;
        # PyTorch 2.9 oneDNN can otherwise emit NaN gradients for these small
        # residual encoders. CUDA behavior is unchanged.
        torch.backends.mkldnn.enabled = False
    if args.updates <= 0:
        raise ValueError("--updates must be > 0")
    if args.critic_warmup_updates < 0:
        raise ValueError("--critic-warmup-updates must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.n_step <= 0:
        raise ValueError("--n-step must be > 0")
    offline_batch_size, online_batch_size = _mixed_batch_sizes(
        args.batch_size,
        args.offline_ratio,
    )
    if args.offline_num_workers < 0:
        raise ValueError("--offline-num-workers must be >= 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    output_dir = Path(args.output).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    if (
        checkpoint_dir.exists()
        and any(checkpoint_dir.iterdir())
        and args.resume is None
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Checkpoint directory is not empty: {checkpoint_dir}. "
            "Use --resume or --overwrite."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    actor, model_config, bc_payload = load_bc_actor(
        args.bc_checkpoint,
        device=device,
        freeze_image_encoder=True,
        freeze_wrench_encoder=False,
    )
    # RL v1 never fine-tunes a visual encoder. Critic visual features start from
    # the same BC representation instead of a random frozen convolution stack.
    actor.state_encoder.set_encoder_freeze(image=True, wrench=False)
    prior = copy.deepcopy(actor)
    critics = build_critics_from_actor(
        actor,
        model_config,
        freeze_image_encoder=True,
        freeze_wrench_encoder=False,
    )

    residual_range = np.asarray(model_config.residual_max) - np.asarray(
        model_config.residual_min
    )
    target_noise = residual_range * float(args.target_noise_fraction)
    target_noise_clip = residual_range * float(args.target_noise_clip_fraction)
    td3_config = TD3Config(
        gamma=args.gamma,
        n_step=args.n_step,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_delay=args.policy_delay,
        target_policy_noise=target_noise.tolist(),
        target_noise_clip=target_noise_clip.tolist(),
        lambda_bc=args.lambda_bc,
        max_grad_norm=args.max_grad_norm,
    )
    td3 = ResidualTD3(
        actor=actor,
        critics=critics,
        config=td3_config,
        bc_prior=prior,
        device=device,
    )

    episode_files = discover_episode_files(args.episodes)
    available_transitions = 0
    for episode_path in episode_files:
        with h5py.File(episode_path, "r") as episode_file:
            available_transitions += int(episode_file["action"].shape[0])
    # The replay buffer eagerly allocates obs and next_obs, including two
    # 224x224 RGB images per transition. Never reserve unused capacity: a
    # nominal 300k buffer would otherwise request about 85 GiB.
    replay_capacity = min(
        max(int(args.replay_capacity), 1),
        max(available_transitions, 1),
    )
    replay = make_replay_buffer(capacity=replay_capacity, seed=args.seed)
    replay_summary = load_into_replay(
        replay,
        episode_files,
        n_step=args.n_step,
        gamma=args.gamma,
    )
    replay_summary["retained_transitions"] = len(replay)
    replay_summary["dropped_transitions"] = max(
        replay_summary["transitions"] - len(replay),
        0,
    )
    if replay_summary["dropped_transitions"] > 0:
        print(
            "Replay capacity retained the newest "
            f"{len(replay)} / {replay_summary['transitions']} transitions"
        )
    if online_batch_size > 0 and len(replay) < online_batch_size:
        raise RuntimeError(
            f"Online replay has {len(replay)} transitions, less than its mixed "
            f"batch share {online_batch_size}"
        )

    offline_dataset = _build_offline_dataset(args, bc_payload)
    offline_loader = None
    offline_iterator = None
    if offline_batch_size > 0:
        if offline_dataset is None:
            raise RuntimeError("offline_ratio requested no offline dataset")
        if len(offline_dataset) == 0:
            raise RuntimeError("Offline demonstration replay is empty")
        offline_generator = torch.Generator().manual_seed(args.seed + 1)
        offline_loader = DataLoader(
            offline_dataset,
            batch_size=offline_batch_size,
            shuffle=True,
            drop_last=len(offline_dataset) >= offline_batch_size,
            num_workers=args.offline_num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.offline_num_workers > 0,
            generator=offline_generator,
        )
        offline_iterator = iter(offline_loader)

    if args.resume is not None:
        resume_payload = load_checkpoint_payload(args.resume, map_location="cpu")
        if resume_payload["kind"] != "td3":
            raise ValueError("--resume must point to a TD3 checkpoint")
        if resume_payload["model_config"] != bc_payload["model_config"]:
            raise ValueError("Resume checkpoint model config differs from BC checkpoint")
        saved_td3_config = resume_payload["td3"].get("config")
        requested_td3_config = asdict(td3_config)
        if saved_td3_config != requested_td3_config:
            raise ValueError(
                "Resume TD3 hyperparameters differ from this run. Keep gamma, tau, "
                "learning rates, policy delay, target noise, lambda_bc, and gradient "
                "limit unchanged so restored optimizer state remains consistent. "
                f"saved={saved_td3_config}, requested={requested_td3_config}"
            )
        td3.load_checkpoint_state(
            resume_payload["td3"],
            strict=True,
            load_optimizers=True,
        )

    metadata = {
        "bc_checkpoint": str(Path(args.bc_checkpoint).expanduser().resolve()),
        "episode_files": [str(path) for path in episode_files],
        "replay_summary": replay_summary,
        "available_transitions": available_transitions,
        "requested_replay_capacity": int(args.replay_capacity),
        "effective_replay_capacity": replay_capacity,
        "critic_warmup_updates": args.critic_warmup_updates,
        "batch_size": args.batch_size,
        "offline_batch_size": offline_batch_size,
        "online_batch_size": online_batch_size,
        "offline_ratio": float(args.offline_ratio),
        "offline_samples": 0 if offline_dataset is None else len(offline_dataset),
        "offline_demo_episodes": (
            0 if offline_dataset is None else len(offline_dataset.demo_names)
        ),
        "n_step": int(args.n_step),
        "td3_config": td3_config.__dict__,
        "visual_encoder_frozen": True,
        "bc_metadata": copy.deepcopy(bc_payload.get("metadata", {})),
    }
    with (output_dir / "run_config.json").open("w") as stream:
        json.dump(
            {"args": vars(args), "metadata": metadata},
            stream,
            indent=2,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else str(value),
        )

    log_path = output_dir / "metrics.jsonl"
    start_update = td3.total_updates
    mix_generator = torch.Generator().manual_seed(args.seed + 2)
    progress = trange(args.updates, desc="Residual TD3 updates")
    for local_update in progress:
        replay_batches = []
        if offline_batch_size > 0:
            assert offline_loader is not None and offline_iterator is not None
            offline_batch, offline_iterator = _next_offline_batch(
                offline_loader,
                offline_iterator,
            )
            replay_batches.append(offline_batch)
        if online_batch_size > 0:
            replay_batches.append(replay.sample(online_batch_size, device="cpu"))
        batch = _concat_batches(replay_batches, generator=mix_generator)
        allow_actor = td3.total_updates >= args.critic_warmup_updates
        metrics = td3.update(batch, update_actor=allow_actor)
        metrics["update"] = td3.total_updates
        metrics["critic_warmup"] = not allow_actor
        metrics["offline_batch_size"] = offline_batch_size
        metrics["online_batch_size"] = online_batch_size
        progress.set_postfix(
            critic=f"{metrics['critic_loss']:.4g}",
            actor=bool(metrics["actor_updated"]),
        )
        if td3.total_updates % args.log_every == 0:
            with log_path.open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
        if (
            args.save_every > 0
            and td3.total_updates % args.save_every == 0
        ):
            save_td3_checkpoint(
                checkpoint_dir / f"update_{td3.total_updates:08d}.pt",
                td3=td3,
                model_config=model_config,
                metadata={**metadata, "start_update": start_update},
            )
            save_td3_checkpoint(
                checkpoint_dir / "latest.pt",
                td3=td3,
                model_config=model_config,
                metadata={**metadata, "start_update": start_update},
            )

    save_td3_checkpoint(
        checkpoint_dir / "latest.pt",
        td3=td3,
        model_config=model_config,
        metadata={**metadata, "start_update": start_update},
    )
    print("Replay:", replay_summary)
    print("Saved:", checkpoint_dir / "latest.pt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        help="Episode HDF5 files or directories containing episode_*.hdf5",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--critic-warmup-updates", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--offline-ratio",
        type=float,
        default=0.5,
        help="Fraction of every batch sampled from successful demonstrations",
    )
    parser.add_argument(
        "--offline-actual-dataset",
        default=None,
        help="Defaults to the actual dataset recorded in the BC checkpoint",
    )
    parser.add_argument(
        "--offline-virtual-dataset",
        default=None,
        help="Defaults to the virtual dataset recorded in the BC checkpoint",
    )
    parser.add_argument(
        "--offline-base-predictions",
        default=None,
        help="Defaults to the aligned base cache recorded in the BC checkpoint",
    )
    parser.add_argument("--offline-num-workers", type=int, default=2)
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=20_000,
        help=(
            "Maximum in-memory transitions (20k is about 5.8 GiB for canonical "
            "obs+next_obs); allocation is additionally capped to loaded data"
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-noise-fraction", type=float, default=0.1)
    parser.add_argument("--target-noise-clip-fraction", type=float, default=0.2)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help=(
            "Keep a numbered update checkpoint every N global updates. "
            "The default 0 writes only checkpoints/latest.pt."
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
