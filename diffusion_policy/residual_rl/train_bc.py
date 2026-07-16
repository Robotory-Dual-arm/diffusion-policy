"""Train the feed-forward actual-to-virtual residual actor with BC."""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from diffusion_policy.residual_rl.bc_dataset import PairedResidualBCDataset
from diffusion_policy.residual_rl.checkpoint import (
    ResidualModelConfig,
    build_actor,
    save_bc_checkpoint,
)
from diffusion_policy.residual_rl.data_identity import (
    action_dataset_sha256,
    file_sha256,
)
from diffusion_policy.residual_rl.stats import (
    episode_split_indices,
    fit_residual_normalizer,
    statistics_to_jsonable,
    symmetric_residual_bounds,
)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_obs(
    observation: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in observation.items()
    }


def _normalized_bc_loss(actor, batch, device: torch.device) -> torch.Tensor:
    observation = _move_obs(batch["obs"], device)
    base_action = batch["base_action"].to(device=device, dtype=torch.float32)
    target = batch["action"].to(device=device, dtype=torch.float32)
    prediction = actor(observation, base_action)
    normalizer = actor.state_encoder.normalizer
    normalized_prediction = normalizer.normalize_field("residual", prediction)
    normalized_target = normalizer.normalize_field("residual", target)
    return F.mse_loss(normalized_prediction, normalized_target)


def validate_base_prediction_cache(
    cache_path: str | Path,
    *,
    actual_dataset: str | Path,
    target_shift: int,
    slow_action_start_index: int,
    fast_steps_per_slow: int,
    base_inference_steps: int,
) -> dict[str, object]:
    """Fail early when a cache was produced with a different online schedule."""

    cache_path = Path(cache_path).expanduser().resolve()
    actual_dataset = Path(actual_dataset).expanduser().resolve()
    with h5py.File(cache_path, "r") as cache:
        required = {
            "alignment": "chunked_like_online_slow_fast_runtime",
            "target_offset": int(target_shift),
            "slow_action_start_index": int(slow_action_start_index),
            "fast_steps_per_slow": int(fast_steps_per_slow),
            "base_num_inference_steps": int(base_inference_steps),
        }
        missing = [key for key in required if key not in cache.attrs]
        if missing:
            raise ValueError(
                f"Base prediction cache is missing schedule metadata: {missing}"
            )
        for key, expected in required.items():
            observed = cache.attrs[key]
            if isinstance(observed, bytes):
                observed = observed.decode()
            if observed != expected:
                raise ValueError(
                    f"Base prediction cache {key}={observed!r}, expected {expected!r}"
                )
        if "source_actual_dataset" not in cache.attrs:
            raise ValueError(
                "Base prediction cache has no source_actual_dataset metadata"
            )
        source = cache.attrs["source_actual_dataset"]
        if isinstance(source, bytes):
            source = source.decode()
        source_path = Path(str(source)).expanduser()
        if source_path.resolve() != actual_dataset:
            warnings.warn(
                "Base cache source path differs from the current dataset path; "
                "checking portable action-content fingerprint instead. "
                f"cache={source_path}, current={actual_dataset}",
                stacklevel=2,
            )
        if "source_actual_action_sha256" not in cache.attrs:
            raise ValueError(
                "Base prediction cache has no portable action fingerprint; "
                "regenerate it with cache_base_predictions"
            )
        cached_fingerprint = cache.attrs["source_actual_action_sha256"]
        if isinstance(cached_fingerprint, bytes):
            cached_fingerprint = cached_fingerprint.decode()
        cached_fingerprint = str(cached_fingerprint)
        current_fingerprint = action_dataset_sha256(actual_dataset)
        if cached_fingerprint != current_fingerprint:
            raise ValueError(
                "Base prediction cache was generated from different actual actions"
            )
        if "source_actual_file_sha256" not in cache.attrs:
            raise ValueError(
                "Base prediction cache has no full source-file fingerprint; "
                "regenerate it with cache_base_predictions"
            )
        cached_file_fingerprint = cache.attrs["source_actual_file_sha256"]
        if isinstance(cached_file_fingerprint, bytes):
            cached_file_fingerprint = cached_file_fingerprint.decode()
        cached_file_fingerprint = str(cached_file_fingerprint)
        if cached_file_fingerprint != file_sha256(actual_dataset):
            raise ValueError(
                "Base prediction cache source HDF5 content differs from the "
                "current actual dataset"
            )
        for key in ("base_checkpoint", "base_checkpoint_sha256", "base_state_key"):
            if key not in cache.attrs:
                raise ValueError(
                    f"Base prediction cache has no {key} provenance metadata"
                )
        return {
            "cache_path": str(cache_path),
            "base_checkpoint": str(cache.attrs.get("base_checkpoint", "")),
            "base_checkpoint_sha256": str(
                cache.attrs["base_checkpoint_sha256"]
            ),
            "base_state_key": str(cache.attrs["base_state_key"]),
            "source_actual_action_sha256": cached_fingerprint,
            "source_actual_file_sha256": cached_file_fingerprint,
            **required,
        }


@torch.no_grad()
def _validate(actor, loader, device: torch.device) -> float:
    actor.eval()
    losses = []
    weights = []
    for batch in loader:
        loss = _normalized_bc_loss(actor, batch, device)
        batch_size = int(batch["action"].shape[0])
        losses.append(float(loss.cpu()) * batch_size)
        weights.append(batch_size)
    return float(sum(losses) / max(sum(weights), 1))


def train(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    device = _device(args.device)
    if device.type == "cpu":
        # PyTorch 2.9 oneDNN can produce non-finite Conv1d/Conv2d gradients for
        # this small encoder even when the forward itself is evaluated with the
        # per-module mkldnn context disabled.  Backward happens after that
        # context exits, so keep native CPU kernels selected for the complete
        # optimization run.  CUDA training is unaffected.
        torch.backends.mkldnn.enabled = False
    output_dir = Path(args.output).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint directory is not empty: {checkpoint_dir}. "
            "Pass --overwrite to replace latest/best checkpoints."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cache_metadata = None
    if args.base_predictions is not None:
        cache_metadata = validate_base_prediction_cache(
            args.base_predictions,
            actual_dataset=args.actual_dataset,
            target_shift=args.target_shift,
            slow_action_start_index=args.slow_action_start_index,
            fast_steps_per_slow=args.fast_steps_per_slow,
            base_inference_steps=args.base_inference_steps,
        )
    base_source = args.base_predictions or "actual"
    dataset = PairedResidualBCDataset(
        args.actual_dataset,
        args.virtual_dataset,
        base_action_source=base_source,
        base_action_key=args.base_action_key,
        action_target_shift=args.target_shift,
        base_action_target_shift=args.target_shift,
    )
    train_indices, val_indices = episode_split_indices(
        dataset.episode_lengths,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if len(train_indices) == 0:
        raise RuntimeError("Episode split produced no training samples")

    # Scale/bounds are dataset metadata, so fit them once before optimization.
    explicit_bounds = args.residual_min is not None or args.residual_max is not None
    if (args.residual_min is None) != (args.residual_max is None):
        raise ValueError("--residual-min and --residual-max must be supplied together")
    normalizer, statistics = fit_residual_normalizer(
        dataset,
        indices=None,
        batch_size=args.stats_batch_size,
        num_workers=args.num_workers,
        residual_min=args.residual_min,
        residual_max=args.residual_max,
        residual_bound_scale=None if explicit_bounds else args.residual_bound_scale,
    )
    # These bounds define the actor's physical tanh parameterization. They are
    # either an explicit controller interval or the user-approved symmetric
    # 1.3x (configurable) envelope around the paired demonstration residuals.
    if explicit_bounds:
        residual_min = np.asarray(args.residual_min, dtype=np.float32)
        residual_max = np.asarray(args.residual_max, dtype=np.float32)
        residual_bound_source = "explicit_cli"
    else:
        residual_min, residual_max = symmetric_residual_bounds(
            statistics["residual"],
            scale=args.residual_bound_scale,
        )
        residual_bound_source = "symmetric_demonstration_max_abs"
    print("Observed residual min:", statistics["residual"]["min"].tolist())
    print("Observed residual max:", statistics["residual"]["max"].tolist())
    print("Actor residual min:", residual_min.tolist())
    print("Actor residual max:", residual_max.tolist())

    model_config = ResidualModelConfig(
        image_feature_dim=args.image_feature_dim,
        wrench_feature_dim=args.wrench_feature_dim,
        actor_hidden_dims=tuple(args.actor_hidden_dims),
        critic_hidden_dims=tuple(args.critic_hidden_dims),
        residual_min=tuple(float(value) for value in residual_min),
        residual_max=tuple(float(value) for value in residual_max),
    )
    actor = build_actor(
        model_config,
        normalizer,
        freeze_image_encoder=False,
        freeze_wrench_encoder=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = None
    if len(val_indices) > 0:
        val_loader = DataLoader(
            Subset(dataset, val_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    stats_json = statistics_to_jsonable(statistics)
    metadata = {
        "actual_dataset": str(dataset.actual_path),
        "virtual_dataset": str(dataset.virtual_path),
        "base_action_source": str(base_source),
        "base_action_key": args.base_action_key,
        "target_shift": args.target_shift,
        "num_episodes": len(dataset.demo_names),
        "num_samples": len(dataset),
        "num_train_samples": len(train_indices),
        "num_val_samples": len(val_indices),
        "physical_residual_min": residual_min.tolist(),
        "physical_residual_max": residual_max.tolist(),
        "residual_bound_source": residual_bound_source,
        "residual_bound_scale": None if explicit_bounds else args.residual_bound_scale,
        "base_prediction_cache": cache_metadata,
    }
    with (output_dir / "run_config.json").open("w") as stream:
        json.dump(
            {
                "args": vars(args),
                "model_config": {
                    **model_config.__dict__,
                    "actor_hidden_dims": list(model_config.actor_hidden_dims),
                    "critic_hidden_dims": list(model_config.critic_hidden_dims),
                    "residual_min": list(model_config.residual_min),
                    "residual_max": list(model_config.residual_max),
                },
                "metadata": metadata,
                "statistics": stats_json,
            },
            stream,
            indent=2,
            default=str,
        )

    best_loss = float("inf")
    global_step = 0
    log_path = output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        actor.train()
        running_loss = 0.0
        running_count = 0
        progress = tqdm(train_loader, desc=f"BC epoch {epoch}/{args.epochs}")
        for batch in progress:
            loss = _normalized_bc_loss(actor, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite BC actor loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in actor.parameters()
            ):
                raise FloatingPointError("Non-finite BC actor gradient")
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            optimizer.step()
            batch_size = int(batch["action"].shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size
            running_count += batch_size
            global_step += 1
            progress.set_postfix(loss=running_loss / running_count)

        train_loss = running_loss / max(running_count, 1)
        val_loss = (
            _validate(actor, val_loader, device)
            if val_loader is not None
            else train_loss
        )
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise FloatingPointError(
                f"Non-finite BC epoch metric: train={train_loss}, val={val_loss}"
            )
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        with log_path.open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(metrics)

        save_bc_checkpoint(
            checkpoint_dir / "latest.pt",
            actor=actor,
            model_config=model_config,
            optimizer=optimizer,
            step=global_step,
            epoch=epoch,
            statistics=stats_json,
            metadata=metadata,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            save_bc_checkpoint(
                checkpoint_dir / "best.pt",
                actor=actor,
                model_config=model_config,
                optimizer=optimizer,
                step=global_step,
                epoch=epoch,
                statistics=stats_json,
                metadata={**metadata, "best_val_loss": best_loss},
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-dataset", required=True)
    parser.add_argument("--virtual-dataset", required=True)
    parser.add_argument("--base-predictions", default=None)
    parser.add_argument("--base-action-key", default="actions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-shift", type=int, default=1)
    parser.add_argument("--slow-action-start-index", type=int, default=1)
    parser.add_argument("--fast-steps-per-slow", type=int, default=6)
    parser.add_argument(
        "--base-inference-steps",
        type=int,
        default=16,
        help="Must match cache generation and real-robot base inference",
    )
    parser.add_argument(
        "--residual-min",
        type=float,
        nargs=6,
        default=None,
        help="Optional explicit delta6 lower limits; must be paired with --residual-max",
    )
    parser.add_argument(
        "--residual-max",
        type=float,
        nargs=6,
        default=None,
        help="Optional explicit delta6 upper limits; must be paired with --residual-min",
    )
    parser.add_argument(
        "--residual-bound-scale",
        type=float,
        default=1.3,
        help="Without explicit bounds, use +/- scale * per-axis demo max(abs(delta))",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stats-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--image-feature-dim", type=int, default=128)
    parser.add_argument("--wrench-feature-dim", type=int, default=64)
    parser.add_argument("--actor-hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--critic-hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
