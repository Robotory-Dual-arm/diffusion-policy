"""Cache slow-policy chunks aligned to the actual/virtual BC dataset.

The cache deliberately mirrors the real slow/fast schedule.  With the default
settings, the base policy is called at observations ``0, 6, 12, ...`` and its
chunk entries ``1:7`` condition six successive fast samples.  Thus sample
``obs_t`` receives the same nominal target it would see online even though the
base policy is not replanned at every fast tick.

Residual supervision is *not* produced here.  The paired BC dataset always
computes that label from ``actual_target -> virtual_target``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusion_policy.residual_rl.base_policy import (
    load_frozen_base_policy,
    predict_base_action_chunk,
)
from diffusion_policy.residual_rl.data_identity import (
    action_dataset_sha256,
    file_sha256,
)


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[-1]), name
    except ValueError:
        return 0, name


def _history_indices(step: int, length: int) -> np.ndarray:
    start = step - length + 1
    return np.maximum(np.arange(start, step + 1), 0)


def _chunk_assignments(
    episode_length: int,
    *,
    target_offset: int,
    slow_action_start_index: int,
    fast_steps_per_slow: int,
) -> list[tuple[int, int, int]]:
    """Return ``(slow_anchor, dataset_target, slow_chunk_index)`` mappings."""

    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if target_offset < 0 or slow_action_start_index < 0:
        raise ValueError("target and slow-action offsets must be non-negative")
    if target_offset != slow_action_start_index:
        raise ValueError(
            "Exact online alignment requires target_offset == "
            "slow_action_start_index"
        )
    if fast_steps_per_slow <= 0:
        raise ValueError("fast_steps_per_slow must be positive")

    assignments: list[tuple[int, int, int]] = []
    usable_observations = max(episode_length - target_offset, 0)
    for anchor in range(0, usable_observations, fast_steps_per_slow):
        for fast_index in range(fast_steps_per_slow):
            observation_index = anchor + fast_index
            target_index = observation_index + target_offset
            if target_index >= episode_length:
                break
            assignments.append(
                (
                    anchor,
                    target_index,
                    slow_action_start_index + fast_index,
                )
            )
    return assignments


def _build_batch(
    demo: h5py.Group,
    steps: np.ndarray,
    *,
    obs_meta,
    n_obs_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs_group = demo["obs"]
    batch: dict[str, torch.Tensor] = {}
    for key, attr in obs_meta.items():
        obs_type = attr.get("type", "low_dim")
        source = obs_group[key]
        if obs_type == "wrench":
            value = np.stack([source[int(step)] for step in steps], axis=0)
            value = value[:, None]
        else:
            value = np.stack(
                [
                    np.stack(
                        [source[int(index)] for index in _history_indices(int(step), n_obs_steps)],
                        axis=0,
                    )
                    for step in steps
                ],
                axis=0,
            )
        if obs_type == "rgb":
            value = np.moveaxis(value, -1, -3).astype(np.float32) / 255.0
        else:
            value = value.astype(np.float32)
        batch[key] = torch.from_numpy(value).to(device=device)
    return batch


@torch.no_grad()
def cache_predictions(args: argparse.Namespace) -> None:
    source_path = Path(args.actual_dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite explicitly"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    loaded = load_frozen_base_policy(
        args.base_checkpoint,
        device=device,
        use_ema=not args.no_ema,
        inference_steps=args.inference_steps,
    )
    policy = loaded.policy
    n_obs_steps = int(policy.n_obs_steps)
    obs_meta = loaded.cfg.task.shape_meta.obs
    action_pose_repr = OmegaConf.select(
        loaded.cfg,
        "task.pose_repr.action_pose_repr",
        default="abs",
    )
    if action_pose_repr != "abs":
        raise RuntimeError(
            "Base prediction caching currently requires an absolute-pose base "
            f"checkpoint, got action_pose_repr={action_pose_repr!r}"
        )
    target_offset = int(args.target_offset)
    slow_action_start_index = int(args.slow_action_start_index)
    fast_steps_per_slow = int(args.fast_steps_per_slow)
    # Validate the schedule before opening the output file.
    _chunk_assignments(
        1,
        target_offset=target_offset,
        slow_action_start_index=slow_action_start_index,
        fast_steps_per_slow=fast_steps_per_slow,
    )

    with h5py.File(source_path, "r") as source, h5py.File(tmp_path, "w") as out:
        out.attrs["source_actual_dataset"] = str(source_path)
        out.attrs["source_actual_action_sha256"] = action_dataset_sha256(
            source_path
        )
        out.attrs["source_actual_file_sha256"] = file_sha256(source_path)
        out.attrs["base_checkpoint"] = str(loaded.checkpoint_path)
        out.attrs["base_checkpoint_sha256"] = file_sha256(
            loaded.checkpoint_path
        )
        out.attrs["base_state_key"] = loaded.state_key
        out.attrs["seed"] = int(args.seed)
        out.attrs["batch_size"] = int(args.batch_size)
        out.attrs["base_num_inference_steps"] = int(policy.num_inference_steps)
        out.attrs["base_n_obs_steps"] = int(n_obs_steps)
        out.attrs["base_n_action_steps"] = int(policy.n_action_steps)
        out.attrs["target_offset"] = target_offset
        out.attrs["slow_action_start_index"] = slow_action_start_index
        out.attrs["fast_steps_per_slow"] = fast_steps_per_slow
        out.attrs["alignment"] = "chunked_like_online_slow_fast_runtime"
        out_data = out.create_group("data")

        demo_names = sorted(source["data"].keys(), key=_demo_sort_key)
        for demo_name in tqdm(demo_names, desc="Caching frozen base predictions"):
            demo = source["data"][demo_name]
            length, action_dim = demo["actions"].shape
            predictions = np.full((length, action_dim), np.nan, dtype=np.float32)
            valid = np.zeros(length, dtype=np.bool_)
            source_anchor = np.full(length, -1, dtype=np.int64)
            assignments = _chunk_assignments(
                length,
                target_offset=target_offset,
                slow_action_start_index=slow_action_start_index,
                fast_steps_per_slow=fast_steps_per_slow,
            )
            anchor_to_assignments: dict[int, list[tuple[int, int]]] = {}
            for anchor, target_index, chunk_index in assignments:
                anchor_to_assignments.setdefault(anchor, []).append(
                    (target_index, chunk_index)
                )
            anchors = np.asarray(sorted(anchor_to_assignments), dtype=np.int64)
            for start in range(0, len(anchors), args.batch_size):
                batch_steps = anchors[start : start + args.batch_size]
                if len(batch_steps) == 0:
                    break
                obs = _build_batch(
                    demo,
                    batch_steps,
                    obs_meta=obs_meta,
                    n_obs_steps=n_obs_steps,
                    device=device,
                )
                chunk = predict_base_action_chunk(loaded, obs).cpu().numpy()
                if chunk.shape[-1] != action_dim:
                    raise RuntimeError(
                        f"{demo_name}: base action dim {chunk.shape[-1]} != {action_dim}"
                    )
                required_chunk_length = (
                    slow_action_start_index + fast_steps_per_slow
                )
                if required_chunk_length > chunk.shape[1]:
                    raise RuntimeError(
                        "Slow chunk is too short for the configured online schedule: "
                        f"need {required_chunk_length}, got {chunk.shape}"
                    )

                for row, anchor in enumerate(batch_steps.tolist()):
                    for target_index, chunk_index in anchor_to_assignments[anchor]:
                        predictions[target_index] = chunk[row, chunk_index]
                        valid[target_index] = True
                        source_anchor[target_index] = anchor

                # Prefix entries are never selected when the BC target shift is
                # ``target_offset``.  Fill them from the first prediction so the
                # sidecar remains a finite, conventional action array.
                if start == 0 and target_offset > 0:
                    prefix = min(target_offset, length, chunk.shape[1])
                    predictions[:prefix] = chunk[0, :prefix]
                    valid[:prefix] = True
                    source_anchor[:prefix] = 0

            if not valid.all() or not np.isfinite(predictions).all():
                bad = np.flatnonzero(~valid | ~np.isfinite(predictions).all(axis=-1))
                raise RuntimeError(f"{demo_name}: invalid predictions at {bad[:10]}")
            out_demo = out_data.create_group(demo_name)
            out_demo.create_dataset("actions", data=predictions, compression="gzip")
            out_demo.create_dataset("valid", data=valid)
            out_demo.create_dataset("source_anchor", data=source_anchor)
            for key, value in demo.attrs.items():
                out_demo.attrs[key] = value

    os.replace(tmp_path, output_path)
    print(f"Wrote aligned base prediction sidecar: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-dataset", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument("--slow-action-start-index", type=int, default=1)
    parser.add_argument("--fast-steps-per-slow", type=int, default=6)
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=16,
        help="Match the existing insert-plug real-robot evaluator (16 steps)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    cache_predictions(build_arg_parser().parse_args())
