if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import json
import pathlib
from typing import Dict, Optional

import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.pose_util import mat_to_pose10d, pose10d_to_mat, pose_to_mat
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_path(path):
    path = pathlib.Path(path).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    return path.resolve()


def default_output_dir(checkpoint: pathlib.Path) -> pathlib.Path:
    stem = checkpoint.stem.replace("=", "_").replace("-", "_")
    return checkpoint.parent.parent / f"action_force_viz_{stem}"


def parse_indices(indices: Optional[str]) -> Optional[np.ndarray]:
    if indices is None or indices.strip() == "":
        return None
    return np.array([int(x.strip()) for x in indices.split(",") if x.strip()], dtype=np.int64)


def choose_indices(dataset_len: int, n_samples: int, seed: int, indices: Optional[str]) -> np.ndarray:
    parsed = parse_indices(indices)
    if parsed is not None:
        if np.any(parsed < 0) or np.any(parsed >= dataset_len):
            raise ValueError(f"indices must be in [0, {dataset_len - 1}]")
        return parsed
    rng = np.random.default_rng(seed)
    count = min(int(n_samples), int(dataset_len))
    return np.sort(rng.choice(dataset_len, size=count, replace=False)).astype(np.int64)


def load_policy(checkpoint: pathlib.Path, output_dir: pathlib.Path, device: torch.device, use_raw: bool):
    payload = torch.load(checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=("optimizer",), strict=True)

    policy = workspace.model if use_raw or not cfg.training.use_ema else workspace.ema_model
    policy.to(device)
    policy.eval()
    return cfg, policy


def get_base_pose_mats(obs: Dict[str, torch.Tensor], arm: str = "R") -> Optional[np.ndarray]:
    pose_key = f"robot_pose_{arm}"
    quat_key = f"robot_quat_{arm}"
    if pose_key not in obs or quat_key not in obs:
        return None

    import scipy.spatial.transform as st

    pos = obs[pose_key][:, -1].detach().cpu().numpy()
    quat = obs[quat_key][:, -1].detach().cpu().numpy()
    rotvec = st.Rotation.from_quat(quat).as_rotvec()
    pose = np.concatenate([pos, rotvec], axis=-1)
    return pose_to_mat(pose)


def action_to_abs_positions(
        action: np.ndarray,
        action_pose_repr: str,
        base_pose_mats: Optional[np.ndarray]) -> np.ndarray:
    if action_pose_repr == "relative" and base_pose_mats is not None:
        abs_pose = []
        for batch_idx in range(action.shape[0]):
            rel_mat = pose10d_to_mat(action[batch_idx, :, :9])
            abs_mat = convert_pose_mat_rep(
                pose_mat=rel_mat,
                base_pose_mat=base_pose_mats[batch_idx],
                pose_rep="relative",
                backward=True,
            )
            abs_pose.append(mat_to_pose10d(abs_mat))
        return np.stack(abs_pose, axis=0)[..., :3]
    return action[..., :3].copy()


def set_axes_equal(ax, *groups):
    pts = []
    for group in groups:
        arr = np.asarray(group)
        if arr.size > 0:
            pts.append(arr.reshape(-1, 3))
    if len(pts) == 0:
        return
    pts = np.concatenate(pts, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def summarize_metrics(gt_action, pred_action, gt_abs_pos, pred_abs_pos, exec_slice):
    pos_err_m = np.linalg.norm(pred_abs_pos - gt_abs_pos, axis=-1)
    force_err = pred_action[..., 9:12] - gt_action[..., 9:12]
    torque_err = pred_action[..., 12:15] - gt_action[..., 12:15]

    def stats_for(mask_slice):
        pe = pos_err_m[:, mask_slice]
        fe = force_err[:, mask_slice]
        te = torque_err[:, mask_slice]
        return {
            "pos_mean_mm": float(pe.mean() * 1000),
            "pos_median_mm": float(np.median(pe) * 1000),
            "pos_p95_mm": float(np.percentile(pe, 95) * 1000),
            "force_rmse": float(np.sqrt(np.mean(fe ** 2))),
            "force_mae": float(np.mean(np.abs(fe))),
            "torque_rmse": float(np.sqrt(np.mean(te ** 2))),
            "torque_mae": float(np.mean(np.abs(te))),
        }

    return {
        "full_horizon": stats_for(slice(None)),
        "exec_horizon": stats_for(exec_slice),
        "per_step": {
            "pos_mean_mm": (pos_err_m.mean(axis=0) * 1000).tolist(),
            "force_rmse": np.sqrt(np.mean(force_err ** 2, axis=(0, 2))).tolist(),
            "torque_rmse": np.sqrt(np.mean(torque_err ** 2, axis=(0, 2))).tolist(),
        },
    }


def save_overview(output_dir, metrics, exec_slice):
    steps = np.arange(len(metrics["per_step"]["pos_mean_mm"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, metrics["per_step"]["pos_mean_mm"], "-o", color="#2563eb")
    axes[0].axvspan(exec_slice.start, exec_slice.stop - 1, color="#dbeafe", alpha=0.5)
    axes[0].set_ylabel("pos err (mm)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(steps, metrics["per_step"]["force_rmse"], "-o", color="#dc2626")
    axes[1].axvspan(exec_slice.start, exec_slice.stop - 1, color="#fee2e2", alpha=0.5)
    axes[1].set_ylabel("force RMSE")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(steps, metrics["per_step"]["torque_rmse"], "-o", color="#7c3aed")
    axes[2].axvspan(exec_slice.start, exec_slice.stop - 1, color="#ede9fe", alpha=0.5)
    axes[2].set_ylabel("torque RMSE")
    axes[2].set_xlabel("diffusion horizon step")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(
        "GT vs prediction overview "
        f"| exec steps {exec_slice.start}:{exec_slice.stop}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "overview_errors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_sample_plots(
        output_dir,
        indices,
        gt_action,
        pred_action,
        gt_abs_pos,
        pred_abs_pos,
        exec_slice):
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    force_labels = ("Fx", "Fy", "Fz")
    torque_labels = ("Tx", "Ty", "Tz")
    colors = ("#2563eb", "#16a34a", "#dc2626")
    steps = np.arange(gt_action.shape[1])

    for batch_idx, dataset_idx in enumerate(indices):
        fig = plt.figure(figsize=(16, 10))
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        gt_xyz = gt_abs_pos[batch_idx, exec_slice]
        pred_xyz = pred_abs_pos[batch_idx, exec_slice]
        ax_3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], "-o", color="#111827", label="GT")
        ax_3d.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], "-o", color="#dc2626", label="Pred")
        ax_3d.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color="#111827", s=65, marker="o")
        ax_3d.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], color="#111827", s=65, marker="s")
        ax_3d.scatter(pred_xyz[0, 0], pred_xyz[0, 1], pred_xyz[0, 2], color="#dc2626", s=65, marker="o")
        ax_3d.scatter(pred_xyz[-1, 0], pred_xyz[-1, 1], pred_xyz[-1, 2], color="#dc2626", s=65, marker="s")
        set_axes_equal(ax_3d, gt_xyz, pred_xyz)
        ax_3d.set_title("absolute position, exec horizon")
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("z")
        ax_3d.legend()
        ax_3d.grid(True, alpha=0.25)

        ax_pos = fig.add_subplot(2, 2, 2)
        for comp, label, color in zip(range(3), ("x", "y", "z"), colors):
            ax_pos.plot(steps, gt_abs_pos[batch_idx, :, comp], "-", color=color, alpha=0.65, label=f"GT {label}")
            ax_pos.plot(steps, pred_abs_pos[batch_idx, :, comp], "--", color=color, label=f"Pred {label}")
        ax_pos.axvspan(exec_slice.start, exec_slice.stop - 1, color="#e5e7eb", alpha=0.45)
        ax_pos.set_title("absolute xyz over horizon")
        ax_pos.set_xlabel("horizon step")
        ax_pos.set_ylabel("m")
        ax_pos.grid(True, alpha=0.25)
        ax_pos.legend(ncol=3, fontsize=8)

        ax_force = fig.add_subplot(2, 2, 3)
        for comp, label, color in zip(range(3), force_labels, colors):
            ax_force.plot(steps, gt_action[batch_idx, :, 9 + comp], "-", color=color, alpha=0.65, label=f"GT {label}")
            ax_force.plot(steps, pred_action[batch_idx, :, 9 + comp], "--", color=color, label=f"Pred {label}")
        ax_force.axvspan(exec_slice.start, exec_slice.stop - 1, color="#fee2e2", alpha=0.4)
        ax_force.set_title("wrist force")
        ax_force.set_xlabel("horizon step")
        ax_force.grid(True, alpha=0.25)
        ax_force.legend(ncol=3, fontsize=8)

        ax_torque = fig.add_subplot(2, 2, 4)
        for comp, label, color in zip(range(3), torque_labels, colors):
            ax_torque.plot(steps, gt_action[batch_idx, :, 12 + comp], "-", color=color, alpha=0.65, label=f"GT {label}")
            ax_torque.plot(steps, pred_action[batch_idx, :, 12 + comp], "--", color=color, label=f"Pred {label}")
        ax_torque.axvspan(exec_slice.start, exec_slice.stop - 1, color="#ede9fe", alpha=0.45)
        ax_torque.set_title("wrist torque")
        ax_torque.set_xlabel("horizon step")
        ax_torque.grid(True, alpha=0.25)
        ax_torque.legend(ncol=3, fontsize=8)

        pos_mean_mm = np.linalg.norm(
            pred_abs_pos[batch_idx, exec_slice] - gt_abs_pos[batch_idx, exec_slice],
            axis=-1,
        ).mean() * 1000
        force_rmse = np.sqrt(np.mean(
            (pred_action[batch_idx, exec_slice, 9:12] - gt_action[batch_idx, exec_slice, 9:12]) ** 2
        ))
        fig.suptitle(
            f"dataset idx {int(dataset_idx)} | exec pos mean {pos_mean_mm:.2f} mm "
            f"| exec force RMSE {force_rmse:.3f}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = sample_dir / f"sample_{int(dataset_idx):06d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_npz(output_dir, indices, gt_action, pred_action, gt_abs_pos, pred_abs_pos):
    path = output_dir / "predictions.npz"
    np.savez_compressed(
        path,
        indices=indices,
        gt_action=gt_action,
        pred_action=pred_action,
        gt_abs_pos=gt_abs_pos,
        pred_abs_pos=pred_abs_pos,
    )
    return path


def write_index_html(output_dir, metrics, overview_path, sample_paths):
    def rel(path):
        return pathlib.Path(path).relative_to(output_dir).as_posix()

    cards = []
    for path in sample_paths:
        cards.append(
            f'<figure><img src="{rel(path)}" alt="{path.name}">'
            f'<figcaption>{path.stem}</figcaption></figure>'
        )

    exec_m = metrics["exec_horizon"]
    full_m = metrics["full_horizon"]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Action Force Prediction Visualization</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #111827; }}
    h1 {{ font-size: 22px; margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; max-width: 900px; }}
    .metric {{ border: 1px solid #d1d5db; padding: 10px; border-radius: 6px; }}
    .label {{ color: #4b5563; font-size: 12px; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 2px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
    figure {{ margin: 0 0 24px 0; }}
    figcaption {{ color: #4b5563; font-size: 13px; margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(720px, 1fr)); gap: 24px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Action + Wrist Wrench Prediction</h1>
  <p><code>{metrics["checkpoint"]}</code></p>
  <p>Dataset: <code>{metrics["dataset_path"]}</code></p>
  <p>Indices: <code>{metrics["indices"]}</code>, exec slice: <code>{metrics["exec_slice"]}</code></p>
  <div class="metrics">
    <div class="metric"><div class="label">Exec Pos Mean</div><div class="value">{exec_m["pos_mean_mm"]:.2f} mm</div></div>
    <div class="metric"><div class="label">Exec Pos P95</div><div class="value">{exec_m["pos_p95_mm"]:.2f} mm</div></div>
    <div class="metric"><div class="label">Exec Force RMSE</div><div class="value">{exec_m["force_rmse"]:.3f}</div></div>
    <div class="metric"><div class="label">Exec Torque RMSE</div><div class="value">{exec_m["torque_rmse"]:.3f}</div></div>
    <div class="metric"><div class="label">Full Pos Mean</div><div class="value">{full_m["pos_mean_mm"]:.2f} mm</div></div>
    <div class="metric"><div class="label">Full Pos P95</div><div class="value">{full_m["pos_p95_mm"]:.2f} mm</div></div>
    <div class="metric"><div class="label">Full Force RMSE</div><div class="value">{full_m["force_rmse"]:.3f}</div></div>
    <div class="metric"><div class="label">Full Torque RMSE</div><div class="value">{full_m["torque_rmse"]:.3f}</div></div>
  </div>
  <h2>Overview</h2>
  <figure><img src="{rel(overview_path)}" alt="overview"><figcaption>Per-step mean position error and wrench RMSE. Shaded area is the executed action window.</figcaption></figure>
  <h2>Samples</h2>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "data/outputs/2026.07.06/22.00.05_train_diffusion_unet_hybrid_"
            "bbbae_dualarm_erase_board_no_wrench_predict_wrench/checkpoints/"
            "epoch=0300-train_loss=0.005.ckpt"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--raw", action="store_true", help="Use raw model instead of EMA model.")
    args = parser.parse_args()

    checkpoint = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else default_output_dir(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    cfg, policy = load_policy(checkpoint, output_dir, device, use_raw=args.raw)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    indices = choose_indices(len(dataset), args.n_samples, args.seed, args.indices)
    exec_start = int(cfg.n_obs_steps) - 1
    exec_stop = exec_start + int(cfg.n_action_steps)
    exec_slice = slice(exec_start, exec_stop)

    all_gt_action = []
    all_pred_action = []
    all_gt_abs_pos = []
    all_pred_abs_pos = []

    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            samples = [dataset[int(idx)] for idx in batch_indices]
            batch = default_collate(samples)
            obs = dict_apply(batch["obs"], lambda x: x.to(device, non_blocking=True))
            gt_action = batch["action"].detach().cpu().numpy()
            result = policy.predict_action(obs)
            pred_action = result["action_pred"].detach().cpu().numpy()

            base_pose_mats = get_base_pose_mats(batch["obs"], arm="R")
            action_pose_repr = cfg.task.pose_repr.get("action_pose_repr", "abs")
            gt_abs_pos = action_to_abs_positions(gt_action, action_pose_repr, base_pose_mats)
            pred_abs_pos = action_to_abs_positions(pred_action, action_pose_repr, base_pose_mats)

            all_gt_action.append(gt_action)
            all_pred_action.append(pred_action)
            all_gt_abs_pos.append(gt_abs_pos)
            all_pred_abs_pos.append(pred_abs_pos)

    gt_action = np.concatenate(all_gt_action, axis=0)
    pred_action = np.concatenate(all_pred_action, axis=0)
    gt_abs_pos = np.concatenate(all_gt_abs_pos, axis=0)
    pred_abs_pos = np.concatenate(all_pred_abs_pos, axis=0)

    metrics = summarize_metrics(gt_action, pred_action, gt_abs_pos, pred_abs_pos, exec_slice)
    metrics.update({
        "checkpoint": str(checkpoint),
        "dataset_path": str(cfg.task.dataset_path),
        "indices": [int(x) for x in indices],
        "exec_slice": [int(exec_slice.start), int(exec_slice.stop)],
        "used_model": "raw" if args.raw else "ema",
    })

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    overview_path = save_overview(output_dir, metrics, exec_slice)
    sample_paths = save_sample_plots(
        output_dir, indices, gt_action, pred_action, gt_abs_pos, pred_abs_pos, exec_slice)
    npz_path = write_npz(output_dir, indices, gt_action, pred_action, gt_abs_pos, pred_abs_pos)
    index_path = write_index_html(output_dir, metrics, overview_path, sample_paths)

    print(f"checkpoint: {checkpoint}")
    print(f"dataset: {cfg.task.dataset_path}")
    print(f"indices: {indices.tolist()}")
    print(f"metrics: {metrics_path}")
    print(f"overview: {overview_path}")
    print(f"samples: {sample_paths[0].parent if len(sample_paths) else output_dir / 'samples'}")
    print(f"index: {index_path}")
    print(f"npz: {npz_path}")
    print("exec metrics:", json.dumps(metrics["exec_horizon"], indent=2))


if __name__ == "__main__":
    main()
