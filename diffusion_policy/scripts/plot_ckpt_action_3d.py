if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import json
import pathlib
from typing import Dict, Iterable, Optional, Tuple

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.pose_util import pose10d_to_mat, pose_to_mat, mat_to_pose10d
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_indices(indices: Optional[str]) -> Optional[np.ndarray]:
    if indices is None or indices.strip() == "":
        return None
    return np.array([int(x.strip()) for x in indices.split(",") if x.strip()], dtype=np.int64)


def choose_indices(
        dataset_len: int,
        num_samples: int,
        seed: int,
        indices: Optional[str]) -> np.ndarray:
    parsed = parse_indices(indices)
    if parsed is not None:
        if np.any(parsed < 0) or np.any(parsed >= dataset_len):
            raise ValueError(f"indices must be in [0, {dataset_len - 1}]")
        return parsed

    count = min(num_samples, dataset_len)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_len, size=count, replace=False)).astype(np.int64)


def load_policy(checkpoint: pathlib.Path, output_dir: pathlib.Path, device: torch.device, strict: bool = True):
    payload = torch.load(checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=("optimizer",), strict=strict)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()
    return cfg, policy


def make_dataset(cfg, split: str):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if split == "val":
        dataset = dataset.get_validation_dataset()
    elif split != "train":
        raise ValueError(f"Unsupported split: {split}")
    return dataset


def get_base_pose_mats(obs: Dict[str, torch.Tensor], arm: str) -> Optional[np.ndarray]:
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


def action_to_positions(
        action: np.ndarray,
        action_pose_repr: str,
        base_pose_mats: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rel_pos = action[..., :3]
    if action_pose_repr == "relative" and base_pose_mats is not None:
        abs_pose = []
        for i in range(action.shape[0]):
            rel_mat = pose10d_to_mat(action[i, :, :9])
            abs_mat = convert_pose_mat_rep(
                pose_mat=rel_mat,
                base_pose_mat=base_pose_mats[i],
                pose_rep="relative",
                backward=True)
            abs_pose.append(mat_to_pose10d(abs_mat))
        abs_pose = np.stack(abs_pose, axis=0)
        return rel_pos, abs_pose[..., :3]
    return rel_pos, rel_pos.copy()


def equalize_scene_ranges(*position_groups: Iterable[np.ndarray]):
    points = []
    for group in position_groups:
        if group is None:
            continue
        arr = np.asarray(group)
        if arr.size > 0:
            points.append(arr.reshape(-1, 3))
    if len(points) == 0:
        return {}
    pts = np.concatenate(points, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 1e-4)
    return {
        "xaxis": {"range": [center[0] - radius, center[0] + radius], "title": "x"},
        "yaxis": {"range": [center[1] - radius, center[1] + radius], "title": "y"},
        "zaxis": {"range": [center[2] - radius, center[2] + radius], "title": "z"},
        "aspectmode": "cube",
    }


def set_matplotlib_axes_equal(ax, *position_groups: Iterable[np.ndarray]):
    points = []
    for group in position_groups:
        if group is None:
            continue
        arr = np.asarray(group)
        if arr.size > 0:
            points.append(arr.reshape(-1, 3))
    if len(points) == 0:
        return

    pts = np.concatenate(points, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_matplotlib_trace(ax, gt_pos: np.ndarray, pred_pos: np.ndarray, base_pos: Optional[np.ndarray], title: str):
    steps = np.arange(gt_pos.shape[0])
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], "-o", color="#2563eb", label="GT", linewidth=2, markersize=4)
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], "-o", color="#dc2626", label="Pred", linewidth=2, markersize=4)
    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], color="#1d4ed8", s=70, marker="o")
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], color="#1d4ed8", s=70, marker="s")
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], color="#b91c1c", s=70, marker="o")
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], color="#b91c1c", s=70, marker="s")
    if base_pos is not None:
        ax.scatter(base_pos[0], base_pos[1], base_pos[2], color="#111827", s=70, marker="D", label="Base")

    for step, pos in zip(steps, gt_pos):
        ax.text(pos[0], pos[1], pos[2], str(int(step)), color="#1d4ed8", fontsize=7)
    for step, pos in zip(steps, pred_pos):
        ax.text(pos[0], pos[1], pos[2], str(int(step)), color="#b91c1c", fontsize=7)

    set_matplotlib_axes_equal(ax, gt_pos, pred_pos, None if base_pos is None else base_pos[None])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=-60)
    ax.grid(True, alpha=0.25)


def save_sample_images(
        output_dir: pathlib.Path,
        indices: np.ndarray,
        gt_rel_pos: np.ndarray,
        pred_rel_pos: np.ndarray,
        gt_abs_pos: np.ndarray,
        pred_abs_pos: np.ndarray,
        base_pose_mats: Optional[np.ndarray],
        step_slice: slice,
        plot_mode: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_dir = output_dir / "sample_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    use_relative = plot_mode in ("relative", "both")
    use_absolute = plot_mode in ("absolute", "both")
    cols = int(use_relative) + int(use_absolute)
    paths = []

    for i, idx in enumerate(indices):
        fig = plt.figure(figsize=(7 * cols, 6))
        col = 1
        if use_relative:
            ax = fig.add_subplot(1, cols, col, projection="3d")
            plot_matplotlib_trace(
                ax,
                gt_rel_pos[i, step_slice],
                pred_rel_pos[i, step_slice],
                np.zeros(3, dtype=np.float32),
                "relative action",
            )
            col += 1
        if use_absolute:
            ax = fig.add_subplot(1, cols, col, projection="3d")
            base_pos = None
            if base_pose_mats is not None:
                base_pos = base_pose_mats[i, :3, 3]
            plot_matplotlib_trace(
                ax,
                gt_abs_pos[i, step_slice],
                pred_abs_pos[i, step_slice],
                base_pos,
                "absolute reconstructed",
            )

        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3)
        fig.suptitle(f"sample {int(idx)} | action steps {step_slice.start}:{step_slice.stop}", y=0.98)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93))
        path = image_dir / f"sample_{int(idx):06d}_pred_vs_gt_action_3d.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def add_trace_pair(
        fig,
        row: int,
        col: int,
        gt_pos: np.ndarray,
        pred_pos: np.ndarray,
        sample_name: str,
        showlegend: bool):
    import plotly.graph_objects as go

    steps = np.arange(gt_pos.shape[0])
    hover_gt = [f"{sample_name}<br>GT step {int(s)}" for s in steps]
    hover_pred = [f"{sample_name}<br>pred step {int(s)}" for s in steps]

    fig.add_trace(
        go.Scatter3d(
            x=gt_pos[:, 0],
            y=gt_pos[:, 1],
            z=gt_pos[:, 2],
            mode="lines+markers",
            name="GT",
            legendgroup="GT",
            showlegend=showlegend,
            line={"color": "#2563eb", "width": 6},
            marker={"size": 3, "color": "#2563eb"},
            text=hover_gt,
            hoverinfo="text+x+y+z",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pred_pos[:, 0],
            y=pred_pos[:, 1],
            z=pred_pos[:, 2],
            mode="lines+markers",
            name="Pred",
            legendgroup="Pred",
            showlegend=showlegend,
            line={"color": "#dc2626", "width": 6},
            marker={"size": 3, "color": "#dc2626"},
            text=hover_pred,
            hoverinfo="text+x+y+z",
        ),
        row=row,
        col=col,
    )


def make_plot(
        output_html: pathlib.Path,
        indices: np.ndarray,
        gt_rel_pos: np.ndarray,
        pred_rel_pos: np.ndarray,
        gt_abs_pos: np.ndarray,
        pred_abs_pos: np.ndarray,
        base_pose_mats: Optional[np.ndarray],
        exec_slice: slice,
        plot_mode: str):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    use_relative = plot_mode in ("relative", "both")
    use_absolute = plot_mode in ("absolute", "both")
    cols = int(use_relative) + int(use_absolute)
    rows = len(indices)

    subplot_titles = []
    for idx in indices:
        if use_relative:
            subplot_titles.append(f"sample {idx} relative")
        if use_absolute:
            subplot_titles.append(f"sample {idx} absolute")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=min(0.04, 0.8 / max(rows - 1, 1)),
    )

    for r, idx in enumerate(indices, start=1):
        col = 1
        sample_name = f"sample {idx}"
        showlegend = r == 1
        if use_relative:
            add_trace_pair(
                fig, r, col,
                gt_rel_pos[r - 1, exec_slice],
                pred_rel_pos[r - 1, exec_slice],
                sample_name,
                showlegend,
            )
            rel_scene = equalize_scene_ranges(
                gt_rel_pos[r - 1, exec_slice],
                pred_rel_pos[r - 1, exec_slice],
                np.zeros((1, 3), dtype=np.float32),
            )
            fig.update_scenes(rel_scene, row=r, col=col)
            fig.add_trace(
                go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[0],
                    mode="markers",
                    name="base",
                    legendgroup="base",
                    showlegend=showlegend,
                    marker={"size": 5, "color": "#111827", "symbol": "diamond"},
                ),
                row=r,
                col=col,
            )
            col += 1

        if use_absolute:
            add_trace_pair(
                fig, r, col,
                gt_abs_pos[r - 1, exec_slice],
                pred_abs_pos[r - 1, exec_slice],
                sample_name,
                showlegend,
            )
            base_pos = None
            if base_pose_mats is not None:
                base_pos = base_pose_mats[r - 1, :3, 3][None]
                fig.add_trace(
                    go.Scatter3d(
                        x=base_pos[:, 0],
                        y=base_pos[:, 1],
                        z=base_pos[:, 2],
                        mode="markers",
                        name="obs base",
                        legendgroup="obs base",
                        showlegend=showlegend,
                        marker={"size": 5, "color": "#111827", "symbol": "diamond"},
                    ),
                    row=r,
                    col=col,
                )
            abs_scene = equalize_scene_ranges(
                gt_abs_pos[r - 1, exec_slice],
                pred_abs_pos[r - 1, exec_slice],
                base_pos,
            )
            fig.update_scenes(abs_scene, row=r, col=col)

    fig.update_layout(
        title=f"Predicted vs GT Action 3D (steps {exec_slice.start}:{exec_slice.stop})",
        height=max(420, rows * 420),
        width=1400 if cols == 2 else 820,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 10, "r": 10, "t": 80, "b": 10},
    )
    fig.write_html(str(output_html), include_plotlyjs=True)


def write_metrics(
        output_path: pathlib.Path,
        indices: np.ndarray,
        gt_rel_pos: np.ndarray,
        pred_rel_pos: np.ndarray,
        gt_abs_pos: np.ndarray,
        pred_abs_pos: np.ndarray,
        exec_slice: slice):
    rows = []
    for i, idx in enumerate(indices):
        full_rel_rmse = float(np.sqrt(np.mean((pred_rel_pos[i] - gt_rel_pos[i]) ** 2)))
        exec_rel_rmse = float(np.sqrt(np.mean((pred_rel_pos[i, exec_slice] - gt_rel_pos[i, exec_slice]) ** 2)))
        full_abs_rmse = float(np.sqrt(np.mean((pred_abs_pos[i] - gt_abs_pos[i]) ** 2)))
        exec_abs_rmse = float(np.sqrt(np.mean((pred_abs_pos[i, exec_slice] - gt_abs_pos[i, exec_slice]) ** 2)))
        rows.append({
            "dataset_index": int(idx),
            "full_relative_pos_rmse": full_rel_rmse,
            "exec_relative_pos_rmse": exec_rel_rmse,
            "full_absolute_pos_rmse": full_abs_rmse,
            "exec_absolute_pos_rmse": exec_abs_rmse,
        })
    summary = {
        "exec_slice": [exec_slice.start, exec_slice.stop],
        "samples": rows,
        "mean_exec_relative_pos_rmse": float(np.mean([x["exec_relative_pos_rmse"] for x in rows])),
        "mean_exec_absolute_pos_rmse": float(np.mean([x["exec_absolute_pos_rmse"] for x in rows])),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot checkpoint predictions against dataset GT actions in 3D.")
    parser.add_argument("-c", "--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("-d", "--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--indices", default=None, help="Comma-separated sampler indices. Overrides --num-samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arm", choices=["R", "L"], default="R")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--plot-mode", choices=["relative", "absolute", "both"], default="both")
    parser.add_argument("--step-mode", choices=["exec", "all"], default="exec")
    parser.add_argument("--save-sample-images", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--non-strict-load", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg, policy = load_policy(args.checkpoint, args.output_dir, device, strict=not args.non_strict_load)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps

    dataset = make_dataset(cfg, args.split)
    indices = choose_indices(len(dataset), args.num_samples, args.seed, args.indices)
    batch = default_collate([dataset[int(i)] for i in indices])

    obs_device = dict_apply(batch["obs"], lambda x: x.to(device, non_blocking=True))
    with torch.no_grad():
        result = policy.predict_action(obs_device)

    gt_action = batch["action"].detach().cpu().numpy()
    pred_action = result["action_pred"].detach().cpu().numpy()

    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default="abs")
    base_pose_mats = get_base_pose_mats(batch["obs"], args.arm)
    gt_rel_pos, gt_abs_pos = action_to_positions(gt_action, action_pose_repr, base_pose_mats)
    pred_rel_pos, pred_abs_pos = action_to_positions(pred_action, action_pose_repr, base_pose_mats)

    if args.step_mode == "all":
        step_slice = slice(0, int(cfg.horizon))
    else:
        start = int(cfg.n_obs_steps) - 1
        stop = start + int(cfg.n_action_steps)
        step_slice = slice(start, stop)

    html_path = args.output_dir / "pred_vs_gt_action_3d.html"
    metrics_path = args.output_dir / "pred_vs_gt_action_3d_metrics.json"
    npz_path = args.output_dir / "pred_vs_gt_action_3d_samples.npz"

    if not args.skip_html:
        make_plot(
            output_html=html_path,
            indices=indices,
            gt_rel_pos=gt_rel_pos,
            pred_rel_pos=pred_rel_pos,
            gt_abs_pos=gt_abs_pos,
            pred_abs_pos=pred_abs_pos,
            base_pose_mats=base_pose_mats,
            exec_slice=step_slice,
            plot_mode=args.plot_mode,
        )
    summary = write_metrics(
        output_path=metrics_path,
        indices=indices,
        gt_rel_pos=gt_rel_pos,
        pred_rel_pos=pred_rel_pos,
        gt_abs_pos=gt_abs_pos,
        pred_abs_pos=pred_abs_pos,
        exec_slice=step_slice,
    )
    image_paths = []
    if args.save_sample_images:
        image_paths = save_sample_images(
            output_dir=args.output_dir,
            indices=indices,
            gt_rel_pos=gt_rel_pos,
            pred_rel_pos=pred_rel_pos,
            gt_abs_pos=gt_abs_pos,
            pred_abs_pos=pred_abs_pos,
            base_pose_mats=base_pose_mats,
            step_slice=step_slice,
            plot_mode=args.plot_mode,
        )
    np.savez_compressed(
        npz_path,
        indices=indices,
        gt_action=gt_action,
        pred_action_pred=pred_action,
        gt_rel_pos=gt_rel_pos,
        pred_rel_pos=pred_rel_pos,
        gt_abs_pos=gt_abs_pos,
        pred_abs_pos=pred_abs_pos,
        base_pose_mats=base_pose_mats,
    )

    if not args.skip_html:
        print(f"saved html: {html_path}")
    if len(image_paths) > 0:
        print("saved images:")
        for path in image_paths:
            print(f"  {path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved samples: {npz_path}")
    print(f"sample indices: {indices.tolist()}")
    print(f"mean exec relative pos RMSE: {summary['mean_exec_relative_pos_rmse']:.6f}")
    print(f"mean exec absolute pos RMSE: {summary['mean_exec_absolute_pos_rmse']:.6f}")


if __name__ == "__main__":
    main()
