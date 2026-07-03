#!/usr/bin/env python3
if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import dill
import h5py
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from diffusion_policy.residual_policy.pose_util import (
    abs_pose9_to_relative_pose9,
    apply_residual_action_to_pose9,
    current_obs_to_pose9,
    pose6_to_pose9,
    pose9_to_mat,
    pose_like_to_pose9,
    relative_pose9_to_abs_pose9,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


COLORS = {
    "gt_actual": "#2563eb",
    "gt_virtual": "#dc2626",
    "slow_pred_actual": "#f59e0b",
    "fast_pred_virtual": "#16a34a",
}

SQRT2 = np.sqrt(2) / 2
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ],
    dtype=np.float32,
)


def sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def parse_int_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    value = str(value).strip()
    if len(value) == 0:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_policy_from_ckpt(ckpt_path, device, use_ema=True):
    ckpt_path = Path(ckpt_path).expanduser()
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key], strict=False)
    policy.to(device)
    policy.eval()
    del payload
    return cfg, policy


def to_torch_obs(obs_np, device):
    out = {}
    for key, value in obs_np.items():
        out[key] = torch.from_numpy(value).to(device=device, dtype=torch.float32)
    return out


def window_indices(t, n_obs_steps):
    start = t - n_obs_steps + 1
    return np.asarray([max(0, start + i) for i in range(n_obs_steps)], dtype=np.int64)


def image_to_chw_float(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    return np.moveaxis(image, -1, 1)


def build_policy_obs(policy, obs_group, t, device, base_action_rel=None, n_obs_steps_override=None):
    n_obs_steps = int(n_obs_steps_override or getattr(policy, "n_obs_steps", 1))
    idxs = window_indices(t, n_obs_steps)
    obs_np = {}

    for key in getattr(policy, "rgb_keys", []):
        if key not in obs_group:
            continue
        obs_np[key] = image_to_chw_float(np.asarray(obs_group[key])[idxs])[None]

    for key in getattr(policy, "low_dim_keys", []):
        if key not in obs_group:
            continue
        obs_np[key] = np.asarray(obs_group[key])[idxs].astype(np.float32)[None]

    for key in getattr(policy, "wrench_keys", []):
        if key not in obs_group:
            continue
        # Wrench-encoder policies use only the latest force-history window.
        obs_np[key] = np.asarray(obs_group[key])[t:t + 1].astype(np.float32)[None]

    if base_action_rel is not None:
        key = getattr(policy, "base_action_key", "base_action_rel")
        obs_np[key] = base_action_rel.astype(np.float32).reshape(1, 1, -1)

    return to_torch_obs(obs_np, device)


def build_context_sequence_obs(policy, obs_group, context_t, step_indices, device, base_action_rel_seq):
    step_indices = np.asarray(step_indices, dtype=np.int64)
    context_t = int(context_t)
    obs_np = {}
    seq_len = len(step_indices)

    for key in getattr(policy, "rgb_keys", []):
        if key not in obs_group:
            continue
        image = np.asarray(obs_group[key])[context_t:context_t + 1]
        image = np.repeat(image, seq_len, axis=0)
        obs_np[key] = image_to_chw_float(image)[None]

    for key in getattr(policy, "low_dim_keys", []):
        if key not in obs_group:
            continue
        if getattr(policy, "include_step_low_dim", False):
            value = np.asarray(obs_group[key])[step_indices]
        else:
            value = np.asarray(obs_group[key])[context_t:context_t + 1]
            value = np.repeat(value, seq_len, axis=0)
        obs_np[key] = value.astype(np.float32)[None]

    for key in getattr(policy, "wrench_keys", []):
        if key not in obs_group:
            continue
        obs_np[key] = np.asarray(obs_group[key])[step_indices].astype(np.float32)[None]

    key = getattr(policy, "base_action_key", "base_action_rel")
    obs_np[key] = np.asarray(base_action_rel_seq, dtype=np.float32)[None]
    return to_torch_obs(obs_np, device)


def collate_policy_obs(obs_list):
    keys = obs_list[0].keys()
    return {
        key: torch.cat([obs[key] for obs in obs_list], dim=0)
        for key in keys
    }


def raw_env_obs_for_step(obs_group, t, arm):
    return {
        f"robot_pose_{arm}": np.asarray(obs_group[f"robot_pose_{arm}"])[t:t + 1],
        f"robot_quat_{arm}": np.asarray(obs_group[f"robot_quat_{arm}"])[t:t + 1],
    }


def current_pose9_for_step(obs_group, t, arm):
    pos = np.asarray(obs_group[f"robot_pose_{arm}"])[t]
    quat = np.asarray(obs_group[f"robot_quat_{arm}"])[t]
    rotvec = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate([pos, rotvec], axis=-1))


def slow_action_to_abs_and_rel(slow_action, current_pose9, action_pose_repr):
    slow_action = pose_like_to_pose9(slow_action)
    if action_pose_repr == "relative":
        slow_abs = relative_pose9_to_abs_pose9(current_pose9, slow_action)
        slow_rel = slow_action
    elif action_pose_repr == "abs":
        slow_abs = slow_action
        slow_rel = abs_pose9_to_relative_pose9(current_pose9, slow_abs)
    else:
        raise ValueError(f"Unsupported slow action_pose_repr: {action_pose_repr}")
    return slow_abs.astype(np.float32), slow_rel.astype(np.float32)


def pose_errors(pred_pose9, gt_pose9):
    pred_mat = pose9_to_mat(pred_pose9)
    gt_mat = pose9_to_mat(gt_pose9)
    pos_err_mm = np.linalg.norm(pred_mat[..., :3, 3] - gt_mat[..., :3, 3], axis=-1) * 1000.0
    rel_rot = np.linalg.inv(pred_mat) @ gt_mat
    rot_err_deg = np.rad2deg(Rotation.from_matrix(rel_rot[..., :3, :3]).magnitude())
    return pos_err_mm, rot_err_deg


def resolve_virtual_position_scale(scale_arg, gt_actual, gt_virtual):
    if scale_arg != "auto":
        return float(scale_arg)
    actual_norm = float(np.median(np.linalg.norm(gt_actual[:, :3], axis=-1)))
    virtual_norm = float(np.median(np.linalg.norm(gt_virtual[:, :3], axis=-1)))
    if actual_norm < 10.0 and virtual_norm > 10.0:
        return 0.001
    return 1.0


def scale_pose_position(pose9, scale):
    out = np.asarray(pose9).copy()
    out[..., :3] *= float(scale)
    return out


def right_robot_pose_to_world(pose9):
    out = np.asarray(pose9).copy()
    out[..., :3] = (RIGHT_ROBOT_TO_WORLD @ out[..., :3].T).T
    return out


def axis_ranges(*pose_groups):
    points = np.concatenate([pose[:, :3] for pose in pose_groups], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max() / 2.0) * 1.08, 1e-3)
    return centers, radius


def add_traj(fig, pose9, name, color, width=2.0, dash=None, opacity=1.0):
    fig.add_trace(go.Scatter3d(
        x=pose9[:, 0],
        y=pose9[:, 1],
        z=pose9[:, 2],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=2.5, color=color),
        opacity=opacity,
        hovertemplate=(
            name + "<br>"
            "step=%{pointNumber}<br>"
            "x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>"
        ),
    ))


def write_trajectory_html(path, demo_name, gt_actual, gt_virtual, slow_pred, fast_pred):
    centers, radius = axis_ranges(gt_actual, gt_virtual, slow_pred, fast_pred)
    fig = go.Figure()
    add_traj(fig, gt_actual, "GT actual", COLORS["gt_actual"], width=2.5)
    add_traj(fig, gt_virtual, "GT virtual", COLORS["gt_virtual"], width=2.5)
    add_traj(fig, slow_pred, "slow predicted actual", COLORS["slow_pred_actual"], width=2.0, dash="dash")
    add_traj(fig, fast_pred, "fast predicted virtual", COLORS["fast_pred_virtual"], width=2.0, dash="dash")
    fig.update_layout(
        title=f"{demo_name}: GT actual/virtual vs slow/fast predictions",
        width=1100,
        height=850,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis=dict(title="x (m)", range=[centers[0] - radius, centers[0] + radius]),
            yaxis=dict(title="y (m)", range=[centers[1] - radius, centers[1] + radius]),
            zaxis=dict(title="z (m)", range=[centers[2] - radius, centers[2] + radius]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=-1.65, z=1.05)),
        ),
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def add_marker_traj(fig, pose9, name, color, symbol="circle", size=5):
    custom = np.arange(len(pose9))
    fig.add_trace(go.Scatter3d(
        x=pose9[:, 0],
        y=pose9[:, 1],
        z=pose9[:, 2],
        mode="markers",
        name=name,
        marker=dict(color=color, size=size, symbol=symbol),
        customdata=custom,
        hovertemplate=(
            name + "<br>"
            "h=%{customdata}<br>"
            "x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>"
        ),
    ))


def add_line_traj(fig, pose9, name, color, width=2.0, dash=None, opacity=1.0):
    fig.add_trace(go.Scatter3d(
        x=pose9[:, 0],
        y=pose9[:, 1],
        z=pose9[:, 2],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=3, color=color),
        opacity=opacity,
    ))


def add_residual_segments(fig, slow_pose9, fast_pose9):
    x = []
    y = []
    z = []
    custom = []
    for i, (slow, fast) in enumerate(zip(slow_pose9, fast_pose9)):
        x.extend([slow[0], fast[0], None])
        y.extend([slow[1], fast[1], None])
        z.extend([slow[2], fast[2], None])
        custom.extend([i, i, None])
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name="fast residual correction",
        line=dict(color=COLORS["fast_pred_virtual"], width=3),
        opacity=0.42,
        customdata=custom,
        hovertemplate="residual pair h=%{customdata}<extra></extra>",
    ))


def write_window_html(path, title, gt_actual, gt_virtual, slow_pred, fast_pred):
    centers, radius = axis_ranges(gt_actual, gt_virtual, slow_pred, fast_pred)
    fig = go.Figure()
    add_line_traj(fig, gt_actual, "GT actual 16-step", COLORS["gt_actual"], width=2.2)
    add_line_traj(fig, gt_virtual, "GT virtual 16-step", COLORS["gt_virtual"], width=2.2)
    add_marker_traj(fig, slow_pred, "slow action points", COLORS["slow_pred_actual"], symbol="diamond", size=5)
    add_marker_traj(fig, fast_pred, "slow + fast residual points", COLORS["fast_pred_virtual"], symbol="circle", size=5)
    add_residual_segments(fig, slow_pred, fast_pred)
    fig.update_layout(
        title=title,
        width=1050,
        height=820,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=0, r=0, t=65, b=0),
        scene=dict(
            xaxis=dict(title="x (m)", range=[centers[0] - radius, centers[0] + radius]),
            yaxis=dict(title="y (m)", range=[centers[1] - radius, centers[1] + radius]),
            zaxis=dict(title="z (m)", range=[centers[2] - radius, centers[2] + radius]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=-1.65, z=1.05)),
        ),
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def add_chunk_markers(fig, pose9, chunk_ids, step_ids, name, color, symbol="circle", size=5):
    custom = np.column_stack([chunk_ids, step_ids])
    fig.add_trace(go.Scatter3d(
        x=pose9[:, 0],
        y=pose9[:, 1],
        z=pose9[:, 2],
        mode="markers",
        name=name,
        marker=dict(color=color, size=size, symbol=symbol),
        customdata=custom,
        hovertemplate=(
            name + "<br>"
            "chunk=%{customdata[0]} step=%{customdata[1]}<br>"
            "x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>"
        ),
    ))


def add_chunk_residual_segments(fig, slow_pose9, fast_pose9, chunk_ids, step_ids):
    x = []
    y = []
    z = []
    custom = []
    for slow, fast, chunk_id, step_id in zip(slow_pose9, fast_pose9, chunk_ids, step_ids):
        x.extend([slow[0], fast[0], None])
        y.extend([slow[1], fast[1], None])
        z.extend([slow[2], fast[2], None])
        custom.extend([f"{chunk_id}:{step_id}", f"{chunk_id}:{step_id}", None])
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name="slow -> fast residual",
        line=dict(color=COLORS["fast_pred_virtual"], width=3),
        opacity=0.4,
        customdata=custom,
        hovertemplate="chunk:step=%{customdata}<extra></extra>",
    ))


def write_chunked_html(path, title, gt_actual, gt_virtual, slow_pred, fast_pred, chunk_ids, step_ids):
    centers, radius = axis_ranges(gt_actual, gt_virtual, slow_pred, fast_pred)
    fig = go.Figure()
    add_line_traj(fig, gt_actual, "GT actual over accepted steps", COLORS["gt_actual"], width=2.3)
    add_line_traj(fig, gt_virtual, "GT virtual over accepted steps", COLORS["gt_virtual"], width=2.3)
    add_chunk_markers(
        fig,
        slow_pred,
        chunk_ids,
        step_ids,
        "accepted slow points",
        COLORS["slow_pred_actual"],
        symbol="diamond",
        size=5,
    )
    add_chunk_markers(
        fig,
        fast_pred,
        chunk_ids,
        step_ids,
        "accepted slow + fast residual points",
        COLORS["fast_pred_virtual"],
        symbol="circle",
        size=5,
    )
    add_chunk_residual_segments(fig, slow_pred, fast_pred, chunk_ids, step_ids)
    fig.update_layout(
        title=title,
        width=1150,
        height=860,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=0, r=0, t=65, b=0),
        scene=dict(
            xaxis=dict(title="x (m)", range=[centers[0] - radius, centers[0] + radius]),
            yaxis=dict(title="y (m)", range=[centers[1] - radius, centers[1] + radius]),
            zaxis=dict(title="z (m)", range=[centers[2] - radius, centers[2] + radius]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=-1.65, z=1.05)),
        ),
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def write_error_png(path, demo_name, slow_err, fast_err):
    slow_pos, slow_rot = slow_err
    fast_pos, fast_rot = fast_err
    t = np.arange(len(slow_pos))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, slow_pos, color=COLORS["slow_pred_actual"], label="slow vs GT actual")
    axes[0].plot(t, fast_pos, color=COLORS["fast_pred_virtual"], label="fast vs GT virtual")
    axes[0].set_ylabel("position error (mm)")
    axes[1].plot(t, slow_rot, color=COLORS["slow_pred_actual"], label="slow vs GT actual")
    axes[1].plot(t, fast_rot, color=COLORS["fast_pred_virtual"], label="fast vs GT virtual")
    axes[1].set_ylabel("rotation error (deg)")
    axes[1].set_xlabel("dataset step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{demo_name}: prediction error")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize_errors(pos_err, rot_err):
    return {
        "pos_mm_mean": float(np.mean(pos_err)),
        "pos_mm_median": float(np.median(pos_err)),
        "pos_mm_max": float(np.max(pos_err)),
        "rot_deg_mean": float(np.mean(rot_err)),
        "rot_deg_median": float(np.median(rot_err)),
        "rot_deg_max": float(np.max(rot_err)),
    }


def summarize_norms(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def summarize_by_local_step(step_ids, slow_err, fast_err):
    slow_pos, slow_rot = slow_err
    fast_pos, fast_rot = fast_err
    out = {}
    for step_id in np.unique(step_ids):
        mask = step_ids == step_id
        out[str(int(step_id))] = {
            "count": int(mask.sum()),
            "slow_pos_mm_mean": float(np.mean(slow_pos[mask])),
            "slow_rot_deg_mean": float(np.mean(slow_rot[mask])),
            "fast_pos_mm_mean": float(np.mean(fast_pos[mask])),
            "fast_rot_deg_mean": float(np.mean(fast_rot[mask])),
        }
    return out


def residual_rotation_norm_deg(residual_pred):
    residual_pred = np.asarray(residual_pred)
    if residual_pred.shape[-1] == 6:
        return np.rad2deg(np.linalg.norm(residual_pred[..., 3:], axis=-1))
    if residual_pred.shape[-1] == 9:
        residual_mat = pose9_to_mat(residual_pred)
        return np.rad2deg(Rotation.from_matrix(residual_mat[..., :3, :3]).magnitude())
    raise ValueError(f"Expected residual action with 6 or 9 dims, got {residual_pred.shape}")


def write_chunk_local_step_error_png(path, demo_name, step_ids, slow_err, fast_err):
    local_steps = np.asarray(sorted(np.unique(step_ids)), dtype=np.int64)
    slow_pos, slow_rot = slow_err
    fast_pos, fast_rot = fast_err

    slow_pos_mean = [np.mean(slow_pos[step_ids == step]) for step in local_steps]
    fast_pos_mean = [np.mean(fast_pos[step_ids == step]) for step in local_steps]
    slow_rot_mean = [np.mean(slow_rot[step_ids == step]) for step in local_steps]
    fast_rot_mean = [np.mean(fast_rot[step_ids == step]) for step in local_steps]

    x = np.arange(len(local_steps))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x - width / 2, slow_pos_mean, width, color=COLORS["slow_pred_actual"], label="slow vs GT actual")
    axes[0].bar(x + width / 2, fast_pos_mean, width, color=COLORS["fast_pred_virtual"], label="fast vs GT virtual")
    axes[0].set_ylabel("position error (mm)")
    axes[1].bar(x - width / 2, slow_rot_mean, width, color=COLORS["slow_pred_actual"], label="slow vs GT actual")
    axes[1].bar(x + width / 2, fast_rot_mean, width, color=COLORS["fast_pred_virtual"], label="fast vs GT virtual")
    axes[1].set_ylabel("rotation error (deg)")
    axes[1].set_xlabel("local step inside accepted chunk")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(step)) for step in local_steps])
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle(f"{demo_name}: chunk local-step error")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_window_step_error_png(path, demo_name, anchor, gt_indices, slow_err, fast_err):
    local_steps = np.arange(len(gt_indices), dtype=np.int64)
    slow_pos, slow_rot = slow_err
    fast_pos, fast_rot = fast_err

    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(
        local_steps - width / 2,
        slow_pos,
        width,
        color=COLORS["slow_pred_actual"],
        label="slow vs GT actual",
    )
    axes[0].bar(
        local_steps + width / 2,
        fast_pos,
        width,
        color=COLORS["fast_pred_virtual"],
        label="fast vs GT virtual",
    )
    axes[0].set_ylabel("position error (mm)")
    axes[1].bar(
        local_steps - width / 2,
        slow_rot,
        width,
        color=COLORS["slow_pred_actual"],
        label="slow vs GT actual",
    )
    axes[1].bar(
        local_steps + width / 2,
        fast_rot,
        width,
        color=COLORS["fast_pred_virtual"],
        label="fast vs GT virtual",
    )
    axes[1].set_ylabel("rotation error (deg)")
    axes[1].set_xlabel("local step inside 16-step slow prediction")
    axes[1].set_xticks(local_steps)
    axes[1].set_xticklabels([str(int(step)) for step in local_steps])
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle(f"{demo_name} anchor={anchor}: 16-step local error")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def predict_demo(
        demo_group,
        slow_policy,
        fast_policy,
        device,
        arm,
        slow_action_pose_repr,
        slow_action_index,
        stride,
        max_steps,
        batch_size,
        virtual_position_scale,
        fast_action_target_shift):
    obs_group = demo_group["obs"]
    length = len(demo_group["actions"])
    fast_action_target_shift = int(fast_action_target_shift)
    indices = np.arange(0, max(0, length - fast_action_target_shift), stride, dtype=np.int64)
    if max_steps is not None:
        indices = indices[:max_steps]
    target_indices = indices + fast_action_target_shift

    gt_actual = pose_like_to_pose9(np.asarray(obs_group["actual_target_abs"])[target_indices])
    gt_virtual = pose_like_to_pose9(np.asarray(obs_group["virtual_target_abs"])[target_indices])

    slow_pred = []
    fast_pred = []
    residual_pred = []
    base_action_rel = []

    desc = f"{demo_group.name.split('/')[-1]} prediction"
    for start in tqdm(range(0, len(indices), batch_size), desc=desc):
        batch_indices = indices[start:start + batch_size]
        slow_obs = collate_policy_obs([
            build_policy_obs(slow_policy, obs_group, int(t), device)
            for t in batch_indices
        ])
        slow_result = slow_policy.predict_action(slow_obs)
        target_action_index = slow_action_index + fast_action_target_shift
        if target_action_index >= slow_result["action"].shape[1]:
            raise IndexError(
                f"slow_action_index + fast_action_target_shift = {target_action_index} "
                f"but slow action length is {slow_result['action'].shape[1]}"
            )
        slow_actions = slow_result["action"][:, target_action_index].detach().cpu().numpy()

        batch_slow_abs = []
        for t, slow_action in zip(batch_indices, slow_actions):
            current_pose9 = current_pose9_for_step(obs_group, int(t), arm)
            slow_abs, _ = slow_action_to_abs_and_rel(
                slow_action,
                current_pose9,
                slow_action_pose_repr,
            )
            batch_slow_abs.append(slow_abs)

        batch_base_rel = []
        for t, slow_abs in zip(batch_indices, batch_slow_abs):
            current_pose9 = current_pose9_for_step(obs_group, int(t), arm)
            batch_base_rel.append(abs_pose9_to_relative_pose9(current_pose9, slow_abs))

        fast_obs = collate_policy_obs([
            build_policy_obs(
                fast_policy,
                obs_group,
                int(t),
                device,
                base_action_rel=base_rel,
            )
            for t, base_rel in zip(batch_indices, batch_base_rel)
        ])
        fast_result = fast_policy.predict_action(fast_obs)
        batch_residual = fast_result["action"][:, 0].detach().cpu().numpy()

        for slow_abs, base_rel, this_residual in zip(batch_slow_abs, batch_base_rel, batch_residual):
            this_fast_abs = apply_residual_action_to_pose9(slow_abs, this_residual)
            slow_pred.append(slow_abs)
            fast_pred.append(this_fast_abs)
            residual_pred.append(this_residual)
            base_action_rel.append(base_rel)

    slow_pred = np.asarray(slow_pred, dtype=np.float32)
    fast_pred = np.asarray(fast_pred, dtype=np.float32)
    residual_pred = np.asarray(residual_pred, dtype=np.float32)
    base_action_rel = np.asarray(base_action_rel, dtype=np.float32)

    resolved_virtual_scale = resolve_virtual_position_scale(
        virtual_position_scale,
        gt_actual,
        gt_virtual,
    )
    gt_virtual_plot = scale_pose_position(gt_virtual, resolved_virtual_scale)
    fast_pred_plot = scale_pose_position(fast_pred, resolved_virtual_scale)

    slow_err = pose_errors(slow_pred, gt_actual)
    fast_err = pose_errors(fast_pred_plot, gt_virtual_plot)
    metrics = {
        "num_steps": int(len(indices)),
        "stride": int(stride),
        "virtual_position_scale": float(resolved_virtual_scale),
        "slow_vs_gt_actual": summarize_errors(*slow_err),
        "fast_vs_gt_virtual": summarize_errors(*fast_err),
    }

    return {
        "indices": indices,
        "target_indices": target_indices,
        "gt_actual": gt_actual,
        "gt_virtual": gt_virtual,
        "gt_virtual_plot": gt_virtual_plot,
        "slow_pred_actual": slow_pred,
        "fast_pred_virtual": fast_pred,
        "fast_pred_virtual_plot": fast_pred_plot,
        "base_action_rel": base_action_rel,
        "residual_pred": residual_pred,
        "slow_err": slow_err,
        "fast_err": fast_err,
        "metrics": metrics,
    }


@torch.no_grad()
def predict_windows(
        demo_group,
        slow_policy,
        fast_policy,
        device,
        arm,
        slow_action_pose_repr,
        anchors,
        virtual_position_scale,
        fast_action_target_shift):
    obs_group = demo_group["obs"]
    length = len(demo_group["actions"])
    n_obs_steps = int(getattr(slow_policy, "n_obs_steps", 1))
    fast_action_target_shift = int(fast_action_target_shift)
    out = []

    for anchor in tqdm(anchors, desc=f"{demo_group.name.split('/')[-1]} windows"):
        anchor = int(anchor)
        if anchor >= length:
            continue
        slow_obs = build_policy_obs(slow_policy, obs_group, anchor, device)
        slow_result = slow_policy.predict_action(slow_obs)
        raw_actions = slow_result["action_pred"][0].detach().cpu().numpy()
        horizon = raw_actions.shape[0]

        gt_start = max(0, anchor - (n_obs_steps - 1))
        gt_end = min(gt_start + horizon, length)
        raw_actions = raw_actions[:gt_end - gt_start]
        gt_indices = np.arange(gt_start, gt_end, dtype=np.int64)

        current_pose9 = current_pose9_for_step(obs_group, anchor, arm)
        slow_abs = []
        slow_rel = []
        for action in raw_actions:
            this_abs, this_rel = slow_action_to_abs_and_rel(
                action,
                current_pose9,
                slow_action_pose_repr,
            )
            slow_abs.append(this_abs)
            slow_rel.append(this_rel)
        slow_abs = np.asarray(slow_abs, dtype=np.float32)
        slow_rel = np.asarray(slow_rel, dtype=np.float32)

        target_local_indices = np.arange(
            fast_action_target_shift,
            len(slow_abs),
            dtype=np.int64,
        )
        if len(target_local_indices) == 0:
            continue
        input_local_indices = target_local_indices - fast_action_target_shift
        input_gt_indices = gt_indices[input_local_indices]
        target_gt_indices = gt_indices[target_local_indices]
        target_slow_abs = slow_abs[target_local_indices]

        fast_rel = []
        fast_obs_list = []
        for input_gt_idx, this_slow_abs in zip(input_gt_indices, target_slow_abs):
            fast_current_pose9 = current_pose9_for_step(obs_group, int(input_gt_idx), arm)
            this_fast_rel = abs_pose9_to_relative_pose9(fast_current_pose9, this_slow_abs)
            fast_rel.append(this_fast_rel)
            fast_obs_list.append(build_policy_obs(
                fast_policy,
                obs_group,
                int(input_gt_idx),
                device,
                base_action_rel=this_fast_rel,
            ))

        fast_rel = np.asarray(fast_rel, dtype=np.float32)

        if getattr(fast_policy, "uses_fixed_context_sequence", False):
            fast_obs = build_context_sequence_obs(
                fast_policy,
                obs_group,
                context_t=anchor,
                step_indices=input_gt_indices,
                device=device,
                base_action_rel_seq=fast_rel,
            )
            fast_result = fast_policy.predict_action(fast_obs)
            residual = fast_result.get("action_sequence", fast_result["action"])[0].detach().cpu().numpy()
        elif hasattr(fast_policy, "predict_step"):
            residual = []
            temporal_hidden = None
            for input_gt_idx, this_fast_rel in zip(input_gt_indices, fast_rel):
                fast_obs = build_policy_obs(
                    fast_policy,
                    obs_group,
                    int(input_gt_idx),
                    device,
                    base_action_rel=this_fast_rel,
                    n_obs_steps_override=1,
                )
                fast_result = fast_policy.predict_step(fast_obs, hidden=temporal_hidden)
                temporal_hidden = fast_result["hidden"]
                residual.append(fast_result["action"][0, 0].detach().cpu().numpy())
            residual = np.asarray(residual, dtype=np.float32)
        else:
            fast_obs = collate_policy_obs(fast_obs_list)
            fast_result = fast_policy.predict_action(fast_obs)
            residual = fast_result["action"][:, 0].detach().cpu().numpy()

        fast_abs = np.asarray([
            apply_residual_action_to_pose9(this_slow_abs, this_residual)
            for this_slow_abs, this_residual in zip(target_slow_abs, residual)
        ], dtype=np.float32)

        gt_actual = pose_like_to_pose9(np.asarray(obs_group["actual_target_abs"])[target_gt_indices])
        gt_virtual = pose_like_to_pose9(np.asarray(obs_group["virtual_target_abs"])[target_gt_indices])
        resolved_virtual_scale = resolve_virtual_position_scale(
            virtual_position_scale,
            gt_actual,
            gt_virtual,
        )
        gt_virtual_plot = scale_pose_position(gt_virtual, resolved_virtual_scale)
        fast_abs_plot = scale_pose_position(fast_abs, resolved_virtual_scale)

        slow_err = pose_errors(target_slow_abs, gt_actual)
        fast_err = pose_errors(fast_abs_plot, gt_virtual_plot)
        out.append({
            "anchor": anchor,
            "input_indices": input_gt_indices,
            "gt_indices": target_gt_indices,
            "gt_actual": gt_actual,
            "gt_virtual": gt_virtual,
            "gt_virtual_plot": gt_virtual_plot,
            "slow_pred_actual": target_slow_abs,
            "fast_pred_virtual": fast_abs,
            "fast_pred_virtual_plot": fast_abs_plot,
            "residual_pred": residual,
            "base_action_rel": fast_rel,
            "virtual_position_scale": resolved_virtual_scale,
            "slow_err": slow_err,
            "fast_err": fast_err,
        })
    return out


@torch.no_grad()
def predict_chunked_receding(
        demo_group,
        slow_policy,
        fast_policy,
        device,
        arm,
        slow_action_pose_repr,
        start,
        total_steps,
        exec_steps,
        virtual_position_scale,
        fast_action_target_shift):
    obs_group = demo_group["obs"]
    length = len(demo_group["actions"])
    n_obs_steps = int(getattr(slow_policy, "n_obs_steps", 1))
    raw_start = max(0, n_obs_steps - 1)
    fast_action_target_shift = int(fast_action_target_shift)
    if total_steps is None or int(total_steps) <= 0:
        total_steps = max(0, length - start - fast_action_target_shift)
    else:
        total_steps = int(total_steps)

    gt_indices = []
    input_indices = []
    chunk_ids = []
    step_ids = []
    slow_pred = []
    fast_pred = []
    residual_pred = []
    base_action_rel = []

    anchors = range(start, min(start + total_steps, length), exec_steps)
    for chunk_id, anchor in enumerate(tqdm(list(anchors), desc=f"{demo_group.name.split('/')[-1]} chunked")):
        if anchor >= length:
            break
        slow_obs = build_policy_obs(slow_policy, obs_group, int(anchor), device)
        slow_result = slow_policy.predict_action(slow_obs)
        raw_actions = slow_result["action_pred"][0].detach().cpu().numpy()
        action_chunk = raw_actions[
            raw_start + fast_action_target_shift:
            raw_start + fast_action_target_shift + exec_steps
        ]
        max_available = max(0, min(
            len(action_chunk),
            length - anchor - fast_action_target_shift,
            start + total_steps - anchor,
        ))
        action_chunk = action_chunk[:max_available]
        if len(action_chunk) == 0:
            continue

        anchor_pose9 = current_pose9_for_step(obs_group, int(anchor), arm)
        chunk_slow_abs = []
        for action in action_chunk:
            this_abs, _ = slow_action_to_abs_and_rel(
                action,
                anchor_pose9,
                slow_action_pose_repr,
            )
            chunk_slow_abs.append(this_abs)

        chunk_slow_rel = []
        fast_obs_list = []
        for local_i, this_slow_abs in enumerate(chunk_slow_abs):
            input_gt_idx = int(anchor + local_i)
            fast_current_pose9 = current_pose9_for_step(obs_group, input_gt_idx, arm)
            this_slow_rel = abs_pose9_to_relative_pose9(fast_current_pose9, this_slow_abs)
            chunk_slow_rel.append(this_slow_rel)
            fast_obs_list.append(build_policy_obs(
                fast_policy,
                obs_group,
                input_gt_idx,
                device,
                base_action_rel=this_slow_rel,
            ))

        if getattr(fast_policy, "uses_fixed_context_sequence", False):
            step_indices = np.asarray(
                [int(anchor + local_i) for local_i in range(len(chunk_slow_rel))],
                dtype=np.int64,
            )
            fast_obs = build_context_sequence_obs(
                fast_policy,
                obs_group,
                context_t=anchor,
                step_indices=step_indices,
                device=device,
                base_action_rel_seq=np.asarray(chunk_slow_rel, dtype=np.float32),
            )
            fast_result = fast_policy.predict_action(fast_obs)
            chunk_residual = fast_result.get("action_sequence", fast_result["action"])[0].detach().cpu().numpy()
        elif hasattr(fast_policy, "predict_step"):
            chunk_residual = []
            temporal_hidden = None
            for local_i, this_slow_rel in enumerate(chunk_slow_rel):
                input_gt_idx = int(anchor + local_i)
                fast_obs = build_policy_obs(
                    fast_policy,
                    obs_group,
                    input_gt_idx,
                    device,
                    base_action_rel=this_slow_rel,
                    n_obs_steps_override=1,
                )
                fast_result = fast_policy.predict_step(fast_obs, hidden=temporal_hidden)
                temporal_hidden = fast_result["hidden"]
                chunk_residual.append(fast_result["action"][0, 0].detach().cpu().numpy())
            chunk_residual = np.asarray(chunk_residual, dtype=np.float32)
        else:
            fast_obs = collate_policy_obs(fast_obs_list)
            fast_result = fast_policy.predict_action(fast_obs)
            chunk_residual = fast_result["action"][:, 0].detach().cpu().numpy()
        chunk_fast_abs = [
            apply_residual_action_to_pose9(this_slow_abs, this_residual)
            for this_slow_abs, this_residual in zip(chunk_slow_abs, chunk_residual)
        ]

        for local_i, (this_slow_abs, this_slow_rel, this_fast_abs, this_residual) in enumerate(
                zip(chunk_slow_abs, chunk_slow_rel, chunk_fast_abs, chunk_residual)):
            input_gt_idx = int(anchor + local_i)
            gt_idx = int(anchor + fast_action_target_shift + local_i)
            input_indices.append(input_gt_idx)
            gt_indices.append(gt_idx)
            chunk_ids.append(chunk_id)
            step_ids.append(local_i)
            slow_pred.append(this_slow_abs)
            fast_pred.append(this_fast_abs)
            residual_pred.append(this_residual)
            base_action_rel.append(this_slow_rel)

    gt_indices = np.asarray(gt_indices, dtype=np.int64)
    input_indices = np.asarray(input_indices, dtype=np.int64)
    chunk_ids = np.asarray(chunk_ids, dtype=np.int64)
    step_ids = np.asarray(step_ids, dtype=np.int64)
    slow_pred = np.asarray(slow_pred, dtype=np.float32)
    fast_pred = np.asarray(fast_pred, dtype=np.float32)
    residual_pred = np.asarray(residual_pred, dtype=np.float32)
    base_action_rel = np.asarray(base_action_rel, dtype=np.float32)

    gt_actual = pose_like_to_pose9(np.asarray(obs_group["actual_target_abs"])[gt_indices])
    gt_virtual = pose_like_to_pose9(np.asarray(obs_group["virtual_target_abs"])[gt_indices])
    resolved_virtual_scale = resolve_virtual_position_scale(
        virtual_position_scale,
        gt_actual,
        gt_virtual,
    )
    gt_virtual_plot = scale_pose_position(gt_virtual, resolved_virtual_scale)
    fast_pred_plot = scale_pose_position(fast_pred, resolved_virtual_scale)

    slow_err = pose_errors(slow_pred, gt_actual)
    fast_err = pose_errors(fast_pred_plot, gt_virtual_plot)
    residual_pos_norm_mm = np.linalg.norm(residual_pred[:, :3], axis=-1) * 1000.0
    residual_rot_norm_deg = residual_rotation_norm_deg(residual_pred)
    metrics = {
        "start": int(start),
        "total_steps": int(total_steps),
        "exec_steps": int(exec_steps),
        "raw_action_start_index": int(raw_start),
        "fast_action_target_shift": int(fast_action_target_shift),
        "num_points": int(len(gt_indices)),
        "virtual_position_scale": float(resolved_virtual_scale),
        "slow_vs_gt_actual": summarize_errors(*slow_err),
        "fast_vs_gt_virtual": summarize_errors(*fast_err),
        "per_exec_step": summarize_by_local_step(step_ids, slow_err, fast_err),
        "predicted_residual": {
            "translation_norm_mm": summarize_norms(residual_pos_norm_mm),
            "rotation_norm_deg": summarize_norms(residual_rot_norm_deg),
        },
    }

    return {
        "input_indices": input_indices,
        "gt_indices": gt_indices,
        "chunk_ids": chunk_ids,
        "step_ids": step_ids,
        "gt_actual": gt_actual,
        "gt_virtual": gt_virtual,
        "gt_virtual_plot": gt_virtual_plot,
        "slow_pred_actual": slow_pred,
        "fast_pred_virtual": fast_pred,
        "fast_pred_virtual_plot": fast_pred_plot,
        "base_action_rel": base_action_rel,
        "residual_pred": residual_pred,
        "slow_err": slow_err,
        "fast_err": fast_err,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GT actual/virtual and slow/fast residual predictions on a converted residual dataset."
    )
    parser.add_argument("--dataset", required=True, help="Converted slow/residual HDF5 dataset.")
    parser.add_argument("--slow-ckpt", required=True)
    parser.add_argument("--fast-ckpt", required=True)
    parser.add_argument("--output-dir", default="plots/residual_step_predictions")
    parser.add_argument("--save-npz", action="store_true", default=False)
    parser.add_argument("--demo", action="append", default=None, help="Demo name. Can be passed multiple times.")
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--virtual-position-scale",
        default="auto",
        help="Scale applied to virtual/fast predicted xyz for plotting and metric calculation. Use auto, 1.0, or 0.001.",
    )
    parser.add_argument("--slow-action-index", type=int, default=0)
    parser.add_argument("--arm", default="R", choices=["L", "R"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slow-use-ema", action="store_true", default=True)
    parser.add_argument("--no-slow-ema", dest="slow_use_ema", action="store_false")
    parser.add_argument("--fast-use-ema", action="store_true", default=True)
    parser.add_argument("--no-fast-ema", dest="fast_use_ema", action="store_false")
    parser.add_argument("--continuous-plots", action="store_true", default=False)
    parser.add_argument("--no-continuous-plots", dest="continuous_plots", action="store_false")
    parser.add_argument("--window-plots", action="store_true", default=True)
    parser.add_argument("--no-window-plots", dest="window_plots", action="store_false")
    parser.add_argument("--window-count", type=int, default=6)
    parser.add_argument("--window-step", type=int, default=20)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument(
        "--window-starts",
        default=None,
        help="Comma-separated explicit anchors for 16-step window plots. Overrides window-start/count/step.",
    )
    parser.add_argument("--chunked-plots", action="store_true", default=True)
    parser.add_argument("--no-chunked-plots", dest="chunked_plots", action="store_false")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument(
        "--chunk-starts",
        default=None,
        help="Comma-separated explicit starts for receding chunk plots. Overrides chunk-start.",
    )
    parser.add_argument("--chunk-total-steps", type=int, default=80)
    parser.add_argument("--chunk-exec-steps", type=int, default=8)
    parser.add_argument(
        "--fast-action-target-shift",
        type=int,
        default=None,
        help="Fast residual target shift. Defaults to task.dataset.action_target_shift from the fast checkpoint.",
    )
    parser.add_argument(
        "--organized-output",
        action="store_true",
        default=False,
        help="Write continuous/window/chunked plots into separate subfolders.",
    )
    parser.add_argument(
        "--world-frame",
        action="store_true",
        default=False,
        help="Plot xyz in the right-arm world frame using the existing RIGHT_ROBOT_TO_WORLD rotation.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    continuous_dir = output_dir
    window_dir = output_dir
    chunked_dir = output_dir
    if args.organized_output:
        continuous_dir = output_dir / "continuous"
        window_dir = output_dir / "window16"
        chunked_suffix = "all" if args.chunk_total_steps <= 0 else str(args.chunk_total_steps)
        chunked_dir = output_dir / f"chunked_{chunked_suffix}"
        if args.continuous_plots:
            continuous_dir.mkdir(parents=True, exist_ok=True)
        if args.window_plots:
            window_dir.mkdir(parents=True, exist_ok=True)
        if args.chunked_plots:
            chunked_dir.mkdir(parents=True, exist_ok=True)

    def plot_pose(pose9):
        if args.world_frame:
            return right_robot_pose_to_world(pose9)
        return pose9

    frame_label = "world frame" if args.world_frame else "dataset robot frame"

    slow_cfg, slow_policy = load_policy_from_ckpt(args.slow_ckpt, device=device, use_ema=args.slow_use_ema)
    fast_cfg, fast_policy = load_policy_from_ckpt(args.fast_ckpt, device=device, use_ema=args.fast_use_ema)

    slow_action_pose_repr = OmegaConf.select(
        slow_cfg,
        "task.pose_repr.action_pose_repr",
        default="relative",
    )
    print(f"slow action_pose_repr: {slow_action_pose_repr}")
    fast_action_target_shift = args.fast_action_target_shift
    if fast_action_target_shift is None:
        fast_action_target_shift = OmegaConf.select(
            fast_cfg,
            "task.dataset.action_target_shift",
            default=0,
        )
    fast_action_target_shift = int(fast_action_target_shift)
    if fast_action_target_shift < 0:
        raise ValueError(f"fast_action_target_shift must be >= 0, got {fast_action_target_shift}")
    print(f"fast action_target_shift: {fast_action_target_shift}")

    all_metrics = {}
    with h5py.File(args.dataset, "r") as f:
        demo_names = args.demo
        if demo_names is None:
            demo_names = sorted_demo_keys(f["data"])[:args.num_demos]

        for demo_name in demo_names:
            if demo_name not in f["data"]:
                raise KeyError(f"{demo_name} not found in {args.dataset}")
            print(f"Predicting {demo_name}")
            slug = demo_name.replace("/", "_")
            all_metrics[demo_name] = {}

            if args.continuous_plots:
                result = predict_demo(
                    demo_group=f["data"][demo_name],
                    slow_policy=slow_policy,
                    fast_policy=fast_policy,
                    device=device,
                    arm=args.arm,
                    slow_action_pose_repr=slow_action_pose_repr,
                    slow_action_index=args.slow_action_index,
                    stride=args.stride,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    virtual_position_scale=args.virtual_position_scale,
                    fast_action_target_shift=fast_action_target_shift,
                )
                html_path = continuous_dir / f"{slug}_continuous_gt_slow_fast_3d.html"
                png_path = continuous_dir / f"{slug}_continuous_prediction_errors.png"

                write_trajectory_html(
                    html_path,
                    f"{demo_name} ({frame_label})",
                    plot_pose(result["gt_actual"]),
                    plot_pose(result["gt_virtual_plot"]),
                    plot_pose(result["slow_pred_actual"]),
                    plot_pose(result["fast_pred_virtual_plot"]),
                )
                write_error_png(
                    png_path,
                    demo_name,
                    result["slow_err"],
                    result["fast_err"],
                )
                if args.save_npz:
                    np.savez_compressed(
                        continuous_dir / f"{slug}_continuous.npz",
                        indices=result["indices"],
                        target_indices=result["target_indices"],
                        gt_actual=result["gt_actual"],
                        gt_virtual=result["gt_virtual"],
                        gt_virtual_plot=result["gt_virtual_plot"],
                        slow_pred_actual=result["slow_pred_actual"],
                        fast_pred_virtual=result["fast_pred_virtual"],
                        fast_pred_virtual_plot=result["fast_pred_virtual_plot"],
                        base_action_rel=result["base_action_rel"],
                        residual_pred=result["residual_pred"],
                    )
                all_metrics[demo_name]["continuous"] = result["metrics"]
                print(f"  wrote {html_path}")
                print(f"  wrote {png_path}")

            if args.window_plots:
                explicit_window_starts = parse_int_list(args.window_starts)
                if explicit_window_starts is None:
                    anchors = args.window_start + np.arange(args.window_count) * args.window_step
                else:
                    anchors = np.asarray(explicit_window_starts, dtype=np.int64)
                windows = predict_windows(
                    demo_group=f["data"][demo_name],
                    slow_policy=slow_policy,
                    fast_policy=fast_policy,
                    device=device,
                    arm=args.arm,
                    slow_action_pose_repr=slow_action_pose_repr,
                    anchors=anchors,
                    virtual_position_scale=args.virtual_position_scale,
                    fast_action_target_shift=fast_action_target_shift,
                )
                window_metrics = {}
                for window in windows:
                    anchor = window["anchor"]
                    window_path = window_dir / f"{slug}_w{anchor:04d}.html"
                    window_error_path = window_dir / f"{slug}_w{anchor:04d}_steps.png"
                    write_window_html(
                        window_path,
                        (
                            f"{demo_name} anchor={anchor}: 16-step GT and slow->fast residual corrections ({frame_label})"
                        ),
                        plot_pose(window["gt_actual"]),
                        plot_pose(window["gt_virtual_plot"]),
                        plot_pose(window["slow_pred_actual"]),
                        plot_pose(window["fast_pred_virtual_plot"]),
                    )
                    write_window_step_error_png(
                        window_error_path,
                        demo_name,
                        anchor,
                        window["gt_indices"],
                        window["slow_err"],
                        window["fast_err"],
                    )
                    if args.save_npz:
                        np.savez_compressed(
                            window_dir / f"{slug}_w{anchor:04d}.npz",
                            gt_indices=window["gt_indices"],
                            input_indices=window["input_indices"],
                            gt_actual=window["gt_actual"],
                            gt_virtual=window["gt_virtual"],
                            gt_virtual_plot=window["gt_virtual_plot"],
                            slow_pred_actual=window["slow_pred_actual"],
                            fast_pred_virtual=window["fast_pred_virtual"],
                            fast_pred_virtual_plot=window["fast_pred_virtual_plot"],
                            base_action_rel=window["base_action_rel"],
                            residual_pred=window["residual_pred"],
                        )
                    slow_summary = summarize_errors(*window["slow_err"])
                    fast_summary = summarize_errors(*window["fast_err"])
                    window_metrics[str(anchor)] = {
                        "gt_start": int(window["gt_indices"][0]),
                        "gt_end": int(window["gt_indices"][-1]),
                        "virtual_position_scale": float(window["virtual_position_scale"]),
                        "slow_vs_gt_actual": slow_summary,
                        "fast_vs_gt_virtual": fast_summary,
                    }
                    print(f"  wrote {window_path}")
                    print(f"  wrote {window_error_path}")
                all_metrics[demo_name]["windows"] = window_metrics

            if args.chunked_plots:
                chunk_starts = parse_int_list(args.chunk_starts)
                if chunk_starts is None:
                    chunk_starts = [args.chunk_start]
                chunk_metrics = {}
                for chunk_start in chunk_starts:
                    chunked = predict_chunked_receding(
                        demo_group=f["data"][demo_name],
                        slow_policy=slow_policy,
                        fast_policy=fast_policy,
                        device=device,
                        arm=args.arm,
                        slow_action_pose_repr=slow_action_pose_repr,
                        start=chunk_start,
                        total_steps=args.chunk_total_steps,
                        exec_steps=args.chunk_exec_steps,
                        virtual_position_scale=args.virtual_position_scale,
                        fast_action_target_shift=fast_action_target_shift,
                    )
                    prefix = f"{slug}_c{chunk_start:04d}"
                    chunked_path = chunked_dir / f"{prefix}.html"
                    write_chunked_html(
                        chunked_path,
                        (
                            f"{demo_name} start={chunk_start}: receding chunks, slow 16-step -> accept first "
                            f"{args.chunk_exec_steps} residual-corrected points ({frame_label})"
                        ),
                        plot_pose(chunked["gt_actual"]),
                        plot_pose(chunked["gt_virtual_plot"]),
                        plot_pose(chunked["slow_pred_actual"]),
                        plot_pose(chunked["fast_pred_virtual_plot"]),
                        chunked["chunk_ids"],
                        chunked["step_ids"],
                    )
                    if args.save_npz:
                        np.savez_compressed(
                            chunked_dir / f"{prefix}.npz",
                            gt_indices=chunked["gt_indices"],
                            input_indices=chunked["input_indices"],
                            chunk_ids=chunked["chunk_ids"],
                            step_ids=chunked["step_ids"],
                            gt_actual=chunked["gt_actual"],
                            gt_virtual=chunked["gt_virtual"],
                            gt_virtual_plot=chunked["gt_virtual_plot"],
                            slow_pred_actual=chunked["slow_pred_actual"],
                            fast_pred_virtual=chunked["fast_pred_virtual"],
                            fast_pred_virtual_plot=chunked["fast_pred_virtual_plot"],
                            base_action_rel=chunked["base_action_rel"],
                            residual_pred=chunked["residual_pred"],
                        )
                    chunk_error_path = chunked_dir / f"{prefix}_steps.png"
                    write_chunk_local_step_error_png(
                        chunk_error_path,
                        demo_name,
                        chunked["step_ids"],
                        chunked["slow_err"],
                        chunked["fast_err"],
                    )
                    chunk_metrics[str(chunk_start)] = chunked["metrics"]
                    print(f"  wrote {chunked_path}")
                    print(f"  wrote {chunk_error_path}")
                all_metrics[demo_name]["chunked"] = chunk_metrics

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(metrics_path)


if __name__ == "__main__":
    main()
