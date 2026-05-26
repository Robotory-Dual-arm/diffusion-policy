if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import copy
import csv
import gc
import math
import pathlib
import re
from collections import OrderedDict

import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.pose_util import pose10d_to_mat
from diffusion_policy.real_world.real_inference_util import get_abs_action_from_relative


OmegaConf.register_new_resolver("eval", eval, replace=True)

MODEL_COLORS = {
    "GT": "#111111",
    "Transformer EMA eval": "#1f77b4",
    "Transformer raw model": "#17becf",
    "Transformer v-pred EMA steps=16": "#9467bd",
    "Transformer epsilon EMA steps=16": "#ff7f0e",
    "Transformer EMA eval steps=16": "#9467bd",
    "Transformer raw model steps=16": "#8c564b",
    "Transformer EMA eval steps=32": "#ff7f0e",
    "Transformer raw model steps=32": "#bcbd22",
    "Transformer DiT v-pred e900 EMA steps=16": "#2ca02c",
    "Transformer non-DiT v-pred e900 EMA steps=16": "#9467bd",
    "UNet EMA eval": "#d62728",
}

STEP_MODEL_COLORS = {
    4: "#1f77b4",
    8: "#2ca02c",
    12: "#ff7f0e",
    16: "#9467bd",
    32: "#8c564b",
    64: "#17becf",
}

FALLBACK_MODEL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#e377c2",
]


def preferred_model_color(name):
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    match = re.search(r"steps=(\d+)", name)
    if match is not None:
        return STEP_MODEL_COLORS.get(int(match.group(1)))
    return None


def model_colors_for(model_names):
    colors = {}
    used = {MODEL_COLORS["GT"]}
    fallback_idx = 0
    for name in model_names:
        color = preferred_model_color(name)
        if color is None or color in used:
            while FALLBACK_MODEL_COLORS[fallback_idx % len(FALLBACK_MODEL_COLORS)] in used:
                fallback_idx += 1
            color = FALLBACK_MODEL_COLORS[fallback_idx % len(FALLBACK_MODEL_COLORS)]
            fallback_idx += 1
        colors[name] = color
        used.add(color)
    return colors


DEFAULT_TRANSFORMER_CKPT = (
    "data/outputs/2026.05.05/21.35.58_train_diffusion_transformer_hybrid_"
    "bbbae_dualarm_insert_plug_no_wrench/checkpoints/epoch=1300-train_loss=0.002.ckpt"
)
DEFAULT_UNET_CKPT = (
    "data/outputs/2026.05.02/06.39.56_train_diffusion_unet_hybrid_"
    "bbbae_dualarm_insert_plug_no_wrench/checkpoints/epoch=0900-train_loss=0.000.ckpt"
)


def resolve_path(path):
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        p = pathlib.Path.cwd() / p
    return p.resolve()


def slugify_ckpt(path):
    p = pathlib.Path(path)
    parts = list(p.parts)
    try:
        idx = parts.index("outputs")
        useful = parts[idx + 1:]
    except ValueError:
        useful = parts[-4:]
    stem = "_".join(useful)
    stem = stem.replace(".ckpt", "")
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem.lower()


def run_dir_from_ckpt(ckpt_path):
    ckpt_path = pathlib.Path(ckpt_path)
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def load_cfg_for_ckpt(ckpt_path):
    run_dir = run_dir_from_ckpt(ckpt_path)
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg.is_file():
        return OmegaConf.load(hydra_cfg)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    del payload
    gc.collect()
    return cfg


def make_dataset_cfg(dataset_source_cfg):
    dataset_dict = OmegaConf.to_container(dataset_source_cfg.task.dataset, resolve=True)
    dataset_dict["dataset_path"] = str(resolve_path(dataset_dict["dataset_path"]))
    dataset_dict["use_cache"] = True
    return OmegaConf.create(dataset_dict)


def action_pose_repr_from_cfg(cfg):
    return OmegaConf.to_container(cfg.task.dataset.pose_repr, resolve=True)["action_pose_repr"]


def load_policy_from_checkpoint(ckpt_path, device, strict=True):
    ckpt_path = resolve_path(ckpt_path)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    for key, state_dict in payload["state_dicts"].items():
        if key not in workspace.__dict__ or not hasattr(workspace.__dict__[key], "state_dict"):
            continue
        target_state = workspace.__dict__[key].state_dict()
        for param_name, value in list(state_dict.items()):
            if (
                    param_name.endswith("position_embedding")
                    and param_name in target_state
                    and tuple(value.shape) != tuple(target_state[param_name].shape)):
                resized = target_state[param_name].clone()
                slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, resized.shape))
                resized[slices] = value[slices]
                state_dict[param_name] = resized
    workspace.load_payload(
        payload=payload,
        exclude_keys=("optimizer",),
        include_keys=[],
        strict=strict,
    )
    del payload
    if hasattr(workspace, "optimizer"):
        workspace.optimizer = None
    workspace.model.to(device)
    workspace.model.eval()
    if workspace.ema_model is not None:
        workspace.ema_model.to(device)
        workspace.ema_model.eval()
    gc.collect()
    return workspace


def obs_to_numpy(obs):
    return {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in obs.items()
    }


def action_to_abs(action, env_obs, action_pose_repr):
    if action_pose_repr == "relative":
        return get_abs_action_from_relative(action, env_obs)
    return action


def collate_obs(samples, keys, device):
    obs = {}
    for key in keys:
        obs[key] = torch.stack([sample["obs"][key] for sample in samples], dim=0).to(device)
    return obs


def adapt_obs_for_policy(obs, policy):
    """Handle compatible datasets that store wrench with different history shapes."""
    params_dict = getattr(policy.normalizer, "params_dict", {})
    n_obs_steps = getattr(policy, "n_obs_steps", None)
    low_dim_keys = set(getattr(policy, "low_dim_keys", []))
    for key, value in list(obs.items()):
        if key not in params_dict or key == "action":
            continue
        if value.ndim >= 5:
            continue
        scale = params_dict[key]["scale"]
        expected_dim = int(scale.shape[0])
        actual_dim = int(np.prod(value.shape[2:])) if value.ndim > 2 else 1
        if actual_dim == expected_dim:
            continue

        # Wrench-encoder datasets store one observation as (C, history).
        # Older low-dim wrench policies expect the latest wrench vector (C).
        if value.ndim == 4 and value.shape[2] == expected_dim:
            value = value[..., -1]
        elif value.ndim == 4 and value.shape[2] * value.shape[3] == expected_dim:
            value = value.reshape(value.shape[0], value.shape[1], expected_dim)
        else:
            raise RuntimeError(
                f"Cannot adapt obs key {key}: got shape {tuple(value.shape)}, "
                f"expected trailing dim {expected_dim}"
            )

        if key in low_dim_keys and n_obs_steps is not None and value.shape[1] < n_obs_steps:
            pad = value[:, -1:].repeat(1, n_obs_steps - value.shape[1], *([1] * (value.ndim - 2)))
            value = torch.cat([pad, value], dim=1)
        obs[key] = value
    return obs


def get_policy_obs_keys(policy):
    if hasattr(policy, "rgb_keys") and hasattr(policy, "low_dim_keys") and hasattr(policy, "wrench_keys"):
        return list(policy.rgb_keys) + list(policy.low_dim_keys) + list(policy.wrench_keys)
    return [key for key in policy.normalizer.params_dict.keys() if key != "action"]


def apply_tcp_offset_to_obs(obs, tcp_offset_key, tcp_offset_m, latest_only=True):
    if tcp_offset_m is None or tcp_offset_key not in obs:
        return obs
    value = obs[tcp_offset_key].clone()
    offset = torch.as_tensor(tcp_offset_m, dtype=value.dtype, device=value.device)
    if offset.shape != (3,):
        raise ValueError(f"tcp_offset_m must have 3 values, got {tuple(offset.shape)}")
    if value.shape[-1] < 3:
        raise ValueError(f"{tcp_offset_key} must have at least 3 position dims, got {tuple(value.shape)}")
    if latest_only:
        value[:, -1, :3] = value[:, -1, :3] + offset
    else:
        value[..., :3] = value[..., :3] + offset
    obs[tcp_offset_key] = value
    return obs


def env_obs_with_tcp_offset(env_obs, tcp_offset_key, tcp_offset_m, latest_only=True):
    env_obs = {key: np.array(value, copy=True) for key, value in env_obs.items()}
    if tcp_offset_m is None or tcp_offset_key not in env_obs:
        return env_obs
    offset = np.asarray(tcp_offset_m, dtype=env_obs[tcp_offset_key].dtype)
    if latest_only:
        env_obs[tcp_offset_key][-1, :3] += offset
    else:
        env_obs[tcp_offset_key][..., :3] += offset
    return env_obs


def predict_abs_actions(
        policy,
        samples_by_idx,
        indices,
        device,
        batch_size,
        seed,
        action_pose_repr,
        tcp_offset_m=None,
        tcp_offset_key="robot_pose_R",
        tcp_offset_latest_only=True):
    policy.eval()
    keys = get_policy_obs_keys(policy)
    pred_by_idx = {}
    tcp_offset_m = None if tcp_offset_m is None else tuple(float(v) for v in tcp_offset_m)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_samples = [samples_by_idx[idx] for idx in batch_indices]
            obs = collate_obs(batch_samples, keys, device)
            obs = apply_tcp_offset_to_obs(
                obs,
                tcp_offset_key=tcp_offset_key,
                tcp_offset_m=tcp_offset_m,
                latest_only=tcp_offset_latest_only,
            )
            obs = adapt_obs_for_policy(obs, policy)
            result = policy.predict_action(obs)
            rel_pred = result["action_pred"].detach().cpu().numpy()
            for local_idx, dataset_idx in enumerate(batch_indices):
                env_obs = obs_to_numpy(samples_by_idx[dataset_idx]["obs"])
                env_obs = env_obs_with_tcp_offset(
                    env_obs,
                    tcp_offset_key=tcp_offset_key,
                    tcp_offset_m=tcp_offset_m,
                    latest_only=tcp_offset_latest_only,
                )
                pred_by_idx[dataset_idx] = action_to_abs(
                    rel_pred[local_idx],
                    env_obs,
                    action_pose_repr,
                )
            del obs, result, rel_pred
            gc.collect()
    return pred_by_idx


def predict_with_inference_steps(
        policy,
        steps,
        samples_by_idx,
        indices,
        device,
        batch_size,
        seed,
        action_pose_repr,
        tcp_offset_m=None,
        tcp_offset_key="robot_pose_R",
        tcp_offset_latest_only=True):
    old_steps = getattr(policy, "num_inference_steps", None)
    if steps is not None:
        policy.num_inference_steps = int(steps)
    try:
        return predict_abs_actions(
            policy,
            samples_by_idx,
            indices,
            device,
            batch_size,
            seed,
            action_pose_repr,
            tcp_offset_m=tcp_offset_m,
            tcp_offset_key=tcp_offset_key,
            tcp_offset_latest_only=tcp_offset_latest_only,
        )
    finally:
        if old_steps is not None:
            policy.num_inference_steps = old_steps


def latest_obs_step_in_episode(dataset, dataset_idx):
    buffer_start_idx, buffer_end_idx, sample_start_idx, _ = dataset.sampler.indices[dataset_idx]
    latest_obs_t = int(dataset.n_obs_steps) - 1
    buffer_len = int(buffer_end_idx - buffer_start_idx)
    buffer_offset = latest_obs_t - int(sample_start_idx)
    buffer_offset = min(max(buffer_offset, 0), buffer_len - 1)
    latest_buffer_idx = int(buffer_start_idx) + buffer_offset

    episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    episode_idx = int(np.searchsorted(episode_ends, latest_buffer_idx, side="right"))
    episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    return episode_idx, latest_buffer_idx - episode_start


def select_demo_step_indices(dataset, demo_steps, max_count):
    demo_steps = set(int(step) for step in demo_steps)
    selected = []
    seen = set()
    for dataset_idx in range(len(dataset)):
        episode_idx, step_in_episode = latest_obs_step_in_episode(dataset, dataset_idx)
        key = (episode_idx, step_in_episode)
        if step_in_episode in demo_steps and key not in seen:
            selected.append(dataset_idx)
            seen.add(key)
            if len(selected) >= max_count:
                break
    if len(selected) < max_count:
        raise RuntimeError(
            f"Only found {len(selected)} demo-step samples for steps {sorted(demo_steps)}, "
            f"requested {max_count}."
        )
    return selected


def load_samples(dataset, indices, action_pose_repr):
    samples_by_idx = {}
    gt_abs_by_idx = {}
    base_xyz_by_idx = {}
    for idx in indices:
        sample = dataset[idx]
        samples_by_idx[idx] = sample
        env_obs = obs_to_numpy(sample["obs"])
        rel_action = sample["action"].detach().cpu().numpy()
        gt_abs_by_idx[idx] = action_to_abs(rel_action, env_obs, action_pose_repr)
        if "robot_pose_R" in env_obs:
            base_xyz_by_idx[idx] = env_obs["robot_pose_R"][-1].copy()
        elif "robot_pose_L" in env_obs:
            base_xyz_by_idx[idx] = env_obs["robot_pose_L"][-1].copy()
        else:
            base_xyz_by_idx[idx] = np.zeros(3, dtype=np.float32)
    return samples_by_idx, gt_abs_by_idx, base_xyz_by_idx


def pose_errors(pred_abs, gt_abs):
    pred_pos = pred_abs[..., :3]
    gt_pos = gt_abs[..., :3]
    pos_err = np.linalg.norm(pred_pos - gt_pos, axis=-1)

    pred_mat = pose10d_to_mat(pred_abs[..., :9])[..., :3, :3]
    gt_mat = pose10d_to_mat(gt_abs[..., :9])[..., :3, :3]
    pred_rot = Rotation.from_matrix(pred_mat.reshape(-1, 3, 3))
    gt_rot = Rotation.from_matrix(gt_mat.reshape(-1, 3, 3))
    rot_err = (pred_rot * gt_rot.inv()).magnitude().reshape(pred_abs.shape[:-1])
    rot_err_deg = np.rad2deg(rot_err)
    return {
        "mean_pos_err_m": float(pos_err.mean()),
        "max_pos_err_m": float(pos_err.max()),
        "mean_rot_err_deg": float(rot_err_deg.mean()),
        "max_rot_err_deg": float(rot_err_deg.max()),
    }


def metric_row(model_name, dataset_idx, pred_abs, gt_abs):
    row = {
        "model": model_name,
        "dataset_idx": dataset_idx,
    }
    row.update(pose_errors(pred_abs, gt_abs))
    row["first_x"] = float(pred_abs[0, 0])
    row["first_y"] = float(pred_abs[0, 1])
    row["first_z"] = float(pred_abs[0, 2])
    row["last_x"] = float(pred_abs[-1, 0])
    row["last_y"] = float(pred_abs[-1, 1])
    row["last_z"] = float(pred_abs[-1, 2])
    return row


def format_vec(vec):
    return "[" + ", ".join(f"{x:.6g}" for x in vec) + "]"


def write_sample_metrics(path, transformer_ckpt, unet_ckpt, dataset_path, sample_idx, base_xyz, gt_abs, preds):
    lines = [
        f"transformer_ckpt {transformer_ckpt}",
        f"unet_ckpt {unet_ckpt}",
        f"dataset_path_runtime {dataset_path}",
        f"sample_idx {sample_idx}",
        f"base_xyz {format_vec(base_xyz)}",
        "GT",
        f"  first_xyz {format_vec(gt_abs[0, :3])} last_xyz {format_vec(gt_abs[-1, :3])}",
        "  pos_err_mean/max_m 0.0 0.0 rot_err_mean/max_deg 0.0 0.0",
    ]
    for name, pred_abs in preds.items():
        err = pose_errors(pred_abs, gt_abs)
        lines.extend([
            name,
            f"  first_xyz {format_vec(pred_abs[0, :3])} last_xyz {format_vec(pred_abs[-1, :3])}",
            (
                "  pos_err_mean/max_m "
                f"{err['mean_pos_err_m']:.6f} {err['max_pos_err_m']:.6f} "
                "rot_err_mean/max_deg "
                f"{err['mean_rot_err_deg']:.3f} {err['max_rot_err_deg']:.3f}"
            ),
        ])
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def summarize_metric_rows(rows):
    by_model = OrderedDict()
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    summary = OrderedDict()
    for model, model_rows in by_model.items():
        summary[model] = {}
        for key in ["mean_pos_err_m", "max_pos_err_m", "mean_rot_err_deg", "max_rot_err_deg"]:
            values = np.array([row[key] for row in model_rows], dtype=np.float64)
            summary[model][key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "max": float(values.max()),
            }
    return summary


def write_summary(path, transformer_ckpt, unet_ckpt, dataset_path, dataset_len, n_samples, summary):
    lines = [
        f"transformer_ckpt {transformer_ckpt}",
        f"unet_ckpt {unet_ckpt}",
        f"dataset_path_runtime {dataset_path}",
        f"dataset_len {dataset_len}",
        f"n_samples {n_samples}",
    ]
    for model, metric_summary in summary.items():
        lines.append(model)
        for key, stats in metric_summary.items():
            lines.append(
                f"  {key} mean/median/p90/max "
                f"{stats['mean']:.6f} {stats['median']:.6f} "
                f"{stats['p90']:.6f} {stats['max']:.6f}"
            )
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def write_metrics_csv(path, rows):
    fieldnames = [
        "model", "dataset_idx", "mean_pos_err_m", "max_pos_err_m",
        "mean_rot_err_deg", "max_rot_err_deg",
        "first_x", "first_y", "first_z", "last_x", "last_y", "last_z",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_plotly_traj(fig, xyz, name, color, width=1.5, dash=None, opacity=1.0, legendgroup=None):
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=2, color=color),
        opacity=opacity,
        legendgroup=legendgroup,
    ))


def xyz_limits_from_points(points):
    points = np.concatenate(points, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = float(np.max(maxs - mins) / 2)
    return centers, max(radius * 1.08, 1e-3)


def plotly_equal_scene(centers, radius):
    return dict(
        xaxis=dict(title="x (m)", range=[centers[0] - radius, centers[0] + radius]),
        yaxis=dict(title="y (m)", range=[centers[1] - radius, centers[1] + radius]),
        zaxis=dict(title="z (m)", range=[centers[2] - radius, centers[2] + radius]),
        aspectmode="cube",
    )


def write_sample_plotly(path, gt_abs, preds, title, axis_limits=None):
    colors = model_colors_for(preds.keys())
    fig = go.Figure()
    centers, radius = axis_limits if axis_limits is not None else xyz_limits(gt_abs, preds)
    add_plotly_traj(fig, gt_abs[:, :3], "GT", MODEL_COLORS["GT"], width=2.0)
    for name, pred_abs in preds.items():
        add_plotly_traj(fig, pred_abs[:, :3], name, colors[name], width=1.5)
    fig.update_layout(
        title=title,
        scene=plotly_equal_scene(centers, radius),
        margin=dict(l=0, r=0, b=0, t=45),
    )
    fig.write_html(str(path))


def write_all_plotly(path, indices, gt_abs_by_idx, pred_abs_by_model, axis_limits=None):
    colors = model_colors_for(pred_abs_by_model.keys())
    fig = go.Figure()
    all_points = []
    for dataset_idx in indices:
        all_points.append(gt_abs_by_idx[dataset_idx][:, :3])
        add_plotly_traj(
            fig, gt_abs_by_idx[dataset_idx][:, :3],
            f"GT idx {dataset_idx}",
            MODEL_COLORS["GT"],
            width=1.2,
            opacity=0.25,
            legendgroup="GT",
        )
        for name, pred_by_idx in pred_abs_by_model.items():
            all_points.append(pred_by_idx[dataset_idx][:, :3])
            add_plotly_traj(
                fig, pred_by_idx[dataset_idx][:, :3],
                f"{name} idx {dataset_idx}",
                colors[name],
                width=1.2,
                opacity=0.32,
                legendgroup=name,
            )
    centers, radius = axis_limits if axis_limits is not None else xyz_limits_from_points(all_points)
    fig.update_layout(
        title=f"GT vs model predictions, {len(indices)} dataset samples",
        scene=plotly_equal_scene(centers, radius),
        margin=dict(l=0, r=0, b=0, t=45),
    )
    fig.write_html(str(path))


def plot_traj_3d(ax, xyz, label, color, linewidth=1.4, alpha=1.0):
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label, color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=9, alpha=alpha, depthshade=False)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color=color, s=17, alpha=alpha, depthshade=False)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=color, marker="x", s=23, alpha=alpha, depthshade=False)


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float64)
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    if radius <= 0:
        radius = 1e-3
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])
    ax.set_box_aspect((1, 1, 1))


def write_sample_png(path, gt_abs, preds, title, axis_limits=None):
    colors = model_colors_for(preds.keys())
    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    plot_traj_3d(ax3d, gt_abs[:, :3], "GT", MODEL_COLORS["GT"], linewidth=1.8)
    for name, pred_abs in preds.items():
        plot_traj_3d(ax3d, pred_abs[:, :3], name, colors[name], linewidth=1.4)
    ax3d.set_title("3D TCP trajectory")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    if axis_limits is None:
        set_axes_equal(ax3d)
    else:
        apply_xyz_limits(ax3d, *axis_limits)
    ax3d.legend(fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    t = np.arange(gt_abs.shape[0])
    for dim, style, label in zip(range(3), ["-", "--", ":"], ["GT x", "GT y", "GT z"]):
        ax.plot(t, gt_abs[:, dim], color=MODEL_COLORS["GT"], linestyle=style, linewidth=2.2, label=label)
    for name, pred_abs in preds.items():
        color = colors[name]
        ax.plot(t, pred_abs[:, 0], color=color, linestyle="-", linewidth=1.6, label=f"{name} x")
        ax.plot(t, pred_abs[:, 1], color=color, linestyle="--", linewidth=1.4, label=f"{name} y")
        ax.plot(t, pred_abs[:, 2], color=color, linestyle=":", linewidth=1.4, label=f"{name} z")
    ax.set_title("x/y/z over horizon")
    ax.set_xlabel("horizon step")
    ax.set_ylabel("position (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def xyz_limits(gt_abs, preds):
    return xyz_limits_from_points([gt_abs[:, :3]] + [pred_abs[:, :3] for pred_abs in preds.values()])


def fixed_span_xyz_limits(gt_abs, preds, span_m):
    # Keep the fixed-size view anchored to GT. If a prediction is far away,
    # centering between GT and the outlier can make the plot show empty space.
    points = gt_abs[:, :3]
    centers = (points.min(axis=0) + points.max(axis=0)) / 2
    return centers, float(span_m) / 2.0


def get_sample_context(dataset, dataset_idx, eval_start=0, horizon_len=None):
    if not hasattr(dataset, "sampler") or not hasattr(dataset, "replay_buffer"):
        return None
    replay_buffer = dataset.replay_buffer
    if "action" not in replay_buffer:
        return None

    buffer_start_idx, buffer_end_idx, _, _ = dataset.sampler.indices[dataset_idx]
    episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
    episode_idx = int(np.searchsorted(episode_ends, buffer_start_idx, side="right"))
    episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    episode_end = int(episode_ends[episode_idx])
    episode_len = episode_end - episode_start

    action_start = min(max(int(buffer_start_idx) + int(eval_start), episode_start), episode_end - 1)
    if horizon_len is None:
        action_end = min(int(buffer_end_idx), episode_end)
    else:
        action_end = min(action_start + int(horizon_len), episode_end)
    step_in_episode = action_start - episode_start
    progress = step_in_episode / max(episode_len - 1, 1)

    full_xyz = np.asarray(replay_buffer["action"][episode_start:episode_end, :3], dtype=np.float64)
    window_xyz = np.asarray(replay_buffer["action"][action_start:action_end, :3], dtype=np.float64)
    return {
        "episode_idx": episode_idx,
        "episode_start": episode_start,
        "episode_end": episode_end,
        "episode_len": episode_len,
        "step_in_episode": step_in_episode,
        "progress": progress,
        "full_xyz": full_xyz,
        "window_xyz": window_xyz,
    }


def apply_xyz_limits(ax, centers, radius):
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])
    ax.set_box_aspect((1, 1, 1))


def points_fit_limits(points, centers, radius):
    points = np.asarray(points, dtype=np.float64)
    lower = centers - radius
    upper = centers + radius
    return bool(np.all((points >= lower) & (points <= upper)))


def centered_fixed_span_limits(points, radius):
    points = np.asarray(points, dtype=np.float64)
    centers = (points.min(axis=0) + points.max(axis=0)) / 2
    return centers, radius


def write_sample_panels_png(path, gt_abs, preds, title, axis_limits=None):
    n = len(preds)
    ncols = 3 if n > 2 else max(1, n)
    nrows = int(math.ceil(n / ncols))
    centers, radius = axis_limits if axis_limits is not None else xyz_limits(gt_abs, preds)
    colors = model_colors_for(preds.keys())
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows))
    for plot_idx, (name, pred_abs) in enumerate(preds.items(), start=1):
        ax = fig.add_subplot(nrows, ncols, plot_idx, projection="3d")
        plot_traj_3d(ax, gt_abs[:, :3], "GT", MODEL_COLORS["GT"], linewidth=1.8)
        plot_traj_3d(ax, pred_abs[:, :3], name, colors[name], linewidth=1.4)
        apply_xyz_limits(ax, centers, radius)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ordered_pair_panel_preds(preds):
    preferred = [
        "UNet EMA eval",
        "Transformer v-pred EMA steps=16",
        "Transformer epsilon EMA steps=16",
        "Transformer EMA eval steps=16",
    ]
    ordered = OrderedDict((name, preds[name]) for name in preferred if name in preds)
    for name, pred_abs in preds.items():
        if name not in ordered:
            ordered[name] = pred_abs
    return ordered


def short_model_title(name):
    title_map = {
        "UNet EMA eval": "GT + UNet",
        "Transformer v-pred EMA steps=16": "GT + Transformer v-pred",
        "Transformer epsilon EMA steps=16": "GT + Transformer epsilon",
        "Transformer relative EMA steps=16": "GT + Transformer relative",
        "Transformer DiT v-pred e900 EMA steps=16": "GT + Transformer DiT",
        "Transformer non-DiT v-pred e900 EMA steps=16": "GT + Transformer non-DiT",
        "Transformer EMA eval steps=16": "GT + Transformer",
    }
    return title_map.get(name, f"GT + {name}")


def set_context_axis_equal(ax, points_2d):
    if points_2d.size == 0:
        return
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins) / 2), 1e-4)
    radius *= 1.08
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_demo_context(ax_xy, ax_xz, ax_prog, context):
    full_xyz = context["full_xyz"]
    window_xyz = context["window_xyz"]

    ax_xy.plot(full_xyz[:, 0], full_xyz[:, 1], color="#b0b0b0", linewidth=1.0, label="full demo")
    if len(window_xyz) > 0:
        ax_xy.plot(window_xyz[:, 0], window_xyz[:, 1], color="#111111", linewidth=2.0, label="this window")
        ax_xy.scatter(window_xyz[0, 0], window_xyz[0, 1], color="#111111", s=12)
    ax_xy.set_title("demo context XY")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(fontsize=7)
    set_context_axis_equal(ax_xy, full_xyz[:, [0, 1]])

    ax_xz.plot(full_xyz[:, 0], full_xyz[:, 2], color="#b0b0b0", linewidth=1.0, label="full demo")
    if len(window_xyz) > 0:
        ax_xz.plot(window_xyz[:, 0], window_xyz[:, 2], color="#111111", linewidth=2.0, label="this window")
        ax_xz.scatter(window_xyz[0, 0], window_xyz[0, 2], color="#111111", s=12)
    ax_xz.set_title("demo context XZ")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.grid(True, alpha=0.25)
    ax_xz.legend(fontsize=7)
    set_context_axis_equal(ax_xz, full_xyz[:, [0, 2]])

    progress = context["progress"]
    ax_prog.barh([0], [1.0], color="#eeeeee", height=0.34)
    ax_prog.barh([0], [progress], color="#111111", height=0.34)
    ax_prog.scatter([progress], [0], color="#d62728", s=28, zorder=3)
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(-0.55, 0.55)
    ax_prog.set_yticks([])
    ax_prog.set_xlabel("demo progress")
    ax_prog.set_title(
        f"demo {context['episode_idx']} | step {context['step_in_episode']}/{context['episode_len'] - 1} "
        f"({progress * 100:.1f}%)"
    )
    ax_prog.grid(True, axis="x", alpha=0.25)


def write_pair_panels_with_errors_png(path, gt_abs, preds, title, axis_limits=None, context=None):
    panel_preds = ordered_pair_panel_preds(preds)
    n = len(panel_preds)
    plot_cols = max(3, n) if context is not None else n
    centers, radius = axis_limits if axis_limits is not None else xyz_limits(gt_abs, panel_preds)
    colors = model_colors_for(panel_preds.keys())
    if context is None:
        fig = plt.figure(figsize=(5.1 * plot_cols, 7.8))
        grid = fig.add_gridspec(2, plot_cols, height_ratios=[3.0, 1.35])
        error_row = 1
    else:
        fig = plt.figure(figsize=(5.1 * plot_cols, 10.2))
        grid = fig.add_gridspec(3, plot_cols, height_ratios=[3.0, 1.25, 1.35])
        error_row = 2

    for col, (name, pred_abs) in enumerate(panel_preds.items()):
        ax = fig.add_subplot(grid[0, col], projection="3d")
        plot_traj_3d(ax, gt_abs[:, :3], "GT", MODEL_COLORS["GT"], linewidth=1.8)
        plot_traj_3d(ax, pred_abs[:, :3], name, colors[name], linewidth=1.4)
        panel_centers, panel_radius = centers, radius
        if not points_fit_limits(pred_abs[:, :3], centers, radius):
            panel_centers, panel_radius = centered_fixed_span_limits(pred_abs[:, :3], radius)
        apply_xyz_limits(ax, panel_centers, panel_radius)
        ax.set_title(short_model_title(name), fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)

    if context is not None:
        ax_xy = fig.add_subplot(grid[1, 0])
        ax_xz = fig.add_subplot(grid[1, 1])
        ax_prog = fig.add_subplot(grid[1, 2:])
        plot_demo_context(ax_xy, ax_xz, ax_prog, context)

    t = np.arange(gt_abs.shape[0])
    split_col = max(1, plot_cols // 2)
    ax_pos = fig.add_subplot(grid[error_row, :split_col])
    ax_rot = fig.add_subplot(grid[error_row, split_col:])
    for name, pred_abs in panel_preds.items():
        pos_err_mm, rot_err_deg = trajectory_errors(pred_abs, gt_abs)
        color = colors[name]
        ax_pos.plot(t, pos_err_mm, label=name, color=color, linewidth=1.6)
        ax_rot.plot(t, rot_err_deg, label=name, color=color, linewidth=1.6)
    ax_pos.set_title("position error")
    ax_pos.set_xlabel("horizon step")
    ax_pos.set_ylabel("mm")
    ax_rot.set_title("rotation error")
    ax_rot.set_xlabel("horizon step")
    ax_rot.set_ylabel("deg")
    for ax in (ax_pos, ax_rot):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def trajectory_errors(pred_abs, gt_abs):
    pos_err_mm = np.linalg.norm(pred_abs[:, :3] - gt_abs[:, :3], axis=-1) * 1000.0
    pred_mat = pose10d_to_mat(pred_abs[:, :9])[:, :3, :3]
    gt_mat = pose10d_to_mat(gt_abs[:, :9])[:, :3, :3]
    pred_rot = Rotation.from_matrix(pred_mat)
    gt_rot = Rotation.from_matrix(gt_mat)
    rot_err_deg = np.rad2deg((pred_rot * gt_rot.inv()).magnitude())
    return pos_err_mm, rot_err_deg


def write_error_over_horizon_png(path, gt_abs, preds, title):
    t = np.arange(gt_abs.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = model_colors_for(preds.keys())
    for name, pred_abs in preds.items():
        pos_err_mm, rot_err_deg = trajectory_errors(pred_abs, gt_abs)
        color = colors[name]
        axes[0].plot(t, pos_err_mm, label=name, color=color, linewidth=2.0)
        axes[1].plot(t, rot_err_deg, label=name, color=color, linewidth=2.0)
    axes[0].set_title("position error")
    axes[0].set_ylabel("mm")
    axes[1].set_title("rotation error")
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("horizon step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def subset_preds(preds, names):
    return OrderedDict((name, preds[name]) for name in names if name in preds)


def write_clear_sample_pngs(output_dir, slug, sample_idx, gt_abs, preds, axis_limits=None):
    prefix = output_dir / f"{slug}_sample_{sample_idx:04d}"
    groups = {
        "default_only": [
            "Transformer EMA eval",
            "Transformer raw model",
            "UNet EMA eval",
        ],
        "transformer_ema_steps": [
            "Transformer EMA eval",
            "Transformer EMA eval steps=16",
            "Transformer EMA eval steps=32",
        ],
        "transformer_raw_steps": [
            "Transformer raw model",
            "Transformer raw model steps=16",
            "Transformer raw model steps=32",
        ],
    }
    for suffix, names in groups.items():
        this_preds = subset_preds(preds, names)
        if len(this_preds) > 0:
            write_sample_png(
                prefix.with_name(prefix.name + f"_{suffix}.png"),
                gt_abs,
                this_preds,
                title=f"Dataset idx {sample_idx}: {suffix.replace('_', ' ')}",
                axis_limits=axis_limits,
            )
    write_sample_panels_png(
        prefix.with_name(prefix.name + "_each_model_panels_3d.png"),
        gt_abs,
        preds,
        title=f"Dataset idx {sample_idx}: GT vs one model per panel",
        axis_limits=axis_limits,
    )
    write_error_over_horizon_png(
        prefix.with_name(prefix.name + "_error_over_horizon.png"),
        gt_abs,
        preds,
        title=f"Dataset idx {sample_idx}: per-step errors",
    )


def write_overview_png(path, rows, summary):
    models = list(summary.keys())
    metric_keys = ["mean_pos_err_m", "max_pos_err_m", "mean_rot_err_deg", "max_rot_err_deg"]
    titles = [
        "Mean position error (m)",
        "Max position error (m)",
        "Mean rotation error (deg)",
        "Max rotation error (deg)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric_key, title in zip(axes.ravel(), metric_keys, titles):
        data = [
            [row[metric_key] for row in rows if row["model"] == model]
            for model in models
        ]
        ax.boxplot(data, labels=models, showfliers=False)
        means = [np.mean(values) for values in data]
        ax.scatter(np.arange(1, len(models) + 1), means, color="#d62728", marker="x", zorder=3, label="mean")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=12)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_individual_traj_dir(
        output_dir,
        indices,
        gt_abs_by_idx,
        pred_abs_by_model,
        rows,
        transformer_ckpt,
        unet_ckpt,
        dataset_path,
        dataset_len,
        individual_axis_span_m=None,
        pair_panels=False,
        images_only=False,
        start_plot_idx=0,
        contexts_by_idx=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not images_only:
        metrics_path = output_dir / "traj10_metrics.csv"
        selected_rows = [row for row in rows if row["dataset_idx"] in set(indices)]
        write_metrics_csv(metrics_path, selected_rows)
        summary = summarize_metric_rows(selected_rows)
        write_summary(
            output_dir / "traj10_summary.txt",
            transformer_ckpt=transformer_ckpt,
            unet_ckpt=unet_ckpt,
            dataset_path=dataset_path,
            dataset_len=dataset_len,
            n_samples=len(indices),
            summary=summary,
        )
    for local_plot_idx, dataset_idx in enumerate(indices):
        plot_idx = start_plot_idx + local_plot_idx
        preds = OrderedDict((name, pred_by_idx[dataset_idx]) for name, pred_by_idx in pred_abs_by_model.items())
        axis_limits = (
            fixed_span_xyz_limits(gt_abs_by_idx[dataset_idx], preds, individual_axis_span_m)
            if individual_axis_span_m is not None else None
        )
        output_path = output_dir / f"traj_{plot_idx:02d}_dataset_idx_{dataset_idx:05d}.png"
        if pair_panels:
            write_pair_panels_with_errors_png(
                output_path,
                gt_abs_by_idx[dataset_idx],
                preds,
                title=f"Dataset idx {dataset_idx}",
                axis_limits=axis_limits,
                context=None if contexts_by_idx is None else contexts_by_idx.get(dataset_idx),
            )
        else:
            write_sample_png(
                output_path,
                gt_abs_by_idx[dataset_idx],
                preds,
                title=f"Dataset idx {dataset_idx}",
                axis_limits=axis_limits,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-ckpt", default=DEFAULT_TRANSFORMER_CKPT)
    parser.add_argument("--unet-ckpt", default=DEFAULT_UNET_CKPT)
    parser.add_argument(
        "--unet-only",
        action="store_true",
        help="Evaluate only the UNet checkpoint and skip all Transformer checkpoints.",
    )
    parser.add_argument(
        "--transformer-only",
        action="store_true",
        help="Evaluate only the Transformer checkpoint and skip all UNet checkpoints.",
    )
    parser.add_argument(
        "--skip-unet",
        action="store_true",
        help="Alias for --transformer-only.",
    )
    parser.add_argument("--output-dir", default="data/debug_action_vis")
    parser.add_argument(
        "--dataset-source",
        choices=("unet", "transformer"),
        default="unet",
        help="Checkpoint config to use for constructing the evaluation dataset.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional short output subdirectory and filename prefix.",
    )
    parser.add_argument("--sample-idx", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--random-samples",
        action="store_true",
        help="Select --n-samples dataset indices uniformly at random without replacement.",
    )
    parser.add_argument(
        "--demo-step-samples",
        type=int,
        nargs="*",
        default=None,
        help="If set, select samples whose latest observation is at these per-demo step indices.",
    )
    parser.add_argument(
        "--demo-step-total",
        type=int,
        default=None,
        help="Total number of samples to select with --demo-step-samples. Defaults to --n-samples.",
    )
    parser.add_argument("--traj-start", type=int, default=0)
    parser.add_argument("--traj-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--individual-axis-span-m",
        type=float,
        default=0.10,
        help="Fixed x/y/z axis span in meters for individual trajectory plots.",
    )
    parser.add_argument(
        "--individual-pair-panels",
        action="store_true",
        help="Write traj10 images as GT+one-model 3D panels with error plots below.",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Write PNG images only. Skip txt/csv/html outputs.",
    )
    parser.add_argument(
        "--transformer-extra-inference-steps",
        type=int,
        nargs="*",
        default=[],
        help="Additional Transformer num_inference_steps variants to evaluate.",
    )
    parser.add_argument(
        "--transformer-ema-only-inference-steps",
        type=int,
        nargs="*",
        default=None,
        help=(
            "If set, evaluate only Transformer EMA at these num_inference_steps "
            "and skip default/raw Transformer variants."
        ),
    )
    parser.add_argument(
        "--transformer-ema-label",
        default=None,
        help="Override label for the primary Transformer EMA when using --transformer-ema-only-inference-steps.",
    )
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        nargs=3,
        default=None,
        help="Offset the policy input TCP position by this xyz vector in meters.",
    )
    parser.add_argument(
        "--tcp-offset-key",
        default="robot_pose_R",
        help="Observation key to offset when --tcp-offset-m is set.",
    )
    parser.add_argument(
        "--tcp-offset-all-obs-steps",
        action="store_true",
        help="Apply --tcp-offset-m to all obs steps instead of only the latest TCP obs.",
    )
    parser.add_argument(
        "--tcp-offset-label",
        default="TCP offset",
        help="Label suffix for predictions generated with --tcp-offset-m.",
    )
    parser.add_argument(
        "--include-unoffset-baseline",
        action="store_true",
        help="When --tcp-offset-m is set, also plot the same Transformer without the TCP offset.",
    )
    parser.add_argument(
        "--compare-transformer-ckpt",
        default=None,
        help="Optional second Transformer checkpoint to plot as an EMA comparison.",
    )
    parser.add_argument(
        "--compare-transformer-label",
        default="Transformer epsilon EMA",
        help="Legend label for --compare-transformer-ckpt.",
    )
    parser.add_argument(
        "--compare-transformer-inference-steps",
        type=int,
        default=None,
        help="num_inference_steps for --compare-transformer-ckpt. Defaults to the primary EMA-only step if present.",
    )
    args = parser.parse_args()

    transformer_only = args.transformer_only or args.skip_unet
    if args.unet_only and transformer_only:
        raise ValueError("--unet-only and --transformer-only/--skip-unet are mutually exclusive.")

    transformer_ckpt = resolve_path(args.transformer_ckpt)
    unet_ckpt = resolve_path(args.unet_ckpt)
    output_dir = resolve_path(args.output_dir)
    if args.run_name is not None:
        output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    unet_cfg = None if transformer_only else load_cfg_for_ckpt(unet_ckpt)
    transformer_cfg = None if args.unet_only else load_cfg_for_ckpt(transformer_ckpt)
    dataset_source_cfg = (
        unet_cfg if args.unet_only
        else transformer_cfg if transformer_only
        else unet_cfg if args.dataset_source == "unet"
        else transformer_cfg
    )
    dataset_cfg = make_dataset_cfg(dataset_source_cfg)
    dataset = hydra.utils.instantiate(dataset_cfg)
    dataset_len = len(dataset)
    resolved_dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_path = str(resolve_path(resolved_dataset_cfg["dataset_path"]))
    eval_start = int(resolved_dataset_cfg["n_obs_steps"]) - 1

    if args.demo_step_samples:
        sample_indices = select_demo_step_indices(
            dataset,
            demo_steps=args.demo_step_samples,
            max_count=args.demo_step_total or args.n_samples,
        )
    elif args.random_samples:
        rng = np.random.default_rng(args.seed)
        sample_indices = sorted(rng.choice(dataset_len, size=args.n_samples, replace=False).astype(int).tolist())
    else:
        sample_indices = np.linspace(0, dataset_len - 1, args.n_samples, dtype=int).tolist()
    sample_idx = args.sample_idx
    if (args.demo_step_samples or args.random_samples) and sample_idx not in sample_indices:
        sample_idx = sample_indices[0]
    all_indices = sorted(set(sample_indices + [sample_idx]))
    dataset_action_pose_repr = resolved_dataset_cfg["pose_repr"]["action_pose_repr"]
    transformer_action_pose_repr = None if args.unet_only else action_pose_repr_from_cfg(transformer_cfg)
    unet_action_pose_repr = None if transformer_only else action_pose_repr_from_cfg(unet_cfg)
    samples_by_idx, gt_abs_by_idx, base_xyz_by_idx = load_samples(
        dataset=dataset,
        indices=all_indices,
        action_pose_repr=dataset_action_pose_repr,
    )
    print(f"GT dataset action_pose_repr: {dataset_action_pose_repr}")
    if transformer_action_pose_repr is not None:
        print(f"Transformer action_pose_repr: {transformer_action_pose_repr}")
    if unet_action_pose_repr is not None:
        print(f"UNet action_pose_repr: {unet_action_pose_repr}")
    tcp_offset_latest_only = not args.tcp_offset_all_obs_steps
    if args.tcp_offset_m is not None:
        print(
            f"TCP offset for model input: key={args.tcp_offset_key}, "
            f"xyz_m={args.tcp_offset_m}, latest_only={tcp_offset_latest_only}"
        )

    pred_abs_by_model = OrderedDict()

    if not args.unet_only:
        print(f"Loading Transformer: {transformer_ckpt}")
        transformer_workspace = load_policy_from_checkpoint(transformer_ckpt, device)
        if args.transformer_ema_only_inference_steps is None:
            if transformer_workspace.ema_model is not None:
                pred_abs_by_model["Transformer EMA eval"] = predict_with_inference_steps(
                    transformer_workspace.ema_model,
                    None,
                    samples_by_idx,
                    all_indices,
                    device,
                    args.batch_size,
                    args.seed,
                    transformer_action_pose_repr,
                    tcp_offset_m=args.tcp_offset_m,
                    tcp_offset_key=args.tcp_offset_key,
                    tcp_offset_latest_only=tcp_offset_latest_only,
                )
            pred_abs_by_model["Transformer raw model"] = predict_with_inference_steps(
                transformer_workspace.model,
                None,
                samples_by_idx,
                all_indices,
                    device,
                    args.batch_size,
                    args.seed,
                    transformer_action_pose_repr,
                    tcp_offset_m=args.tcp_offset_m,
                    tcp_offset_key=args.tcp_offset_key,
                    tcp_offset_latest_only=tcp_offset_latest_only,
                )
            for steps in args.transformer_extra_inference_steps:
                if transformer_workspace.ema_model is not None:
                    pred_abs_by_model[f"Transformer EMA eval steps={steps}"] = predict_with_inference_steps(
                        transformer_workspace.ema_model,
                        steps,
                        samples_by_idx,
                        all_indices,
                        device,
                        args.batch_size,
                        args.seed,
                        transformer_action_pose_repr,
                        tcp_offset_m=args.tcp_offset_m,
                        tcp_offset_key=args.tcp_offset_key,
                        tcp_offset_latest_only=tcp_offset_latest_only,
                    )
                pred_abs_by_model[f"Transformer raw model steps={steps}"] = predict_with_inference_steps(
                    transformer_workspace.model,
                    steps,
                    samples_by_idx,
                    all_indices,
                    device,
                    args.batch_size,
                    args.seed,
                    transformer_action_pose_repr,
                    tcp_offset_m=args.tcp_offset_m,
                    tcp_offset_key=args.tcp_offset_key,
                    tcp_offset_latest_only=tcp_offset_latest_only,
                )
        else:
            if transformer_workspace.ema_model is None:
                raise RuntimeError("--transformer-ema-only-inference-steps requires an EMA model in the checkpoint.")
            for steps in args.transformer_ema_only_inference_steps:
                base_model_name = (
                    f"{args.transformer_ema_label} steps={steps}"
                    if args.transformer_ema_label is not None
                    else f"Transformer EMA eval steps={steps}"
                )
                if args.tcp_offset_m is not None and args.include_unoffset_baseline:
                    pred_abs_by_model[base_model_name] = predict_with_inference_steps(
                        transformer_workspace.ema_model,
                        steps,
                        samples_by_idx,
                        all_indices,
                        device,
                        args.batch_size,
                        args.seed,
                        transformer_action_pose_repr,
                    )
                model_name = (
                    f"{base_model_name} {args.tcp_offset_label}"
                    if args.tcp_offset_m is not None
                    else base_model_name
                )
                pred_abs_by_model[model_name] = predict_with_inference_steps(
                    transformer_workspace.ema_model,
                    steps,
                    samples_by_idx,
                    all_indices,
                    device,
                    args.batch_size,
                    args.seed,
                    transformer_action_pose_repr,
                    tcp_offset_m=args.tcp_offset_m,
                    tcp_offset_key=args.tcp_offset_key,
                    tcp_offset_latest_only=tcp_offset_latest_only,
                )
        del transformer_workspace
        gc.collect()

    if args.compare_transformer_ckpt is not None and not args.unet_only:
        compare_transformer_ckpt = resolve_path(args.compare_transformer_ckpt)
        compare_steps = args.compare_transformer_inference_steps
        if compare_steps is None and args.transformer_ema_only_inference_steps:
            compare_steps = args.transformer_ema_only_inference_steps[0]
        compare_name = args.compare_transformer_label
        if compare_steps is not None:
            compare_name = f"{compare_name} steps={compare_steps}"
        print(f"Loading comparison Transformer: {compare_transformer_ckpt}")
        compare_transformer_cfg = load_cfg_for_ckpt(compare_transformer_ckpt)
        compare_action_pose_repr = action_pose_repr_from_cfg(compare_transformer_cfg)
        print(f"Comparison Transformer action_pose_repr: {compare_action_pose_repr}")
        compare_workspace = load_policy_from_checkpoint(compare_transformer_ckpt, device)
        compare_policy = compare_workspace.ema_model if compare_workspace.ema_model is not None else compare_workspace.model
        pred_abs_by_model[compare_name] = predict_with_inference_steps(
            compare_policy,
            compare_steps,
            samples_by_idx,
            all_indices,
            device,
            args.batch_size,
            args.seed,
            compare_action_pose_repr,
            tcp_offset_m=args.tcp_offset_m,
            tcp_offset_key=args.tcp_offset_key,
            tcp_offset_latest_only=tcp_offset_latest_only,
        )
        del compare_workspace
        gc.collect()

    if not transformer_only:
        print(f"Loading UNet: {unet_ckpt}")
        unet_workspace = load_policy_from_checkpoint(unet_ckpt, device, strict=False)
        unet_policy = unet_workspace.ema_model if unet_workspace.ema_model is not None else unet_workspace.model
        pred_abs_by_model["UNet EMA eval"] = predict_with_inference_steps(
            unet_policy,
            None,
            samples_by_idx,
            all_indices,
            device,
            args.batch_size,
            args.seed,
            unet_action_pose_repr,
            tcp_offset_m=args.tcp_offset_m,
            tcp_offset_key=args.tcp_offset_key,
            tcp_offset_latest_only=tcp_offset_latest_only,
        )
        del unet_workspace
        gc.collect()

    gt_eval_by_idx = {
        idx: action[eval_start:]
        for idx, action in gt_abs_by_idx.items()
    }
    pred_eval_by_model = OrderedDict(
        (
            name,
            {idx: action[eval_start:] for idx, action in pred_by_idx.items()},
        )
        for name, pred_by_idx in pred_abs_by_model.items()
    )
    contexts_by_idx = {
        idx: get_sample_context(
            dataset,
            idx,
            eval_start=eval_start,
            horizon_len=next(iter(pred_eval_by_model.values()))[idx].shape[0],
        )
        for idx in all_indices
    }
    rows = []
    for dataset_idx in sample_indices:
        gt_abs = gt_eval_by_idx[dataset_idx]
        for model_name, pred_by_idx in pred_eval_by_model.items():
            rows.append(metric_row(model_name, dataset_idx, pred_by_idx[dataset_idx], gt_abs))

    if args.unet_only:
        slug = "unet_" + slugify_ckpt(unet_ckpt)
    elif transformer_only:
        slug = "transformer_" + slugify_ckpt(transformer_ckpt)
    else:
        slug = (
            "new_" + slugify_ckpt(transformer_ckpt)
            + "_vs_unet_" + re.sub(r"[^A-Za-z0-9]+", "_", unet_ckpt.stem).strip("_").lower()
        )
    if args.transformer_ema_only_inference_steps is not None:
        steps_slug = "_".join(str(step) for step in args.transformer_ema_only_inference_steps)
        slug += f"_transformer_ema_steps_{steps_slug}_unet_only"
    if args.compare_transformer_ckpt is not None:
        slug += "_compare_transformer"
    if args.run_name is not None:
        slug = args.run_name
        metrics_csv = output_dir / f"metrics_{args.n_samples}_samples.csv"
        summary_txt = output_dir / "summary.txt"
        overview_png = output_dir / "overview.png"
        all_html = output_dir / "all_samples_3d.html"
        sample_png = output_dir / f"sample_{sample_idx:04d}_overlay_3d.png"
        sample_panels_png = output_dir / f"sample_{sample_idx:04d}_pair_panels_3d.png"
        sample_html = output_dir / f"sample_{sample_idx:04d}_overlay_3d.html"
        sample_metrics = output_dir / f"sample_{sample_idx:04d}_metrics.txt"
        sample_error_png = output_dir / f"sample_{sample_idx:04d}_error_over_horizon.png"
        traj_dir = output_dir / "traj10_individual"
    else:
        metrics_csv = output_dir / f"{slug}_many_{args.n_samples}_samples_metrics.csv"
        summary_txt = output_dir / f"{slug}_many_{args.n_samples}_samples_summary.txt"
        overview_png = output_dir / f"{slug}_many_{args.n_samples}_samples_overview.png"
        all_html = output_dir / f"{slug}_many_{args.n_samples}_samples_3d_all.html"
        sample_png = output_dir / f"{slug}_sample_{sample_idx:04d}_gt_vs_models_3d.png"
        sample_panels_png = output_dir / f"{slug}_sample_{sample_idx:04d}_pair_panels_3d.png"
        sample_html = output_dir / f"{slug}_sample_{sample_idx:04d}_gt_vs_models_3d.html"
        sample_metrics = output_dir / f"{slug}_sample_{sample_idx:04d}_metrics.txt"
        sample_error_png = output_dir / f"{slug}_sample_{sample_idx:04d}_error_over_horizon.png"
        traj_dir = output_dir / f"{slug}_traj10_individual"

    summary = summarize_metric_rows(rows)
    if not args.images_only:
        write_metrics_csv(metrics_csv, rows)
        write_summary(
            summary_txt,
            transformer_ckpt="" if args.unet_only else str(transformer_ckpt),
            unet_ckpt="" if transformer_only else str(unet_ckpt),
            dataset_path=dataset_path,
            dataset_len=dataset_len,
            n_samples=args.n_samples,
            summary=summary,
        )
        write_all_plotly(all_html, sample_indices, gt_eval_by_idx, pred_eval_by_model)
    write_overview_png(overview_png, rows, summary)

    sample_preds = OrderedDict((name, pred_by_idx[sample_idx]) for name, pred_by_idx in pred_eval_by_model.items())
    sample_axis_limits = fixed_span_xyz_limits(
        gt_eval_by_idx[sample_idx],
        sample_preds,
        args.individual_axis_span_m,
    )
    if not args.images_only:
        write_sample_metrics(
            sample_metrics,
            transformer_ckpt="" if args.unet_only else str(transformer_ckpt),
            unet_ckpt="" if transformer_only else str(unet_ckpt),
            dataset_path=dataset_path,
            sample_idx=sample_idx,
            base_xyz=base_xyz_by_idx[sample_idx],
            gt_abs=gt_eval_by_idx[sample_idx],
            preds=sample_preds,
        )
    write_sample_png(
        sample_png,
        gt_eval_by_idx[sample_idx],
        sample_preds,
        title=f"GT vs new Transformer and U-Net, dataset idx {sample_idx}",
        axis_limits=sample_axis_limits,
    )
    if not args.images_only:
        write_sample_plotly(
            sample_html,
            gt_eval_by_idx[sample_idx],
            sample_preds,
            title=f"GT vs new Transformer and U-Net, dataset idx {sample_idx}",
            axis_limits=sample_axis_limits,
        )
    write_sample_panels_png(
        sample_panels_png,
        gt_eval_by_idx[sample_idx],
        ordered_pair_panel_preds(sample_preds),
        title=f"Dataset idx {sample_idx}: GT paired with each model",
        axis_limits=sample_axis_limits,
    )
    if args.transformer_ema_only_inference_steps is None:
        write_clear_sample_pngs(
            output_dir,
            slug,
            sample_idx,
            gt_eval_by_idx[sample_idx],
            sample_preds,
            axis_limits=sample_axis_limits,
        )
    else:
        write_error_over_horizon_png(
            sample_error_png,
            gt_eval_by_idx[sample_idx],
            sample_preds,
            title=f"Dataset idx {sample_idx}: per-step errors",
        )
    traj_indices = sample_indices[args.traj_start:args.traj_start + args.traj_count]
    write_individual_traj_dir(
        traj_dir,
        traj_indices,
        gt_eval_by_idx,
        pred_eval_by_model,
        rows,
        transformer_ckpt=str(transformer_ckpt),
        unet_ckpt="" if transformer_only else str(unet_ckpt),
        dataset_path=dataset_path,
        dataset_len=dataset_len,
        individual_axis_span_m=args.individual_axis_span_m,
        pair_panels=args.individual_pair_panels,
        images_only=args.images_only,
        start_plot_idx=args.traj_start,
        contexts_by_idx=contexts_by_idx,
    )

    if not args.images_only:
        print(f"Wrote {summary_txt}")
        print(f"Wrote {all_html}")
    print(f"Wrote {overview_png}")
    print(f"Wrote {sample_png}")


if __name__ == "__main__":
    main()
