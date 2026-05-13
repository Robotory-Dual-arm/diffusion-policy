if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import gc
import pathlib
from collections import OrderedDict

import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.pose_util import pose10d_to_mat
from diffusion_policy.real_world.real_inference_util import (
    get_abs_action_from_relative,
    get_relative_action_from_abs,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_TRANSFORMER_CKPT = (
    "data/outputs/2026.05.11/19.51.30_train_diffusion_transformer_hybrid_"
    "bbbae_dualarm_insert_plug_no_wrench/checkpoints/epoch=0900-train_loss=0.006.ckpt"
)

STAGE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#17becf", "#bcbd22", "#e377c2",
]


def resolve_path(path):
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        p = pathlib.Path.cwd() / p
    return p.resolve()


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


def get_policy_obs_keys(policy):
    if hasattr(policy, "rgb_keys") and hasattr(policy, "low_dim_keys") and hasattr(policy, "wrench_keys"):
        return list(policy.rgb_keys) + list(policy.low_dim_keys) + list(policy.wrench_keys)
    return [key for key in policy.normalizer.params_dict.keys() if key != "action"]


def apply_xyz_limits(ax, centers, radius):
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])
    ax.set_box_aspect((1, 1, 1))


def centered_fixed_span_limits(points, radius):
    points = np.asarray(points, dtype=np.float64)
    centers = (points.min(axis=0) + points.max(axis=0)) / 2
    return centers, radius


def collate_single_obs(sample, keys, device):
    return {
        key: sample["obs"][key].unsqueeze(0).to(device)
        for key in keys
    }


def temporary_action_masks(model, token_horizon):
    old_mask = getattr(model, "mask", None)
    old_memory_mask = getattr(model, "memory_mask", None)
    if old_mask is not None:
        model.mask = old_mask[:token_horizon, :token_horizon]
    if old_memory_mask is not None:
        model.memory_mask = old_memory_mask[:token_horizon, :]
    return old_mask, old_memory_mask


def restore_action_masks(model, old_mask, old_memory_mask):
    if old_mask is not None:
        model.mask = old_mask
    if old_memory_mask is not None:
        model.memory_mask = old_memory_mask


def obs_condition(policy, sample, keys, device):
    obs = collate_single_obs(sample, keys, device)
    nobs = policy.normalizer.normalize(obs)
    tobs = policy.n_obs_steps
    this_nobs = dict_apply(nobs, lambda x: x[:, :tobs, ...].reshape(-1, *x.shape[2:]))
    nobs_features = policy.obs_encoder(this_nobs)
    cond = nobs_features.reshape(1, tobs, -1)
    return obs, cond


def denoise_from_sample(policy, sample_tensor, cond, token_horizon, start_index=0):
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    timesteps = scheduler.timesteps[start_index:]
    trajectory = sample_tensor.clone()

    old_mask, old_memory_mask = temporary_action_masks(policy.model, token_horizon)
    try:
        for t in timesteps:
            model_output = policy.model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                **policy.kwargs,
            ).prev_sample
    finally:
        restore_action_masks(policy.model, old_mask, old_memory_mask)
    return trajectory


def model_action_to_abs(policy, model_action_tensor, env_obs, action_pose_repr):
    action = policy.normalizer["action"].unnormalize(model_action_tensor).detach().cpu().numpy()[0]
    return action_to_abs(action, env_obs, action_pose_repr)


def abs_to_model_action(policy, abs_action, env_obs, action_pose_repr, device):
    if action_pose_repr == "relative":
        model_action = get_relative_action_from_abs(abs_action, env_obs)
    else:
        model_action = abs_action
    model_action = torch.from_numpy(model_action.astype(np.float32)).unsqueeze(0).to(device)
    return policy.normalizer["action"].normalize(model_action)


def pure_denoise_plan(policy, sample, keys, horizon, noise_seed, action_pose_repr, device):
    obs, cond = obs_condition(policy, sample, keys, device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))
    initial_noise = torch.randn(
        (1, horizon, policy.action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=policy.dtype)
    denoised = denoise_from_sample(policy, initial_noise, cond, horizon, start_index=0)
    env_obs = obs_to_numpy(sample["obs"])
    abs_plan = model_action_to_abs(policy, denoised, env_obs, action_pose_repr)
    del obs, cond, initial_noise, denoised
    gc.collect()
    return abs_plan


def renoise_tail_and_denoise(policy, clean_abs_tail, sample, keys, noise_level, noise_seed, action_pose_repr, device):
    tail_horizon = clean_abs_tail.shape[0]
    obs, cond = obs_condition(policy, sample, keys, device)
    env_obs = obs_to_numpy(sample["obs"])
    clean_model = abs_to_model_action(policy, clean_abs_tail, env_obs, action_pose_repr, device)

    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    timesteps = scheduler.timesteps
    target_t = int(round((scheduler.config.num_train_timesteps - 1) * float(noise_level)))
    start_index = int(torch.argmin(torch.abs(timesteps.cpu() - target_t)).item())
    noise_t = timesteps[start_index].to(device)
    timestep_batch = noise_t.reshape(1).long()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(
        clean_model.shape,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=clean_model.dtype)
    noised_tail = scheduler.add_noise(clean_model, noise, timestep_batch)
    denoised = denoise_from_sample(policy, noised_tail, cond, tail_horizon, start_index=start_index)
    abs_tail = model_action_to_abs(policy, denoised, env_obs, action_pose_repr)

    del obs, cond, clean_model, noise, noised_tail, denoised
    gc.collect()
    return abs_tail, int(noise_t.detach().cpu().item())


def pose10d_to_pos_quat(pose10d):
    pose_mat = pose10d_to_mat(np.asarray(pose10d[:9], dtype=np.float32)[None])[0]
    quat = Rotation.from_matrix(pose_mat[:3, :3]).as_quat().astype(np.float32)
    return np.asarray(pose10d[:3], dtype=np.float32), quat


def clone_sample_with_predicted_tcp(sample, executed_abs):
    new_sample = {
        "obs": {key: value.clone() for key, value in sample["obs"].items()},
        "action": sample["action"].clone(),
    }
    if "robot_pose_R" not in new_sample["obs"]:
        return new_sample

    obs_len = int(new_sample["obs"]["robot_pose_R"].shape[0])
    tcp_history = np.asarray(executed_abs[-obs_len:], dtype=np.float32)
    if tcp_history.shape[0] < obs_len:
        pad = np.repeat(tcp_history[:1], obs_len - tcp_history.shape[0], axis=0)
        tcp_history = np.concatenate([pad, tcp_history], axis=0)

    pose_values = []
    quat_values = []
    rot6d_values = []
    for pose10d in tcp_history:
        pos, quat = pose10d_to_pos_quat(pose10d)
        pose_values.append(pos)
        quat_values.append(quat)
        rot6d_values.append(np.asarray(pose10d[3:9], dtype=np.float32))

    pose_tensor = new_sample["obs"]["robot_pose_R"]
    new_sample["obs"]["robot_pose_R"] = torch.from_numpy(np.stack(pose_values)).to(dtype=pose_tensor.dtype)
    if "robot_quat_R" in new_sample["obs"]:
        quat_tensor = new_sample["obs"]["robot_quat_R"]
        if quat_tensor.shape[-1] == 4:
            quat_value = np.stack(quat_values)
        elif quat_tensor.shape[-1] == 6:
            quat_value = np.stack(rot6d_values)
        else:
            raise RuntimeError(f"Unsupported robot_quat_R shape: {tuple(quat_tensor.shape)}")
        new_sample["obs"]["robot_quat_R"] = torch.from_numpy(quat_value).to(dtype=quat_tensor.dtype)
    return new_sample


def shifted_dataset_indices(dataset, sample_idx, offsets):
    sampler_indices = np.asarray(dataset.sampler.indices)
    base_start = int(sampler_indices[sample_idx][0])
    start_to_idx = {}
    for idx, row in enumerate(sampler_indices):
        start_to_idx.setdefault(int(row[0]), idx)

    result = {}
    for offset in offsets:
        shifted = start_to_idx.get(base_start + offset)
        if shifted is None:
            shifted = min(sample_idx + offset, len(dataset) - 1)
        result[offset] = shifted
    return result


def load_gt_abs(sample, horizon, action_pose_repr):
    env_obs = obs_to_numpy(sample["obs"])
    action = sample["action"].detach().cpu().numpy()[:horizon]
    return action_to_abs(action, env_obs, action_pose_repr)


def trajectory_errors(pred_abs, gt_abs):
    pos_err_mm = np.linalg.norm(pred_abs[:, :3] - gt_abs[:, :3], axis=-1) * 1000.0
    pred_mat = pose10d_to_mat(pred_abs[:, :9])[:, :3, :3]
    gt_mat = pose10d_to_mat(gt_abs[:, :9])[:, :3, :3]
    pred_rot = Rotation.from_matrix(pred_mat.reshape(-1, 3, 3))
    gt_rot = Rotation.from_matrix(gt_mat.reshape(-1, 3, 3))
    rot_err_deg = np.rad2deg((pred_rot * gt_rot.inv()).magnitude()).reshape(pred_abs.shape[0])
    return pos_err_mm, rot_err_deg


def plot_traj_3d(ax, xyz, label, color, linewidth=1.4, alpha=1.0):
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label, color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=13, alpha=alpha, depthshade=False)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color=color, s=24, alpha=alpha, depthshade=False)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=color, marker="x", s=30, alpha=alpha, depthshade=False)


def label_action_indices(ax, xyz, color):
    for idx, point in enumerate(xyz):
        ax.text(point[0], point[1], point[2], str(idx), color=color, fontsize=7)


def fixed_gt_limits(gt_abs, span_m):
    return centered_fixed_span_limits(gt_abs[:, :3], float(span_m) / 2.0)


def write_receding_png(
        path,
        gt_abs,
        stage_segments,
        stitched_abs,
        sample_idx,
        step_stride,
        noise_level,
        noise_t,
        exec_start):
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.45])
    centers, radius = fixed_gt_limits(gt_abs, 0.035)

    ax_stage = fig.add_subplot(grid[0, 0], projection="3d")
    plot_traj_3d(ax_stage, gt_abs[:, :3], "GT full 16", "#111111", linewidth=2.0)
    for i, (offset, segment_abs) in enumerate(stage_segments.items()):
        color = STAGE_COLORS[i % len(STAGE_COLORS)]
        label = (
            f"t+{offset}: initial tokens"
            if offset == 0
            else f"t+{offset}: executed {offset} + denoise tail"
        )
        plot_traj_3d(
            ax_stage,
            segment_abs[:, :3],
            label,
            color,
            linewidth=1.2,
            alpha=0.82,
        )
    apply_xyz_limits(ax_stage, centers, radius)
    ax_stage.set_title("all re-denoised tails")
    ax_stage.set_xlabel("x")
    ax_stage.set_ylabel("y")
    ax_stage.set_zlabel("z")
    ax_stage.legend(fontsize=6)

    ax_initial = fig.add_subplot(grid[0, 1], projection="3d")
    initial_abs = stage_segments[0]
    plot_traj_3d(ax_initial, gt_abs[:, :3], "GT", "#111111", linewidth=2.0)
    plot_traj_3d(ax_initial, initial_abs[:, :3], "initial 16-step", "#1f77b4", linewidth=1.5)
    apply_xyz_limits(ax_initial, centers, radius)
    ax_initial.set_title("initial full denoise")
    ax_initial.set_xlabel("x")
    ax_initial.set_ylabel("y")
    ax_initial.set_zlabel("z")
    ax_initial.legend(fontsize=7)

    ax_stitched = fig.add_subplot(grid[0, 2], projection="3d")
    plot_traj_3d(ax_stitched, gt_abs[:, :3], "GT", "#111111", linewidth=2.0)
    plot_traj_3d(
        ax_stitched,
        stitched_abs[:, :3],
        f"stitched 2-step receding ({stitched_abs.shape[0]} executable actions)",
        "#d62728",
        linewidth=1.6,
    )
    ax_stitched.scatter(
        stitched_abs[:, 0],
        stitched_abs[:, 1],
        stitched_abs[:, 2],
        color="#d62728",
        s=34,
        depthshade=False,
    )
    label_action_indices(ax_stitched, stitched_abs[:, :3], "#d62728")
    apply_xyz_limits(ax_stitched, centers, radius)
    ax_stitched.set_title(f"stitched receding output ({stitched_abs.shape[0]} executable actions)")
    ax_stitched.set_xlabel("x")
    ax_stitched.set_ylabel("y")
    ax_stitched.set_zlabel("z")
    ax_stitched.legend(fontsize=7)

    ax_pos = fig.add_subplot(grid[1, :2])
    ax_rot = fig.add_subplot(grid[1, 2])
    initial_exec = initial_abs[exec_start:exec_start + stitched_abs.shape[0]]
    gt_exec = gt_abs[exec_start:exec_start + stitched_abs.shape[0]]
    initial_pos, initial_rot = trajectory_errors(initial_exec, gt_exec)
    stitched_pos, stitched_rot = trajectory_errors(stitched_abs, gt_exec)
    t = np.arange(stitched_abs.shape[0])
    ax_pos.plot(t, initial_pos, label="initial 16-step", color="#1f77b4", linewidth=1.8)
    ax_pos.plot(t, stitched_pos, label="stitched receding", color="#d62728", linewidth=1.8)
    for offset in list(stage_segments.keys())[1:]:
        ax_pos.axvline(offset, color="#999999", linewidth=0.7, alpha=0.35)
    ax_pos.set_title("position error")
    ax_pos.set_xlabel("global horizon step")
    ax_pos.set_ylabel("mm")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(fontsize=8)

    ax_rot.plot(t, initial_rot, label="initial 16-step", color="#1f77b4", linewidth=1.8)
    ax_rot.plot(t, stitched_rot, label="stitched receding", color="#d62728", linewidth=1.8)
    for offset in list(stage_segments.keys())[1:]:
        ax_rot.axvline(offset, color="#999999", linewidth=0.7, alpha=0.35)
    ax_rot.set_title("rotation error")
    ax_rot.set_xlabel("global horizon step")
    ax_rot.set_ylabel("deg")
    ax_rot.grid(True, alpha=0.3)
    ax_rot.legend(fontsize=8)

    fig.suptitle(
        f"Dataset idx {sample_idx}: denoise 16, advance {step_stride}, re-noise tail "
        f"{noise_level * 100:.0f}% (DDIM t={noise_t}), repeat | execute from token {exec_start}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_receding_sample_png(
        policy,
        dataset,
        keys,
        action_pose_repr,
        sample_idx,
        args,
        output_dir,
        device):
    offsets = list(range(0, args.horizon, args.step_stride))
    idx_by_offset = shifted_dataset_indices(dataset, sample_idx, offsets)
    samples_by_offset = OrderedDict((offset, dataset[idx_by_offset[offset]]) for offset in offsets)
    base_sample = samples_by_offset[0]
    gt_abs = load_gt_abs(base_sample, args.horizon, action_pose_repr)

    print(f"Sample {sample_idx}: dataset idx by offset: {idx_by_offset}")

    initial_plan = pure_denoise_plan(
        policy=policy,
        sample=base_sample,
        keys=keys,
        horizon=args.horizon,
        noise_seed=args.seed + sample_idx,
        action_pose_repr=action_pose_repr,
        device=device,
    )
    exec_start = int(policy.n_obs_steps) - 1
    current_tail = initial_plan.copy()
    stage_segments = OrderedDict()
    stage_segments[0] = initial_plan.copy()
    stitched_chunks = []
    executed_abs = np.zeros((0, initial_plan.shape[-1]), dtype=np.float32)
    noise_t_used = None

    offset = 0
    while offset < args.horizon:
        available_exec = max(current_tail.shape[0] - exec_start, 0)
        execute_len = min(args.step_stride, available_exec, args.horizon - offset)
        if execute_len <= 0:
            break
        next_chunk = current_tail[exec_start:exec_start + execute_len].copy()
        stitched_chunks.append(next_chunk)
        executed_abs = np.concatenate([executed_abs, next_chunk], axis=0)
        offset += execute_len
        if offset >= args.horizon:
            break

        future_tail = current_tail[exec_start + execute_len:].copy()
        if future_tail.shape[0] <= 0:
            break
        # The transformer sequence includes a current-state-like anchor at token 0.
        # Keep that anchor when re-noising the remaining future tokens.
        clean_tail = np.concatenate([executed_abs[-1:].copy(), future_tail], axis=0)
        remaining = clean_tail.shape[0]

        pred_obs_sample = clone_sample_with_predicted_tcp(samples_by_offset[offset], executed_abs)
        tail_abs, noise_t_used = renoise_tail_and_denoise(
            policy=policy,
            clean_abs_tail=clean_tail,
            sample=pred_obs_sample,
            keys=keys,
            noise_level=args.noise_level,
            noise_seed=args.seed + sample_idx + offset,
            action_pose_repr=action_pose_repr,
            device=device,
        )
        if tail_abs.shape[0] != remaining:
            raise RuntimeError(f"Expected tail length {remaining}, got {tail_abs.shape[0]}")
        # Plot each replan on the global timeline. Drop the tail anchor for plotting
        # because it is the current TCP, not a future action.
        stage_segments[offset] = np.concatenate([executed_abs.copy(), tail_abs[exec_start:].copy()], axis=0)[:args.horizon]
        current_tail = tail_abs.copy()

    stitched_abs = np.concatenate(stitched_chunks, axis=0)
    print(f"Sample {sample_idx}: stitched action count: {stitched_abs.shape[0]}")
    print(f"Sample {sample_idx}: executable token start: {exec_start}")
    print(f"Sample {sample_idx}: stage plan lengths: { {offset: plan.shape[0] for offset, plan in stage_segments.items()} }")
    if noise_t_used is None:
        noise_t_used = 0

    out_path = output_dir / f"sample_{sample_idx:04d}_receding_tail_redenoise.png"
    write_receding_png(
        out_path,
        gt_abs=gt_abs,
        stage_segments=stage_segments,
        stitched_abs=stitched_abs,
        sample_idx=sample_idx,
        step_stride=args.step_stride,
        noise_level=args.noise_level,
        noise_t=noise_t_used,
        exec_start=exec_start,
    )
    print(f"Wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-ckpt", default=DEFAULT_TRANSFORMER_CKPT)
    parser.add_argument("--output-dir", default="data/debug_action_vis")
    parser.add_argument(
        "--run-name",
        default="0511_195130__basic_transformer_e900__receding_tail_50pct_noise_predtcp_sample",
    )
    parser.add_argument("--sample-idx", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--random-samples", action="store_true")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--step-stride", type=int, default=2)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ckpt_path = resolve_path(args.transformer_ckpt)
    output_dir = resolve_path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = load_cfg_for_ckpt(ckpt_path)
    dataset_cfg = make_dataset_cfg(cfg)
    dataset = hydra.utils.instantiate(dataset_cfg)
    action_pose_repr = action_pose_repr_from_cfg(cfg)

    workspace = load_policy_from_checkpoint(ckpt_path, device)
    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    keys = get_policy_obs_keys(policy)

    print(f"Transformer checkpoint: {ckpt_path}")
    print(f"Action pose repr: {action_pose_repr}")
    if args.random_samples:
        max_start = max(len(dataset) - args.horizon, 1)
        rng = np.random.default_rng(args.seed)
        replace = args.num_samples > max_start
        sample_indices = rng.choice(max_start, size=args.num_samples, replace=replace).tolist()
        print(f"Random sample indices: {sample_indices}")
    else:
        sample_indices = range(args.sample_idx, args.sample_idx + args.num_samples)
    for sample_idx in sample_indices:
        make_receding_sample_png(
            policy=policy,
            dataset=dataset,
            keys=keys,
            action_pose_repr=action_pose_repr,
            sample_idx=sample_idx,
            args=args,
            output_dir=output_dir,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
