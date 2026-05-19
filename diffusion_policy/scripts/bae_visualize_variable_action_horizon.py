if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import gc
from collections import OrderedDict

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.scripts.bae_visualize_action_ckpt_compare import (
    action_pose_repr_from_cfg,
    action_to_abs,
    apply_xyz_limits,
    centered_fixed_span_limits,
    get_policy_obs_keys,
    load_cfg_for_ckpt,
    load_policy_from_checkpoint,
    make_dataset_cfg,
    obs_to_numpy,
    pose10d_to_mat,
    resolve_path,
)
from scipy.spatial.transform import Rotation


OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_TRANSFORMER_CKPT = (
    "data/outputs/2026.05.11/19.51.30_train_diffusion_transformer_hybrid_"
    "bbbae_dualarm_insert_plug_no_wrench/checkpoints/epoch=0900-train_loss=0.006.ckpt"
)

MODEL_COLORS = {
    "GT": "#111111",
    4: "#1f77b4",
    8: "#ff7f0e",
    12: "#2ca02c",
    16: "#9467bd",
}


def collate_obs(samples, keys, device):
    return {
        key: torch.stack([sample["obs"][key] for sample in samples], dim=0).to(device)
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


def build_initial_noise_by_idx(indices, max_horizon, action_dim, seed):
    noise_by_idx = {}
    for idx in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(idx))
        noise_by_idx[idx] = torch.randn(
            max_horizon,
            action_dim,
            generator=generator,
            dtype=torch.float32,
        )
    return noise_by_idx


def conditional_sample_from_noise(policy, initial_noise, condition_data, condition_mask, cond):
    model = policy.model
    scheduler = policy.noise_scheduler
    trajectory = initial_noise.clone()
    scheduler.set_timesteps(policy.num_inference_steps)

    for t in scheduler.timesteps:
        trajectory[condition_mask] = condition_data[condition_mask]
        model_output = model(trajectory, t, cond)
        trajectory = scheduler.step(
            model_output,
            t,
            trajectory,
            **policy.kwargs,
        ).prev_sample

    trajectory[condition_mask] = condition_data[condition_mask]
    return trajectory


def predict_token_horizon(
        policy,
        samples_by_idx,
        indices,
        token_horizon,
        initial_noise_by_idx,
        device,
        batch_size,
        action_pose_repr):
    policy.eval()
    keys = get_policy_obs_keys(policy)
    pred_by_idx = {}

    old_steps = getattr(policy, "num_inference_steps", None)
    # Keep denoising iterations fixed. The varying quantity here is action token count.
    policy.num_inference_steps = old_steps

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_samples = [samples_by_idx[idx] for idx in batch_indices]
            obs = collate_obs(batch_samples, keys, device)
            nobs = policy.normalizer.normalize(obs)
            value = next(iter(nobs.values()))
            batch_size_actual = value.shape[0]
            tobs = policy.n_obs_steps
            action_dim = policy.action_dim
            dtype = policy.dtype

            if hasattr(policy, "obs_as_cond") and not policy.obs_as_cond:
                raise RuntimeError("This variable-horizon visualizer currently expects obs_as_cond=True.")

            if hasattr(policy, "obs_as_cond"):
                this_nobs = dict_apply(nobs, lambda x: x[:, :tobs, ...].reshape(-1, *x.shape[2:]))
                nobs_features = policy.obs_encoder(this_nobs)
                cond = nobs_features.reshape(batch_size_actual, tobs, -1)
            else:
                this_nobs = None
                nobs_features = None
                cond = policy.obs_encoder(nobs)
            cond_data = torch.zeros(
                size=(batch_size_actual, token_horizon, action_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            initial_noise = torch.stack(
                [initial_noise_by_idx[idx][:token_horizon] for idx in batch_indices],
                dim=0,
            ).to(device=device, dtype=dtype)

            old_mask, old_memory_mask = temporary_action_masks(policy.model, token_horizon)
            try:
                nsample = conditional_sample_from_noise(
                    policy=policy,
                    initial_noise=initial_noise,
                    condition_data=cond_data,
                    condition_mask=cond_mask,
                    cond=cond,
                )
            finally:
                restore_action_masks(policy.model, old_mask, old_memory_mask)

            naction_pred = nsample[..., :action_dim]
            action_pred = policy.normalizer["action"].unnormalize(naction_pred).detach().cpu().numpy()

            for local_idx, dataset_idx in enumerate(batch_indices):
                env_obs = obs_to_numpy(samples_by_idx[dataset_idx]["obs"])
                pred_by_idx[dataset_idx] = action_to_abs(
                    action_pred[local_idx],
                    env_obs,
                    action_pose_repr,
                )
            del obs, nobs, this_nobs, nobs_features, cond, cond_data, cond_mask
            del initial_noise, nsample, naction_pred, action_pred
            gc.collect()
    return pred_by_idx


def load_samples(dataset, indices, action_pose_repr, max_horizon):
    samples_by_idx = {}
    gt_abs_by_idx = {}
    for idx in indices:
        sample = dataset[idx]
        samples_by_idx[idx] = sample
        env_obs = obs_to_numpy(sample["obs"])
        action = sample["action"].detach().cpu().numpy()[:max_horizon]
        gt_abs_by_idx[idx] = action_to_abs(action, env_obs, action_pose_repr)
    return samples_by_idx, gt_abs_by_idx


def trajectory_errors(pred_abs, gt_abs):
    pos_err_mm = np.linalg.norm(pred_abs[:, :3] - gt_abs[:, :3], axis=-1) * 1000.0
    pred_mat = pose10d_to_mat(pred_abs[:, :9])[:, :3, :3]
    gt_mat = pose10d_to_mat(gt_abs[:, :9])[:, :3, :3]
    pred_rot = Rotation.from_matrix(pred_mat.reshape(-1, 3, 3))
    gt_rot = Rotation.from_matrix(gt_mat.reshape(-1, 3, 3))
    rot_err_deg = np.rad2deg((pred_rot * gt_rot.inv()).magnitude()).reshape(pred_abs.shape[0])
    return pos_err_mm, rot_err_deg


def plot_traj_3d(ax, xyz, label, color, linewidth=1.4):
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label, color=color, linewidth=linewidth)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=14, depthshade=False)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color=color, s=24, depthshade=False)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=color, marker="x", s=30, depthshade=False)


def fixed_limits_for_gt(gt_abs, span_m):
    return centered_fixed_span_limits(gt_abs[:, :3], float(span_m) / 2.0)


def write_sample_png(path, dataset_idx, gt_abs, preds_by_horizon, axis_span_m):
    horizons = list(preds_by_horizon.keys())
    plot_cols = max(2, len(horizons))
    fig = plt.figure(figsize=(5.0 * plot_cols, 8.0))
    grid = fig.add_gridspec(2, plot_cols, height_ratios=[3.0, 1.35])

    for col, horizon in enumerate(horizons):
        pred_abs = preds_by_horizon[horizon]
        gt_slice = gt_abs[:horizon]
        centers, radius = fixed_limits_for_gt(gt_slice, axis_span_m)

        ax = fig.add_subplot(grid[0, col], projection="3d")
        plot_traj_3d(ax, gt_slice[:, :3], "GT", MODEL_COLORS["GT"], linewidth=1.8)
        plot_traj_3d(ax, pred_abs[:, :3], f"{horizon}-step infer", MODEL_COLORS[horizon], linewidth=1.4)
        apply_xyz_limits(ax, centers, radius)
        ax.set_title(f"GT + {horizon}-step infer", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)

    split_col = max(1, plot_cols // 2)
    ax_pos = fig.add_subplot(grid[1, :split_col])
    ax_rot = fig.add_subplot(grid[1, split_col:])
    for horizon, pred_abs in preds_by_horizon.items():
        gt_slice = gt_abs[:horizon]
        pos_err_mm, rot_err_deg = trajectory_errors(pred_abs, gt_slice)
        t = np.arange(horizon)
        color = MODEL_COLORS[horizon]
        ax_pos.plot(t, pos_err_mm, label=f"{horizon}-step", color=color, linewidth=1.6)
        ax_rot.plot(t, rot_err_deg, label=f"{horizon}-step", color=color, linewidth=1.6)

    ax_pos.set_title("position error")
    ax_pos.set_xlabel("horizon step")
    ax_pos.set_ylabel("mm")
    ax_rot.set_title("rotation error")
    ax_rot.set_xlabel("horizon step")
    ax_rot.set_ylabel("deg")
    for ax in (ax_pos, ax_rot):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(f"Dataset idx {dataset_idx}: variable action-token horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-ckpt", default=DEFAULT_TRANSFORMER_CKPT)
    parser.add_argument("--output-dir", default="data/debug_action_vis")
    parser.add_argument(
        "--run-name",
        default="0511_195130__basic_transformer_e900__action_horizon_4_8_12_16__3cm",
    )
    parser.add_argument("--horizons", type=int, nargs="*", default=[4, 8, 12, 16])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--traj-start", type=int, default=0)
    parser.add_argument("--traj-count", type=int, default=30)
    parser.add_argument("--sample-idx", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--axis-span-m", type=float, default=0.03)
    args = parser.parse_args()

    ckpt_path = resolve_path(args.transformer_ckpt)
    output_dir = resolve_path(args.output_dir) / args.run_name
    traj_dir = output_dir / "traj_variable_horizon"
    traj_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = load_cfg_for_ckpt(ckpt_path)
    dataset_cfg = make_dataset_cfg(cfg)
    dataset = hydra.utils.instantiate(dataset_cfg)
    dataset_len = len(dataset)
    resolved_dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    action_pose_repr = action_pose_repr_from_cfg(cfg)

    sample_indices = np.linspace(0, dataset_len - 1, args.n_samples, dtype=int).tolist()
    traj_indices = sample_indices[args.traj_start:args.traj_start + args.traj_count]
    all_indices = sorted(set(traj_indices + [args.sample_idx]))
    max_horizon = max(args.horizons)

    print(f"Transformer checkpoint: {ckpt_path}")
    print(f"Dataset path: {resolved_dataset_cfg['dataset_path']}")
    print(f"Action pose repr: {action_pose_repr}")
    print(f"Token horizons: {args.horizons}")

    samples_by_idx, gt_abs_by_idx = load_samples(
        dataset=dataset,
        indices=all_indices,
        action_pose_repr=action_pose_repr,
        max_horizon=max_horizon,
    )

    workspace = load_policy_from_checkpoint(ckpt_path, device)
    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    initial_noise_by_idx = build_initial_noise_by_idx(
        indices=all_indices,
        max_horizon=max_horizon,
        action_dim=policy.action_dim,
        seed=args.seed,
    )

    preds_by_horizon = OrderedDict()
    for horizon in args.horizons:
        print(f"Predicting {horizon}-token action horizon")
        preds_by_horizon[horizon] = predict_token_horizon(
            policy=policy,
            samples_by_idx=samples_by_idx,
            indices=all_indices,
            token_horizon=horizon,
            initial_noise_by_idx=initial_noise_by_idx,
            device=device,
            batch_size=args.batch_size,
            action_pose_repr=action_pose_repr,
        )

    for local_idx, dataset_idx in enumerate(traj_indices):
        sample_preds = OrderedDict(
            (horizon, preds_by_horizon[horizon][dataset_idx])
            for horizon in args.horizons
        )
        out_path = traj_dir / f"traj_{local_idx:02d}_dataset_idx_{dataset_idx:05d}.png"
        write_sample_png(
            out_path,
            dataset_idx=dataset_idx,
            gt_abs=gt_abs_by_idx[dataset_idx],
            preds_by_horizon=sample_preds,
            axis_span_m=args.axis_span_m,
        )

    sample_preds = OrderedDict(
        (horizon, preds_by_horizon[horizon][args.sample_idx])
        for horizon in args.horizons
    )
    write_sample_png(
        output_dir / f"sample_{args.sample_idx:04d}_variable_horizon.png",
        dataset_idx=args.sample_idx,
        gt_abs=gt_abs_by_idx[args.sample_idx],
        preds_by_horizon=sample_preds,
        axis_span_m=args.axis_span_m,
    )
    print(f"Wrote {traj_dir}")
    print(f"Wrote {output_dir / f'sample_{args.sample_idx:04d}_variable_horizon.png'}")


if __name__ == "__main__":
    main()
