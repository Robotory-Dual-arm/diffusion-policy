if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import copy
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

from diffusion_policy.scripts.bae_visualize_action_ckpt_compare import (
    action_to_abs,
    load_samples,
    make_dataset_cfg,
    predict_abs_actions,
    resolve_path,
    run_dir_from_ckpt,
    trajectory_errors,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


STATIONARY_INDICES = [135, 140, 146, 151, 156, 318]
STRAIGHT_INDICES = [1743, 2119, 2124, 2364, 2974, 4519]


def load_workspace_from_checkpoint(ckpt_path, device, strict=True):
    ckpt_path = resolve_path(ckpt_path)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = copy.deepcopy(payload["cfg"])
    if "obs_encoder" in cfg.policy and "pretrained" in cfg.policy.obs_encoder:
        cfg.policy.obs_encoder.pretrained = False
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
    workspace.model.to(device).eval()
    if workspace.ema_model is not None:
        workspace.ema_model.to(device).eval()
    gc.collect()
    return workspace


def load_cfg_for_ckpt(ckpt_path):
    run_dir = run_dir_from_ckpt(resolve_path(ckpt_path))
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg.is_file():
        return OmegaConf.load(hydra_cfg)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    del payload
    return cfg


def set_equal_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_one(path, title, gt_abs, pred_abs):
    gt_xyz = gt_abs[:, :3]
    pred_xyz = pred_abs[:, :3]
    err_mm, rot_deg = trajectory_errors(pred_abs, gt_abs)
    t = np.arange(gt_abs.shape[0])

    fig = plt.figure(figsize=(15, 5.2))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="#111111", linewidth=3.0, label="GT")
    ax3d.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color="#1f77b4", linewidth=2.4, label="Pred EMA")
    ax3d.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color="#111111", s=32)
    ax3d.scatter(pred_xyz[0, 0], pred_xyz[0, 1], pred_xyz[0, 2], color="#1f77b4", s=32)
    ax3d.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], color="#111111", marker="x", s=52)
    ax3d.scatter(pred_xyz[-1, 0], pred_xyz[-1, 1], pred_xyz[-1, 2], color="#1f77b4", marker="x", s=52)
    set_equal_3d(ax3d, np.concatenate([gt_xyz, pred_xyz], axis=0))
    ax3d.set_title("3D TCP path")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(fontsize=8)

    ax_xyz = fig.add_subplot(1, 3, 2)
    for dim, label, linestyle in zip(range(3), ["x", "y", "z"], ["-", "--", ":"]):
        ax_xyz.plot(t, gt_xyz[:, dim], color="#111111", linestyle=linestyle, linewidth=2.2, label=f"GT {label}")
        ax_xyz.plot(t, pred_xyz[:, dim], color="#1f77b4", linestyle=linestyle, linewidth=1.8, label=f"Pred {label}")
    ax_xyz.set_title("position components")
    ax_xyz.set_xlabel("horizon step")
    ax_xyz.set_ylabel("m")
    ax_xyz.grid(True, alpha=0.3)
    ax_xyz.legend(fontsize=7, ncol=2)

    ax_err = fig.add_subplot(1, 3, 3)
    ax_err.plot(t, err_mm, color="#d62728", linewidth=2.2, label="pos err")
    ax_err.set_xlabel("horizon step")
    ax_err.set_ylabel("position error (mm)", color="#d62728")
    ax_err.tick_params(axis="y", labelcolor="#d62728")
    ax_err.grid(True, alpha=0.3)
    ax_rot = ax_err.twinx()
    ax_rot.plot(t, rot_deg, color="#9467bd", linewidth=2.0, label="rot err")
    ax_rot.set_ylabel("rotation error (deg)", color="#9467bd")
    ax_rot.tick_params(axis="y", labelcolor="#9467bd")
    ax_err.set_title(f"mean {err_mm.mean():.1f} mm, max {err_mm.max():.1f} mm")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_montage(path, image_paths, title):
    imgs = [plt.imread(p) for p in image_paths]
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 9))
    for ax, img, img_path in zip(axes.ravel(), imgs, image_paths):
        ax.imshow(img)
        ax.set_title(pathlib.Path(img_path).stem, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-strict-load", action="store_true")
    args = parser.parse_args()

    ckpt = resolve_path(args.ckpt)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = load_cfg_for_ckpt(ckpt)
    dataset_cfg = make_dataset_cfg(cfg)
    dataset = hydra.utils.instantiate(dataset_cfg)
    dataset_cfg_resolved = OmegaConf.to_container(dataset_cfg, resolve=True)
    action_pose_repr = dataset_cfg_resolved["pose_repr"]["action_pose_repr"]
    indices_by_group = OrderedDict([
        ("stationary", STATIONARY_INDICES),
        ("straight", STRAIGHT_INDICES),
    ])
    all_indices = sorted(set(STATIONARY_INDICES + STRAIGHT_INDICES))
    samples_by_idx, gt_abs_by_idx, _ = load_samples(dataset, all_indices, action_pose_repr)

    workspace = load_workspace_from_checkpoint(ckpt, device, strict=not args.non_strict_load)
    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    pred_by_idx = predict_abs_actions(
        policy=policy,
        samples_by_idx=samples_by_idx,
        indices=all_indices,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        action_pose_repr=action_pose_repr,
    )

    for group, indices in indices_by_group.items():
        group_dir = output_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for idx in indices:
            path = group_dir / f"{group}_idx_{idx:05d}_3d.png"
            plot_one(
                path=path,
                title=f"{group} dataset idx {idx}, 16-step action prediction",
                gt_abs=gt_abs_by_idx[idx],
                pred_abs=pred_by_idx[idx],
            )
            image_paths.append(path)
        make_montage(
            output_dir / f"{group}_montage.png",
            image_paths,
            title=f"{group}: GT vs EMA prediction, {ckpt.name}",
        )

    print(output_dir)


if __name__ == "__main__":
    main()
