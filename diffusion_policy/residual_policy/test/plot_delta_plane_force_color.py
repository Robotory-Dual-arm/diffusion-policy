#!/usr/bin/env python3
if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def corrcoef(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def collect(dataset, actual_key, virtual_key, residual_key, wrench_key, force_stat):
    delta_h = []
    delta_z = []
    delta_norm = []
    angle_deg = []
    force_norm = []
    demo_id = []
    with h5py.File(dataset, "r") as f:
        for i, demo_name in enumerate(sorted_demo_keys(f["data"])):
            obs = f["data"][demo_name]["obs"]
            if actual_key and virtual_key:
                actual = np.asarray(obs[actual_key], dtype=np.float64)[:, :3]
                virtual = np.asarray(obs[virtual_key], dtype=np.float64)[:, :3]
                delta = virtual - actual
            else:
                delta = np.asarray(obs[residual_key], dtype=np.float64)[:, :3]
            wrench = np.asarray(obs[wrench_key], dtype=np.float64)[:, :3]
            if force_stat == "last":
                force = wrench[:, :, -1]
            elif force_stat == "mean":
                force = wrench.mean(axis=-1)
            elif force_stat == "maxabs":
                idx = np.argmax(np.abs(wrench), axis=-1)
                force = np.take_along_axis(wrench, idx[:, :, None], axis=-1)[:, :, 0]
            else:
                raise ValueError(f"Unsupported force_stat: {force_stat}")

            h = np.linalg.norm(delta[:, :2], axis=1)
            z = delta[:, 2]
            norm = np.linalg.norm(delta, axis=1)
            angle = np.degrees(np.arctan2(np.abs(z), np.maximum(h, 1e-12)))
            fn = np.linalg.norm(force, axis=1)

            delta_h.append(h)
            delta_z.append(z)
            delta_norm.append(norm)
            angle_deg.append(angle)
            force_norm.append(fn)
            demo_id.append(np.full(len(h), i, dtype=np.int32))

    return {
        "delta_h": np.concatenate(delta_h),
        "delta_z": np.concatenate(delta_z),
        "delta_norm": np.concatenate(delta_norm),
        "angle_deg": np.concatenate(angle_deg),
        "force_norm": np.concatenate(force_norm),
        "demo_id": np.concatenate(demo_id),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/baetae/260618/slow_erase_board_virtual_m_world_wrench.hdf5")
    parser.add_argument("--output-dir", default="plots/force_delta/world/04_delta_plane")
    parser.add_argument("--actual-key", default="actual_target_abs")
    parser.add_argument("--virtual-key", default="virtual_target_abs")
    parser.add_argument("--residual-key", default="residual_delta6_gt_actual_to_virtual")
    parser.add_argument("--wrench-key", default="wrench_wrist_R")
    parser.add_argument("--force-stat", choices=("last", "mean", "maxabs"), default="last")
    parser.add_argument("--max-points", type=int, default=60000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = collect(
        args.dataset,
        args.actual_key,
        args.virtual_key,
        args.residual_key,
        args.wrench_key,
        args.force_stat,
    )

    n = len(data["force_norm"])
    idx = np.arange(n)
    if n > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=args.max_points, replace=False)

    x_mm = data["delta_h"][idx] * 1000.0
    y_mm = data["delta_z"][idx] * 1000.0
    color = data["force_norm"][idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    sc = axes[0].scatter(
        x_mm,
        y_mm,
        c=color,
        s=6,
        alpha=0.35,
        linewidths=0,
        cmap="viridis",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("horizontal delta length sqrt(dx^2 + dy^2) (mm)")
    axes[0].set_ylabel("vertical delta dz (mm)")
    axes[0].set_title("Each timestep: delta direction/steepness colored by force norm")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[0], label="|force| (N)")

    hb = axes[1].hexbin(
        data["delta_h"] * 1000.0,
        data["delta_z"] * 1000.0,
        C=data["force_norm"],
        reduce_C_function=np.median,
        gridsize=70,
        mincnt=3,
        cmap="viridis",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("horizontal delta length sqrt(dx^2 + dy^2) (mm)")
    axes[1].set_ylabel("vertical delta dz (mm)")
    axes[1].set_title("Binned view: median force norm per delta bin")
    axes[1].grid(True, alpha=0.25)
    fig.colorbar(hb, ax=axes[1], label="median |force| (N)")

    fig.suptitle("Delta horizontal/vertical plane with force magnitude as color", fontsize=15)
    fig.savefig(output_dir / "delta_plane_force_color.png", dpi=180)
    plt.close(fig)

    metrics = {
        "dataset": args.dataset,
        "num_samples": int(n),
        "force_stat": args.force_stat,
        "corr": {
            "force_vs_delta_horizontal": {
                "pearson": corrcoef(data["force_norm"], data["delta_h"]),
                "spearman": float(spearmanr(data["force_norm"], data["delta_h"], nan_policy="omit").correlation),
            },
            "force_vs_delta_z_signed": {
                "pearson": corrcoef(data["force_norm"], data["delta_z"]),
                "spearman": float(spearmanr(data["force_norm"], data["delta_z"], nan_policy="omit").correlation),
            },
            "force_vs_abs_delta_z": {
                "pearson": corrcoef(data["force_norm"], np.abs(data["delta_z"])),
                "spearman": float(spearmanr(data["force_norm"], np.abs(data["delta_z"]), nan_policy="omit").correlation),
            },
            "force_vs_vertical_angle_deg": {
                "pearson": corrcoef(data["force_norm"], data["angle_deg"]),
                "spearman": float(spearmanr(data["force_norm"], data["angle_deg"], nan_policy="omit").correlation),
            },
        },
        "summary": {
            "force_norm_median": float(np.median(data["force_norm"])),
            "delta_horizontal_median_mm": float(np.median(data["delta_h"]) * 1000.0),
            "delta_z_median_mm": float(np.median(data["delta_z"]) * 1000.0),
            "vertical_angle_median_deg": float(np.median(data["angle_deg"])),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(output_dir / "delta_plane_force_color.png")


if __name__ == "__main__":
    main()
