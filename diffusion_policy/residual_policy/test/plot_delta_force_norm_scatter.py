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
    rows = []
    with h5py.File(dataset, "r") as f:
        for demo_name in sorted_demo_keys(f["data"]):
            obs = f["data"][demo_name]["obs"]
            actual = np.asarray(obs[actual_key], dtype=np.float64)[:, :3]
            virtual = np.asarray(obs[virtual_key], dtype=np.float64)[:, :3]
            residual = np.asarray(obs[residual_key], dtype=np.float64)
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

            delta_vec = virtual - actual
            rows.append({
                "demo": demo_name,
                "delta_pos_norm": np.linalg.norm(delta_vec, axis=1),
                "residual_pos_norm": np.linalg.norm(residual[:, :3], axis=1),
                "residual_rot_norm": np.linalg.norm(residual[:, 3:6], axis=1),
                "force_norm": np.linalg.norm(force, axis=1),
            })
    return rows


def plot_scatter(x, y, xlabel, ylabel, title, output_path):
    pearson = corrcoef(x, y)
    spearman = float(spearmanr(x, y, nan_policy="omit").correlation)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    rng = np.random.default_rng(0)
    idx = np.arange(len(x))
    if len(idx) > 20000:
        idx = rng.choice(idx, size=20000, replace=False)

    axes[0].scatter(x[idx], y[idx], s=5, alpha=0.15, linewidths=0, color="#2563eb")
    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 100)
        axes[0].plot(xs, coef[0] * xs + coef[1], color="#dc2626", linewidth=2)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"sampled scatter\nPearson={pearson:.3f}, Spearman={spearman:.3f}")
    axes[0].grid(True, alpha=0.25)

    hb = axes[1].hexbin(x, y, gridsize=80, bins="log", cmap="viridis", mincnt=1)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title("density view (log count)")
    axes[1].grid(True, alpha=0.2)
    fig.colorbar(hb, ax=axes[1], label="log10 count")

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"pearson": pearson, "spearman": spearman}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/baetae/260618/slow_erase_board_virtual_m_world_wrench.hdf5")
    parser.add_argument("--output-dir", default="plots/force_delta/world/03_norm_scatter")
    parser.add_argument("--actual-key", default="actual_target_abs")
    parser.add_argument("--virtual-key", default="virtual_target_abs")
    parser.add_argument("--residual-key", default="residual_delta6_gt_actual_to_virtual")
    parser.add_argument("--wrench-key", default="wrench_wrist_R")
    parser.add_argument("--force-stat", choices=("last", "mean", "maxabs"), default="last")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(
        dataset=args.dataset,
        actual_key=args.actual_key,
        virtual_key=args.virtual_key,
        residual_key=args.residual_key,
        wrench_key=args.wrench_key,
        force_stat=args.force_stat,
    )
    delta_pos = np.concatenate([row["delta_pos_norm"] for row in rows])
    residual_pos = np.concatenate([row["residual_pos_norm"] for row in rows])
    residual_rot = np.concatenate([row["residual_rot_norm"] for row in rows])
    force = np.concatenate([row["force_norm"] for row in rows])

    metrics = {
        "dataset": args.dataset,
        "force_stat": args.force_stat,
        "num_demos": len(rows),
        "num_samples": int(len(force)),
        "plots": {},
    }
    metrics["plots"]["actual_to_virtual_delta_pos_vs_force_norm"] = plot_scatter(
        delta_pos,
        force,
        "|virtual xyz - actual xyz| (m)",
        "|force| (N)",
        "All demos: actual-to-virtual delta position length vs force norm",
        output_dir / "delta_pos_norm_vs_force_norm.png",
    )
    metrics["plots"]["residual_pos_vs_force_norm"] = plot_scatter(
        residual_pos,
        force,
        "|residual position delta| (m)",
        "|force| (N)",
        "All demos: residual position delta length vs force norm",
        output_dir / "residual_pos_norm_vs_force_norm.png",
    )
    metrics["plots"]["residual_rot_vs_force_norm"] = plot_scatter(
        residual_rot,
        force,
        "|residual rotation delta| (rad)",
        "|force| (N)",
        "All demos: residual rotation delta length vs force norm",
        output_dir / "residual_rot_norm_vs_force_norm.png",
    )

    # Per-demo median overview: useful when individual demos have different force offsets/magnitudes.
    demo_delta = np.asarray([np.median(row["delta_pos_norm"]) for row in rows])
    demo_force = np.asarray([np.median(row["force_norm"]) for row in rows])
    metrics["plots"]["per_demo_median_delta_pos_vs_force_norm"] = plot_scatter(
        demo_delta,
        demo_force,
        "per-demo median |virtual xyz - actual xyz| (m)",
        "per-demo median |force| (N)",
        "Demo-level medians: delta position length vs force norm",
        output_dir / "per_demo_median_delta_pos_norm_vs_force_norm.png",
    )

    metrics["summary"] = {
        "force_norm_median": float(np.median(force)),
        "force_norm_mean": float(np.mean(force)),
        "delta_pos_norm_median_m": float(np.median(delta_pos)),
        "delta_pos_norm_mean_m": float(np.mean(delta_pos)),
        "residual_rot_norm_median_rad": float(np.median(residual_rot)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
