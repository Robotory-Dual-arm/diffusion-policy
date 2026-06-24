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

from diffusion_policy.residual_policy.test.analyze_force_delta_relation import (
    corrcoef,
    sorted_demo_keys,
    wrench_features,
)


TARGET_KEY = "residual_delta6_gt_actual_to_virtual"
TARGET_LABELS = ["dx", "dy", "dz", "dRx", "dRy", "dRz"]
FEATURE_FOR_TARGET = {
    0: "last_Fx",
    1: "last_Fz",
    2: "last_Fx",
    3: "last_Fy",
    4: "last_Fz",
    5: "last_Fz",
}


def load_samples(dataset, wrench_key):
    features_all = []
    targets_all = []
    with h5py.File(dataset, "r") as f:
        for demo_name in sorted_demo_keys(f["data"]):
            obs = f["data"][demo_name]["obs"]
            features, feature_names = wrench_features(np.asarray(obs[wrench_key]))
            features_all.append(features)
            targets_all.append(np.asarray(obs[TARGET_KEY], dtype=np.float64))
    return np.concatenate(features_all), np.concatenate(targets_all), feature_names


def density_scatter(ax, x, y, xlabel, ylabel, title, corr):
    rng = np.random.default_rng(0)
    if len(x) > 5000:
        idx = rng.choice(len(x), size=5000, replace=False)
        x = x[idx]
        y = y[idx]
    ax.scatter(x, y, s=4, alpha=0.18, linewidths=0, color="#2563eb")
    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#dc2626", linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nr={corr:.2f}", fontsize=10)
    ax.grid(True, alpha=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/baetae/260618/slow_erase_board_virtual_m_world_wrench.hdf5")
    parser.add_argument("--metrics", default="plots/force_delta/world/01_summary/full_metrics.json")
    parser.add_argument("--output-dir", default="plots/force_delta/world/01_summary")
    parser.add_argument("--wrench-key", default="wrench_wrist_R")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(args.metrics).read_text())
    target_metrics = metrics[TARGET_KEY]["baseline_metrics"]

    features, targets, feature_names = load_samples(args.dataset, args.wrench_key)
    feature_index = {name: i for i, name in enumerate(feature_names)}

    r2_force = np.asarray(target_metrics["force_only"]["r2_axis"], dtype=float)
    r2_base = np.asarray(target_metrics["base_only"]["r2_axis"], dtype=float)
    r2_bf = np.asarray(target_metrics["base_plus_force"]["r2_axis"], dtype=float)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.15, 1.15])

    ax = fig.add_subplot(gs[0, :3])
    x = np.arange(len(TARGET_LABELS))
    width = 0.25
    ax.bar(x - width, r2_force, width, label="force only", color="#2563eb")
    ax.bar(x, r2_base, width, label="base action only", color="#f59e0b")
    ax.bar(x + width, r2_bf, width, label="base + force", color="#16a34a")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS)
    ax.set_ylabel("Validation R2")
    ax.set_title("Can force explain the residual action?")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[0, 3:])
    pos_means = [r2_force[:3].mean(), r2_base[:3].mean(), r2_bf[:3].mean()]
    rot_means = [r2_force[3:].mean(), r2_base[3:].mean(), r2_bf[3:].mean()]
    labels = ["force", "base", "base+force"]
    xx = np.arange(len(labels))
    ax.bar(xx - 0.18, pos_means, 0.36, label="position xyz", color="#22c55e")
    ax.bar(xx + 0.18, rot_means, 0.36, label="rotation xyz", color="#ef4444")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean validation R2")
    ax.set_title("Position has force signal; rotation does not")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    for target_i in range(3):
        feature_name = FEATURE_FOR_TARGET[target_i]
        feat = features[:, feature_index[feature_name]]
        target = targets[:, target_i]
        corr = corrcoef(feat, target)
        ax = fig.add_subplot(gs[1, target_i * 2:(target_i + 1) * 2])
        density_scatter(
            ax,
            feat,
            target,
            feature_name,
            f"residual {TARGET_LABELS[target_i]}",
            f"Position residual {TARGET_LABELS[target_i]}",
            corr,
        )

    for target_i in range(3, 6):
        feature_name = FEATURE_FOR_TARGET[target_i]
        feat = features[:, feature_index[feature_name]]
        target = targets[:, target_i]
        corr = corrcoef(feat, target)
        ax = fig.add_subplot(gs[2, (target_i - 3) * 2:(target_i - 2) * 2])
        density_scatter(
            ax,
            feat,
            target,
            feature_name,
            f"residual {TARGET_LABELS[target_i]}",
            f"Rotation residual {TARGET_LABELS[target_i]}",
            corr,
        )

    fig.suptitle(
        "World-frame wrist force vs fast residual target\n"
        "Strong relation in translation residuals; weak/no relation in rotation residuals",
        fontsize=16,
    )
    out = output_dir / "force_delta_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    text = {
        "short_read": [
            "Force contains a strong signal for position residuals dx/dy/dz.",
            "Force contains little to no signal for rotation residuals dRx/dRy/dRz.",
            "Adding force to base action improves position R2 from %.3f to %.3f."
            % (float(r2_base[:3].mean()), float(r2_bf[:3].mean())),
        ],
        "position_r2": {
            "force_only": float(r2_force[:3].mean()),
            "base_only": float(r2_base[:3].mean()),
            "base_plus_force": float(r2_bf[:3].mean()),
        },
        "rotation_r2": {
            "force_only": float(r2_force[3:].mean()),
            "base_only": float(r2_base[3:].mean()),
            "base_plus_force": float(r2_bf[3:].mean()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(text, indent=2))
    print(out)


if __name__ == "__main__":
    main()
