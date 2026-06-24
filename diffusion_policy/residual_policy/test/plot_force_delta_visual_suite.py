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
import plotly.express as px
from scipy.stats import binned_statistic, spearmanr


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


def load_data(dataset, actual_key, virtual_key, wrench_key, force_stat):
    rows = []
    with h5py.File(dataset, "r") as f:
        for demo_i, demo_name in enumerate(sorted_demo_keys(f["data"])):
            obs = f["data"][demo_name]["obs"]
            actual = np.asarray(obs[actual_key], dtype=np.float64)[:, :3]
            virtual = np.asarray(obs[virtual_key], dtype=np.float64)[:, :3]
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

            delta = virtual - actual
            delta_h = np.linalg.norm(delta[:, :2], axis=1)
            delta_z = delta[:, 2]
            delta_norm = np.linalg.norm(delta, axis=1)
            force_norm = np.linalg.norm(force, axis=1)
            angle_deg = np.degrees(np.arctan2(np.abs(delta_z), np.maximum(delta_h, 1e-12)))
            slope = delta_z / np.maximum(delta_h, 1e-12)
            rows.append({
                "demo": np.asarray([demo_name] * len(delta_h), dtype=object),
                "demo_i": np.full(len(delta_h), demo_i, dtype=np.int32),
                "step": np.arange(len(delta_h), dtype=np.int32),
                "delta_x_mm": delta[:, 0] * 1000.0,
                "delta_y_mm": delta[:, 1] * 1000.0,
                "delta_z_mm": delta_z * 1000.0,
                "delta_h_mm": delta_h * 1000.0,
                "delta_norm_mm": delta_norm * 1000.0,
                "angle_deg": angle_deg,
                "slope": slope,
                "force_norm": force_norm,
                "force_x": force[:, 0],
                "force_y": force[:, 1],
                "force_z": force[:, 2],
            })
    data = {}
    for key in rows[0]:
        data[key] = np.concatenate([row[key] for row in rows])
    return data


def filter_by_force(data, min_force_norm):
    if min_force_norm <= 0:
        return data
    mask = data["force_norm"] >= min_force_norm
    return {key: value[mask] for key, value in data.items()}


def sample_indices(n, max_points):
    idx = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_points, replace=False)
    return idx


def add_corr_title(ax, title, x, y):
    p = corrcoef(x, y)
    s = float(spearmanr(x, y, nan_policy="omit").correlation)
    ax.set_title(f"{title}\nr={p:.2f}, rho={s:.2f}", fontsize=10)


def plot_overview(data, out_dir, max_points):
    idx = sample_indices(len(data["force_norm"]), max_points)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    pairs = [
        ("force_norm", "delta_norm_mm", "|F| (N)", "|delta| (mm)", "force vs total delta"),
        ("force_norm", "delta_h_mm", "|F| (N)", "horizontal delta (mm)", "force vs horizontal delta"),
        ("force_norm", "delta_z_mm", "|F| (N)", "signed dz (mm)", "force vs vertical dz"),
        ("force_norm", "angle_deg", "|F| (N)", "vertical angle (deg)", "force vs steepness"),
        ("delta_h_mm", "delta_z_mm", "horizontal delta (mm)", "signed dz (mm)", "delta plane, color=force"),
        ("force_z", "delta_h_mm", "Fz world (N)", "horizontal delta (mm)", "Fz vs horizontal delta"),
    ]
    for ax, (xk, yk, xlabel, ylabel, title) in zip(axes.ravel(), pairs):
        if title.startswith("delta plane"):
            sc = ax.scatter(
                data[xk][idx],
                data[yk][idx],
                c=data["force_norm"][idx],
                cmap="viridis",
                s=6,
                alpha=0.35,
                linewidths=0,
            )
            fig.colorbar(sc, ax=ax, label="|F| (N)")
        else:
            ax.scatter(data[xk][idx], data[yk][idx], s=5, alpha=0.18, linewidths=0, color="#2563eb")
            mask = np.isfinite(data[xk]) & np.isfinite(data[yk])
            if np.std(data[xk][mask]) > 1e-12 and np.std(data[yk][mask]) > 1e-12:
                coef = np.polyfit(data[xk][mask], data[yk][mask], deg=1)
                xs = np.linspace(np.percentile(data[xk][mask], 1), np.percentile(data[xk][mask], 99), 100)
                ax.plot(xs, coef[0] * xs + coef[1], color="#dc2626", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        add_corr_title(ax, title, data[xk], data[yk])
        ax.grid(True, alpha=0.25)
    fig.suptitle("Force magnitude vs delta decomposition overview", fontsize=16)
    path = out_dir / "01_overview_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def force_bin_labels(force):
    qs = np.quantile(force, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.digitize(force, qs[1:-1], right=True)
    labels = np.asarray([
        f"Q{i+1}: {qs[i]:.1f}-{qs[i+1]:.1f}N"
        for i in range(4)
    ], dtype=object)
    return bins, labels, qs


def plot_force_quartiles(data, out_dir, max_points):
    bins, labels, qs = force_bin_labels(data["force_norm"])
    idx = sample_indices(len(data["force_norm"]), max_points)
    colors = np.asarray(["#93c5fd", "#38bdf8", "#f59e0b", "#dc2626"])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for bin_i in range(4):
        mask = idx[bins[idx] == bin_i]
        ax.scatter(
            data["delta_h_mm"][mask],
            data["delta_z_mm"][mask],
            s=7,
            alpha=0.22,
            linewidths=0,
            color=colors[bin_i],
            label=labels[bin_i],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("horizontal delta (mm)")
    ax.set_ylabel("signed dz (mm)")
    ax.set_title("Delta plane split by force quartile")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "02_delta_plane_force_quartiles.png", dpi=180)
    plt.close(fig)
    return qs


def plot_binned_response(data, out_dir):
    force = data["force_norm"]
    edges = np.quantile(force, np.linspace(0, 1, 16))
    edges = np.unique(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    signals = [
        ("delta_h_mm", "horizontal delta (mm)", "#2563eb"),
        ("delta_z_mm", "signed dz (mm)", "#16a34a"),
        ("angle_deg", "vertical angle (deg)", "#dc2626"),
        ("delta_norm_mm", "total delta (mm)", "#7c3aed"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (key, ylabel, color) in zip(axes.ravel(), signals):
        med, _, _ = binned_statistic(force, data[key], statistic="median", bins=edges)
        q25, _, _ = binned_statistic(force, data[key], statistic=lambda x: np.percentile(x, 25), bins=edges)
        q75, _, _ = binned_statistic(force, data[key], statistic=lambda x: np.percentile(x, 75), bins=edges)
        ax.plot(centers, med, marker="o", color=color, linewidth=2)
        ax.fill_between(centers, q25, q75, color=color, alpha=0.18)
        ax.set_xlabel("|force| bin center (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"median {ylabel} by force bin")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Binned response: what changes as force gets larger?", fontsize=16)
    fig.savefig(out_dir / "03_binned_response_by_force.png", dpi=180)
    plt.close(fig)


def plot_distributions(data, out_dir):
    bins, labels, _ = force_bin_labels(data["force_norm"])
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    keys = [
        ("delta_norm_mm", "|delta| (mm)"),
        ("delta_h_mm", "horizontal delta (mm)"),
        ("delta_z_mm", "signed dz (mm)"),
        ("angle_deg", "vertical angle (deg)"),
    ]
    for ax, (key, ylabel) in zip(axes, keys):
        values = [data[key][bins == i] for i in range(4)]
        ax.boxplot(values, labels=[f"Q{i+1}" for i in range(4)], showfliers=False)
        ax.set_xlabel("force quartile")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Delta distribution by force quartile", fontsize=16)
    fig.savefig(out_dir / "04_boxplots_by_force_quartile.png", dpi=180)
    plt.close(fig)


def plot_demo_medians(data, out_dir):
    demos = np.unique(data["demo_i"])
    demo_h = []
    demo_z = []
    demo_force = []
    demo_angle = []
    for demo_i in demos:
        mask = data["demo_i"] == demo_i
        demo_h.append(np.median(data["delta_h_mm"][mask]))
        demo_z.append(np.median(data["delta_z_mm"][mask]))
        demo_force.append(np.median(data["force_norm"][mask]))
        demo_angle.append(np.median(data["angle_deg"][mask]))
    demo_h = np.asarray(demo_h)
    demo_z = np.asarray(demo_z)
    demo_force = np.asarray(demo_force)
    demo_angle = np.asarray(demo_angle)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc = axes[0].scatter(demo_h, demo_z, c=demo_force, cmap="viridis", s=45, alpha=0.9)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("demo median horizontal delta (mm)")
    axes[0].set_ylabel("demo median signed dz (mm)")
    axes[0].set_title("Each dot is one demo, color=median force")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[0], label="median |force| (N)")

    axes[1].scatter(demo_force, demo_angle, s=45, alpha=0.85, color="#7c3aed")
    axes[1].set_xlabel("demo median |force| (N)")
    axes[1].set_ylabel("demo median vertical angle (deg)")
    add_corr_title(axes[1], "demo median force vs angle", demo_force, demo_angle)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Demo-level view", fontsize=16)
    fig.savefig(out_dir / "05_demo_level_medians.png", dpi=180)
    plt.close(fig)


def plot_interactive(data, out_dir, max_points):
    idx = sample_indices(len(data["force_norm"]), max_points)
    hover = {
        "demo": data["demo"][idx],
        "step": data["step"][idx],
        "force_norm": data["force_norm"][idx],
        "delta_h_mm": data["delta_h_mm"][idx],
        "delta_z_mm": data["delta_z_mm"][idx],
        "angle_deg": data["angle_deg"][idx],
    }
    fig = px.scatter(
        x=data["delta_h_mm"][idx],
        y=data["delta_z_mm"][idx],
        color=data["force_norm"][idx],
        color_continuous_scale="Viridis",
        labels={
            "x": "horizontal delta (mm)",
            "y": "signed dz (mm)",
            "color": "|force| (N)",
        },
        title="Interactive delta plane: color is force norm",
        hover_data=hover,
        opacity=0.45,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(width=1050, height=760, template="plotly_white")
    fig.write_html(out_dir / "06_interactive_delta_plane.html", include_plotlyjs="cdn", full_html=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/baetae/260618/slow_erase_board_virtual_m_world_wrench.hdf5")
    parser.add_argument("--output-dir", default="plots/force_delta/world/02_visual_suite")
    parser.add_argument("--actual-key", default="actual_target_abs")
    parser.add_argument("--virtual-key", default="virtual_target_abs")
    parser.add_argument("--wrench-key", default="wrench_wrist_R")
    parser.add_argument("--force-stat", choices=("last", "mean", "maxabs"), default="last")
    parser.add_argument("--min-force-norm", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=60000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.dataset, args.actual_key, args.virtual_key, args.wrench_key, args.force_stat)
    total_samples = len(data["force_norm"])
    data = filter_by_force(data, args.min_force_norm)
    if len(data["force_norm"]) == 0:
        raise ValueError(f"No samples left after min-force-norm={args.min_force_norm}")

    plot_overview(data, out_dir, args.max_points)
    qs = plot_force_quartiles(data, out_dir, args.max_points)
    plot_binned_response(data, out_dir)
    plot_distributions(data, out_dir)
    plot_demo_medians(data, out_dir)
    plot_interactive(data, out_dir, min(args.max_points, 20000))

    metrics = {
        "dataset": args.dataset,
        "min_force_norm": float(args.min_force_norm),
        "num_samples_before_filter": int(total_samples),
        "num_samples": int(len(data["force_norm"])),
        "force_quartile_edges": qs.tolist(),
        "correlations": {
            "force_vs_delta_norm": {
                "pearson": corrcoef(data["force_norm"], data["delta_norm_mm"]),
                "spearman": float(spearmanr(data["force_norm"], data["delta_norm_mm"], nan_policy="omit").correlation),
            },
            "force_vs_delta_horizontal": {
                "pearson": corrcoef(data["force_norm"], data["delta_h_mm"]),
                "spearman": float(spearmanr(data["force_norm"], data["delta_h_mm"], nan_policy="omit").correlation),
            },
            "force_vs_signed_dz": {
                "pearson": corrcoef(data["force_norm"], data["delta_z_mm"]),
                "spearman": float(spearmanr(data["force_norm"], data["delta_z_mm"], nan_policy="omit").correlation),
            },
            "force_vs_vertical_angle": {
                "pearson": corrcoef(data["force_norm"], data["angle_deg"]),
                "spearman": float(spearmanr(data["force_norm"], data["angle_deg"], nan_policy="omit").correlation),
            },
        },
        "summary": {
            "force_median": float(np.median(data["force_norm"])),
            "delta_norm_median_mm": float(np.median(data["delta_norm_mm"])),
            "delta_h_median_mm": float(np.median(data["delta_h_mm"])),
            "delta_z_median_mm": float(np.median(data["delta_z_mm"])),
            "angle_median_deg": float(np.median(data["angle_deg"])),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
