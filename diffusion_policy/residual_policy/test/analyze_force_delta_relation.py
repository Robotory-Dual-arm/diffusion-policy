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
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def wrench_features(wrench):
    wrench = np.asarray(wrench, dtype=np.float32)
    force = wrench[:, :3]
    torque = wrench[:, 3:6]
    pieces = []
    names = []
    stats = {
        "last": wrench[:, :, -1],
        "mean": wrench.mean(axis=-1),
        "std": wrench.std(axis=-1),
        "min": wrench.min(axis=-1),
        "max": wrench.max(axis=-1),
        "delta": wrench[:, :, -1] - wrench[:, :, 0],
        "absmax": np.max(np.abs(wrench), axis=-1),
    }
    axes = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    for stat_name, value in stats.items():
        pieces.append(value)
        names.extend([f"{stat_name}_{axis}" for axis in axes])

    norm_stats = {
        "last_force_norm": np.linalg.norm(force[:, :, -1], axis=1),
        "mean_force_norm": np.linalg.norm(force.mean(axis=-1), axis=1),
        "std_force_norm": force.std(axis=-1).mean(axis=1),
        "last_torque_norm": np.linalg.norm(torque[:, :, -1], axis=1),
        "mean_torque_norm": np.linalg.norm(torque.mean(axis=-1), axis=1),
        "std_torque_norm": torque.std(axis=-1).mean(axis=1),
    }
    pieces.append(np.stack(list(norm_stats.values()), axis=1))
    names.extend(norm_stats.keys())
    return np.concatenate(pieces, axis=1), names


def standardize(train, val):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (val - mean) / std


def ridge_fit_predict(x_train, y_train, x_val, alpha):
    x_train = np.concatenate([x_train, np.ones((len(x_train), 1), dtype=x_train.dtype)], axis=1)
    x_val = np.concatenate([x_val, np.ones((len(x_val), 1), dtype=x_val.dtype)], axis=1)
    eye = np.eye(x_train.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + alpha * eye, x_train.T @ y_train)
    return x_val @ coef


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return r2


def load_arrays(path, wrench_key, target_keys, base_key):
    demo_arrays = []
    with h5py.File(path, "r") as f:
        for demo_name in sorted_demo_keys(f["data"]):
            obs = f["data"][demo_name]["obs"]
            features, feature_names = wrench_features(np.asarray(obs[wrench_key]))
            row = {
                "demo": demo_name,
                "force_features": features.astype(np.float64),
                "base": np.asarray(obs[base_key], dtype=np.float64),
            }
            for key in target_keys:
                row[key] = np.asarray(obs[key], dtype=np.float64)
            demo_arrays.append(row)
    return demo_arrays, feature_names


def concat_by_demos(demos, key):
    return np.concatenate([demo[key] for demo in demos], axis=0)


def heatmap(matrix, xlabels, ylabels, title, path, vmin=-1.0, vmax=1.0):
    fig_w = max(8, len(xlabels) * 0.25)
    fig_h = max(4, len(ylabels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_target(all_demos, train_demos, val_demos, feature_names, target_key, output_dir):
    target = concat_by_demos(all_demos, target_key)
    force = concat_by_demos(all_demos, "force_features")
    base = concat_by_demos(all_demos, "base")
    target_names = [f"{target_key}_{i}" for i in range(target.shape[1])]

    pearson = np.zeros((target.shape[1], force.shape[1]), dtype=np.float64)
    spearman = np.zeros_like(pearson)
    for i in range(target.shape[1]):
        for j in range(force.shape[1]):
            pearson[i, j] = corrcoef(force[:, j], target[:, i])
            spearman[i, j] = spearmanr(force[:, j], target[:, i], nan_policy="omit").correlation

    heatmap(
        pearson,
        feature_names,
        target_names,
        f"Pearson corr: force features vs {target_key}",
        output_dir / f"{target_key}_pearson.png",
    )
    heatmap(
        spearman,
        feature_names,
        target_names,
        f"Spearman corr: force features vs {target_key}",
        output_dir / f"{target_key}_spearman.png",
    )

    x_force_train = concat_by_demos(train_demos, "force_features")
    x_force_val = concat_by_demos(val_demos, "force_features")
    x_base_train = concat_by_demos(train_demos, "base")
    x_base_val = concat_by_demos(val_demos, "base")
    y_train = concat_by_demos(train_demos, target_key)
    y_val = concat_by_demos(val_demos, target_key)

    x_force_train, x_force_val = standardize(x_force_train, x_force_val)
    x_base_train, x_base_val = standardize(x_base_train, x_base_val)
    y_mean = y_train.mean(axis=0, keepdims=True)
    baselines = {
        "constant": np.repeat(y_mean, len(y_val), axis=0),
        "force_only": ridge_fit_predict(x_force_train, y_train, x_force_val, alpha=10.0),
        "base_only": ridge_fit_predict(x_base_train, y_train, x_base_val, alpha=10.0),
        "base_plus_force": ridge_fit_predict(
            np.concatenate([x_base_train, x_force_train], axis=1),
            y_train,
            np.concatenate([x_base_val, x_force_val], axis=1),
            alpha=10.0,
        ),
    }
    baseline_metrics = {}
    for name, pred in baselines.items():
        mse_axis = np.mean((y_val - pred) ** 2, axis=0)
        r2_axis = r2_score(y_val, pred)
        baseline_metrics[name] = {
            "mse_mean": float(np.mean(mse_axis)),
            "r2_mean": float(np.mean(r2_axis)),
            "mse_axis": mse_axis.tolist(),
            "r2_axis": r2_axis.tolist(),
        }

    best = []
    for i, target_name in enumerate(target_names):
        order = np.argsort(-np.abs(pearson[i]))[:8]
        best.append({
            "target": target_name,
            "best_abs_pearson": [
                {"feature": feature_names[j], "corr": float(pearson[i, j])}
                for j in order
            ],
        })

    return {
        "num_samples": int(len(target)),
        "target_dim": int(target.shape[1]),
        "best_correlations": best,
        "baseline_metrics": baseline_metrics,
    }


def lag_analysis(all_demos, target_key, output_dir, max_lag):
    axis_names = ["last_Fx", "last_Fy", "last_Fz", "last_Tx", "last_Ty", "last_Tz"]
    target_dim = all_demos[0][target_key].shape[1]
    lags = np.arange(-max_lag, max_lag + 1)
    lag_corr = np.zeros((len(lags), len(axis_names), target_dim), dtype=np.float64)
    for lag_i, lag in enumerate(lags):
        xs = []
        ys = []
        for demo in all_demos:
            force_last = demo["force_features"][:, :6]
            target = demo[target_key]
            if lag >= 0:
                x = force_last[lag:]
                y = target[:len(target) - lag]
            else:
                x = force_last[:len(force_last) + lag]
                y = target[-lag:]
            if len(x) > 0:
                xs.append(x)
                ys.append(y)
        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        for a in range(len(axis_names)):
            for t in range(target_dim):
                lag_corr[lag_i, a, t] = corrcoef(x_all[:, a], y_all[:, t])

    target_names = [f"{target_key}_{i}" for i in range(target_dim)]
    summary = {}
    for t, target_name in enumerate(target_names):
        mat = lag_corr[:, :, t]
        idx = np.unravel_index(np.nanargmax(np.abs(mat)), mat.shape)
        summary[target_name] = {
            "best_lag": int(lags[idx[0]]),
            "feature": axis_names[idx[1]],
            "corr": float(mat[idx]),
        }
        heatmap(
            mat.T,
            [str(int(lag)) for lag in lags],
            axis_names,
            f"Lag corr force[t+lag] vs {target_name}",
            output_dir / f"{target_name}_lag.png",
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/baetae/260618/slow_erase_board_virtual_m_world_wrench.hdf5")
    parser.add_argument("--output-dir", default="plots/force_delta/world/06_corr_lag/raw")
    parser.add_argument("--wrench-key", default="wrench_wrist_R")
    parser.add_argument("--base-key", default="actual_action_rel")
    parser.add_argument(
        "--target-key",
        action="append",
        default=None,
        help="Target under obs. Can repeat. Defaults to actual_action_rel and residual_delta6_gt_actual_to_virtual.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-lag", type=int, default=10)
    args = parser.parse_args()

    target_keys = args.target_key or ["actual_action_rel", "residual_delta6_gt_actual_to_virtual"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_demos, feature_names = load_arrays(args.dataset, args.wrench_key, target_keys, args.base_key)
    num_val = max(1, int(round(len(all_demos) * args.val_ratio)))
    train_demos = all_demos[:-num_val]
    val_demos = all_demos[-num_val:]

    results = {
        "dataset": args.dataset,
        "wrench_key": args.wrench_key,
        "base_key": args.base_key,
        "num_demos": len(all_demos),
        "train_demos": [demo["demo"] for demo in train_demos],
        "val_demos": [demo["demo"] for demo in val_demos],
    }
    for target_key in target_keys:
        target_dir = output_dir / target_key
        target_dir.mkdir(parents=True, exist_ok=True)
        results[target_key] = analyze_target(
            all_demos,
            train_demos,
            val_demos,
            feature_names,
            target_key,
            target_dir,
        )
        results[target_key]["lag_summary"] = lag_analysis(
            all_demos,
            target_key,
            target_dir,
            max_lag=args.max_lag,
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(metrics_path)


if __name__ == "__main__":
    main()
