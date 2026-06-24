#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from train_force_distribution import (
    build_force_model_from_kwargs,
    demo_sort_key,
    make_device,
    preprocess_image,
)


DEFAULT_WINDOWS = "start:0.05:0.15,erase:0.40:0.60,end:0.90:1.00"


def parse_windows(text: str) -> list[tuple[str, float, float]]:
    windows = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, lo, hi = chunk.split(":")
        windows.append((name, float(lo), float(hi)))
    if not windows:
        raise ValueError("At least one phase window is required.")
    return windows


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    explained = (s[:2] ** 2) / max(x.shape[0] - 1, 1)
    return coords.astype(np.float32), explained.astype(np.float32)


@torch.no_grad()
def encode_phase_windows(
    model,
    checkpoint: dict,
    hdf5_path: str,
    windows: list[tuple[str, float, float]],
    samples_per_window: int,
    batch_size: int,
    device: torch.device,
    post_blur_radius_override: float | None,
) -> tuple[np.ndarray, list[dict]]:
    image_key = checkpoint["image_key"]
    image_size = int(checkpoint.get("image_size", 224))
    crop_ratio = float(checkpoint.get("image_crop_ratio", 0.95))
    bgr_to_rgb = bool(checkpoint.get("bgr_to_rgb", True))
    post_blur_radius = (
        float(checkpoint.get("post_blur_radius", 0.0))
        if post_blur_radius_override is None
        else float(post_blur_radius_override)
    )

    tensors = []
    rows = []
    features = []
    with h5py.File(hdf5_path, "r") as h5:
        demos = sorted(h5["data"].keys(), key=demo_sort_key)
        for demo in tqdm(demos, desc="sample/encode", dynamic_ncols=True):
            obs = h5["data"][demo]["observations"]
            image_count = int(obs[image_key].shape[0])
            last = max(image_count - 1, 1)
            for window_name, lo_frac, hi_frac in windows:
                lo = int(np.floor(max(0.0, lo_frac) * last))
                hi = int(np.ceil(min(1.0, hi_frac) * last))
                frames = np.unique(
                    np.linspace(lo, hi, int(samples_per_window), dtype=np.int64)
                )
                for frame in frames:
                    image = np.asarray(obs[image_key][int(frame)])
                    tensors.append(
                        preprocess_image(
                            image=image,
                            image_size=image_size,
                            augment=False,
                            crop_ratio=crop_ratio,
                            bgr_to_rgb=bgr_to_rgb,
                            color_jitter=False,
                            grayscale_p=0.0,
                            post_blur_radius=post_blur_radius,
                        )
                    )
                    rows.append(
                        {
                            "demo": demo,
                            "frame": int(frame),
                            "phase": float(frame / last),
                            "window": window_name,
                        }
                    )
                    if len(tensors) >= batch_size:
                        x = torch.stack(tensors, dim=0).to(device)
                        raw = model.attention_pool_2d(model.vision_encoder(x))
                        feat = model.feature_norm(raw)
                        features.append(feat.cpu().numpy().astype(np.float32))
                        tensors = []
    if tensors:
        x = torch.stack(tensors, dim=0).to(device)
        raw = model.attention_pool_2d(model.vision_encoder(x))
        feat = model.feature_norm(raw)
        features.append(feat.cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0), rows


def distance_summary(features: np.ndarray, rows: list[dict]) -> dict:
    x = np.asarray(features, dtype=np.float32)
    windows = sorted({row["window"] for row in rows})
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))
    out = {}
    for window in windows:
        ids = np.asarray([i for i, row in enumerate(rows) if row["window"] == window])
        if len(ids) > 1:
            d = dist[np.ix_(ids, ids)]
            mask = ~np.eye(len(ids), dtype=bool)
            out[f"within_{window}_mean"] = float(d[mask].mean())
            out[f"within_{window}_median"] = float(np.median(d[mask]))
    for i, a in enumerate(windows):
        for b in windows[i + 1 :]:
            ia = np.asarray([idx for idx, row in enumerate(rows) if row["window"] == a])
            ib = np.asarray([idx for idx, row in enumerate(rows) if row["window"] == b])
            out[f"between_{a}_{b}_mean"] = float(dist[np.ix_(ia, ib)].mean())
            out[f"between_{a}_{b}_median"] = float(np.median(dist[np.ix_(ia, ib)]))
    return out


def plot_windows(coords: np.ndarray, rows: list[dict], out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = list(dict.fromkeys(row["window"] for row in rows))
    colors = {
        "start": "tab:green",
        "erase": "tab:red",
        "end": "tab:blue",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    for window in windows:
        ids = np.asarray([i for i, row in enumerate(rows) if row["window"] == window])
        ax.scatter(
            coords[ids, 0],
            coords[ids, 1],
            s=18,
            alpha=0.62,
            color=colors.get(window, None),
            label=f"{window} ({len(ids)})",
        )
        center = coords[ids].mean(axis=0)
        ax.scatter(center[0], center[1], s=180, marker="X", edgecolor="black", linewidth=1.0, color=colors.get(window, None))
    ax.set_title("Policy feature PCA by coarse phase window")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    demos = sorted({row["demo"] for row in rows}, key=demo_sort_key)
    demo_to_color = {demo: plt.get_cmap("tab20")(i % 20) for i, demo in enumerate(demos)}
    for demo in demos:
        ids = np.asarray([i for i, row in enumerate(rows) if row["demo"] == demo])
        ax.plot(coords[ids, 0], coords[ids, 1], color=demo_to_color[demo], alpha=0.22, linewidth=0.7)
        ax.scatter(coords[ids, 0], coords[ids, 1], color=demo_to_color[demo], alpha=0.35, s=10)
    ax.set_title("Same points connected within each demo")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=150, facecolor="white", edgecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5", default=None)
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--samples-per-window", type=int, default=2)
    parser.add_argument(
        "--post-blur-radius",
        type=float,
        default=None,
        help="Override checkpoint blur radius before encoding. Use 8 to preview blur8 without training.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = make_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_force_model_from_kwargs(checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if not hasattr(model, "vision_encoder"):
        raise RuntimeError("Phase-window visualization requires policy-frozen checkpoint.")

    hdf5_path = args.hdf5 or checkpoint["hdf5_path"]
    windows = parse_windows(args.windows)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / "feature_phase_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    features, rows = encode_phase_windows(
        model=model,
        checkpoint=checkpoint,
        hdf5_path=hdf5_path,
        windows=windows,
        samples_per_window=args.samples_per_window,
        batch_size=args.batch_size,
        device=device,
        post_blur_radius_override=args.post_blur_radius,
    )
    coords, explained = pca_2d(features)
    plot_path = out_dir / "phase_window_feature_pca.png"
    plot_windows(coords, rows, plot_path)
    summary = {
        "checkpoint": str(args.checkpoint),
        "hdf5_path": str(hdf5_path),
        "windows": windows,
        "post_blur_radius": (
            float(checkpoint.get("post_blur_radius", 0.0))
            if args.post_blur_radius is None
            else float(args.post_blur_radius)
        ),
        "samples_per_window": int(args.samples_per_window),
        "sample_count": int(len(rows)),
        "feature_dim": int(features.shape[1]),
        "pca_explained_variance": explained.astype(float).tolist(),
        "distance_summary": distance_summary(features, rows),
        "rows": rows,
    }
    summary_path = out_dir / "phase_window_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {plot_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
