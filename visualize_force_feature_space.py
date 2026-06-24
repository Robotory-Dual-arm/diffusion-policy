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


def resolve_demos(h5: h5py.File, checkpoint: dict, demo_arg: str) -> list[str]:
    all_demos = sorted(h5["data"].keys(), key=demo_sort_key)
    if demo_arg == "all":
        return all_demos
    if demo_arg == "val":
        val = checkpoint.get("val_demos", [])
        return [demo for demo in val if demo in h5["data"]] or all_demos
    demos = []
    for item in demo_arg.replace(",", " ").split():
        name = item if item.startswith("demo_") else f"demo_{item}"
        if name not in h5["data"]:
            raise KeyError(f"{name} not found in HDF5.")
        demos.append(name)
    return demos


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].astype(np.float32)
    coords = centered @ components.T
    denom = max(x.shape[0] - 1, 1)
    explained = (s[:2] ** 2) / denom
    return coords.astype(np.float32), components, explained.astype(np.float32)


def apply_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mean) @ components.T


@torch.no_grad()
def encode_images(
    model,
    checkpoint: dict,
    hdf5_path: str,
    demos: list[str],
    samples_per_demo: int,
    phase_min: float,
    phase_max: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict]]:
    image_key = checkpoint["image_key"]
    image_size = int(checkpoint.get("image_size", 224))
    crop_ratio = float(checkpoint.get("image_crop_ratio", 0.95))
    bgr_to_rgb = bool(checkpoint.get("bgr_to_rgb", True))
    post_blur_radius = float(checkpoint.get("post_blur_radius", 0.0))

    tensors = []
    rows = []
    features = []
    with h5py.File(hdf5_path, "r") as h5:
        for demo in demos:
            obs = h5["data"][demo]["observations"]
            image_count = int(obs[image_key].shape[0])
            lo = int(np.floor(max(0.0, phase_min) * max(image_count - 1, 1)))
            hi = int(np.ceil(min(1.0, phase_max) * max(image_count - 1, 1)))
            frames = np.unique(
                np.linspace(lo, hi, int(samples_per_demo), dtype=np.int64)
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
                        "phase": float(frame / max(image_count - 1, 1)),
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


def pairwise_stats(features: np.ndarray, rows: list[dict]) -> dict:
    x = np.asarray(features, dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))
    mask = ~np.eye(len(x), dtype=bool)
    same_demo = np.zeros_like(mask)
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            same_demo[i, j] = i != j and a["demo"] == b["demo"]
    different_demo = mask & ~same_demo
    same_phase = np.zeros_like(mask)
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            same_phase[i, j] = i != j and abs(a["phase"] - b["phase"]) <= 0.05
    out = {
        "pairwise_mean": float(dist[mask].mean()),
        "pairwise_median": float(np.median(dist[mask])),
        "same_demo_mean": float(dist[same_demo].mean()) if same_demo.any() else None,
        "different_demo_mean": float(dist[different_demo].mean()) if different_demo.any() else None,
        "same_phase_mean": float(dist[same_phase].mean()) if same_phase.any() else None,
    }
    return out


def plot_feature_pca(coords: np.ndarray, rows: list[dict], out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demos = sorted({row["demo"] for row in rows}, key=demo_sort_key)
    demo_to_id = {demo: i for i, demo in enumerate(demos)}
    demo_ids = np.asarray([demo_to_id[row["demo"]] for row in rows])
    phases = np.asarray([row["phase"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax = axes[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=phases, cmap="viridis", s=34, alpha=0.85)
    ax.set_title("Policy image features PCA, colored by progress")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="normalized progress")

    ax = axes[1]
    cmap = plt.get_cmap("tab20", max(len(demos), 1))
    for demo in demos:
        mask = demo_ids == demo_to_id[demo]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=34,
            alpha=0.80,
            color=cmap(demo_to_id[demo] % 20),
            label=demo,
        )
    ax.set_title("Same PCA, colored by demo")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    if len(demos) <= 12:
        ax.legend(fontsize=8, loc="best")
    fig.savefig(out_path, dpi=150, facecolor="white", edgecolor="white")
    plt.close(fig)


def plot_noise_cloud(
    features: np.ndarray,
    rows: list[dict],
    noise_std: float,
    samples_per_point: int,
    out_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    base_count = min(8, len(features))
    base_idx = np.linspace(0, len(features) - 1, base_count, dtype=np.int64)
    base = features[base_idx]
    noisy = []
    noisy_owner = []
    for owner, feat in enumerate(base):
        noise = rng.normal(
            loc=0.0,
            scale=float(noise_std),
            size=(int(samples_per_point), feat.shape[0]),
        ).astype(np.float32)
        noisy.append(feat[None] + noise)
        noisy_owner.extend([owner] * int(samples_per_point))
    noisy = np.concatenate(noisy, axis=0)
    all_features = np.concatenate([features, noisy], axis=0)
    mean = all_features.mean(axis=0, keepdims=True)
    coords, components, _ = pca_2d(all_features)
    original_coords = coords[: len(features)]
    noisy_coords = coords[len(features) :]
    base_coords = apply_pca(base, mean, components)
    noisy_owner = np.asarray(noisy_owner)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.scatter(original_coords[:, 0], original_coords[:, 1], s=20, color="lightgray", alpha=0.6, label="all originals")
    cmap = plt.get_cmap("tab10", max(base_count, 1))
    for owner in range(base_count):
        mask = noisy_owner == owner
        label = f"{rows[int(base_idx[owner])]['demo']} f{rows[int(base_idx[owner])]['frame']}"
        ax.scatter(noisy_coords[mask, 0], noisy_coords[mask, 1], s=12, alpha=0.35, color=cmap(owner), label=label)
        ax.scatter(base_coords[owner, 0], base_coords[owner, 1], s=95, color=cmap(owner), edgecolor="black", linewidth=0.8)
    ax.set_title(f"Feature noise cloud, std={noise_std}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(out_path, dpi=150, facecolor="white", edgecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5", default=None)
    parser.add_argument("--demos", default="val", help="'val', 'all', or comma/space separated demo ids.")
    parser.add_argument("--samples-per-demo", type=int, default=20)
    parser.add_argument("--phase-min", type=float, default=0.0)
    parser.add_argument("--phase-max", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--noise-samples", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = make_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_force_model_from_kwargs(checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if not hasattr(model, "vision_encoder"):
        raise RuntimeError("Feature visualization requires --encoder-backend policy-frozen checkpoint.")

    hdf5_path = args.hdf5 or checkpoint["hdf5_path"]
    with h5py.File(hdf5_path, "r") as h5:
        demos = resolve_demos(h5, checkpoint, args.demos)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / "feature_space"
    out_dir.mkdir(parents=True, exist_ok=True)

    features, rows = encode_images(
        model=model,
        checkpoint=checkpoint,
        hdf5_path=hdf5_path,
        demos=demos,
        samples_per_demo=args.samples_per_demo,
        phase_min=args.phase_min,
        phase_max=args.phase_max,
        batch_size=args.batch_size,
        device=device,
    )
    coords, _, explained = pca_2d(features)
    plot_feature_pca(coords, rows, out_dir / "feature_pca.png")
    plot_noise_cloud(
        features=features,
        rows=rows,
        noise_std=args.noise_std,
        samples_per_point=args.noise_samples,
        out_path=out_dir / f"feature_noise_std_{args.noise_std:g}.png",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "hdf5_path": str(hdf5_path),
        "demos": demos,
        "sample_count": len(rows),
        "feature_dim": int(features.shape[1]),
        "pca_explained_variance": explained.astype(float).tolist(),
        "distance_stats": pairwise_stats(features, rows),
        "rows": rows,
    }
    (out_dir / "feature_space_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {out_dir / 'feature_pca.png'}")
    print(f"saved: {out_dir / f'feature_noise_std_{args.noise_std:g}.png'}")
    print(f"saved: {out_dir / 'feature_space_summary.json'}")


if __name__ == "__main__":
    main()
