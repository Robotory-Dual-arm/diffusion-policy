#!/usr/bin/env python3
"""Create interactive Plotly 3D robot-pose trajectories from a robomimic-style HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go


DEFAULT_DATASET = "obs/robot_pose_R"


def parse_demo_indices(text: str) -> list[int]:
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(part))
    return indices


def equal_axis_ranges(xyz: np.ndarray, pad_ratio: float = 0.12) -> dict[str, list[float]]:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1e-3
    radius = span * (1.0 + pad_ratio) / 2.0
    return {
        "x": [float(center[0] - radius), float(center[0] + radius)],
        "y": [float(center[1] - radius), float(center[1] + radius)],
        "z": [float(center[2] - radius), float(center[2] + radius)],
    }


def dataset_label(dataset_path: str) -> str:
    return dataset_path.strip("/").replace("/", "_")


def build_hover_text(xyz: np.ndarray, extra_fields: dict[str, np.ndarray]) -> list[str]:
    hover = []
    for idx, point in enumerate(xyz):
        lines = [
            f"frame: {idx}",
            f"x: {point[0]:.6f} m",
            f"y: {point[1]:.6f} m",
            f"z: {point[2]:.6f} m",
        ]
        for name, values_by_frame in extra_fields.items():
            if idx >= len(values_by_frame):
                continue
            values = np.asarray(values_by_frame[idx]).reshape(-1)
            text = ", ".join(f"{value:.4f}" for value in values[:6])
            lines.append(f"{name}: [{text}]")
        hover.append("<br>".join(lines))
    return hover


def make_figure(
    hdf5_path: Path,
    demo_name: str,
    dataset_path: str,
    xyz: np.ndarray,
    extra_fields: dict[str, np.ndarray],
) -> go.Figure:
    ranges = equal_axis_ranges(xyz)
    displacement = np.zeros(len(xyz), dtype=np.float64)
    if len(xyz) > 1:
        displacement[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    total_path_length = float(displacement.sum())
    frame_idx = np.arange(len(xyz))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="lines",
            name="trajectory line",
            line=dict(color="rgba(35, 74, 166, 0.55)", width=5),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            name=f"{dataset_label(dataset_path)} points",
            marker=dict(
                size=5,
                color=frame_idx,
                colorscale="Viridis",
                colorbar=dict(title="frame", len=0.74),
                opacity=0.9,
            ),
            text=build_hover_text(xyz, extra_fields),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[xyz[0, 0], xyz[-1, 0]],
            y=[xyz[0, 1], xyz[-1, 1]],
            z=[xyz[0, 2], xyz[-1, 2]],
            mode="markers+text",
            name="start/end",
            marker=dict(size=[9, 10], color=["#2ca02c", "#d62728"], symbol=["circle", "diamond"]),
            text=["start", "end"],
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            f"{hdf5_path.name} {demo_name} {dataset_path} full 3D trajectory"
            f"<br><sup>{len(xyz)} points, path length {total_path_length:.4f}m</sup>"
        ),
        width=1200,
        height=900,
        margin=dict(l=0, r=0, t=72, b=0),
        legend=dict(x=0.02, y=0.98),
        scene=dict(
            xaxis=dict(title="X (m)", range=ranges["x"], backgroundcolor="#f8f9fb", gridcolor="#d9dee7"),
            yaxis=dict(title="Y (m)", range=ranges["y"], backgroundcolor="#f8f9fb", gridcolor="#d9dee7"),
            zaxis=dict(title="Z (m)", range=ranges["z"], backgroundcolor="#f8f9fb", gridcolor="#d9dee7"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.35, z=1.05)),
        ),
    )
    return fig


def load_demo_data(file: h5py.File, demo_name: str, dataset_path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    group_path = f"data/{demo_name}"
    if group_path not in file:
        raise KeyError(f"Missing demo group: {group_path}")

    full_dataset_path = f"{group_path}/{dataset_path}"
    if full_dataset_path not in file:
        raise KeyError(f"Missing dataset: {full_dataset_path}")

    xyz = np.asarray(file[full_dataset_path], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"{full_dataset_path} must have shape (T, >=3); got {xyz.shape}.")
    xyz = xyz[:, :3]

    extra_fields = {}
    for name in ("wrench_wrist_R", "gripper"):
        field_path = f"{group_path}/obs/{name}"
        if field_path in file:
            extra_fields[name] = np.asarray(file[field_path])
    return xyz, extra_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save interactive Plotly HTML 3D trajectories for HDF5 demos."
    )
    parser.add_argument("hdf5_path", help="Input HDF5 path.")
    parser.add_argument("--demos", default="1-10", help="Demo indices, e.g. 1-10 or 0,3,5.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset under data/demo_N. Default: obs/robot_pose_R.")
    parser.add_argument("--output-dir", default="data", help="Directory for output HTML files.")
    parser.add_argument("--prefix", default=None, help="Optional filename prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or hdf5_path.stem

    demo_indices = parse_demo_indices(args.demos)
    with h5py.File(hdf5_path, "r") as file:
        for demo_idx in demo_indices:
            demo_name = f"demo_{demo_idx}"
            xyz, extra_fields = load_demo_data(file, demo_name, args.dataset)
            fig = make_figure(hdf5_path, demo_name, args.dataset, xyz, extra_fields)
            output_path = output_dir / f"{prefix}_{demo_name}_{dataset_label(args.dataset)}_trajectory_3d.html"
            fig.write_html(output_path, include_plotlyjs=True, full_html=True)
            print(f"Saved {output_path} ({len(xyz)} points)")


if __name__ == "__main__":
    main()
