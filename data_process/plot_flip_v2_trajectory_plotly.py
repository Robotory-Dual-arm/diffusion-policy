#!/usr/bin/env python3
"""Create an interactive Plotly 3D TCP trajectory for one flip_v2 episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from irdp_data_check import (
    collect_field,
    field_value,
    get_messages,
    load_pickle,
    message_field_names,
    timestamp_array,
)


DEFAULT_FIELD = "rightRobotTCP"


def resolve_episode_path(data_dir: Path, episode: str) -> Path:
    path = Path(episode).expanduser()
    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")
    if not path.is_absolute():
        path = data_dir / path.name
    if not path.exists():
        raise FileNotFoundError(f"Episode pickle not found: {path}")
    return path


def load_xyz_trajectory(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = load_pickle(path)
    messages = get_messages(data)
    if len(messages) == 0:
        raise ValueError(f"{path} has no messages.")

    available = set(message_field_names(messages[0]))
    if field not in available:
        raise KeyError(f"{field!r} is not available in {path.name}. Available: {sorted(available)}")

    time_s = timestamp_array(messages)
    trajectory = collect_field(messages, field)
    if trajectory.dtype == object or trajectory.ndim != 2 or trajectory.shape[1] < 3:
        raise ValueError(
            f"{field} must be a numeric array with shape (T, >=3); got {trajectory.shape} "
            f"with dtype {trajectory.dtype}."
        )

    gripper = None
    gripper_field = field.replace("TCP", "GripperState")
    if gripper_field in available:
        gripper = collect_field(messages, gripper_field)

    return time_s, trajectory[:, :3].astype(np.float64), gripper


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


def build_hover_text(time_s: np.ndarray, xyz: np.ndarray, gripper: np.ndarray | None) -> list[str]:
    hover = []
    for idx, (t, point) in enumerate(zip(time_s, xyz)):
        lines = [
            f"frame: {idx}",
            f"time: {t:.3f} s",
            f"x: {point[0]:.6f} m",
            f"y: {point[1]:.6f} m",
            f"z: {point[2]:.6f} m",
        ]
        if gripper is not None and idx < len(gripper):
            grip = np.asarray(gripper[idx]).reshape(-1)
            grip_text = ", ".join(f"{value:.4f}" for value in grip[:4])
            lines.append(f"gripper: [{grip_text}]")
        hover.append("<br>".join(lines))
    return hover


def make_figure(path: Path, field: str, time_s: np.ndarray, xyz: np.ndarray, gripper: np.ndarray | None) -> go.Figure:
    ranges = equal_axis_ranges(xyz)
    displacement = np.zeros(len(xyz), dtype=np.float64)
    if len(xyz) > 1:
        displacement[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    total_path_length = float(displacement.sum())

    hover_text = build_hover_text(time_s, xyz, gripper)

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
            name="TCP points",
            marker=dict(
                size=5,
                color=time_s,
                colorscale="Viridis",
                colorbar=dict(title="time (s)", len=0.74),
                opacity=0.9,
            ),
            text=hover_text,
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
            f"{path.name} {field} full 3D trajectory"
            f"<br><sup>{len(xyz)} points, duration {time_s[-1]:.3f}s, path length {total_path_length:.4f}m</sup>"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save an interactive Plotly HTML 3D trajectory for one flip_v2 pickle episode."
    )
    parser.add_argument("--data-dir", default="data/flip_v2", help="Directory containing flip_v2 pkl files.")
    parser.add_argument("--episode", default="1", help="Episode id or pkl path, e.g. 1 or data/flip_v2/1.pkl.")
    parser.add_argument("--field", default=DEFAULT_FIELD, help="Message field to plot. Default: rightRobotTCP.")
    parser.add_argument("--output-dir", default="data/flip_v2_check", help="Directory for the output HTML.")
    parser.add_argument("--output", default=None, help="Optional explicit HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    episode_path = resolve_episode_path(data_dir, args.episode)

    time_s, xyz, gripper = load_xyz_trajectory(episode_path, args.field)
    fig = make_figure(episode_path, args.field, time_s, xyz, gripper)

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        output_path = output_dir / f"{episode_path.stem}_{args.field}_trajectory_3d.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)

    print(f"Saved interactive 3D trajectory: {output_path}")
    print(f"Episode: {episode_path}")
    print(f"Field: {args.field}")
    print(f"Points: {len(xyz)}")
    print(f"Duration: {time_s[-1]:.3f} s")


if __name__ == "__main__":
    main()
