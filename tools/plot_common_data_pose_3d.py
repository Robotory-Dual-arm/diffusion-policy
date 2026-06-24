#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import plotly.graph_objects as go
import roboticstoolbox as rtb


SQRT2 = np.sqrt(2) / 2
SAMPLE_HZ = 10.0
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ]
)


def right_robot_to_world(pos_robot: np.ndarray) -> np.ndarray:
    pos_robot = np.asarray(pos_robot)
    return (RIGHT_ROBOT_TO_WORLD @ pos_robot.T).T


def sample_nearest_at_hz(time_s: np.ndarray, *arrays: np.ndarray, hz: float) -> tuple[np.ndarray, ...]:
    time_s = np.asarray(time_s, dtype=float)
    time_s = time_s - time_s[0]

    sample_period = 1.0 / hz
    query_t = np.arange(0.0, time_s[-1] + sample_period * 0.5, sample_period)
    indices = np.searchsorted(time_s, query_t, side="left")
    indices = np.clip(indices, 0, len(time_s) - 1)

    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(time_s[previous] - query_t) < np.abs(time_s[indices] - query_t)
    indices[use_previous] = previous[use_previous]
    indices = np.unique(indices)

    return (time_s[indices], *(array[indices] for array in arrays))


def axis_ranges(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 1e-3) * 1.08

    return (
        [centers[0] - radius, centers[0] + radius],
        [centers[1] - radius, centers[1] + radius],
        [centers[2] - radius, centers[2] + radius],
    )


def load_demo(path: Path, demo: str, urdf_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = rtb.ERobot.URDF(str(urdf_path))

    with h5py.File(path, "r") as f:
        obs = f[f"data/{demo}/observations"]
        time_s = np.asarray(obs["timestamp_robot"])
        virtual_xyz = np.asarray(obs["desired_pose"])[:, :3] / 1000.0
        joint_r = np.asarray(obs["joint_R"])

    actual_xyz = np.asarray(robot.fkine(joint_r).t)
    time_s, virtual_xyz, actual_xyz = sample_nearest_at_hz(
        time_s, virtual_xyz, actual_xyz, hz=SAMPLE_HZ
    )

    virtual_xyz = right_robot_to_world(virtual_xyz)
    actual_xyz = right_robot_to_world(actual_xyz)

    return time_s, virtual_xyz, actual_xyz


def plot_demo(path: Path, demo: str, urdf_path: Path, output_path: Path) -> None:
    time_s, virtual_xyz, actual_xyz = load_demo(path, demo, urdf_path)
    x_range, y_range, z_range = axis_ranges(np.concatenate([virtual_xyz, actual_xyz], axis=0))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=virtual_xyz[:, 0],
            y=virtual_xyz[:, 1],
            z=virtual_xyz[:, 2],
            mode="markers",
            name="virtual target",
            marker=dict(color="red", size=4, opacity=0.85),
            customdata=np.round(time_s, 3),
            hovertemplate=(
                "virtual target<br>"
                "t=%{customdata:.3f}s<br>"
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=actual_xyz[:, 0],
            y=actual_xyz[:, 1],
            z=actual_xyz[:, 2],
            mode="markers",
            name="actual pose",
            marker=dict(color="blue", size=4, opacity=0.85),
            customdata=np.round(time_s, 3),
            hovertemplate=(
                "actual pose<br>"
                "t=%{customdata:.3f}s<br>"
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[virtual_xyz[0, 0], actual_xyz[0, 0]],
            y=[virtual_xyz[0, 1], actual_xyz[0, 1]],
            z=[virtual_xyz[0, 2], actual_xyz[0, 2]],
            mode="markers",
            name="start points",
            marker=dict(color=["red", "blue"], size=8, symbol="diamond"),
        )
    )

    fig.update_layout(
        title=(
            f"{path.stem} {demo}: desired_pose vs FK actual, world frame, "
            f"{SAMPLE_HZ:g} Hz points, actual from correct URDF"
        ),
        width=1100,
        height=850,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(title="x (m)", range=x_range, backgroundcolor="#f7f7f7", gridcolor="#d9d9d9"),
            yaxis=dict(title="y (m)", range=y_range, backgroundcolor="#f7f7f7", gridcolor="#d9d9d9"),
            zaxis=dict(title="z (m)", range=z_range, backgroundcolor="#f7f7f7", gridcolor="#d9d9d9"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=-1.65, z=1.05)),
        ),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "Downloads/common_data_height.hdf5")
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--urdf", type=Path, default=Path("m0609.white.urdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/wrench_downloads"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"new_{args.input.stem}_{args.demo}_pose_3d.html"
    plot_demo(args.input, args.demo, args.urdf.resolve(), output_path)
    print(output_path)


if __name__ == "__main__":
    main()
