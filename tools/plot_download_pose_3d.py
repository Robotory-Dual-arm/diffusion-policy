#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import plotly.graph_objects as go
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3


GROUPS = ("nowrench", "wrench")
SQRT2 = np.sqrt(2) / 2
SAMPLE_HZ = 10.0

# Right arm robot base frame -> world frame.
# Same convention as data_process/plot_height_actual_vs_desired_pose.py.
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ]
)


def files_for_group(input_dir: Path, group_name: str) -> list[Path]:
    files = [input_dir / f"{group_name}{index}.hdf5" for index in range(1, 6)]
    missing = [path for path in files if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing expected files: {missing_names}")
    return files


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

    return (time_s[indices], *(np.asarray(array)[indices] for array in arrays))


def recalculate_actual_with_correct_urdf(
    actual_xyz_weird: np.ndarray,
    actual_quat_xyzw: np.ndarray,
    weird_robot: rtb.ERobot,
    correct_robot: rtb.ERobot,
) -> np.ndarray:
    corrected_xyz = []
    q0 = None
    for xyz, quat in zip(actual_xyz_weird, actual_quat_xyzw):
        target_pose = SE3.Rt(R.from_quat(quat).as_matrix(), xyz)
        solution = weird_robot.ikine_LM(target_pose, q0=q0)
        if not solution.success:
            solution = weird_robot.ikine_LM(target_pose)
        if not solution.success:
            raise RuntimeError(f"IK failed for actual pose {xyz}")
        q0 = solution.q
        corrected_xyz.append(correct_robot.fkine(solution.q).t)
    return np.asarray(corrected_xyz)


def load_pose_paths(
    path: Path,
    weird_robot: rtb.ERobot,
    correct_robot: rtb.ERobot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        target_xyz = np.asarray(f["action_virtual_target/action"])[:, :3]
        target_time_s = np.asarray(f["action_virtual_target/elapsed_s"])
        actual_xyz = np.asarray(f["actual/robot_pose_R"])
        actual_quat = np.asarray(f["actual/robot_quat_R"])
        actual_time_s = np.asarray(f["actual/elapsed_s"])

    target_time_s, target_xyz = sample_nearest_at_hz(target_time_s, target_xyz, hz=SAMPLE_HZ)
    actual_time_s, actual_xyz, actual_quat = sample_nearest_at_hz(
        actual_time_s, actual_xyz, actual_quat, hz=SAMPLE_HZ
    )

    target_xyz = right_robot_to_world(target_xyz)
    actual_xyz = recalculate_actual_with_correct_urdf(
        actual_xyz, actual_quat, weird_robot, correct_robot
    )
    actual_xyz = right_robot_to_world(actual_xyz)

    return target_time_s, target_xyz, actual_time_s, actual_xyz


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


def plot_trial(
    path: Path,
    output_path: Path,
    weird_robot: rtb.ERobot,
    correct_robot: rtb.ERobot,
) -> None:
    target_time_s, target_xyz, actual_time_s, actual_xyz = load_pose_paths(
        path, weird_robot, correct_robot
    )
    x_range, y_range, z_range = axis_ranges(np.concatenate([target_xyz, actual_xyz], axis=0))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=target_xyz[:, 0],
            y=target_xyz[:, 1],
            z=target_xyz[:, 2],
            mode="markers",
            name="virtual target",
            marker=dict(color="red", size=4, opacity=0.85),
            customdata=np.round(target_time_s, 3),
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
            customdata=np.round(actual_time_s, 3),
            hovertemplate=(
                "actual pose<br>"
                "t=%{customdata:.3f}s<br>"
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[target_xyz[0, 0], actual_xyz[0, 0]],
            y=[target_xyz[0, 1], actual_xyz[0, 1]],
            z=[target_xyz[0, 2], actual_xyz[0, 2]],
            mode="markers",
            name="start",
            marker=dict(color=["red", "blue"], size=5, symbol="circle"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[target_xyz[-1, 0], actual_xyz[-1, 0]],
            y=[target_xyz[-1, 1], actual_xyz[-1, 1]],
            z=[target_xyz[-1, 2], actual_xyz[-1, 2]],
            mode="markers",
            name="end",
            marker=dict(color=["red", "blue"], size=7, symbol="diamond"),
            showlegend=True,
        )
    )

    fig.update_layout(
        title=(
            f"{path.stem}: virtual target vs actual pose, world frame, "
            f"{SAMPLE_HZ:g} Hz points, actual recalculated with correct URDF"
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
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/wrench_downloads"))
    parser.add_argument("--weird-urdf", type=Path, default=Path("m0609.white_weird.urdf"))
    parser.add_argument("--correct-urdf", type=Path, default=Path("m0609.white.urdf"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weird_robot = rtb.ERobot.URDF(str(args.weird_urdf.resolve()))
    correct_robot = rtb.ERobot.URDF(str(args.correct_urdf.resolve()))
    output_paths = []
    for group_name in GROUPS:
        for path in files_for_group(args.input_dir, group_name):
            output_path = args.output_dir / f"new_{path.stem}_pose_3d.html"
            plot_trial(path, output_path, weird_robot, correct_robot)
            output_paths.append(output_path)

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
