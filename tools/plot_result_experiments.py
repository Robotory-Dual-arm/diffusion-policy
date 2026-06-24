#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3


SQRT2 = np.sqrt(2) / 2
SAMPLE_HZ = 10.0
WRENCH_CALIB_COUNT = 10
WRENCH_EMA_ALPHA = 0.01
POSE_CAMERA_EYE = dict(x=0.0, y=-2.4, z=0.42)

RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ]
)

FORCE_LABELS = ("x", "y", "z")
TORQUE_LABELS = ("rx", "ry", "rz")
COMPONENT_COLORS = ("red", "blue", "green")


def right_robot_to_world(pos_robot: np.ndarray) -> np.ndarray:
    return (RIGHT_ROBOT_TO_WORLD @ np.asarray(pos_robot).T).T


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


def ema_filter(wrench: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.empty_like(wrench)
    filtered[0] = wrench[0]
    for index in range(1, len(wrench)):
        filtered[index] = alpha * wrench[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def apply_controller_wrench_filter(time_s: np.ndarray, wrench: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offset = np.mean(wrench[:WRENCH_CALIB_COUNT], axis=0)
    corrected = wrench - offset
    corrected = corrected[WRENCH_CALIB_COUNT:]
    time_s = time_s[WRENCH_CALIB_COUNT:]
    filtered = ema_filter(corrected, WRENCH_EMA_ALPHA)
    return time_s - time_s[0], filtered


def axis_ranges(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max() / 2.0), 1e-3) * 1.08
    return (
        [centers[0] - radius, centers[0] + radius],
        [centers[1] - radius, centers[1] + radius],
        [centers[2] - radius, centers[2] + radius],
    )


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


def write_pose_png_from_html(html_path: Path, png_path: Path) -> None:
    subprocess.run(
        [
            "google-chrome",
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=1100,850",
            "--virtual-time-budget=5000",
            f"--screenshot={png_path.resolve()}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def plot_pose(
    path: Path,
    html_output_path: Path,
    png_output_path: Path,
    weird_robot: rtb.ERobot,
    correct_robot: rtb.ERobot,
) -> None:
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
    actual_xyz = right_robot_to_world(
        recalculate_actual_with_correct_urdf(actual_xyz, actual_quat, weird_robot, correct_robot)
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
            hovertemplate="virtual target<br>t=%{customdata:.3f}s<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
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
            hovertemplate="actual pose<br>t=%{customdata:.3f}s<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{path.stem}: virtual target vs actual pose, world frame, {SAMPLE_HZ:g} Hz points",
        width=1100,
        height=850,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(
                title="x (m)",
                range=x_range,
                backgroundcolor="#f7f7f7",
                gridcolor="#d9d9d9",
            ),
            yaxis=dict(
                title="y (m)",
                range=y_range,
                backgroundcolor="#f7f7f7",
                gridcolor="#d9d9d9",
            ),
            zaxis=dict(
                title="z (m)",
                range=z_range,
                backgroundcolor="#f7f7f7",
                gridcolor="#d9d9d9",
            ),
            aspectmode="cube",
            camera=dict(eye=POSE_CAMERA_EYE),
        ),
    )
    fig.write_html(html_output_path, include_plotlyjs=True, full_html=True)
    write_pose_png_from_html(html_output_path, png_output_path)


def plot_wrench(path: Path, output_path: Path) -> None:
    with h5py.File(path, "r") as f:
        time_s = np.asarray(f["wrist_ft/elapsed_s"], dtype=float)
        wrench = np.asarray(f["wrist_ft/wrench_wrist_R"], dtype=float)

    time_s, wrench = apply_controller_wrench_filter(time_s, wrench)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    force_axis, torque_axis = axes

    for component_index, (label, color) in enumerate(zip(FORCE_LABELS, COMPONENT_COLORS)):
        force_axis.plot(time_s, wrench[:, component_index], color=color, linewidth=1.1, alpha=0.8, label=label)

    for component_index, (label, color) in enumerate(zip(TORQUE_LABELS, COMPONENT_COLORS), start=3):
        torque_axis.plot(time_s, wrench[:, component_index], color=color, linewidth=1.1, alpha=0.8, label=label)

    force_axis.set_title("Force xyz")
    force_axis.set_ylabel("force (N)")
    torque_axis.set_title("Torque rx ry rz")
    torque_axis.set_ylabel("torque (Nm)")
    torque_axis.set_xlabel("time (s)")

    for axis in axes:
        axis.set_facecolor("white")
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.55)
        axis.legend(loc="best", fontsize=9, ncol=3)

    fig.suptitle(
        f"{path.stem}: wrist wrench, calib {WRENCH_CALIB_COUNT} samples + EMA alpha {WRENCH_EMA_ALPHA}"
    )
    fig.savefig(output_path, dpi=180, facecolor="white", edgecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "Downloads/result")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Downloads/result_visualizations")
    parser.add_argument("--weird-urdf", type=Path, default=Path("m0609.white_weird.urdf"))
    parser.add_argument("--correct-urdf", type=Path, default=Path("m0609.white.urdf"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weird_robot = rtb.ERobot.URDF(str(args.weird_urdf.resolve()))
    correct_robot = rtb.ERobot.URDF(str(args.correct_urdf.resolve()))

    for path in sorted(args.input_dir.glob("*.hdf5")):
        pose_output = args.output_dir / f"{path.stem}_pose_3d.html"
        pose_png_output = args.output_dir / f"{path.stem}_pose_3d.png"
        wrench_output = args.output_dir / f"{path.stem}_wrench.png"
        plot_pose(path, pose_output, pose_png_output, weird_robot, correct_robot)
        plot_wrench(path, wrench_output)
        print(pose_output)
        print(pose_png_output)
        print(wrench_output)


if __name__ == "__main__":
    main()
