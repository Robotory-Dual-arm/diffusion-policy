#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


SQRT2 = np.sqrt(2.0) / 2.0
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ],
    dtype=float,
)

DEFAULT_ACTUAL_Z_CORRECTION_M = -0.018
FORCE_OFFSET_COUNT = 10
FORCE_EMA_ALPHA = 0.03


def right_robot_to_world(vectors_robot: np.ndarray) -> np.ndarray:
    vectors_robot = np.asarray(vectors_robot, dtype=float)
    return (RIGHT_ROBOT_TO_WORLD @ vectors_robot.T).T


def ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def preprocess_wrench(
    time_s: np.ndarray,
    wrench: np.ndarray,
    offset_count: int,
    ema_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=float)
    wrench = np.asarray(wrench, dtype=float)

    if offset_count > 0:
        offset = np.mean(wrench[:offset_count], axis=0)
        wrench = wrench - offset
        wrench = wrench[offset_count:]
        time_s = time_s[offset_count:]

    if ema_alpha > 0.0:
        wrench = ema_filter(wrench, ema_alpha)

    return time_s, wrench


def nearest_indices(query_time: np.ndarray, sample_time: np.ndarray) -> np.ndarray:
    insert = np.searchsorted(sample_time, query_time, side="left")
    insert = np.clip(insert, 1, len(sample_time) - 1)
    left = insert - 1
    right = insert
    use_right = np.abs(sample_time[right] - query_time) < np.abs(query_time - sample_time[left])
    return np.where(use_right, right, left)


def horizontal_speed_mps(time_s: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
    speed = np.zeros(len(time_s), dtype=float)
    if len(time_s) < 2:
        return speed
    dt = np.diff(time_s)
    dt = np.where(dt <= 1e-9, np.nan, dt)
    dxy = np.linalg.norm(np.diff(xyz_world[:, :2], axis=0), axis=1)
    speed[1:] = dxy / dt
    speed[0] = speed[1]
    return np.nan_to_num(speed)


def contact_spans(time_s: np.ndarray, contact: np.ndarray) -> list[tuple[float, float]]:
    spans = []
    start = None
    for index, is_contact in enumerate(contact):
        if is_contact and start is None:
            start = float(time_s[index])
        if start is not None and (not is_contact or index == len(contact) - 1):
            end_index = index if is_contact else max(index - 1, 0)
            spans.append((start, float(time_s[end_index])))
            start = None
    return spans


def load_trial(
    path: Path,
    wrench_frame: str,
    force_offset_count: int,
    force_ema_alpha: float,
    actual_z_correction_m: float,
    contact_force_threshold: float,
    min_xy_speed: float,
) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as hdf:
        action_time = np.asarray(hdf["action_virtual_target/elapsed_s"], dtype=float)
        action_xyz_robot = np.asarray(hdf["action_virtual_target/action"], dtype=float)[:, :3]
        actual_time = np.asarray(hdf["actual/elapsed_s"], dtype=float)
        actual_xyz_robot = np.asarray(hdf["actual/robot_pose_R"], dtype=float)
        actual_quat_xyzw = np.asarray(hdf["actual/robot_quat_R"], dtype=float)
        wrench_time_raw = np.asarray(hdf["wrist_ft/elapsed_s"], dtype=float)
        wrench_raw = np.asarray(hdf["wrist_ft/wrench_wrist_R"], dtype=float)

    actual_xyz_robot = actual_xyz_robot.copy()
    actual_xyz_robot[:, 2] += actual_z_correction_m

    wrench_time, wrench = preprocess_wrench(
        wrench_time_raw,
        wrench_raw,
        force_offset_count,
        force_ema_alpha,
    )

    valid_actual = (actual_time >= action_time[0]) & (actual_time <= action_time[-1])
    valid_actual &= (actual_time >= wrench_time[0]) & (actual_time <= wrench_time[-1])
    actual_time = actual_time[valid_actual]
    actual_xyz_robot = actual_xyz_robot[valid_actual]
    actual_quat_xyzw = actual_quat_xyzw[valid_actual]

    action_idx = nearest_indices(actual_time, action_time)
    wrench_idx = nearest_indices(actual_time, wrench_time)
    desired_xyz_robot = action_xyz_robot[action_idx]
    force_input = wrench[wrench_idx, :3]

    if wrench_frame == "ee":
        ee_to_robot = R.from_quat(actual_quat_xyzw).as_matrix()
        force_robot = np.einsum("nij,nj->ni", ee_to_robot, force_input)
        force_world = right_robot_to_world(force_robot)
    elif wrench_frame == "robot":
        force_world = right_robot_to_world(force_input)
    elif wrench_frame == "world":
        force_world = force_input
    else:
        raise ValueError(f"unsupported wrench frame: {wrench_frame}")

    desired_world_m = right_robot_to_world(desired_xyz_robot)
    actual_world_m = right_robot_to_world(actual_xyz_robot)
    xy_speed = horizontal_speed_mps(actual_time, desired_world_m)
    force_z_world = force_world[:, 2]
    contact_moving = (np.abs(force_z_world) >= contact_force_threshold) & (xy_speed >= min_xy_speed)

    return {
        "time_s": actual_time - actual_time[0],
        "desired_world_z_mm": desired_world_m[:, 2] * 1000.0,
        "actual_world_z_mm": actual_world_m[:, 2] * 1000.0,
        "z_error_mm": (desired_world_m[:, 2] - actual_world_m[:, 2]) * 1000.0,
        "force_z_world_n": force_z_world,
        "xy_speed_mps": xy_speed,
        "contact_moving": contact_moving,
    }


def plot_z_error_force(row: dict[str, np.ndarray], path: Path, output_html: Path, output_png: Path, wrench_frame: str) -> None:
    time_s = row["time_s"]
    title = f"{path.name}: desired-actual z error and world-frame wrist Fz (wrench frame={wrench_frame})"

    fig = go.Figure()
    for start, end in contact_spans(time_s, row["contact_moving"]):
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(31,119,180,0.10)", line_width=0, layer="below")

    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=row["z_error_mm"],
            mode="lines",
            name="desired z - actual z",
            line=dict(color="#d62728", width=2.4),
            customdata=np.column_stack(
                [row["desired_world_z_mm"], row["actual_world_z_mm"], row["xy_speed_mps"] * 1000.0]
            ),
            hovertemplate=(
                "t=%{x:.3f}s<br>"
                "desired-actual z=%{y:.3f} mm<br>"
                "desired z=%{customdata[0]:.3f} mm<br>"
                "actual z=%{customdata[1]:.3f} mm<br>"
                "xy speed=%{customdata[2]:.1f} mm/s<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=row["force_z_world_n"],
            mode="lines",
            name="wrist Fz world",
            line=dict(color="#111111", width=2.0),
            yaxis="y2",
            customdata=row["z_error_mm"],
            hovertemplate=(
                "t=%{x:.3f}s<br>"
                "Fz world=%{y:.3f} N<br>"
                "desired-actual z=%{customdata:.3f} mm<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        width=1300,
        height=620,
        template="plotly_white",
        xaxis=dict(title="time (s)"),
        yaxis=dict(title="desired z - actual z (mm)", side="left"),
        yaxis2=dict(title="world Fz (N)", side="right", overlaying="y", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=80, r=75, t=90, b=60),
    )
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)

    mpl_fig, ax_error = plt.subplots(figsize=(14, 5.6), constrained_layout=True)
    ax_f = ax_error.twinx()
    for start, end in contact_spans(time_s, row["contact_moving"]):
        ax_error.axvspan(start, end, color="#1f77b4", alpha=0.10, linewidth=0)

    error_line, = ax_error.plot(
        time_s,
        row["z_error_mm"],
        color="#d62728",
        linewidth=2.0,
        label="desired z - actual z",
    )
    force_line, = ax_f.plot(
        time_s,
        row["force_z_world_n"],
        color="#111111",
        linewidth=1.8,
        label="wrist Fz world",
    )
    ax_error.axhline(0.0, color="#d62728", alpha=0.25, linewidth=0.9)
    ax_f.axhline(0.0, color="#111111", alpha=0.20, linewidth=0.9)
    ax_error.set_xlabel("time (s)")
    ax_error.set_ylabel("desired z - actual z (mm)")
    ax_f.set_ylabel("world Fz (N)")
    ax_error.grid(True, alpha=0.28)
    ax_error.set_title(title)
    ax_error.legend(handles=[error_line, force_line], loc="upper right")
    mpl_fig.savefig(output_png, dpi=180)
    plt.close(mpl_fig)


def default_files(input_dir: Path) -> list[Path]:
    files = [input_dir / f"wrench{index}.hdf5" for index in range(1, 6)]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--files", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("plots/inference_wrench_force"))
    parser.add_argument("--wrench-frame", choices=("ee", "robot", "world"), default="ee")
    parser.add_argument("--force-offset-count", type=int, default=FORCE_OFFSET_COUNT)
    parser.add_argument("--force-ema-alpha", type=float, default=FORCE_EMA_ALPHA)
    parser.add_argument("--actual-z-correction-m", type=float, default=DEFAULT_ACTUAL_Z_CORRECTION_M)
    parser.add_argument("--contact-force-threshold", type=float, default=2.0)
    parser.add_argument("--min-xy-speed", type=float, default=0.005)
    args = parser.parse_args()

    files = [path.expanduser().resolve() for path in args.files] if args.files else default_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for path in files:
        row = load_trial(
            path,
            args.wrench_frame,
            args.force_offset_count,
            args.force_ema_alpha,
            args.actual_z_correction_m,
            args.contact_force_threshold,
            args.min_xy_speed,
        )
        output_html = args.output_dir / f"{path.stem}_z_error_and_world_fz_timeseries.html"
        output_png = args.output_dir / f"{path.stem}_z_error_and_world_fz_timeseries.png"
        plot_z_error_force(row, path, output_html, output_png, args.wrench_frame)
        output_paths.extend([output_html, output_png])

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
