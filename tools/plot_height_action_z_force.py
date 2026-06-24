#!/usr/bin/env python3
import argparse
import csv
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SQRT2 = np.sqrt(2.0) / 2.0
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ],
    dtype=float,
)

DEFAULT_TIMESERIES_DEMOS = ("demo_0", "demo_20", "demo_40", "demo_60", "demo_80")
FORCE_OFFSET_COUNT = 10
FORCE_EMA_ALPHA = 0.03


@dataclass
class JointSpec:
    name: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def parse_xyz(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(value) for value in text.split()], dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ]
    )


def transform_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def rotation_transform(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


class SerialUrdfFk:
    def __init__(self, urdf_path: Path, base_link: str = "base", tip_link: str = "link_6"):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.tip_link = tip_link
        self.joints = self._load_chain()
        self.revolute_names = [joint.name for joint in self.joints if joint.joint_type != "fixed"]
        if len(self.revolute_names) != 6:
            raise ValueError(
                f"expected 6 moving joints in {urdf_path}, found {len(self.revolute_names)}: "
                f"{self.revolute_names}"
            )

    def _load_chain(self) -> list[JointSpec]:
        root = ET.parse(self.urdf_path).getroot()
        child_to_joint = {}
        parent_to_children = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            child_to_joint[child] = joint
            parent_to_children.setdefault(parent, []).append(child)

        chain_joints = []
        current = self.tip_link
        while current != self.base_link:
            if current not in child_to_joint:
                raise ValueError(f"cannot find joint connecting to {current} in {self.urdf_path}")
            joint = child_to_joint[current]
            origin = joint.find("origin")
            axis = joint.find("axis")
            chain_joints.append(
                JointSpec(
                    name=joint.attrib["name"],
                    joint_type=joint.attrib.get("type", "fixed"),
                    xyz=parse_xyz(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                    rpy=parse_xyz(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                    axis=parse_xyz(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                )
            )
            current = joint.find("parent").attrib["link"]

        chain_joints.reverse()
        return chain_joints

    def fk(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint_positions = np.asarray(joint_positions, dtype=float)
        positions = np.empty((len(joint_positions), 3), dtype=float)
        rotations = np.empty((len(joint_positions), 3, 3), dtype=float)

        for row_index, q in enumerate(joint_positions):
            transform = np.eye(4)
            moving_index = 0
            for joint in self.joints:
                transform = transform @ transform_matrix(joint.xyz, joint.rpy)
                if joint.joint_type != "fixed":
                    transform = transform @ rotation_transform(
                        axis_angle_matrix(joint.axis, float(q[moving_index]))
                    )
                    moving_index += 1
            positions[row_index] = transform[:3, 3]
            rotations[row_index] = transform[:3, :3]
        return positions, rotations


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


def load_demo(
    hdf_path: Path,
    demo: str,
    fk: SerialUrdfFk,
    wrench_frame: str,
    offset_count: int,
    ema_alpha: float,
    contact_force_threshold: float,
    min_xy_speed: float,
) -> dict[str, np.ndarray]:
    with h5py.File(hdf_path, "r") as hdf:
        obs = hdf[f"data/{demo}/observations"]
        time_robot = np.asarray(obs["timestamp_robot"], dtype=float)
        joint_r = np.asarray(obs["joint_R"], dtype=float)
        desired_robot_m = np.asarray(obs["desired_pose"], dtype=float)[:, :3] / 1000.0
        time_wrench_raw = np.asarray(obs["timestamp_wrench"], dtype=float)
        wrench_raw = np.asarray(obs["wrench_wrist_R"], dtype=float)

    time_wrench, wrench = preprocess_wrench(time_wrench_raw, wrench_raw, offset_count, ema_alpha)
    valid_robot = (time_robot >= time_wrench[0]) & (time_robot <= time_wrench[-1])
    time_robot = time_robot[valid_robot]
    joint_r = joint_r[valid_robot]
    desired_robot_m = desired_robot_m[valid_robot]

    actual_robot_m, actual_robot_rot = fk.fk(joint_r)
    idx = nearest_indices(time_robot, time_wrench)
    force_input = wrench[idx, :3]

    if wrench_frame == "ee":
        force_robot = np.einsum("nij,nj->ni", actual_robot_rot, force_input)
        force_world = right_robot_to_world(force_robot)
    elif wrench_frame == "robot":
        force_robot = force_input
        force_world = right_robot_to_world(force_robot)
    elif wrench_frame == "world":
        force_robot = right_robot_to_world(force_input)
        force_world = force_input
    else:
        raise ValueError(f"unsupported wrench frame: {wrench_frame}")

    desired_world_m = right_robot_to_world(desired_robot_m)
    actual_world_m = right_robot_to_world(actual_robot_m)
    xy_speed = horizontal_speed_mps(time_robot, desired_world_m)
    force_z_world = force_world[:, 2]
    contact_moving = (np.abs(force_z_world) >= contact_force_threshold) & (xy_speed >= min_xy_speed)

    if np.any(contact_moving):
        desired_contact_median = float(np.median(desired_world_m[contact_moving, 2]))
    else:
        desired_contact_median = float(np.median(desired_world_m[:, 2]))

    return {
        "demo": np.full(len(time_robot), demo),
        "time_s": time_robot - time_robot[0],
        "desired_world_x_mm": desired_world_m[:, 0] * 1000.0,
        "desired_world_y_mm": desired_world_m[:, 1] * 1000.0,
        "desired_world_z_mm": desired_world_m[:, 2] * 1000.0,
        "actual_world_x_mm": actual_world_m[:, 0] * 1000.0,
        "actual_world_y_mm": actual_world_m[:, 1] * 1000.0,
        "actual_world_z_mm": actual_world_m[:, 2] * 1000.0,
        "x_error_mm": (desired_world_m[:, 0] - actual_world_m[:, 0]) * 1000.0,
        "y_error_mm": (desired_world_m[:, 1] - actual_world_m[:, 1]) * 1000.0,
        "z_error_mm": (desired_world_m[:, 2] - actual_world_m[:, 2]) * 1000.0,
        "desired_z_delta_mm": (desired_world_m[:, 2] - desired_contact_median) * 1000.0,
        "force_x_world_n": force_world[:, 0],
        "force_y_world_n": force_world[:, 1],
        "force_z_world_n": force_z_world,
        "force_z_abs_world_n": np.abs(force_z_world),
        "xy_speed_mps": xy_speed,
        "contact_moving": contact_moving,
    }


def available_demos(hdf_path: Path) -> list[str]:
    with h5py.File(hdf_path, "r") as hdf:
        demos = list(hdf["data"].keys())
    return sorted(demos, key=lambda name: int(name.split("_")[1]))


def concatenate_demo_data(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {key: np.concatenate([row[key] for row in rows]) for key in keys}


def binned_stats(x: np.ndarray, y: np.ndarray, bins: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < bins:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    centers = []
    means = []
    p25 = []
    p75 = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x <= right if right == edges[-1] else x < right)
        if np.count_nonzero(mask) < 5:
            continue
        centers.append(float(np.mean(x[mask])))
        means.append(float(np.mean(y[mask])))
        p25.append(float(np.percentile(y[mask], 25)))
        p75.append(float(np.percentile(y[mask], 75)))

    return np.asarray(centers), np.asarray(means), np.asarray(p25), np.asarray(p75)


def add_scatter_with_bins(
    fig: go.Figure,
    data: dict[str, np.ndarray],
    x_key: str,
    x_title: str,
    row: int,
    col: int,
    contact_only: bool,
) -> None:
    mask = data["contact_moving"] if contact_only else np.ones(len(data["force_z_world_n"]), dtype=bool)
    label = "contact + horizontal motion" if contact_only else "all aligned samples"
    color = data["xy_speed_mps"][mask] * 1000.0
    custom = np.column_stack(
        [
            data["desired_world_z_mm"][mask],
            data["actual_world_z_mm"][mask],
            data["z_error_mm"][mask],
            data["xy_speed_mps"][mask] * 1000.0,
        ]
    )
    fig.add_trace(
        go.Scattergl(
            x=data[x_key][mask],
            y=data["force_z_world_n"][mask],
            mode="markers",
            name=label,
            marker=dict(
                size=5 if contact_only else 4,
                color=color,
                colorscale="Viridis",
                showscale=contact_only,
                colorbar=dict(title="xy speed<br>mm/s") if contact_only else None,
                opacity=0.55 if contact_only else 0.20,
            ),
            customdata=custom,
            hovertemplate=(
                f"{x_title}: %{{x:.3f}} mm<br>"
                "Fz world: %{y:.3f} N<br>"
                "desired z: %{customdata[0]:.3f} mm<br>"
                "actual z: %{customdata[1]:.3f} mm<br>"
                "desired-actual z: %{customdata[2]:.3f} mm<br>"
                "xy speed: %{customdata[3]:.1f} mm/s<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )

    centers, means, p25, p75 = binned_stats(data[x_key][mask], data["force_z_world_n"][mask])
    if len(centers):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([centers, centers[::-1]]),
                y=np.concatenate([p75, p25[::-1]]),
                fill="toself",
                mode="lines",
                line=dict(color="rgba(30, 30, 30, 0.0)"),
                fillcolor="rgba(30, 30, 30, 0.15)",
                name=f"{label} IQR",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=means,
                mode="lines+markers",
                line=dict(color="#111111", width=3),
                marker=dict(size=5),
                name=f"{label} binned mean",
                hovertemplate=f"{x_title}: %{{x:.3f}} mm<br>mean Fz: %{{y:.3f}} N<extra></extra>",
            ),
            row=row,
            col=col,
        )


def plot_interactive(
    data: dict[str, np.ndarray],
    demo_rows: dict[str, dict[str, np.ndarray]],
    timeseries_demos: tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Action world z delta vs world Fz",
            "Desired minus actual world z vs world Fz",
            "Selected demos: desired/actual world z",
            "Selected demos: world Fz",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.13,
    )

    add_scatter_with_bins(
        fig,
        data,
        "desired_z_delta_mm",
        "action z delta",
        row=1,
        col=1,
        contact_only=False,
    )
    add_scatter_with_bins(
        fig,
        data,
        "desired_z_delta_mm",
        "action z delta",
        row=1,
        col=1,
        contact_only=True,
    )
    add_scatter_with_bins(
        fig,
        data,
        "z_error_mm",
        "desired-actual z",
        row=1,
        col=2,
        contact_only=True,
    )

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for index, demo in enumerate(timeseries_demos):
        if demo not in demo_rows:
            continue
        row = demo_rows[demo]
        color = palette[index % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=row["time_s"],
                y=row["desired_world_z_mm"],
                mode="lines",
                name=f"{demo} desired z",
                line=dict(color=color, width=2),
                legendgroup=demo,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=row["time_s"],
                y=row["actual_world_z_mm"],
                mode="lines",
                name=f"{demo} actual z",
                line=dict(color=color, width=1, dash="dot"),
                legendgroup=demo,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=row["time_s"],
                y=row["force_z_world_n"],
                mode="lines",
                name=f"{demo} Fz world",
                line=dict(color=color, width=2),
                legendgroup=demo,
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="action z delta from per-demo contact median (mm)", row=1, col=1)
    fig.update_xaxes(title_text="desired_world_z - actual_world_z (mm)", row=1, col=2)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=2)
    fig.update_yaxes(title_text="world Fz (N)", row=1, col=1)
    fig.update_yaxes(title_text="world Fz (N)", row=1, col=2)
    fig.update_yaxes(title_text="world z (mm)", row=2, col=1)
    fig.update_yaxes(title_text="world Fz (N)", row=2, col=2)

    contact = data["contact_moving"]
    if np.any(contact):
        for col, x_key in ((1, "desired_z_delta_mm"), (2, "z_error_mm")):
            low, high = np.percentile(data[x_key][contact], [0.5, 99.5])
            pad = max((high - low) * 0.08, 1.0)
            fig.update_xaxes(range=[low - pad, high + pad], row=1, col=col)

    fig.update_layout(
        title=title,
        width=1500,
        height=1000,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0.0),
        margin=dict(l=70, r=40, t=80, b=150),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def plot_png(data: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    specs = [
        ("desired_z_delta_mm", "action z delta from per-demo contact median (mm)"),
        ("z_error_mm", "desired_world_z - actual_world_z (mm)"),
    ]
    contact = data["contact_moving"]

    for axis, (x_key, x_label) in zip(axes, specs):
        axis.scatter(
            data[x_key],
            data["force_z_world_n"],
            s=4,
            alpha=0.08,
            color="0.45",
            label="all",
        )
        axis.scatter(
            data[x_key][contact],
            data["force_z_world_n"][contact],
            s=7,
            alpha=0.35,
            color="#1f77b4",
            label="contact + horizontal",
        )
        centers, means, p25, p75 = binned_stats(data[x_key][contact], data["force_z_world_n"][contact])
        if len(centers):
            axis.fill_between(centers, p25, p75, color="black", alpha=0.14, linewidth=0)
            axis.plot(centers, means, color="black", linewidth=2.0, label="binned mean")
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel(x_label)
        axis.set_ylabel("world Fz (N)")
        axis.legend(fontsize=8)
        if np.any(contact):
            low, high = np.percentile(data[x_key][contact], [0.5, 99.5])
            pad = max((high - low) * 0.08, 1.0)
            axis.set_xlim(low - pad, high + pad)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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


def plot_single_demo_timeseries(row: dict[str, np.ndarray], output_html: Path, output_png: Path, title: str) -> None:
    time_s = row["time_s"]

    fig = go.Figure()
    for start, end in contact_spans(time_s, row["contact_moving"]):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(31, 119, 180, 0.10)",
            line_width=0,
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=row["desired_world_z_mm"],
            mode="lines",
            name="desired/action world z",
            line=dict(color="#d62728", width=2.4),
            hovertemplate="t=%{x:.3f}s<br>desired z=%{y:.3f} mm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=row["actual_world_z_mm"],
            mode="lines",
            name="actual TCP world z",
            line=dict(color="#1f77b4", width=2.0, dash="dot"),
            hovertemplate="t=%{x:.3f}s<br>actual z=%{y:.3f} mm<extra></extra>",
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
            customdata=np.column_stack(
                [
                    row["desired_world_z_mm"],
                    row["actual_world_z_mm"],
                    row["z_error_mm"],
                    row["xy_speed_mps"] * 1000.0,
                ]
            ),
            hovertemplate=(
                "t=%{x:.3f}s<br>"
                "Fz world=%{y:.3f} N<br>"
                "desired z=%{customdata[0]:.3f} mm<br>"
                "actual z=%{customdata[1]:.3f} mm<br>"
                "desired-actual z=%{customdata[2]:.3f} mm<br>"
                "xy speed=%{customdata[3]:.1f} mm/s<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        width=1300,
        height=620,
        template="plotly_white",
        xaxis=dict(title="time (s)"),
        yaxis=dict(title="world z (mm)", side="left"),
        yaxis2=dict(title="world Fz (N)", side="right", overlaying="y", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=70, r=75, t=90, b=60),
    )
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)

    mpl_fig, ax_z = plt.subplots(figsize=(14, 5.6), constrained_layout=True)
    ax_f = ax_z.twinx()
    for start, end in contact_spans(time_s, row["contact_moving"]):
        ax_z.axvspan(start, end, color="#1f77b4", alpha=0.10, linewidth=0)

    z_desired_line, = ax_z.plot(
        time_s,
        row["desired_world_z_mm"],
        color="#d62728",
        linewidth=2.0,
        label="desired/action world z",
    )
    z_actual_line, = ax_z.plot(
        time_s,
        row["actual_world_z_mm"],
        color="#1f77b4",
        linewidth=1.8,
        linestyle=":",
        label="actual TCP world z",
    )
    force_line, = ax_f.plot(
        time_s,
        row["force_z_world_n"],
        color="#111111",
        linewidth=1.8,
        label="wrist Fz world",
    )

    ax_z.set_xlabel("time (s)")
    ax_z.set_ylabel("world z (mm)")
    ax_f.set_ylabel("world Fz (N)")
    ax_z.grid(True, alpha=0.28)
    ax_z.set_title(title)
    ax_z.legend(handles=[z_desired_line, z_actual_line, force_line], loc="upper right")
    mpl_fig.savefig(output_png, dpi=180)
    plt.close(mpl_fig)


def plot_single_demo_z_error_force(
    row: dict[str, np.ndarray],
    output_html: Path,
    output_png: Path,
    title: str,
) -> None:
    time_s = row["time_s"]

    fig = go.Figure()
    for start, end in contact_spans(time_s, row["contact_moving"]):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(31, 119, 180, 0.10)",
            line_width=0,
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=row["z_error_mm"],
            mode="lines",
            name="desired z - actual z",
            line=dict(color="#d62728", width=2.4),
            customdata=np.column_stack(
                [
                    row["desired_world_z_mm"],
                    row["actual_world_z_mm"],
                    row["xy_speed_mps"] * 1000.0,
                ]
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
            customdata=np.column_stack(
                [
                    row["desired_world_z_mm"],
                    row["actual_world_z_mm"],
                    row["z_error_mm"],
                    row["xy_speed_mps"] * 1000.0,
                ]
            ),
            hovertemplate=(
                "t=%{x:.3f}s<br>"
                "Fz world=%{y:.3f} N<br>"
                "desired-actual z=%{customdata[2]:.3f} mm<br>"
                "desired z=%{customdata[0]:.3f} mm<br>"
                "actual z=%{customdata[1]:.3f} mm<br>"
                "xy speed=%{customdata[3]:.1f} mm/s<extra></extra>"
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


def plot_single_demo_xyz_error_force(
    row: dict[str, np.ndarray],
    output_html: Path,
    output_png: Path,
    title: str,
) -> None:
    time_s = row["time_s"]
    axes = (
        ("x", "x_error_mm", "force_x_world_n", "#d62728"),
        ("y", "y_error_mm", "force_y_world_n", "#2ca02c"),
        ("z", "z_error_mm", "force_z_world_n", "#1f77b4"),
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "world x: desired-actual and Fx",
            "world y: desired-actual and Fy",
            "world z: desired-actual and Fz",
        ),
        vertical_spacing=0.08,
    )

    for row_index, (axis_name, error_key, force_key, color) in enumerate(axes, start=1):
        for start, end in contact_spans(time_s, row["contact_moving"]):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(31, 119, 180, 0.08)",
                line_width=0,
                layer="below",
                row=row_index,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=time_s,
                y=row[error_key],
                mode="lines",
                name=f"desired {axis_name} - actual {axis_name}",
                line=dict(color=color, width=2.2),
                hovertemplate=f"t=%{{x:.3f}}s<br>{axis_name} error=%{{y:.3f}} mm<extra></extra>",
            ),
            row=row_index,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=time_s,
                y=row[force_key],
                mode="lines",
                name=f"world F{axis_name}",
                line=dict(color="#111111", width=1.8),
                hovertemplate=f"t=%{{x:.3f}}s<br>F{axis_name} world=%{{y:.3f}} N<extra></extra>",
            ),
            row=row_index,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(title_text=f"{axis_name} error (mm)", row=row_index, col=1, secondary_y=False)
        fig.update_yaxes(title_text=f"F{axis_name} world (N)", row=row_index, col=1, secondary_y=True)

    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_layout(
        title=title,
        width=1400,
        height=1000,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=80, r=80, t=110, b=60),
    )
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)

    mpl_fig, mpl_axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for ax_error, (axis_name, error_key, force_key, color) in zip(mpl_axes, axes):
        ax_force = ax_error.twinx()
        for start, end in contact_spans(time_s, row["contact_moving"]):
            ax_error.axvspan(start, end, color="#1f77b4", alpha=0.08, linewidth=0)

        error_line, = ax_error.plot(
            time_s,
            row[error_key],
            color=color,
            linewidth=2.0,
            label=f"desired {axis_name} - actual {axis_name}",
        )
        force_line, = ax_force.plot(
            time_s,
            row[force_key],
            color="#111111",
            linewidth=1.7,
            label=f"world F{axis_name}",
        )
        ax_error.axhline(0.0, color=color, alpha=0.22, linewidth=0.9)
        ax_force.axhline(0.0, color="#111111", alpha=0.18, linewidth=0.9)
        ax_error.set_ylabel(f"{axis_name} error (mm)")
        ax_force.set_ylabel(f"F{axis_name} world (N)")
        ax_error.grid(True, alpha=0.28)
        ax_error.legend(handles=[error_line, force_line], loc="upper right")

    mpl_axes[-1].set_xlabel("time (s)")
    mpl_fig.suptitle(title)
    mpl_fig.savefig(output_png, dpi=180)
    plt.close(mpl_fig)


def write_samples_csv(data: dict[str, np.ndarray], output_path: Path) -> None:
    fields = [
        "demo",
        "time_s",
        "desired_world_z_mm",
        "actual_world_z_mm",
        "z_error_mm",
        "desired_z_delta_mm",
        "force_x_world_n",
        "force_y_world_n",
        "force_z_world_n",
        "force_z_abs_world_n",
        "xy_speed_mps",
        "contact_moving",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fields)
        for index in range(len(data["time_s"])):
            writer.writerow([data[field][index] for field in fields])


def write_summary_csv(demo_rows: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    fields = [
        "demo",
        "samples",
        "contact_moving_samples",
        "force_z_world_mean_n",
        "force_z_world_std_n",
        "force_z_world_abs_mean_n",
        "z_error_mean_mm",
        "z_error_std_mm",
        "desired_z_delta_std_mm",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for demo, row in demo_rows.items():
            contact = row["contact_moving"]
            stat_mask = contact if np.any(contact) else np.ones(len(contact), dtype=bool)
            writer.writerow(
                {
                    "demo": demo,
                    "samples": len(row["time_s"]),
                    "contact_moving_samples": int(np.count_nonzero(contact)),
                    "force_z_world_mean_n": float(np.mean(row["force_z_world_n"][stat_mask])),
                    "force_z_world_std_n": float(np.std(row["force_z_world_n"][stat_mask])),
                    "force_z_world_abs_mean_n": float(np.mean(row["force_z_abs_world_n"][stat_mask])),
                    "z_error_mean_mm": float(np.mean(row["z_error_mm"][stat_mask])),
                    "z_error_std_mm": float(np.std(row["z_error_mm"][stat_mask])),
                    "desired_z_delta_std_mm": float(np.std(row["desired_z_delta_mm"][stat_mask])),
                }
            )


def parse_demos(raw_demos: list[str] | None, hdf_path: Path) -> list[str]:
    demos = available_demos(hdf_path)
    if not raw_demos:
        return demos
    demo_set = set(demos)
    parsed = []
    for raw in raw_demos:
        demo = raw if raw.startswith("demo_") else f"demo_{raw}"
        if demo not in demo_set:
            raise ValueError(f"{demo} not found in {hdf_path}")
        parsed.append(demo)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot desired/action world z differences against wrist FT force transformed into the world frame."
        )
    )
    parser.add_argument("--input", type=Path, default=Path.home() / "Downloads/common_data_height.hdf5")
    parser.add_argument("--urdf", type=Path, default=Path("m0609.white.urdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/height_force"))
    parser.add_argument("--demos", nargs="*", help="Demo ids, e.g. demo_0 demo_20 or 0 20. Default: all.")
    parser.add_argument("--timeseries-demos", nargs="*", default=list(DEFAULT_TIMESERIES_DEMOS))
    parser.add_argument("--single-demo", default="demo_0", help="Demo id for one-plot z/Fz time-series output.")
    parser.add_argument(
        "--wrench-frame",
        choices=("ee", "robot", "world"),
        default="ee",
        help="Frame of wrench_wrist_R force. ee uses URDF FK link_6 rotation before world rotation.",
    )
    parser.add_argument("--force-offset-count", type=int, default=FORCE_OFFSET_COUNT)
    parser.add_argument("--force-ema-alpha", type=float, default=FORCE_EMA_ALPHA)
    parser.add_argument("--contact-force-threshold", type=float, default=2.0)
    parser.add_argument("--min-xy-speed", type=float, default=0.005)
    args = parser.parse_args()

    hdf_path = args.input.expanduser().resolve()
    urdf_path = args.urdf.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demos = parse_demos(args.demos, hdf_path)
    timeseries_demos = tuple(
        demo if demo.startswith("demo_") else f"demo_{demo}" for demo in args.timeseries_demos
    )
    single_demo = args.single_demo if args.single_demo.startswith("demo_") else f"demo_{args.single_demo}"
    if single_demo not in set(available_demos(hdf_path)):
        raise ValueError(f"{single_demo} not found in {hdf_path}")
    if single_demo not in demos:
        demos.append(single_demo)

    fk = SerialUrdfFk(urdf_path)
    demo_rows = {}
    rows = []
    for demo in demos:
        row = load_demo(
            hdf_path,
            demo,
            fk,
            args.wrench_frame,
            args.force_offset_count,
            args.force_ema_alpha,
            args.contact_force_threshold,
            args.min_xy_speed,
        )
        demo_rows[demo] = row
        rows.append(row)

    data = concatenate_demo_data(rows)
    contact_count = int(np.count_nonzero(data["contact_moving"]))
    total_count = len(data["time_s"])
    title = (
        f"{hdf_path.name}: action z vs world-frame wrist Fz "
        f"({len(demos)} demos, {contact_count}/{total_count} contact-moving samples, "
        f"wrench frame={args.wrench_frame})"
    )

    stem = hdf_path.stem
    html_path = args.output_dir / f"{stem}_action_z_vs_world_fz.html"
    png_path = args.output_dir / f"{stem}_action_z_vs_world_fz.png"
    samples_csv_path = args.output_dir / f"{stem}_action_z_vs_world_fz_samples.csv"
    summary_csv_path = args.output_dir / f"{stem}_action_z_vs_world_fz_summary.csv"
    single_html_path = args.output_dir / f"{stem}_{single_demo}_z_and_world_fz_timeseries.html"
    single_png_path = args.output_dir / f"{stem}_{single_demo}_z_and_world_fz_timeseries.png"
    error_html_path = args.output_dir / f"{stem}_{single_demo}_z_error_and_world_fz_timeseries.html"
    error_png_path = args.output_dir / f"{stem}_{single_demo}_z_error_and_world_fz_timeseries.png"
    xyz_error_html_path = args.output_dir / f"{stem}_{single_demo}_xyz_error_and_world_force_timeseries.html"
    xyz_error_png_path = args.output_dir / f"{stem}_{single_demo}_xyz_error_and_world_force_timeseries.png"

    plot_interactive(data, demo_rows, timeseries_demos, html_path, title)
    plot_png(data, png_path, title)
    plot_single_demo_timeseries(
        demo_rows[single_demo],
        single_html_path,
        single_png_path,
        (
            f"{hdf_path.name} {single_demo}: desired/action z and world-frame wrist Fz "
            f"(wrench frame={args.wrench_frame})"
        ),
    )
    plot_single_demo_z_error_force(
        demo_rows[single_demo],
        error_html_path,
        error_png_path,
        (
            f"{hdf_path.name} {single_demo}: desired-actual z error and world-frame wrist Fz "
            f"(wrench frame={args.wrench_frame})"
        ),
    )
    plot_single_demo_xyz_error_force(
        demo_rows[single_demo],
        xyz_error_html_path,
        xyz_error_png_path,
        (
            f"{hdf_path.name} {single_demo}: world-frame xyz desired-actual error and wrist force "
            f"(wrench frame={args.wrench_frame})"
        ),
    )
    write_samples_csv(data, samples_csv_path)
    write_summary_csv(demo_rows, summary_csv_path)

    print(html_path)
    print(png_path)
    print(single_html_path)
    print(single_png_path)
    print(error_html_path)
    print(error_png_path)
    print(xyz_error_html_path)
    print(xyz_error_png_path)
    print(samples_csv_path)
    print(summary_csv_path)


if __name__ == "__main__":
    main()
