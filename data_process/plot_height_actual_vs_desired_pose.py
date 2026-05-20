import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_INPUT = "/data/baetae/260519/diffusion_data_height_wrench_encoder_R_image_desired_pose_action.hdf5"
DEFAULT_OUTPUT = "visualizations/height_demo_actual_vs_desired_pose.html"

SQRT2 = np.sqrt(2) / 2

# Right arm robot base frame -> world frame. Same convention as data_process/plot_pose_hdf.py.
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ]
)


def right_robot_to_world(pos_robot):
    pos_robot = np.asarray(pos_robot)
    return (RIGHT_ROBOT_TO_WORLD @ pos_robot.T).T


def set_equal_axes(fig, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = float(np.max(maxs - mins) / 2.0)
    half_range = max(half_range, 1e-3)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[center[0] - half_range, center[0] + half_range], title="X (m)"),
            yaxis=dict(range=[center[1] - half_range, center[1] + half_range], title="Y (m)"),
            zaxis=dict(range=[center[2] - half_range, center[2] + half_range], title="Z (m)"),
            aspectmode="cube",
        )
    )


def axis_ranges(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = float(np.max(maxs - mins) / 2.0)
    half_range = max(half_range, 1e-3)
    return {
        "xaxis": dict(range=[center[0] - half_range, center[0] + half_range], title="X (m)"),
        "yaxis": dict(range=[center[1] - half_range, center[1] + half_range], title="Y (m)"),
        "zaxis": dict(range=[center[2] - half_range, center[2] + half_range], title="Z (m)"),
        "aspectmode": "cube",
    }


def load_demo(path, demo):
    with h5py.File(path, "r") as f:
        group = f["data"][demo]
        actual_xyz = np.asarray(group["obs"]["robot_pose_R"])
        actions = np.asarray(group["actions"])

    if actions.shape[1] >= 9:
        desired_xyz_m = actions[:, :3]
        desired_pose = actions.copy()
        desired_pose[:, :3] *= 1000.0
    else:
        desired_xyz_m = actions[:, :3] / 1000.0
        desired_pose = actions
    n = min(len(actual_xyz), len(desired_xyz_m))
    return actual_xyz[:n], desired_xyz_m[:n], desired_pose[:n]


def make_figure(actual_xyz, desired_xyz_m, desired_pose, demo, frame_name):
    steps = np.arange(len(actual_xyz))
    err = np.linalg.norm(actual_xyz - desired_xyz_m, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=actual_xyz[:, 0],
            y=actual_xyz[:, 1],
            z=actual_xyz[:, 2],
            mode="lines+markers",
            name="Actual TCP pose",
            line=dict(color="#1f77b4", width=6),
            marker=dict(size=2, color=steps, colorscale="Blues", showscale=False),
            hovertemplate="actual<br>step=%{customdata}<br>x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<extra></extra>",
            customdata=steps,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=desired_xyz_m[:, 0],
            y=desired_xyz_m[:, 1],
            z=desired_xyz_m[:, 2],
            mode="lines+markers",
            name="Desired pose action",
            line=dict(color="#d62728", width=6),
            marker=dict(size=2, color=steps, colorscale="Reds", showscale=False),
            hovertemplate=(
                "desired<br>step=%{customdata[0]}<br>"
                "x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<br>"
                "raw=[%{customdata[1]:.2f}, %{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>"
            ),
            customdata=np.column_stack([steps, desired_pose[:, :3]]),
        )
    )

    for xyz, label, color in [
        (actual_xyz[0], "actual start", "#1f77b4"),
        (actual_xyz[-1], "actual end", "#0b3d91"),
        (desired_xyz_m[0], "desired start", "#d62728"),
        (desired_xyz_m[-1], "desired end", "#8b0000"),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[xyz[0]],
                y=[xyz[1]],
                z=[xyz[2]],
                mode="markers+text",
                name=label,
                marker=dict(size=6, color=color),
                text=[label],
                textposition="top center",
                showlegend=False,
            )
        )

    all_points = np.vstack([actual_xyz, desired_xyz_m])
    set_equal_axes(fig, all_points)
    fig.update_layout(
        title=(
            f"{demo}: actual TCP pose vs desired pose action ({frame_name} frame) "
            f"(mean error {err.mean():.4f} m, max {err.max():.4f} m)"
        ),
        width=1100,
        height=850,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return fig


def add_demo_traces(fig, actual_xyz, desired_xyz_m, desired_pose, demo, row, col, showlegend):
    steps = np.arange(len(actual_xyz))
    err = np.linalg.norm(actual_xyz - desired_xyz_m, axis=1)

    fig.add_trace(
        go.Scatter3d(
            x=actual_xyz[:, 0],
            y=actual_xyz[:, 1],
            z=actual_xyz[:, 2],
            mode="lines",
            name="Actual TCP pose",
            legendgroup="actual",
            showlegend=showlegend,
            line=dict(color="#1f77b4", width=5),
            hovertemplate=(
                f"{demo} actual<br>step=%{{customdata}}<br>"
                "x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<extra></extra>"
            ),
            customdata=steps,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter3d(
            x=desired_xyz_m[:, 0],
            y=desired_xyz_m[:, 1],
            z=desired_xyz_m[:, 2],
            mode="lines",
            name="Desired pose action",
            legendgroup="desired",
            showlegend=showlegend,
            line=dict(color="#d62728", width=5),
            hovertemplate=(
                f"{demo} desired<br>step=%{{customdata[0]}}<br>"
                "x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<br>"
                "raw=[%{customdata[1]:.2f}, %{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>"
            ),
            customdata=np.column_stack([steps, desired_pose[:, :3]]),
        ),
        row=row,
        col=col,
    )

    for xyz, label, color, symbol in [
        (actual_xyz[0], "actual start", "#1f77b4", "circle"),
        (desired_xyz_m[0], "desired start", "#d62728", "diamond"),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[xyz[0]],
                y=[xyz[1]],
                z=[xyz[2]],
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=showlegend,
                marker=dict(size=5, color=color, symbol=symbol),
                hovertemplate=f"{demo} {label}<br>x=%{{x:.4f}} m<br>y=%{{y:.4f}} m<br>z=%{{z:.4f}} m<extra></extra>",
            ),
            row=row,
            col=col,
        )

    return err


def make_multi_figure(demos_data, frame_name):
    n = len(demos_data)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    titles = []
    for demo, actual_xyz, desired_xyz_m, _ in demos_data:
        err = np.linalg.norm(actual_xyz - desired_xyz_m, axis=1)
        titles.append(f"{demo}<br>mean {err.mean() * 1000:.1f} mm / max {err.max() * 1000:.1f} mm")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for idx, (demo, actual_xyz, desired_xyz_m, desired_pose) in enumerate(demos_data):
        row = idx // cols + 1
        col = idx % cols + 1
        add_demo_traces(fig, actual_xyz, desired_xyz_m, desired_pose, demo, row, col, showlegend=(idx == 0))
        scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
        fig.update_layout({scene_name: axis_ranges(np.vstack([actual_xyz, desired_xyz_m]))})

    fig.update_layout(
        title=f"Actual TCP pose vs desired pose action ({frame_name} frame)",
        width=520 * cols,
        height=480 * rows,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


def resolve_demos(path, demo, demos):
    if demos:
        return demos
    if "," in demo:
        return [item.strip() for item in demo.split(",") if item.strip()]
    return [demo]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--demo", default="demo_29")
    parser.add_argument("--demos", nargs="+", help="Plot multiple demos in one HTML.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frame",
        choices=["world", "robot"],
        default="world",
        help="Plot in right-arm robot base frame or world frame. Default: world.",
    )
    args = parser.parse_args()

    demo_names = resolve_demos(args.input, args.demo, args.demos)
    demos_data = []
    for demo in demo_names:
        actual_xyz, desired_xyz_m, desired_pose = load_demo(args.input, demo)
        if args.frame == "world":
            actual_xyz = right_robot_to_world(actual_xyz)
            desired_xyz_m = right_robot_to_world(desired_xyz_m)
        demos_data.append((demo, actual_xyz, desired_xyz_m, desired_pose))

    if len(demos_data) == 1:
        demo, actual_xyz, desired_xyz_m, desired_pose = demos_data[0]
        fig = make_figure(actual_xyz, desired_xyz_m, desired_pose, demo, args.frame)
    else:
        fig = make_multi_figure(demos_data, args.frame)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn")
    print(output.resolve())


if __name__ == "__main__":
    main()
