#!/usr/bin/env python3
if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go


SQRT2 = np.sqrt(2.0) / 2.0
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ],
    dtype=np.float32,
)


def sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def to_world(points):
    points = np.asarray(points, dtype=np.float32)
    return (RIGHT_ROBOT_TO_WORLD @ points.T).T


def axis_ranges(*arrays, pad_ratio=0.18):
    pts = np.concatenate([np.asarray(a)[:, :3] for a in arrays if len(a) > 0], axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius * (1.0 + pad_ratio), 0.03)
    return centers, radius


def add_points(fig, xyz, name, color, symbol, size=4):
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=3),
        marker=dict(color=color, size=size, symbol=symbol),
        hovertemplate=(
            name + "<br>"
            "step=%{customdata}<br>"
            "x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>"
        ),
        customdata=np.arange(len(xyz)),
    ))


def add_vector_lines(fig, starts, vectors, name, color, width=4, opacity=0.7, scale=1.0):
    x, y, z, custom = [], [], [], []
    ends = starts + vectors * scale
    for i, (s, e) in enumerate(zip(starts, ends)):
        x.extend([s[0], e[0], None])
        y.extend([s[1], e[1], None])
        z.extend([s[2], e[2], None])
        custom.extend([i, i, None])
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        opacity=opacity,
        customdata=custom,
        hovertemplate=name + " step=%{customdata}<extra></extra>",
    ))
    return ends


def add_cones(fig, starts, vectors, name, color, scale=1.0, sizeref=0.006, opacity=0.75):
    v = vectors * scale
    norms = np.linalg.norm(v, axis=1)
    keep = norms > 1e-9
    if not np.any(keep):
        return
    fig.add_trace(go.Cone(
        x=starts[keep, 0],
        y=starts[keep, 1],
        z=starts[keep, 2],
        u=v[keep, 0],
        v=v[keep, 1],
        w=v[keep, 2],
        name=name,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode="absolute",
        sizeref=sizeref,
        anchor="tail",
        opacity=opacity,
        hovertemplate=name + "<extra></extra>",
    ))


def make_plot(
    dataset,
    demo,
    output,
    start,
    count,
    stride,
    world_frame,
    actual_key,
    virtual_key,
    wrench_key,
    force_scale,
    force_stat,
):
    with h5py.File(dataset, "r") as f:
        if demo is None:
            demo = sorted_demo_keys(f["data"])[0]
        obs = f["data"][demo]["obs"]
        length = len(obs[actual_key])
        idx = np.arange(start, min(length, start + count * stride), stride, dtype=np.int64)
        actual = np.asarray(obs[actual_key], dtype=np.float32)[idx, :3]
        virtual = np.asarray(obs[virtual_key], dtype=np.float32)[idx, :3]
        wrench = np.asarray(obs[wrench_key], dtype=np.float32)[idx, :3]

        wrench_frame = f.attrs.get("wrench_frame", "unknown")

    if force_stat == "last":
        force = wrench[:, :, -1]
    elif force_stat == "mean":
        force = wrench.mean(axis=-1)
    elif force_stat == "maxabs":
        max_i = np.argmax(np.abs(wrench), axis=-1)
        force = np.take_along_axis(wrench, max_i[:, :, None], axis=-1)[:, :, 0]
    else:
        raise ValueError(f"Unsupported force_stat: {force_stat}")

    frame_label = "robot frame"
    if world_frame:
        actual_plot = to_world(actual)
        virtual_plot = to_world(virtual)
        if wrench_frame == "world":
            force_plot = force
        else:
            force_plot = to_world(force)
        frame_label = "world frame"
    else:
        actual_plot = actual
        virtual_plot = virtual
        force_plot = force

    delta = virtual_plot - actual_plot
    force_norm = np.linalg.norm(force_plot, axis=1)
    delta_norm = np.linalg.norm(delta, axis=1)
    if force_scale <= 0:
        median_delta = np.median(delta_norm[delta_norm > 1e-9]) if np.any(delta_norm > 1e-9) else 0.01
        median_force = np.median(force_norm[force_norm > 1e-9]) if np.any(force_norm > 1e-9) else 1.0
        force_scale = 0.8 * median_delta / median_force

    force_end = actual_plot + force_plot * force_scale
    centers, radius = axis_ranges(actual_plot, virtual_plot, force_end)
    delta_sizeref = max(radius * 0.055, 0.003)
    force_sizeref = max(radius * 0.055, 0.003)

    fig = go.Figure()
    add_points(fig, actual_plot, "actual target", "#2563eb", "circle", size=4)
    add_points(fig, virtual_plot, "virtual target", "#dc2626", "diamond", size=4)
    add_vector_lines(fig, actual_plot, delta, "delta actual -> virtual", "#16a34a", width=4, opacity=0.65)
    add_cones(fig, actual_plot, delta, "delta arrow head", "#16a34a", scale=1.0, sizeref=delta_sizeref, opacity=0.58)
    add_vector_lines(
        fig,
        actual_plot,
        force_plot,
        f"force vector from actual ({force_stat}, scale={force_scale:.5g})",
        "#7c3aed",
        width=4,
        opacity=0.75,
        scale=force_scale,
    )
    add_cones(
        fig,
        actual_plot,
        force_plot,
        "force arrow head",
        "#7c3aed",
        scale=force_scale,
        sizeref=force_sizeref,
        opacity=0.72,
    )

    fig.update_layout(
        title=(
            f"{demo}: actual/virtual, residual delta, and FT force vectors ({frame_label})<br>"
            f"dataset={Path(dataset).name}, steps={int(idx[0])}-{int(idx[-1])}, stride={stride}, wrench_frame={wrench_frame}"
        ),
        width=1200,
        height=900,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=0, r=0, t=85, b=0),
        scene=dict(
            xaxis=dict(title="x (m)", range=[centers[0] - radius, centers[0] + radius]),
            yaxis=dict(title="y (m)", range=[centers[1] - radius, centers[1] + radius]),
            zaxis=dict(title="z (m)", range=[centers[2] - radius, centers[2] + radius]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=-1.65, z=1.05)),
        ),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    print(output)


def main():
    parser = argparse.ArgumentParser(
        description="3D actual/virtual trajectory with delta arrows and FT force vectors from actual points."
    )
    parser.add_argument("--dataset", default="data/baetae/260618/slow_erase_board_virtual_m_world_wrench.hdf5")
    parser.add_argument("--demo", default=None)
    parser.add_argument("--output", default="plots/force_delta/world/05_3d_arrows/demo_0.html")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--actual-key", default="actual_target_abs")
    parser.add_argument("--virtual-key", default="virtual_target_abs")
    parser.add_argument("--wrench-key", default="wrench_wrist_R")
    parser.add_argument("--force-stat", choices=("last", "mean", "maxabs"), default="last")
    parser.add_argument(
        "--force-scale",
        type=float,
        default=0.0,
        help="Meters per Newton for display. <=0 chooses an automatic visual scale.",
    )
    parser.add_argument("--world-frame", action="store_true", default=True)
    parser.add_argument("--robot-frame", dest="world_frame", action="store_false")
    args = parser.parse_args()

    make_plot(
        dataset=args.dataset,
        demo=args.demo,
        output=args.output,
        start=args.start,
        count=args.count,
        stride=args.stride,
        world_frame=args.world_frame,
        actual_key=args.actual_key,
        virtual_key=args.virtual_key,
        wrench_key=args.wrench_key,
        force_scale=args.force_scale,
        force_stat=args.force_stat,
    )


if __name__ == "__main__":
    main()
