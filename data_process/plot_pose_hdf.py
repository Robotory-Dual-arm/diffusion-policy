import h5py
import numpy as np
import argparse
import plotly.graph_objects as go

sqrt2 = np.sqrt(2) / 2

# Left-hand rotation (robot frame → world)
R_L = np.array([
    [0,        sqrt2,  sqrt2],
    [1,        0,      0     ],
    [0,        sqrt2, -sqrt2]
])

# Right-hand rotation (robot frame → world)
R_R = np.array([
    [0,       -sqrt2, -sqrt2],
    [-1,       0,      0     ],
    [0,        sqrt2, -sqrt2]
])

def apply_rotation_only(pos_robot, R):
    # pos_robot: (T,3)
    return (R @ pos_robot.T).T

def plot_3d_trajectory_plotly(hdf_path, demo_name="demo_0"):
    with h5py.File(hdf_path, "r") as f:
        base = f[f"data/{demo_name}/obs"]

        tcp_L = base["robot_pose_L"][:, :3]
        tcp_R = base["robot_pose_R"][:, :3]

        world_L = apply_rotation_only(tcp_L, R_L)
        world_R = apply_rotation_only(tcp_R, R_R)

        world_L[:, 0] += 0.5

    xL, yL, zL = world_L[:, 0], world_L[:, 1], world_L[:, 2]
    xR, yR, zR = world_R[:, 0], world_R[:, 1], world_R[:, 2]

    # 왼손 궤적
    trace_L = go.Scatter3d(
        x=xL, y=yL, z=zL,
        mode='lines+markers',
        name='Left Hand',
        line=dict(width=4, color='blue'),
        marker=dict(size=3, color='blue')
    )

    # 오른손 궤적
    trace_R = go.Scatter3d(
        x=xR, y=yR, z=zR,
        mode='lines+markers',
        name='Right Hand',
        line=dict(width=4, color='red'),
        marker=dict(size=3, color='red')
    )

    # 시작점/끝점 별도 표시 (원하면)
    start_L = go.Scatter3d(
        x=[xL[0]], y=[yL[0]], z=[zL[0]],
        mode='markers',
        name='Left Start',
        marker=dict(size=6, color='green', symbol='diamond')
    )
    end_L = go.Scatter3d(
        x=[xL[-1]], y=[yL[-1]], z=[zL[-1]],
        mode='markers',
        name='Left End',
        marker=dict(size=6, color='black', symbol='x')
    )

    start_R = go.Scatter3d(
        x=[xR[0]], y=[yR[0]], z=[zR[0]],
        mode='markers',
        name='Right Start',
        marker=dict(size=6, color='green', symbol='diamond-open')
    )
    end_R = go.Scatter3d(
        x=[xR[-1]], y=[yR[-1]], z=[zR[-1]],
        mode='markers',
        name='Right End',
        marker=dict(size=6, color='black', symbol='x')
    )

    data = [trace_L, trace_R, start_L, end_L, start_R, end_R]

    # 축 스케일 맞추기
    x_all = np.concatenate([xL, xR])
    y_all = np.concatenate([yL, yR])
    z_all = np.concatenate([zL, zR])

    max_range = max(
        x_all.max() - x_all.min(),
        y_all.max() - y_all.min(),
        z_all.max() - z_all.min()
    )
    mid_x = (x_all.max() + x_all.min()) * 0.5
    mid_y = (y_all.max() + y_all.min()) * 0.5
    mid_z = (z_all.max() + z_all.min()) * 0.5

    axis_range = [
        mid_x - max_range/2, mid_x + max_range/2,
        mid_y - max_range/2, mid_y + max_range/2,
        mid_z - max_range/2, mid_z + max_range/2,
    ]

    layout = go.Layout(
        title=f"World Frame 3D Trajectory — {hdf_path} - {demo_name}",
        scene=dict(
            xaxis=dict(title="X", range=axis_range[0:2]),
            yaxis=dict(title="Y", range=axis_range[2:4]),
            zaxis=dict(title="Z", range=axis_range[4:6]),
            aspectmode='cube'
        ),
        legend=dict(x=0, y=1)
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()  # 브라우저에서 인터랙티브 뷰어 오픈


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf_path", type=str)
    parser.add_argument("--demo", type=str, default="demo_0")
    args = parser.parse_args()

    plot_3d_trajectory_plotly(args.hdf_path, args.demo)
