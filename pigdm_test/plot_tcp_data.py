#!/usr/bin/env python3
"""
월드 좌표계 TCP Position 데이터를 Plotly로 시각화
"""
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os


def plot_world_position(hdf5_file):
    """
    월드 좌표계 TCP position 데이터를 plotly로 시각화
    
    Args:
        hdf5_file: world_position_*.hdf5 파일 경로
    """
    # 데이터 로드
    print(f"Loading: {hdf5_file}")
    with h5py.File(hdf5_file, 'r') as f:
        tcp_L_world = np.array(f['tcp_L_world'])  # (N, 3)
        tcp_R_world = np.array(f['tcp_R_world'])  # (N, 3)
        timestamps = np.array(f['timestamp'])     # (N,)
        
        duration = f.attrs['duration']
        num_samples = f.attrs['num_samples']
        sample_rate = f.attrs['sample_rate']
    
    print(f"Loaded {num_samples} samples")
    print(f"Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")
    
    # 3D Trajectory Plot
    fig_3d = go.Figure()
    
    # Left arm trajectory
    fig_3d.add_trace(go.Scatter3d(
        x=tcp_L_world[:, 0],
        y=tcp_L_world[:, 1],
        z=tcp_L_world[:, 2],
        mode='lines+markers',
        name='Left TCP',
        line=dict(color='blue', width=3),
        marker=dict(size=2, color='blue'),
    ))
    
    # Right arm trajectory
    fig_3d.add_trace(go.Scatter3d(
        x=tcp_R_world[:, 0],
        y=tcp_R_world[:, 1],
        z=tcp_R_world[:, 2],
        mode='lines+markers',
        name='Right TCP',
        line=dict(color='red', width=3),
        marker=dict(size=2, color='red'),
    ))
    
    # Start points (larger markers)
    fig_3d.add_trace(go.Scatter3d(
        x=[tcp_L_world[0, 0]],
        y=[tcp_L_world[0, 1]],
        z=[tcp_L_world[0, 2]],
        mode='markers',
        name='Left Start',
        marker=dict(size=10, color='darkblue', symbol='diamond'),
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[tcp_R_world[0, 0]],
        y=[tcp_R_world[0, 1]],
        z=[tcp_R_world[0, 2]],
        mode='markers',
        name='Right Start',
        marker=dict(size=10, color='darkred', symbol='diamond'),
    ))
    
    # Robot base positions
    fig_3d.add_trace(go.Scatter3d(
        x=[0], y=[0.5], z=[0],
        mode='markers',
        name='Left Base',
        marker=dict(size=15, color='lightblue', symbol='x'),
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[0], y=[-0.5], z=[0],
        mode='markers',
        name='Right Base',
        marker=dict(size=15, color='lightcoral', symbol='x'),
    ))
    
    fig_3d.update_layout(
        title=f'TCP Trajectories in World Frame<br><sub>Duration: {duration:.2f}s, Samples: {num_samples}</sub>',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        width=1200,
        height=800,
    )
    
    # Position vs Time plots
    fig_time = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Left X', 'Right X', 'Left Y', 'Right Y', 'Left Z', 'Right Z'),
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )
    
    # X positions
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=tcp_L_world[:, 0], name='Left X', 
                   line=dict(color='blue')),
        row=1, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=tcp_R_world[:, 0], name='Right X',
                   line=dict(color='red')),
        row=1, col=2
    )
    
    # Y positions
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=tcp_L_world[:, 1], name='Left Y',
                   line=dict(color='blue')),
        row=2, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=tcp_R_world[:, 1], name='Right Y',
                   line=dict(color='red')),
        row=2, col=2
    )
    
    # Z positions
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=tcp_L_world[:, 2], name='Left Z',
                   line=dict(color='blue')),
        row=3, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=tcp_R_world[:, 2], name='Right Z',
                   line=dict(color='red')),
        row=3, col=2
    )
    
    # Update axes
    for i in range(1, 4):
        fig_time.update_xaxes(title_text='Time (s)', row=i, col=1)
        fig_time.update_xaxes(title_text='Time (s)', row=i, col=2)
        fig_time.update_yaxes(title_text='Position (m)', row=i, col=1)
        fig_time.update_yaxes(title_text='Position (m)', row=i, col=2)
    
    fig_time.update_layout(
        title=f'TCP Positions vs Time<br><sub>Sample rate: {sample_rate}Hz</sub>',
        height=900,
        showlegend=False,
    )
    
    # Velocity calculation and plot
    dt = np.diff(timestamps)
    vel_L = np.diff(tcp_L_world, axis=0) / dt[:, np.newaxis]  # (N-1, 3)
    vel_R = np.diff(tcp_R_world, axis=0) / dt[:, np.newaxis]
    
    speed_L = np.linalg.norm(vel_L, axis=1)  # (N-1,)
    speed_R = np.linalg.norm(vel_R, axis=1)
    
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(
        x=timestamps[1:], y=speed_L,
        name='Left Speed', line=dict(color='blue', width=2)
    ))
    fig_vel.add_trace(go.Scatter(
        x=timestamps[1:], y=speed_R,
        name='Right Speed', line=dict(color='red', width=2)
    ))
    
    fig_vel.update_layout(
        title='TCP Speed over Time',
        xaxis_title='Time (s)',
        yaxis_title='Speed (m/s)',
        height=400,
    )
    
    # Show all plots
    print("\nShowing plots...")
    fig_3d.show()
    fig_time.show()
    fig_vel.show()
    
    # Statistics
    print("\n=== Statistics ===")
    print(f"Left TCP:")
    print(f"  X range: [{tcp_L_world[:, 0].min():.3f}, {tcp_L_world[:, 0].max():.3f}] m")
    print(f"  Y range: [{tcp_L_world[:, 1].min():.3f}, {tcp_L_world[:, 1].max():.3f}] m")
    print(f"  Z range: [{tcp_L_world[:, 2].min():.3f}, {tcp_L_world[:, 2].max():.3f}] m")
    print(f"  Max speed: {speed_L.max():.3f} m/s")
    print(f"  Avg speed: {speed_L.mean():.3f} m/s")
    
    print(f"\nRight TCP:")
    print(f"  X range: [{tcp_R_world[:, 0].min():.3f}, {tcp_R_world[:, 0].max():.3f}] m")
    print(f"  Y range: [{tcp_R_world[:, 1].min():.3f}, {tcp_R_world[:, 1].max():.3f}] m")
    print(f"  Z range: [{tcp_R_world[:, 2].min():.3f}, {tcp_R_world[:, 2].max():.3f}] m")
    print(f"  Max speed: {speed_R.max():.3f} m/s")
    print(f"  Avg speed: {speed_R.mean():.3f} m/s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_world_position.py <world_position_hdf5_file>")
        print("\nExample:")
        print("  python plot_world_position.py data/joint_recordings/world_position_20260213_120000.hdf5")
        return
    
    hdf5_file = sys.argv[1]
    
    if not os.path.exists(hdf5_file):
        print(f"Error: File not found: {hdf5_file}")
        return
    
    plot_world_position(hdf5_file)


if __name__ == '__main__':
    main()