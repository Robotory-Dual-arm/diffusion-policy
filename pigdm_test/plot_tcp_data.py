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


def plot_world_position(hdf5_file, camera_preset='default', custom_camera=None):
    """
    월드 좌표계 TCP position 데이터를 plotly로 시각화
    
    Args:
        hdf5_file: world_position_*.hdf5 파일 경로
        camera_preset: 카메라 프리셋 ('default', 'top', 'front', 'side', 'isometric')
        custom_camera: 커스텀 카메라 설정 dict (eye, center, up)
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
    
    # 카메라 프리셋 정의
    camera_presets = {
        'default': dict(
            eye=dict(x=2.0, y=0.25, z=0.37),
            center=dict(x=0.0, y=0.1, z=-0.03),
            up=dict(x=0, y=0, z=1)
        ),
        'top': dict(
            eye=dict(x=0, y=0, z=2.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
        'front': dict(
            eye=dict(x=2.5, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        'side': dict(
            eye=dict(x=0, y=2.5, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        'isometric': dict(
            eye=dict(x=1.7, y=1.7, z=1.7),
            center=dict(x=0, y=0, z=0.5),
            up=dict(x=0, y=0, z=1)
        ),
    }
    
    # 카메라 설정 선택
    if custom_camera is not None:
        camera = custom_camera
    else:
        camera = camera_presets.get(camera_preset, camera_presets['default'])
    
    print(f"\nUsing camera preset: {camera_preset}")
    print(f"  eye: {camera['eye']}")
    print(f"  center: {camera['center']}")
    print(f"  up: {camera['up']}")
    
    # 3D Trajectory Plot
    fig_3d = go.Figure()
    
    # Left arm trajectory
    fig_3d.add_trace(go.Scatter3d(
        x=tcp_L_world[:, 0],
        y=tcp_L_world[:, 1],
        z=tcp_L_world[:, 2],
        # mode='lines+markers',
        mode='markers',
        name='Left TCP',
        line=dict(color='blue', width=3),
        marker=dict(size=2, color='blue'),
    ))
    
    # Right arm trajectory
    fig_3d.add_trace(go.Scatter3d(
        x=tcp_R_world[:, 0],
        y=tcp_R_world[:, 1],
        z=tcp_R_world[:, 2],
        # mode='lines+markers',
        mode='markers',
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
        marker=dict(size=7, color='darkblue', symbol='diamond'),
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[tcp_R_world[0, 0]],
        y=[tcp_R_world[0, 1]],
        z=[tcp_R_world[0, 2]],
        mode='markers',
        name='Right Start',
        marker=dict(size=7, color='darkred', symbol='diamond'),
    ))
    
    # End points (larger markers)
    fig_3d.add_trace(go.Scatter3d(
        x=[tcp_L_world[-1, 0]],
        y=[tcp_L_world[-1, 1]],
        z=[tcp_L_world[-1, 2]],
        mode='markers',
        name='Left End',
        marker=dict(size=5, color='darkblue', symbol='x'),
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[tcp_R_world[-1, 0]],
        y=[tcp_R_world[-1, 1]],
        z=[tcp_R_world[-1, 2]],
        mode='markers',
        name='Right End',
        marker=dict(size=5, color='darkred', symbol='x'),
    ))
    
    # Robot base positions
    # fig_3d.add_trace(go.Scatter3d(
    #     x=[0], y=[0.30  ], z=[0],
    #     mode='markers',
    #     name='Left Base',
    #     marker=dict(size=15, color='lightblue', symbol='x'),
    # ))
    
    # fig_3d.add_trace(go.Scatter3d(
    #     x=[0], y=[-0.30], z=[0],
    #     mode='markers',
    #     name='Right Base',
    #     marker=dict(size=15, color='lightcoral', symbol='x'),
    # ))
    
    # 축 범위 계산 (궤적보다 10% 더 넓게) - 양손 모두 포함
    margin = 0.1
    x_min = min(tcp_L_world[:, 0].min(), tcp_R_world[:, 0].min())
    x_max = max(tcp_L_world[:, 0].max(), tcp_R_world[:, 0].max())
    y_min = min(tcp_L_world[:, 1].min(), tcp_R_world[:, 1].min())
    y_max = max(tcp_L_world[:, 1].max(), tcp_R_world[:, 1].max())
    z_min = min(tcp_L_world[:, 2].min(), tcp_R_world[:, 2].min())
    z_max = max(tcp_L_world[:, 2].max(), tcp_R_world[:, 2].max())
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    fig_3d.update_layout(
        title=f'TCP Trajectories in World Frame<br><sub>Duration: {duration:.2f}s, Samples: {num_samples}</sub>',
        scene=dict(
            xaxis=dict(
                title='X (m)',
                range=[0.4 , 0.7 ]
            ),
            yaxis=dict(
                title='Y (m)',
                range=[-0.3 , 0.5 ]
            ),
            zaxis=dict(
                title='Z (m)',
                range=[-0.4 , 0.0 ]
            ),
            aspectmode='data',
            camera=camera,  # 카메라 설정 추가
        ),
        width=1600,
        height=1000,
        autosize=True,
    )
    
    # Show 3D plot only
    print("\nShowing 3D trajectory plot...")
    fig_3d.show()
    
    # Statistics
    print("\n=== Statistics ===")
    print(f"Left TCP:")
    print(f"  X range: [{tcp_L_world[:, 0].min():.3f}, {tcp_L_world[:, 0].max():.3f}] m")
    print(f"  Y range: [{tcp_L_world[:, 1].min():.3f}, {tcp_L_world[:, 1].max():.3f}] m")
    print(f"  Z range: [{tcp_L_world[:, 2].min():.3f}, {tcp_L_world[:, 2].max():.3f}] m")
    
    print(f"\nRight TCP:")
    print(f"  X range: [{tcp_R_world[:, 0].min():.3f}, {tcp_R_world[:, 0].max():.3f}] m")
    print(f"  Y range: [{tcp_R_world[:, 1].min():.3f}, {tcp_R_world[:, 1].max():.3f}] m")
    print(f"  Z range: [{tcp_R_world[:, 2].min():.3f}, {tcp_R_world[:, 2].max():.3f}] m")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_tcp_data.py <hdf5_file> [camera_preset]")
        print("\nCamera presets:")
        print("  default    - 기본 시점 (x=1.5, y=1.5, z=1.5)")
        print("  top        - 위에서 내려다봄 (z=2.5)")
        print("  front      - 앞에서 봄 (x=2.5)")
        print("  side       - 옆에서 봄 (y=2.5)")
        print("  isometric  - 등각투상 시점")
        print("\nExample:")
        print("  python plot_tcp_data.py data/world_tcp.hdf5")
        print("  python plot_tcp_data.py data/world_tcp.hdf5 top")
        print("\nCustom camera (코드 수정 필요):")
        print("  custom_camera = dict(")
        print("      eye=dict(x=2.0, y=2.0, z=1.0),")
        print("      center=dict(x=0, y=0, z=0.5),")
        print("      up=dict(x=0, y=0, z=1)")
        print("  )")
        return
    
    hdf5_file = sys.argv[1]
    camera_preset = sys.argv[2] if len(sys.argv) > 2 else 'default'
    
    if not os.path.exists(hdf5_file):
        print(f"Error: File not found: {hdf5_file}")
        return
    
    # 커스텀 카메라 예시 (필요시 주석 해제하고 값 수정)
    # custom_camera = dict(
    #     eye=dict(x=2.0, y=2.0, z=1.0),      # 카메라 위치
    #     center=dict(x=0, y=0, z=0.5),       # 카메라가 보는 중심점
    #     up=dict(x=0, y=0, z=1)              # 카메라 위쪽 방향
    # )
    # plot_world_position(hdf5_file, custom_camera=custom_camera)
    
    plot_world_position(hdf5_file, camera_preset=camera_preset)


if __name__ == '__main__':
    main()