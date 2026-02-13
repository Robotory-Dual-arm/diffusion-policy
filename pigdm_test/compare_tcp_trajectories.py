#!/usr/bin/env python3
"""
두 개의 TCP trajectory HDF5 파일을 비교
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_trajectory(filepath):
    """
    HDF5 파일 로드
    
    Args:
        filepath: HDF5 파일 경로
        
    Returns:
        data: dict with trajectory data
    """
    print(f"\nLoading: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        tcp_L = np.array(f['tcp_L_world'])
        tcp_R = np.array(f['tcp_R_world'])
        timestamp = np.array(f['timestamp'])
        
        print(f"  Samples: {len(timestamp)}")
        print(f"  Duration: {timestamp[-1]:.2f}s")
    
    return {
        'tcp_L': tcp_L,
        'tcp_R': tcp_R,
        'timestamp': timestamp
    }


def plot_comparison(data1, data2, label1='Trajectory 1', label2='Trajectory 2'):
    """
    두 trajectory 비교 플롯 (2개의 창)
    Window 1: Position 비교 (6개 그래프 - 양팔 X,Y,Z)
    Window 2: 속도/가속도 비교 (4개 그래프 - 양팔 speed, acceleration)
    
    Args:
        data1, data2: load_trajectory의 출력
        label1, label2: 각 데이터의 라벨
    """
    # 속도 계산
    def compute_speed(positions, timestamps):
        """위치에서 속도 계산"""
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6  # 0으로 나누기 방지
        vel = np.diff(positions, axis=0) / dt[:, None]
        speed = np.linalg.norm(vel, axis=1)
        return speed
    
    # 가속도 계산
    def compute_acceleration(positions, timestamps):
        """위치에서 가속도 계산 (속도를 먼저 구한 후 미분)"""
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6
        vel = np.diff(positions, axis=0) / dt[:, None]
        
        # 속도를 한번 더 미분
        dt2 = np.diff(timestamps[:-1])
        dt2[dt2 == 0] = 1e-6
        acc = np.diff(vel, axis=0) / dt2[:, None]
        acc_magnitude = np.linalg.norm(acc, axis=1)
        return acc_magnitude
    
    speed_L1 = compute_speed(data1['tcp_L'], data1['timestamp'])
    speed_R1 = compute_speed(data1['tcp_R'], data1['timestamp'])
    speed_L2 = compute_speed(data2['tcp_L'], data2['timestamp'])
    speed_R2 = compute_speed(data2['tcp_R'], data2['timestamp'])
    
    acc_L1 = compute_acceleration(data1['tcp_L'], data1['timestamp'])
    acc_R1 = compute_acceleration(data1['tcp_R'], data1['timestamp'])
    acc_L2 = compute_acceleration(data2['tcp_L'], data2['timestamp'])
    acc_R2 = compute_acceleration(data2['tcp_R'], data2['timestamp'])
    
    # 시간축 (속도는 n-1, 가속도는 n-2)
    time1_speed = data1['timestamp'][:-1]
    time2_speed = data2['timestamp'][:-1]
    time1_acc = data1['timestamp'][:-2]
    time2_acc = data2['timestamp'][:-2]
    
    # ========== Window 1: 왼팔 Position (3개 그래프) ==========
    fig1, axes1 = plt.subplots(3, 1, figsize=(16, 12))
    fig1.suptitle('Left TCP Position Comparison', fontsize=14, fontweight='bold')
    
    ax_LX, ax_LY, ax_LZ = axes1
    
    # ===== 왼팔 Position =====
    ax_LX.plot(data1['timestamp'], data1['tcp_L'][:, 0], 'r-', linewidth=2, alpha=0.7)
    ax_LX.plot(data2['timestamp'], data2['tcp_L'][:, 0], 'b-', linewidth=2, alpha=0.7)
    ax_LX.set_ylabel('X (m)', fontsize=11)
    ax_LX.set_title('Left TCP - X Position', fontsize=12, fontweight='bold')
    ax_LX.grid(True, alpha=0.3)
    
    ax_LY.plot(data1['timestamp'], data1['tcp_L'][:, 1], 'r-', linewidth=2, alpha=0.7)
    ax_LY.plot(data2['timestamp'], data2['tcp_L'][:, 1], 'b-', linewidth=2, alpha=0.7)
    ax_LY.set_ylabel('Y (m)', fontsize=11)
    ax_LY.set_title('Left TCP - Y Position', fontsize=12, fontweight='bold')
    ax_LY.grid(True, alpha=0.3)
    
    ax_LZ.plot(data1['timestamp'], data1['tcp_L'][:, 2], 'r-', linewidth=2, alpha=0.7)
    ax_LZ.plot(data2['timestamp'], data2['tcp_L'][:, 2], 'b-', linewidth=2, alpha=0.7)
    ax_LZ.set_xlabel('Time (s)', fontsize=11)
    ax_LZ.set_ylabel('Z (m)', fontsize=11)
    ax_LZ.set_title('Left TCP - Z Position', fontsize=12, fontweight='bold')
    ax_LZ.grid(True, alpha=0.3)
    
    # 범례를 Figure 레벨에 추가 (한 번만)
    fig1.legend([label1, label2], loc='upper right', ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.98, 0.98))
    
    plt.figure(fig1.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 범례 공간 확보
    
    # ========== Window 2: 오른팔 Position (3개 그래프) ==========
    fig2, axes2 = plt.subplots(3, 1, figsize=(16, 12))
    fig2.suptitle('Right TCP Position Comparison', fontsize=14, fontweight='bold')
    
    ax_RX, ax_RY, ax_RZ = axes2
    
    # ===== 오른팔 Position =====
    ax_RX.plot(data1['timestamp'], data1['tcp_R'][:, 0], 'r-', linewidth=2, alpha=0.7)
    ax_RX.plot(data2['timestamp'], data2['tcp_R'][:, 0], 'b-', linewidth=2, alpha=0.7)
    ax_RX.set_ylabel('X (m)', fontsize=11)
    ax_RX.set_title('Right TCP - X Position', fontsize=12, fontweight='bold')
    ax_RX.grid(True, alpha=0.3)
    
    ax_RY.plot(data1['timestamp'], data1['tcp_R'][:, 1], 'r-', linewidth=2, alpha=0.7)
    ax_RY.plot(data2['timestamp'], data2['tcp_R'][:, 1], 'b-', linewidth=2, alpha=0.7)
    ax_RY.set_ylabel('Y (m)', fontsize=11)
    ax_RY.set_title('Right TCP - Y Position', fontsize=12, fontweight='bold')
    ax_RY.grid(True, alpha=0.3)
    
    ax_RZ.plot(data1['timestamp'], data1['tcp_R'][:, 2], 'r-', linewidth=2, alpha=0.7)
    ax_RZ.plot(data2['timestamp'], data2['tcp_R'][:, 2], 'b-', linewidth=2, alpha=0.7)
    ax_RZ.set_xlabel('Time (s)', fontsize=11)
    ax_RZ.set_ylabel('Z (m)', fontsize=11)
    ax_RZ.set_title('Right TCP - Z Position', fontsize=12, fontweight='bold')
    ax_RZ.grid(True, alpha=0.3)
    
    # 범례를 Figure 레벨에 추가 (한 번만)
    fig2.legend([label1, label2], loc='upper right', ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.98, 0.98))
    
    plt.figure(fig2.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 범례 공간 확보
    
    # ========== Window 3: 속도/가속도 비교 (4개 그래프) ==========
    fig3, axes3 = plt.subplots(4, 1, figsize=(16, 16))
    fig3.suptitle('TCP Velocity & Acceleration Comparison', fontsize=14, fontweight='bold')
    
    ax_Lspeed = axes3[0]
    ax_Rspeed = axes3[1]
    ax_Lacc = axes3[2]
    ax_Racc = axes3[3]
    
    # ===== 왼팔 속도 =====
    ax_Lspeed.plot(time1_speed, speed_L1, 'r-', linewidth=2, alpha=0.7)
    ax_Lspeed.plot(time2_speed, speed_L2, 'b-', linewidth=2, alpha=0.7)
    ax_Lspeed.set_ylabel('Speed (m/s)', fontsize=11)
    ax_Lspeed.set_title('Left TCP - Speed', fontsize=12, fontweight='bold')
    ax_Lspeed.grid(True, alpha=0.3)
    
    # ===== 오른팔 속도 =====
    ax_Rspeed.plot(time1_speed, speed_R1, 'r-', linewidth=2, alpha=0.7)
    ax_Rspeed.plot(time2_speed, speed_R2, 'b-', linewidth=2, alpha=0.7)
    ax_Rspeed.set_ylabel('Speed (m/s)', fontsize=11)
    ax_Rspeed.set_title('Right TCP - Speed', fontsize=12, fontweight='bold')
    ax_Rspeed.grid(True, alpha=0.3)
    
    # ===== 왼팔 가속도 =====
    ax_Lacc.plot(time1_acc, acc_L1, 'r-', linewidth=2, alpha=0.7)
    ax_Lacc.plot(time2_acc, acc_L2, 'b-', linewidth=2, alpha=0.7)
    ax_Lacc.set_xlabel('Time (s)', fontsize=11)
    ax_Lacc.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax_Lacc.set_title('Left TCP - Acceleration', fontsize=12, fontweight='bold')
    ax_Lacc.grid(True, alpha=0.3)
    
    # ===== 오른팔 가속도 =====
    ax_Racc.plot(time1_acc, acc_R1, 'r-', linewidth=2, alpha=0.7)
    ax_Racc.plot(time2_acc, acc_R2, 'b-', linewidth=2, alpha=0.7)
    ax_Racc.set_xlabel('Time (s)', fontsize=11)
    ax_Racc.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax_Racc.set_title('Right TCP - Acceleration', fontsize=12, fontweight='bold')
    ax_Racc.grid(True, alpha=0.3)
    
    # 범례를 Figure 레벨에 추가 (한 번만)
    fig3.legend([label1, label2], loc='upper right', ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.98, 0.98))
    
    plt.figure(fig3.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 범례 공간 확보
    
    plt.show()


def compute_errors(data1, data2):
    """
    두 trajectory 간 오차 계산
    """
    # 시간 정렬 (더 짧은 쪽에 맞춤)
    min_len = min(len(data1['timestamp']), len(data2['timestamp']))
    
    tcp_L1 = data1['tcp_L'][:min_len]
    tcp_L2 = data2['tcp_L'][:min_len]
    tcp_R1 = data1['tcp_R'][:min_len]
    tcp_R2 = data2['tcp_R'][:min_len]
    
    # 유클리드 거리 오차
    error_L = np.linalg.norm(tcp_L1 - tcp_L2, axis=1)
    error_R = np.linalg.norm(tcp_R1 - tcp_R2, axis=1)
    
    print("\n=== Trajectory Comparison Errors ===")
    print(f"Compared samples: {min_len}")
    print(f"\nLeft TCP:")
    print(f"  Mean error: {np.mean(error_L)*1000:.2f} mm")
    print(f"  Max error:  {np.max(error_L)*1000:.2f} mm")
    print(f"  Std error:  {np.std(error_L)*1000:.2f} mm")
    print(f"\nRight TCP:")
    print(f"  Mean error: {np.mean(error_R)*1000:.2f} mm")
    print(f"  Max error:  {np.max(error_R)*1000:.2f} mm")
    print(f"  Std error:  {np.std(error_R)*1000:.2f} mm")
    
    return error_L, error_R


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_tcp_trajectories.py <file1.hdf5> <file2.hdf5>")
        print("\nExample:")
        print("  python compare_tcp_trajectories.py traj1.hdf5 traj2.hdf5")
        return
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    if not os.path.exists(file1):
        print(f"Error: File not found: {file1}")
        return
    if not os.path.exists(file2):
        print(f"Error: File not found: {file2}")
        return
    
    # 데이터 로드
    data1 = load_trajectory(file1)
    data2 = load_trajectory(file2)
    
    # 오차 계산
    error_L, error_R = compute_errors(data1, data2)
    
    # 비교 플롯
    label1 = os.path.basename(file1).replace('world_position_', '').replace('.hdf5', '')
    label2 = os.path.basename(file2).replace('world_position_', '').replace('.hdf5', '')
    
    plot_comparison(data1, data2, label1, label2)


if __name__ == '__main__':
    main()
