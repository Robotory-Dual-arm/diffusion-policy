#!/usr/bin/env python3
"""
두 개의 trajectory HDF5 파일에서 손가락 joint 비교
thumb joint 1,2,3 / index joint 2,3 / middle joint 2,3
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_hand_joints(filepath):
    """
    HDF5 파일에서 손가락 joint 데이터 로드
    
    Args:
        filepath: HDF5 파일 경로
        
    Returns:
        data: dict with hand joint data
    """
    print(f"\nLoading: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        if 'hand_L' not in f.keys() or 'hand_R' not in f.keys():
            print(f"  ❌ Error: hand data not found in file")
            return None
        
        hand_L = np.array(f['hand_L'])  # (N, 15) - hand joints only
        hand_R = np.array(f['hand_R'])  # (N, 15)
        timestamp = np.array(f['timestamp'])
        
        print(f"  Samples: {len(timestamp)}")
        print(f"  Duration: {timestamp[-1]:.2f}s")
        print(f"  Hand joint shape: {hand_L.shape}")
    
    # Hand joints 인덱스 (15개 = 5개 손가락 × 3개 joint)
    # thumb: joint 1,2,3 → 인덱스 0,1,2
    # index: joint 1,2,3 → 인덱스 3,4,5
    # middle: joint 1,2,3 → 인덱스 6,7,8
    # ring: joint 1,2,3 → 인덱스 9,10,11
    # pinky: joint 1,2,3 → 인덱스 12,13,14
    
    return {
        'hand_L': hand_L,
        'hand_R': hand_R,
        'timestamp': timestamp,
        # 왼손
        'thumb_L_j1': hand_L[:, 0],   # thumb joint1
        'thumb_L_j2': hand_L[:, 1],   # thumb joint2
        'thumb_L_j3': hand_L[:, 2],   # thumb joint3
        'index_L_j2': hand_L[:, 4],   # index joint2
        'index_L_j3': hand_L[:, 5],   # index joint3
        'middle_L_j2': hand_L[:, 7],  # middle joint2
        'middle_L_j3': hand_L[:, 8],  # middle joint3
        # 오른손
        'thumb_R_j1': hand_R[:, 0],
        'thumb_R_j2': hand_R[:, 1],
        'thumb_R_j3': hand_R[:, 2],
        'index_R_j2': hand_R[:, 4],
        'index_R_j3': hand_R[:, 5],
        'middle_R_j2': hand_R[:, 7],
        'middle_R_j3': hand_R[:, 8],
    }


def plot_hand_comparison(data1, data2, label1='Trajectory 1', label2='Trajectory 2'):
    """
    손가락 joint 비교 플롯
    
    Args:
        data1, data2: load_hand_joints의 출력
        label1, label2: 각 데이터의 라벨
    """
    # ========== Window 1: 왼손 손가락 joints (7개 그래프) ==========
    fig1, axes1 = plt.subplots(7, 1, figsize=(12, 18))
    fig1.suptitle('Left Hand Joint Comparison', fontsize=14, fontweight='bold')
    
    joints_left = [
        ('thumb_L_j1', 'Thumb Joint 1'),
        ('thumb_L_j2', 'Thumb Joint 2'),
        ('thumb_L_j3', 'Thumb Joint 3'),
        ('index_L_j2', 'Index Joint 2'),
        ('index_L_j3', 'Index Joint 3'),
        ('middle_L_j2', 'Middle Joint 2'),
        ('middle_L_j3', 'Middle Joint 3'),
    ]
    
    for i, (key, title) in enumerate(joints_left):
        ax = axes1[i]
        ax.plot(data1['timestamp'], data1[key], 'r-', linewidth=2, alpha=0.7)
        ax.plot(data2['timestamp'], data2[key], 'b-', linewidth=2, alpha=0.7)
        ax.set_ylabel('Angle (rad)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == len(joints_left) - 1:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    fig1.legend([label1, label2], loc='upper right', ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.98, 0.99))
    plt.figure(fig1.number)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # ========== Window 2: 오른손 손가락 joints (7개 그래프) ==========
    fig2, axes2 = plt.subplots(7, 1, figsize=(12, 18))
    fig2.suptitle('Right Hand Joint Comparison', fontsize=14, fontweight='bold')
    
    joints_right = [
        ('thumb_R_j1', 'Thumb Joint 1'),
        ('thumb_R_j2', 'Thumb Joint 2'),
        ('thumb_R_j3', 'Thumb Joint 3'),
        ('index_R_j2', 'Index Joint 2'),
        ('index_R_j3', 'Index Joint 3'),
        ('middle_R_j2', 'Middle Joint 2'),
        ('middle_R_j3', 'Middle Joint 3'),
    ]
    
    for i, (key, title) in enumerate(joints_right):
        ax = axes2[i]
        ax.plot(data1['timestamp'], data1[key], 'r-', linewidth=2, alpha=0.7)
        ax.plot(data2['timestamp'], data2[key], 'b-', linewidth=2, alpha=0.7)
        ax.set_ylabel('Angle (rad)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == len(joints_right) - 1:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    fig2.legend([label1, label2], loc='upper right', ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.98, 0.99))
    plt.figure(fig2.number)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.show()


def compute_joint_errors(data1, data2):
    """
    손가락 joint 오차 계산
    """
    min_len = min(len(data1['timestamp']), len(data2['timestamp']))
    
    joint_keys = [
        'thumb_L_j1', 'thumb_L_j2', 'thumb_L_j3',
        'index_L_j2', 'index_L_j3',
        'middle_L_j2', 'middle_L_j3',
        'thumb_R_j1', 'thumb_R_j2', 'thumb_R_j3',
        'index_R_j2', 'index_R_j3',
        'middle_R_j2', 'middle_R_j3',
    ]
    
    print("\n=== Hand Joint Comparison Errors ===")
    print(f"Compared samples: {min_len}\n")
    
    for key in joint_keys:
        j1 = data1[key][:min_len]
        j2 = data2[key][:min_len]
        error = np.abs(j1 - j2)
        
        print(f"{key}:")
        print(f"  Mean error: {np.mean(error):.4f} rad ({np.rad2deg(np.mean(error)):.2f}°)")
        print(f"  Max error:  {np.max(error):.4f} rad ({np.rad2deg(np.max(error)):.2f}°)")


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_hand_joints.py <file1.hdf5> <file2.hdf5>")
        print("\nExample:")
        print("  python compare_hand_joints.py traj1.hdf5 traj2.hdf5")
        print("\nCompares:")
        print("  - Left/Right Thumb: joint 1, 2, 3")
        print("  - Left/Right Index: joint 2, 3")
        print("  - Left/Right Middle: joint 2, 3")
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
    data1 = load_hand_joints(file1)
    data2 = load_hand_joints(file2)
    
    if data1 is None or data2 is None:
        return
    
    # 오차 계산
    compute_joint_errors(data1, data2)
    
    # 비교 플롯
    label1 = os.path.basename(file1).replace('world_position_', '').replace('.hdf5', '')
    label2 = os.path.basename(file2).replace('world_position_', '').replace('.hdf5', '')
    
    plot_hand_comparison(data1, data2, label1, label2)


if __name__ == '__main__':
    main()
