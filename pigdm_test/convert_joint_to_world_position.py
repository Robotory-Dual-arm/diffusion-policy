#!/usr/bin/env python3
"""
Joint 데이터를 월드 좌표계 TCP Position으로 변환
"""
import h5py
import numpy as np
import roboticstoolbox as rtb
import os
import sys


# 좌표계 변환 정의
sqrt2 = np.sqrt(2) / 2

# 왼팔: x축=월드 -X, y축=월드 Y+Z(45°), z축=월드 Y-Z(45°)
R_L = np.array([
    [0.0,      1.0,      0.0    ],   # x_W
    [sqrt2,    0.0,      sqrt2  ],   # y_W
    [sqrt2,    0.0,     -sqrt2  ],   # z_W
])

# 오른팔: x축=월드 +X, y축=월드 Z-Y(45°), z축=월드 -Z-Y(45°)
R_R = np.array([
    [0.0,      -1.0,      0.0    ],   # x_W
    [-sqrt2,    0.0,      sqrt2  ],   # y_W
    [-sqrt2,    0.0,     -sqrt2  ],   # z_W
])

# Translation: 왼팔 베이스는 월드 +Y 0.5m, 오른팔은 월드 -Y 0.5m
T_L = np.array([0.35, 0.0, 0.0])  # 월드 좌표계에서 왼팔 베이스 위치
T_R = np.array([-0.35, -0.0, 0.0]) # 월드 좌표계에서 오른팔 베이스 위치


def transform_left_to_world(p_L):
    """왼팔 베이스 좌표 → 월드 좌표 (회전 + translation)"""
    p_L = np.asarray(p_L).reshape(3,)
    return R_L @ p_L + T_L


def transform_right_to_world(p_R):
    """오른팔 베이스 좌표 → 월드 좌표 (회전 + translation)"""
    p_R = np.asarray(p_R).reshape(3,)
    return R_R @ p_R + T_R


def convert_joint_to_world_position(input_file, output_file=None):
    """
    Joint 데이터를 월드 좌표계 TCP position으로 변환
    
    Args:
        input_file: joint_record_*.hdf5 파일 경로
        output_file: 출력 파일 경로 (None이면 자동 생성)
    """
    # URDF 로봇 모델 로드
    urdf_path = "/home/vision/dualarm_ws/src/doosan-robot2/dsr_description2/urdf/m0609.white.urdf"
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        print("Please update the urdf_path in the script")
        return
    
    robot = rtb.ERobot.URDF(urdf_path)
    
    # 입력 파일 읽기
    print(f"Loading: {input_file}")
    with h5py.File(input_file, 'r') as f:
        joint_L = np.array(f['joint_L'])  # (N, 6)
        joint_R = np.array(f['joint_R'])  # (N, 6)
        timestamps = np.array(f['timestamp'])  # (N,)
        
        duration = f.attrs['duration']
        num_samples = f.attrs['num_samples']
        sample_rate = f.attrs['sample_rate']
    
    print(f"Loaded {num_samples} samples, Duration: {duration:.1f}s, Rate: {sample_rate}Hz")
    
    # Joint → TCP position 변환
    print("Converting joints to TCP positions...")
    tcp_L_world = []
    tcp_R_world = []
    
    for i in range(len(joint_L)):
        # Left arm: joint → TCP (베이스 좌표계) → 월드 좌표계
        T_L_base = robot.fkine(joint_L[i])
        pos_L_base = np.array(T_L_base.t).flatten()  # (3,)
        pos_L_world = transform_left_to_world(pos_L_base)
        tcp_L_world.append(pos_L_world)
        
        # Right arm: joint → TCP (베이스 좌표계) → 월드 좌표계
        T_R_base = robot.fkine(joint_R[i])
        pos_R_base = np.array(T_R_base.t).flatten()  # (3,)
        pos_R_world = transform_right_to_world(pos_R_base)
        tcp_R_world.append(pos_R_world)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(joint_L)} processed")
    
    tcp_L_world = np.array(tcp_L_world)  # (N, 3)
    tcp_R_world = np.array(tcp_R_world)  # (N, 3)
    
    # 출력 파일명 생성
    if output_file is None:
        base_name = os.path.basename(input_file).replace('joint_record_', 'world_position_')
        output_file = os.path.join(os.path.dirname(input_file), base_name)
    
    # 저장
    print(f"Saving: {output_file}")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('tcp_L_world', data=tcp_L_world)  # (N, 3) - 월드 좌표계
        f.create_dataset('tcp_R_world', data=tcp_R_world)  # (N, 3) - 월드 좌표계
        f.create_dataset('timestamp', data=timestamps)
        
        # 원본 joint 데이터도 저장
        f.create_dataset('joint_L', data=joint_L)
        f.create_dataset('joint_R', data=joint_R)
        
        # Metadata
        f.attrs['duration'] = duration
        f.attrs['num_samples'] = num_samples
        f.attrs['sample_rate'] = sample_rate
        f.attrs['coordinate_frame'] = 'Both tcp_L_world and tcp_R_world are in world frame'
        f.attrs['left_arm_base'] = 'World +Y 0.5m'
        f.attrs['right_arm_base'] = 'World -Y 0.5m'
    
    print(f"\nConversion complete!")
    print(f"  Left TCP (world frame):   shape {tcp_L_world.shape}")
    print(f"  Right TCP (world frame):  shape {tcp_R_world.shape}")
    print(f"  Timestamps:               shape {timestamps.shape}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_joint_to_world_position.py <input_hdf5_file> [output_file]")
        print("\nExample:")
        print("  python convert_joint_to_world_position.py data/joint_recordings/joint_record_20260213_120000.hdf5")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    convert_joint_to_world_position(input_file, output_file)


if __name__ == '__main__':
    main()
