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

# 왼팔 베이스: x축=-X_w, y축=Y_w+Z_w, z축=Y_w-Z_w (정규화 필요)
R_L = np.array([
    [-1.0,     0.0,      0.0    ],   # 왼팔 x → 월드 -x
    [0.0,      sqrt2,    sqrt2  ],   # 왼팔 y → 월드 (y+z)/√2
    [0.0,      sqrt2,   -sqrt2  ],   # 왼팔 z → 월드 (y-z)/√2
])

# 오른팔 베이스: x축=+X_w, y축=Z_w-Y_w, z축=-Y_w-Z_w (정규화 필요)
# 열벡터가 각 축 방향: [오른팔x의 월드표현 | 오른팔y의 월드표현 | 오른팔z의 월드표현]
R_R = np.array([
    [1.0,      0.0,      0.0    ],   # 오른팔 x → 월드 +x (1, 0, 0)
    [0.0,     -sqrt2,   -sqrt2  ],   # 오른팔 y → 월드 (z-y)/√2 = (0, -y, +z)/√2
    [0.0,      sqrt2,   -sqrt2  ],   # 오른팔 z → 월드 (-y-z)/√2 = (0, -y, -z)/√2
])

# Translation: 실제 베이스 간격 0.6m (각각 ±0.3m)
T_L = np.array([0.0, 0.15, 0.0])   # 월드 좌표계에서 왼팔 베이스 위치
T_R = np.array([0.0, -0.15, 0.0])  # 월드 좌표계에서 오른팔 베이스 위치


def transform_left_to_world(p_L):
    """왼팔 베이스 좌표 → 월드 좌표 (회전 + translation)"""
    p_L = np.asarray(p_L).reshape(3,)
    return R_L @ p_L + T_L


def transform_right_to_world(p_R):
    """오른팔 베이스 좌표 → 월드 좌표 (회전 + translation)"""
    p_R = np.asarray(p_R).reshape(3,)
    return R_R @ p_R + T_R


def find_movement_start_from_tcp(tcp_L, threshold=0.005):
    """
    왼손 TCP가 첫 위치에서 threshold 이상 이동한 시점 찾기
    
    Args:
        tcp_L: (N, 3) 왼손 TCP 위치 배열
        threshold: 움직임 감지 임계값 (m, 유클리드 거리)
        
    Returns:
        start_idx: 움직임 시작 인덱스
    """
    if len(tcp_L) == 0:
        return 0
    
    # 첫 번째 위치를 기준으로
    initial_pos = tcp_L[0]
    
    # 각 시점에서 첫 위치로부터의 거리 계산
    distances = np.linalg.norm(tcp_L - initial_pos, axis=1)
    
    # threshold 이상인 첫 시점 찾기
    moving_indices = np.where(distances > threshold)[0]
    
    if len(moving_indices) > 0:
        return moving_indices[0]
    else:
        return 0


def convert_joint_to_world_position(input_file, output_file=None, movement_threshold=0.01):
    """
    Joint 데이터를 월드 좌표계 TCP position으로 변환
    
    Args:
        input_file: joint_record_*.hdf5 파일 경로
        output_file: 출력 파일 경로 (None이면 자동 생성)
    """
    # URDF 로봇 모델 로드
    # URDF 경로 (현재 작업 디렉토리 기준 상대 경로)
    urdf_path = "../m0609.white.urdf"
    # 절대 경로로 변환하여 roboticstoolbox에 전달
    urdf_path_abs = os.path.abspath(urdf_path)
    
    if not os.path.exists(urdf_path_abs):
        print(f"Error: URDF file not found at {urdf_path_abs}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please check the file exists at ../m0609.white.urdf")
        return
    
    print(f"Loading URDF from: {urdf_path}")
    robot = rtb.ERobot.URDF(urdf_path_abs)
    
    # 디버깅: zero position에서 TCP 확인
    print("\n=== DEBUG: Zero position TCP ===")
    zero_joints = np.zeros(6)
    T_zero = robot.fkine(zero_joints)
    pos_zero = np.array(T_zero.t).flatten()
    print(f"TCP at zero joints (base frame): {pos_zero}")
    print(f"Expected: Both arms should have similar TCP position in their base frames")
    print("================================\n")
    
    # 입력 파일 읽기
    print(f"Loading: {input_file}")
    with h5py.File(input_file, 'r') as f:
        joint_L = np.array(f['joint_L'])  # (N, 6) - arm joints only
        joint_R = np.array(f['joint_R'])  # (N, 6)
        hand_L = np.array(f['hand_L'])    # (N, 15) - hand joints (5 fingers × 3)
        hand_R = np.array(f['hand_R'])    # (N, 15)
        timestamps = np.array(f['timestamp'])  # (N,)
        
        duration = f.attrs['duration']
        num_samples = f.attrs['num_samples']
        sample_rate = f.attrs['sample_rate']
    
    print(f"Loaded {num_samples} samples, Duration: {duration:.1f}s, Rate: {sample_rate}Hz")
    print(f"Arm joint shape: {joint_L.shape}")
    print(f"Hand joint shape: {hand_L.shape}")
    
    # Arm joints만 사용 (TCP 계산용)
    arm_L = joint_L
    arm_R = joint_R
    
    # Joint → TCP position 변환 (먼저 전체 변환)
    print("\nConverting joints to TCP positions...")
    tcp_L_world = []
    tcp_R_world = []
    
    for i in range(len(arm_L)):
        # Left arm: joint → TCP (베이스 좌표계) → 월드 좌표계
        T_L_base = robot.fkine(arm_L[i])
        pos_L_base = np.array(T_L_base.t).flatten()  # (3,)
        pos_L_world = transform_left_to_world(pos_L_base)
        tcp_L_world.append(pos_L_world)
        
        # Right arm: joint → TCP (베이스 좌표계) → 월드 좌표계
        T_R_base = robot.fkine(arm_R[i])
        pos_R_base = np.array(T_R_base.t).flatten()  # (3,)
        pos_R_world = transform_right_to_world(pos_R_base)
        tcp_R_world.append(pos_R_world)
    
    tcp_L_world = np.array(tcp_L_world)  # (N, 3)
    tcp_R_world = np.array(tcp_R_world)  # (N, 3)
    
    # 움직임 시작 지점 찾기 (왼손 TCP 기준)
    print(f"\nDetecting movement start from Left TCP (threshold: {movement_threshold} m)...")
    start_idx = find_movement_start_from_tcp(tcp_L_world, threshold=movement_threshold)
    print(f"Movement starts at index {start_idx} (t={timestamps[start_idx]:.2f}s)")
    print(f"Initial position: {tcp_L_world[0]}")
    print(f"Movement start position: {tcp_L_world[start_idx]}")
    print(f"Distance: {np.linalg.norm(tcp_L_world[start_idx] - tcp_L_world[0]):.4f} m")
    
    # 움직임 시작점부터 데이터 추출
    tcp_L_world = tcp_L_world[start_idx:]
    tcp_R_world = tcp_R_world[start_idx:]
    timestamps = timestamps[start_idx:] - timestamps[start_idx]  # 0초부터 시작
    arm_L = arm_L[start_idx:]  # Arm joint 데이터도 같이 자르기
    arm_R = arm_R[start_idx:]
    hand_L = hand_L[start_idx:]  # Hand joint 데이터도 같이 자르기
    hand_R = hand_R[start_idx:]
    
    print(f"After trimming: {len(timestamps)} samples, Duration: {timestamps[-1]:.1f}s")
    
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
        
        # Arm joint 데이터 저장 (6개)
        f.create_dataset('joint_L', data=arm_L)  # (N, 6) - arm only
        f.create_dataset('joint_R', data=arm_R)  # (N, 6) - arm only
        
        # Hand joint 데이터 저장 (15개 = 5 fingers × 3 joints)
        f.create_dataset('hand_L', data=hand_L)  # (N, 15) - hand only
        f.create_dataset('hand_R', data=hand_R)  # (N, 15) - hand only
        
        # Metadata
        f.attrs['duration'] = timestamps[-1]
        f.attrs['num_samples'] = len(timestamps)
        f.attrs['sample_rate'] = sample_rate
        f.attrs['coordinate_frame'] = 'Both tcp_L_world and tcp_R_world are in world frame'
        f.attrs['left_arm_base'] = 'World +Y 0.3m'
        f.attrs['right_arm_base'] = 'World -Y 0.3m'
        f.attrs['movement_start_trimmed'] = 'Data starts from movement detection'
    
    print(f"\nConversion complete!")
    print(f"  Left TCP (world frame):   shape {tcp_L_world.shape}")
    print(f"  Right TCP (world frame):  shape {tcp_R_world.shape}")
    print(f"  Arm joints:               shape {arm_L.shape}")
    print(f"  Hand joints:              shape {hand_L.shape}")
    print(f"  Timestamps:               shape {timestamps.shape}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_joint_to_world_position.py <input_hdf5_file> [output_file] [movement_threshold]")
        print("\nExample:")
        print("  python convert_joint_to_world_position.py data/joint_recordings/joint_record_20260213_120000.hdf5")
        print("  python convert_joint_to_world_position.py data/joint_recordings/joint_record_20260213_120000.hdf5 output.hdf5")
        print("  python convert_joint_to_world_position.py data/joint_recordings/joint_record_20260213_120000.hdf5 output.hdf5 0.01")
        print("\nArguments:")
        print("  movement_threshold: 움직임 감지 임계값 (default: 0.005 m, 왼손 TCP 기준)")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    movement_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0005
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    convert_joint_to_world_position(input_file, output_file, movement_threshold)


if __name__ == '__main__':
    main()
