#!/usr/bin/env python3
"""
TCP trajectory HDF5 파일에서 시간 구간을 지정하여 자르기
"""
import h5py
import numpy as np
import sys
import os


def trim_trajectory(input_file, output_file, t1, t2):
    """
    HDF5 trajectory 파일에서 시간 구간 [t1, t2]만 추출
    
    Args:
        input_file: 입력 HDF5 파일 경로
        output_file: 출력 HDF5 파일 경로
        t1: 시작 시간 (초)
        t2: 종료 시간 (초)
    """
    print(f"\n=== Trimming Trajectory ===")
    print(f"Input: {input_file}")
    print(f"Time range: [{t1:.3f}s, {t2:.3f}s]")
    
    # 입력 파일 로드 - 모든 키 읽기
    data_dict = {}
    attrs_dict = {}
    with h5py.File(input_file, 'r') as f:
        print(f"\nDatasets in file:")
        
        # timestamp 먼저 로드 (필수)
        if 'timestamp' not in f.keys():
            print(f"❌ Error: 'timestamp' dataset not found in file")
            return
        
        timestamp = np.array(f['timestamp'])
        n_samples = len(timestamp)
        
        # 모든 데이터셋 로드 및 shape 확인
        for key in f.keys():
            data = np.array(f[key])
            data_dict[key] = data
            print(f"  {key}: shape={data.shape}")
        
        # Attributes 로드
        print(f"\nAttributes in file:")
        for key in f.attrs.keys():
            attrs_dict[key] = f.attrs[key]
            print(f"  {key}: {f.attrs[key]}")
    
    print(f"\nOriginal data:")
    print(f"  Total samples: {n_samples}")
    print(f"  Time range: [{timestamp[0]:.3f}s, {timestamp[-1]:.3f}s]")
    print(f"  Duration: {timestamp[-1] - timestamp[0]:.3f}s")
    
    # 시간 범위 검증
    if t1 < timestamp[0] or t2 > timestamp[-1]:
        print(f"\n⚠️  Warning: Requested time range [{t1:.3f}s, {t2:.3f}s] exceeds data range [{timestamp[0]:.3f}s, {timestamp[-1]:.3f}s]")
        t1 = max(t1, timestamp[0])
        t2 = min(t2, timestamp[-1])
        print(f"    Adjusted to [{t1:.3f}s, {t2:.3f}s]")
    
    if t1 >= t2:
        print(f"\n❌ Error: t1 ({t1:.3f}s) must be less than t2 ({t2:.3f}s)")
        return
    
    # 시간 구간에 해당하는 인덱스 찾기
    mask = (timestamp >= t1) & (timestamp <= t2)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        print(f"\n❌ Error: No data found in time range [{t1:.3f}s, {t2:.3f}s]")
        return
    
    # 모든 데이터 자르기
    trimmed_dict = {}
    for key, data in data_dict.items():
        # timestamp와 길이가 같은 데이터만 마스크 적용
        if len(data) == n_samples:
            if key == 'timestamp':
                # 시간은 0부터 시작하도록 재조정
                trimmed_data = data[mask]
                trimmed_dict[key] = trimmed_data - trimmed_data[0]
            else:
                # 나머지 데이터는 마스크 적용
                trimmed_dict[key] = data[mask]
        else:
            # 길이가 다른 데이터는 그대로 복사 (경고 출력)
            print(f"  ⚠️  Warning: '{key}' has different length ({len(data)} vs {n_samples}), copying without trimming")
            trimmed_dict[key] = data
    
    timestamp_trimmed = trimmed_dict['timestamp']
    
    print(f"\nTrimmed data:")
    print(f"  Samples: {len(timestamp_trimmed)}")
    print(f"  Time range: [{timestamp_trimmed[0]:.3f}s, {timestamp_trimmed[-1]:.3f}s]")
    print(f"  Duration: {timestamp_trimmed[-1]:.3f}s")
    
    # 출력 파일 저장
    if output_file is None:
        # 자동으로 파일명 생성
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_trim_{t1:.2f}to{t2:.2f}s.hdf5"
    
    with h5py.File(output_file, 'w') as f:
        # 모든 데이터셋 저장
        for key, data in trimmed_dict.items():
            f.create_dataset(key, data=data)
        
        # Attributes 저장 (업데이트)
        for key, value in attrs_dict.items():
            # duration, num_samples는 trimmed 데이터로 업데이트
            if key == 'duration':
                f.attrs[key] = timestamp_trimmed[-1]
            elif key == 'num_samples':
                f.attrs[key] = len(timestamp_trimmed)
            elif key == 'sample_rate':
                # sample_rate는 유지
                f.attrs[key] = value
            else:
                # 나머지는 그대로 복사
                f.attrs[key] = value
        
        # 추가 정보: trimming 정보
        f.attrs['trimmed'] = True
        f.attrs['trim_time_range'] = f'[{t1:.3f}s, {t2:.3f}s]'
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"   Datasets: {list(trimmed_dict.keys())}")
    print(f"   Duration: {timestamp_trimmed[-1]:.3f}s, Samples: {len(timestamp_trimmed)}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python trim_tcp_trajectory.py <input.hdf5> <t1> <t2> [output.hdf5]")
        print("\nArguments:")
        print("  input.hdf5  : 입력 HDF5 파일 (tcp_L_world, tcp_R_world, timestamp 포함)")
        print("  t1          : 시작 시간 (초)")
        print("  t2          : 종료 시간 (초)")
        print("  output.hdf5 : (선택) 출력 파일명 (생략시 자동 생성)")
        print("\nExample:")
        print("  python trim_tcp_trajectory.py world_position.hdf5 1.5 5.0")
        print("  python trim_tcp_trajectory.py world_position.hdf5 1.5 5.0 trimmed.hdf5")
        return
    
    input_file = sys.argv[1]
    t1 = float(sys.argv[2])
    t2 = float(sys.argv[3])
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return
    
    trim_trajectory(input_file, output_file, t1, t2)


if __name__ == '__main__':
    main()
