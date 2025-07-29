import h5py
import sys

if len(sys.argv) < 2:
    print("Usage: python get_hdf_info.py <file.hdf5>")
    sys.exit(1)

hdf5_name = sys.argv[1]

def inspect_group(group, indent=0):
    prefix = " " * indent
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            data = obj[()] 
            print(f"{prefix}- dataset: {name}, shape: {data.shape}")
        elif isinstance(obj, h5py.Group):
            print(f"{prefix}- group:   {name}/")
            # 재귀 호출로 깊이 탐색
            inspect_group(obj, indent+4)

with h5py.File(hdf5_name, 'r') as f:
    print(f"Inspecting file: {hdf5_name}")
    inspect_group(f)