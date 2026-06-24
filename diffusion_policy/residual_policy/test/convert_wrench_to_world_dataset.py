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
from scipy.spatial.transform import Rotation
from tqdm import tqdm


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


def copy_h5(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key, item in src.items():
        if isinstance(item, h5py.Dataset):
            src.copy(item, dst, name=key)
        elif isinstance(item, h5py.Group):
            group = dst.create_group(key)
            copy_h5(item, group)


def parse_matrix(values):
    if values is None:
        return np.eye(3, dtype=np.float32)
    arr = np.asarray([float(x) for x in values], dtype=np.float32)
    if arr.size != 9:
        raise ValueError("Expected 9 values for a 3x3 matrix")
    return arr.reshape(3, 3)


def transform_wrench_history(wrench, quat, robot_to_world, sensor_to_tcp):
    """Rotate wrench force/torque histories from sensor/EEF frame to world.

    wrench: (T, 6, H), with force rows 0:3 and torque rows 3:6.
    quat: (T, 4), scipy xyzw quaternion of TCP/EEF in robot-base frame.
    sensor_to_tcp: fixed rotation from sensor frame into TCP frame.
    """
    wrench = np.asarray(wrench, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    if wrench.ndim != 3 or wrench.shape[1] != 6:
        raise ValueError(f"Expected wrench shape (T, 6, H), got {wrench.shape}")
    if quat.shape[0] != wrench.shape[0]:
        raise ValueError(f"quat length {quat.shape[0]} does not match wrench length {wrench.shape[0]}")

    robot_tcp = Rotation.from_quat(quat).as_matrix().astype(np.float32)
    world_sensor = np.einsum("ij,tjk,kl->til", robot_to_world, robot_tcp, sensor_to_tcp)

    force_world = np.einsum("tij,tjh->tih", world_sensor, wrench[:, :3])
    torque_world = np.einsum("tij,tjh->tih", world_sensor, wrench[:, 3:6])
    return np.concatenate([force_world, torque_world], axis=1).astype(np.float32)


def convert_dataset(input_path, output_path, wrench_keys, quat_key, overwrite, sensor_to_tcp):
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_h5(src, dst)
        dst.attrs["wrench_frame_original"] = "sensor_or_ee"
        dst.attrs["wrench_frame"] = "world"
        dst.attrs["wrench_world_rotation"] = "RIGHT_ROBOT_TO_WORLD @ R_robot_tcp @ R_sensor_to_tcp"
        dst.attrs["right_robot_to_world"] = RIGHT_ROBOT_TO_WORLD
        dst.attrs["sensor_to_tcp"] = sensor_to_tcp

        for demo_name in tqdm(sorted_demo_keys(dst["data"]), desc="converting wrench to world"):
            obs = dst["data"][demo_name]["obs"]
            if quat_key not in obs:
                raise KeyError(f"{demo_name}/obs/{quat_key} not found")
            quat = np.asarray(obs[quat_key], dtype=np.float32)
            for key in wrench_keys:
                if key not in obs:
                    continue
                wrench_world = transform_wrench_history(
                    np.asarray(obs[key], dtype=np.float32),
                    quat,
                    RIGHT_ROBOT_TO_WORLD,
                    sensor_to_tcp,
                )
                del obs[key]
                obs.create_dataset(key, data=wrench_world, compression="gzip", compression_opts=4)
                obs[key].attrs["frame"] = "world"


def main():
    parser = argparse.ArgumentParser(
        description="Copy a residual HDF5 dataset and rotate wrench histories into the world frame."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quat-key", default="robot_quat_R")
    parser.add_argument(
        "--wrench-key",
        action="append",
        default=None,
        help="Wrench key under obs. Can be repeated. Default: wrench_wrist_R.",
    )
    parser.add_argument(
        "--sensor-to-tcp",
        nargs=9,
        default=None,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="Optional fixed 3x3 rotation mapping sensor-frame vectors into TCP-frame vectors. Default identity.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wrench_keys = args.wrench_key or ["wrench_wrist_R"]
    sensor_to_tcp = parse_matrix(args.sensor_to_tcp)
    convert_dataset(
        input_path=args.input,
        output_path=args.output,
        wrench_keys=wrench_keys,
        quat_key=args.quat_key,
        overwrite=args.overwrite,
        sensor_to_tcp=sensor_to_tcp,
    )
    print(f"Wrote world-wrench dataset: {args.output}")


if __name__ == "__main__":
    main()
