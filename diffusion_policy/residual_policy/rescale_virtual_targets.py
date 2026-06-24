#!/usr/bin/env python3
if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import pathlib

import h5py
import numpy as np
from tqdm import tqdm

from diffusion_policy.residual_policy.pose_util import delta6_from_base_to_target


def sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def replace_dataset(group, key, data):
    if key in group:
        del group[key]
    group.create_dataset(key, data=data)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy a residual-policy HDF5 and rescale obs/virtual_target_abs xyz, "
            "then recompute obs/residual_delta6_gt_actual_to_virtual."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).expanduser()
    output_path = pathlib.Path(args.output).expanduser()
    if not input_path.is_absolute():
        input_path = pathlib.Path.cwd() / input_path
    if not output_path.is_absolute():
        output_path = pathlib.Path.cwd() / output_path

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        src.copy("data", dst)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["virtual_position_scale_fix"] = args.position_scale
        dst.attrs["obs/virtual_target_abs"] = (
            f"next_virtual_target_pose9; xyz scaled by {args.position_scale}"
        )
        dst.attrs["obs/residual_delta6_gt_actual_to_virtual"] = (
            "delta6_from_next_actual_pose9_to_scaled_next_virtual_target_pose9"
        )

        data_group = dst["data"]
        for demo_name in tqdm(sorted_demo_keys(data_group), desc="Rescaling virtual targets"):
            obs = data_group[demo_name]["obs"]
            virtual = np.asarray(obs["virtual_target_abs"]).astype(np.float32)
            actual = np.asarray(obs["actual_target_abs"]).astype(np.float32)
            virtual[:, :3] *= args.position_scale
            residual = delta6_from_base_to_target(actual, virtual)
            replace_dataset(obs, "virtual_target_abs", virtual)
            replace_dataset(obs, "residual_delta6_gt_actual_to_virtual", residual)

    print(f"Wrote corrected residual dataset: {output_path}")


if __name__ == "__main__":
    main()
