#!/usr/bin/env python3
"""Build a residual-policy dataset from aligned actual/virtual action HDF5s.

The output keeps the full actual action (for example pose9 + hand7), while all
residual targets and base actions are deliberately computed from the leading
end-effector pose9 only.  Any action tail is therefore owned by the slow policy.
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse
import pathlib
import shutil

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from diffusion_policy.residual_policy.pose_util import (
    abs_pose9_to_relative_pose9,
    delta6_from_base_to_target,
    pose6_to_pose9,
    pose_like_to_pose9,
)


def sorted_demo_keys(data_group):
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def current_pose9(obs_group, arm):
    pos = np.asarray(obs_group[f"robot_pose_{arm}"], dtype=np.float32)
    quat = np.asarray(obs_group[f"robot_quat_{arm}"], dtype=np.float32)
    rotvec = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate([pos, rotvec], axis=-1))


def replace_dataset(group, name, data):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def copy_actual_dataset(actual_path, output_path, demo_limit):
    if demo_limit is None:
        shutil.copy2(actual_path, output_path)
        return

    with h5py.File(actual_path, "r") as src, h5py.File(output_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst_data = dst.create_group("data")
        demo_keys = sorted_demo_keys(src["data"])[:demo_limit]
        for demo_name in tqdm(demo_keys, desc="Copying selected actual demos"):
            src.copy(src["data"][demo_name], dst_data, name=demo_name)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy an actual-action diffusion HDF5 and add pose-only residual keys "
            "from an aligned virtual-action HDF5."
        )
    )
    parser.add_argument("--actual", required=True)
    parser.add_argument("--virtual", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="R", choices=["L", "R"])
    parser.add_argument("--demo-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    actual_path = pathlib.Path(args.actual).expanduser().resolve()
    virtual_path = pathlib.Path(args.virtual).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    if output_path in (actual_path, virtual_path):
        raise ValueError("Output must differ from both input datasets.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
    if args.demo_limit is not None and args.demo_limit <= 0:
        raise ValueError(f"demo_limit must be positive, got {args.demo_limit}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    copy_actual_dataset(actual_path, output_path, args.demo_limit)

    hand_max_abs_diff = 0.0
    frame_count = 0
    with h5py.File(output_path, "r+") as output_file, h5py.File(virtual_path, "r") as virtual_file:
        output_data = output_file["data"]
        virtual_data = virtual_file["data"]
        demo_keys = sorted_demo_keys(output_data)
        missing = [name for name in demo_keys if name not in virtual_data]
        if missing:
            raise KeyError(f"Virtual dataset is missing demos: {missing[:5]}")

        for demo_name in tqdm(demo_keys, desc="Adding EE residual targets"):
            demo = output_data[demo_name]
            virtual_demo = virtual_data[demo_name]
            actual_action = np.asarray(demo["actions"])
            virtual_action = np.asarray(virtual_demo["actions"])
            if actual_action.shape != virtual_action.shape:
                raise ValueError(
                    f"{demo_name} action shape mismatch: "
                    f"actual={actual_action.shape}, virtual={virtual_action.shape}"
                )
            if actual_action.shape[-1] < 9:
                raise ValueError(f"{demo_name} needs action pose9, got {actual_action.shape}")

            obs = demo["obs"]
            actual_target_abs = pose_like_to_pose9(actual_action)
            virtual_target_abs = pose_like_to_pose9(virtual_action)
            current_target_base = current_pose9(obs, args.arm)
            if len(current_target_base) != len(actual_target_abs):
                raise ValueError(
                    f"{demo_name} obs/action length mismatch: "
                    f"obs={len(current_target_base)}, actions={len(actual_target_abs)}"
                )

            actual_action_rel = abs_pose9_to_relative_pose9(
                current_target_base,
                actual_target_abs,
            )
            residual_delta6 = delta6_from_base_to_target(
                actual_target_abs,
                virtual_target_abs,
            )
            replace_dataset(obs, "actual_target_abs", actual_target_abs.astype(np.float32))
            replace_dataset(obs, "virtual_target_abs", virtual_target_abs.astype(np.float32))
            replace_dataset(obs, "actual_action_rel", actual_action_rel.astype(np.float32))
            replace_dataset(
                obs,
                "residual_delta6_gt_actual_to_virtual",
                residual_delta6.astype(np.float32),
            )

            if actual_action.shape[-1] > 9:
                hand_max_abs_diff = max(
                    hand_max_abs_diff,
                    float(np.max(np.abs(actual_action[..., 9:] - virtual_action[..., 9:]))),
                )
            frame_count += len(actual_action)

        output_file.attrs["actions"] = "full_actual_action; leading pose9 plus unchanged slow-owned tail"
        output_file.attrs["obs/actual_target_abs"] = "actual action leading end-effector pose9"
        output_file.attrs["obs/virtual_target_abs"] = "virtual action leading end-effector pose9"
        output_file.attrs["obs/actual_action_rel"] = "actual EE pose9 relative to current observed EE pose"
        output_file.attrs["obs/residual_delta6_gt_actual_to_virtual"] = (
            "EE-only delta6 from actual target pose9 to virtual target pose9"
        )
        output_file.attrs["residual_actual_dataset"] = str(actual_path)
        output_file.attrs["residual_virtual_dataset"] = str(virtual_path)
        output_file.attrs["residual_corrected_action_dims"] = 9
        output_file.attrs["slow_owned_action_tail_dims"] = max(
            0,
            int(output_data[demo_keys[0]]["actions"].shape[-1]) - 9,
        )

    print("Wrote residual dataset:", output_path)
    print("demos:", len(demo_keys))
    print("frames:", frame_count)
    print("actual/virtual action-tail max abs diff:", hand_max_abs_diff)


if __name__ == "__main__":
    main()
