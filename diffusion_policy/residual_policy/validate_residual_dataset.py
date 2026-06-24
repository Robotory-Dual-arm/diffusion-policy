#!/usr/bin/env python3
if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

import argparse

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.residual_policy.pose_util import (
    apply_delta6_to_pose9,
    pose6_to_pose9,
    pose9_to_mat,
    relative_pose9_to_abs_pose9,
)


def sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def pose_error(a, b):
    mat_a = pose9_to_mat(a)
    mat_b = pose9_to_mat(b)
    pos_err = np.linalg.norm(mat_a[..., :3, 3] - mat_b[..., :3, 3], axis=-1)
    rel = np.linalg.inv(mat_a) @ mat_b
    rot_err = Rotation.from_matrix(rel[..., :3, :3]).magnitude()
    return pos_err, rot_err


def current_pose9_from_obs(obs_group, arm):
    pos = np.asarray(obs_group[f"robot_pose_{arm}"])
    quat = np.asarray(obs_group[f"robot_quat_{arm}"])
    rotvec = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate([pos, rotvec], axis=-1))


def update_max(current, values):
    if values.size == 0:
        return current
    return max(current, float(np.max(values)))


def main():
    parser = argparse.ArgumentParser(
        description="Validate slow/fast residual HDF5 pose timing and SE(3) conversions."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arm", default="R", choices=["L", "R"])
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--pos-tol", type=float, default=1.0e-3)
    parser.add_argument("--rot-tol", type=float, default=1.0e-4)
    parser.add_argument("--max-residual-translation-m", type=float, default=1.0)
    parser.add_argument("--skip-unit-check", action="store_true")
    args = parser.parse_args()

    stats = {
        "actions_vs_actual_pos": 0.0,
        "actions_vs_actual_rot": 0.0,
        "relative_actual_pos": 0.0,
        "relative_actual_rot": 0.0,
        "residual_virtual_pos": 0.0,
        "residual_virtual_rot": 0.0,
    }
    checked_frames = 0
    actual_xyz_norm = []
    virtual_xyz_norm = []
    residual_xyz_norm = []

    with h5py.File(args.dataset, "r") as f:
        demo_names = sorted_demo_keys(f["data"])
        if args.max_demos is not None:
            demo_names = demo_names[:args.max_demos]

        for demo_name in demo_names:
            demo = f["data"][demo_name]
            obs = demo["obs"]

            actions = np.asarray(demo["actions"])
            actual_target = np.asarray(obs["actual_target_abs"])
            virtual_target = np.asarray(obs["virtual_target_abs"])
            actual_action_rel = np.asarray(obs["actual_action_rel"])
            residual_delta6 = np.asarray(obs["residual_delta6_gt_actual_to_virtual"])
            current_pose9 = current_pose9_from_obs(obs, args.arm)
            actual_xyz_norm.append(np.linalg.norm(actual_target[..., :3], axis=-1))
            virtual_xyz_norm.append(np.linalg.norm(virtual_target[..., :3], axis=-1))
            residual_xyz_norm.append(np.linalg.norm(residual_delta6[..., :3], axis=-1))

            pos_err, rot_err = pose_error(actions, actual_target)
            stats["actions_vs_actual_pos"] = update_max(stats["actions_vs_actual_pos"], pos_err)
            stats["actions_vs_actual_rot"] = update_max(stats["actions_vs_actual_rot"], rot_err)

            reconstructed_actual = relative_pose9_to_abs_pose9(
                current_pose9,
                actual_action_rel,
            )
            pos_err, rot_err = pose_error(reconstructed_actual, actual_target)
            stats["relative_actual_pos"] = update_max(stats["relative_actual_pos"], pos_err)
            stats["relative_actual_rot"] = update_max(stats["relative_actual_rot"], rot_err)

            reconstructed_virtual = apply_delta6_to_pose9(
                actual_target,
                residual_delta6,
            )
            pos_err, rot_err = pose_error(reconstructed_virtual, virtual_target)
            stats["residual_virtual_pos"] = update_max(stats["residual_virtual_pos"], pos_err)
            stats["residual_virtual_rot"] = update_max(stats["residual_virtual_rot"], rot_err)

            checked_frames += len(actions)

    print(f"checked_frames: {checked_frames}")
    for key, value in stats.items():
        print(f"{key}: {value:.9g}")

    actual_median = float(np.median(np.concatenate(actual_xyz_norm)))
    virtual_median = float(np.median(np.concatenate(virtual_xyz_norm)))
    residual_median = float(np.median(np.concatenate(residual_xyz_norm)))
    print(f"actual_xyz_norm_median: {actual_median:.9g}")
    print(f"virtual_xyz_norm_median: {virtual_median:.9g}")
    print(f"residual_translation_norm_median: {residual_median:.9g}")

    if not args.skip_unit_check:
        if actual_median < 10.0 and virtual_median > 10.0:
            raise SystemExit(
                "Validation failed: actual pose looks like meters but virtual target looks like millimeters. "
                "Reconvert with --virtual-position-scale 0.001 or use rescale_virtual_targets.py."
            )
        if residual_median > args.max_residual_translation_m:
            raise SystemExit(
                f"Validation failed: residual translation median {residual_median:.6g} m exceeds "
                f"--max-residual-translation-m={args.max_residual_translation_m}."
            )

    pos_ok = (
        stats["actions_vs_actual_pos"] <= args.pos_tol
        and stats["relative_actual_pos"] <= args.pos_tol
        and stats["residual_virtual_pos"] <= args.pos_tol
    )
    rot_ok = (
        stats["actions_vs_actual_rot"] <= args.rot_tol
        and stats["relative_actual_rot"] <= args.rot_tol
        and stats["residual_virtual_rot"] <= args.rot_tol
    )
    if not (pos_ok and rot_ok):
        raise SystemExit(
            f"Validation failed with pos_tol={args.pos_tol}, rot_tol={args.rot_tol}"
        )
    print("validation: ok")


if __name__ == "__main__":
    main()
