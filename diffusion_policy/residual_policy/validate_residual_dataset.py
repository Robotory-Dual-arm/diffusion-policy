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


def summarize(values, scale=1.0):
    values = np.concatenate(values) * scale
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def print_summary(name, stats):
    print(
        f"{name}: "
        f"mean={stats['mean']:.6g} "
        f"p50={stats['p50']:.6g} "
        f"p90={stats['p90']:.6g} "
        f"p99={stats['p99']:.6g} "
        f"max={stats['max']:.6g}"
    )


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
    parser.add_argument("--max-residual-rotation-deg", type=float, default=30.0)
    parser.add_argument("--skip-residual-rotation-check", action="store_true")
    parser.add_argument(
        "--reference-dataset",
        default=None,
        help="Optional diffusion HDF5 whose data/demo_*/actions should match obs/virtual_target_abs.",
    )
    parser.add_argument(
        "--max-reference-rotation-deg",
        type=float,
        default=None,
        help="Optional max allowed virtual_target_abs vs reference actions rotation error in degrees.",
    )
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
    residual_rot_norm = []
    reference_virtual_pos_err = []
    reference_virtual_rot_err = []

    ref_file = h5py.File(args.reference_dataset, "r") if args.reference_dataset else None
    try:
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
                _, actual_to_virtual_rot = pose_error(actual_target, virtual_target)
                residual_rot_norm.append(actual_to_virtual_rot)

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

                if ref_file is not None:
                    ref_action = np.asarray(ref_file["data"][demo_name]["actions"])
                    ref_action = ref_action[:len(virtual_target)]
                    pos_err, rot_err = pose_error(virtual_target[:len(ref_action)], ref_action)
                    reference_virtual_pos_err.append(pos_err)
                    reference_virtual_rot_err.append(rot_err)

                checked_frames += len(actions)
    finally:
        if ref_file is not None:
            ref_file.close()

    print(f"checked_frames: {checked_frames}")
    for key, value in stats.items():
        print(f"{key}: {value:.9g}")

    actual_median = float(np.median(np.concatenate(actual_xyz_norm)))
    virtual_median = float(np.median(np.concatenate(virtual_xyz_norm)))
    residual_median = float(np.median(np.concatenate(residual_xyz_norm)))
    print(f"actual_xyz_norm_median: {actual_median:.9g}")
    print(f"virtual_xyz_norm_median: {virtual_median:.9g}")
    print(f"residual_translation_norm_median: {residual_median:.9g}")
    residual_translation_stats = summarize(residual_xyz_norm)
    residual_rotation_stats = summarize(residual_rot_norm, scale=180.0 / np.pi)
    print_summary("residual_translation_norm_m", residual_translation_stats)
    print_summary("actual_to_virtual_rotation_deg", residual_rotation_stats)

    if reference_virtual_pos_err:
        reference_pos_stats = summarize(reference_virtual_pos_err, scale=1000.0)
        reference_rot_stats = summarize(reference_virtual_rot_err, scale=180.0 / np.pi)
        print_summary("virtual_vs_reference_actions_pos_mm", reference_pos_stats)
        print_summary("virtual_vs_reference_actions_rot_deg", reference_rot_stats)

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
        if (
            not args.skip_residual_rotation_check
            and residual_rotation_stats["p99"] > args.max_residual_rotation_deg
        ):
            raise SystemExit(
                f"Validation failed: actual->virtual residual rotation p99 "
                f"{residual_rotation_stats['p99']:.6g} deg exceeds "
                f"--max-residual-rotation-deg={args.max_residual_rotation_deg}."
            )
        if (
            args.max_reference_rotation_deg is not None
            and reference_virtual_rot_err
            and reference_rot_stats["max"] > args.max_reference_rotation_deg
        ):
            raise SystemExit(
                f"Validation failed: virtual_target_abs vs reference actions max rotation "
                f"{reference_rot_stats['max']:.6g} deg exceeds "
                f"--max-reference-rotation-deg={args.max_reference_rotation_deg}."
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
