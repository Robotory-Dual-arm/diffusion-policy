#!/usr/bin/env python3
if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT_DIR))
    os.chdir(ROOT_DIR)

from typing import Tuple, Any, cast
import argparse
import math
import os
import pathlib

import cv2
import h5py
import numpy as np
import roboticstoolbox as rtb
import tqdm
from scipy.spatial.transform import Rotation

from diffusion_policy.residual_policy.pose_util import (
    abs_pose9_to_relative_pose9,
    delta6_from_base_to_target,
    mat_to_pose9,
    pose_like_to_pose9,
)


urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "m0609.white.urdf"))
robot: Any = rtb.ERobot.URDF(urdf_path)


def get_image_transform(
        input_res: Tuple[int, int] = (640, 480),
        output_res: Tuple[int, int] = (224, 224),
        bgr_to_rgb: bool = True):
    iw, ih = input_res
    ow, oh = output_res
    interp_method = cv2.INTER_AREA

    if (iw / ih) >= (ow / oh):
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR

    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None, None, -1) if bgr_to_rgb else slice(None)

    def transform_single(img):
        assert img.shape == (ih, iw, 3), f"Unexpected image shape: {img.shape}, expected {(ih, iw, 3)}"
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        return img[h_slice, w_slice, c_slice]

    return transform_single


def quat_to_6d(quats):
    rotmat = Rotation.from_quat(quats).as_matrix()
    return np.concatenate([rotmat[:, :, 0], rotmat[:, :, 1]], axis=1)


def subtract_offset(wrench_data, mean_number):
    return wrench_data - np.mean(wrench_data[:mean_number], axis=0)


def ema_filter(wrench_data, alpha):
    result = [wrench_data[0]]
    for i in range(1, len(wrench_data)):
        result.append(alpha * wrench_data[i] + (1 - alpha) * result[i - 1])
    return np.asarray(result)


def find_nearest_wrench_indices(robot_timestamps, wrench_timestamps):
    insert_idx = np.searchsorted(wrench_timestamps, robot_timestamps, side="left")
    insert_idx = np.clip(insert_idx, 1, len(wrench_timestamps) - 1)
    left_idx = insert_idx - 1
    right_idx = insert_idx
    choose_right = (
        np.abs(wrench_timestamps[right_idx] - robot_timestamps)
        < np.abs(robot_timestamps - wrench_timestamps[left_idx])
    )
    return np.where(choose_right, right_idx, left_idx)


def stack_wrench_history(wrench_data, wrench_indices, history_len):
    return np.stack([
        np.transpose(wrench_data[wrench_idx - history_len + 1:wrench_idx + 1])
        for wrench_idx in wrench_indices
    ])


def sorted_demo_indices(data_group):
    out = []
    for key in data_group.keys():
        if key.startswith("demo_"):
            out.append(int(key.split("_")[-1]))
    return sorted(out)


def virtual_pose_to_pose9(pose, rotation_format):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] >= 9:
        return pose[..., :9].astype(np.float32)
    if pose.shape[-1] != 6:
        raise ValueError(f"Expected virtual pose with 6 or >=9 dims, got {pose.shape}")

    if rotation_format == "rotvec_rad":
        return pose_like_to_pose9(pose).astype(np.float32)

    pos = pose[..., :3]
    rot_data = pose[..., 3:6]
    flat_rot_data = rot_data.reshape(-1, 3)

    if rotation_format == "rotvec_deg":
        rot = Rotation.from_rotvec(np.deg2rad(flat_rot_data))
    elif rotation_format == "euler_ZYX_deg":
        rot = Rotation.from_euler("ZYX", flat_rot_data, degrees=True)
    elif rotation_format == "euler_zyx_deg":
        rot = Rotation.from_euler("zyx", flat_rot_data, degrees=True)
    elif rotation_format == "euler_XYZ_deg":
        rot = Rotation.from_euler("XYZ", flat_rot_data, degrees=True)
    elif rotation_format == "euler_xyz_deg":
        rot = Rotation.from_euler("xyz", flat_rot_data, degrees=True)
    else:
        raise ValueError(f"Unknown virtual rotation format: {rotation_format}")

    mat = np.zeros(pos.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix().astype(np.float32).reshape(pos.shape[:-1] + (3, 3))
    mat[..., 3, 3] = 1.0
    return mat_to_pose9(mat).astype(np.float32)


def convert_one_demo(
        input_demo,
        output_demo,
        args,
        transform):
    input_obs = cast(h5py.Group, input_demo["observations"])
    output_obs = output_demo.create_group("obs")

    input_timestamp_robot = np.asarray(input_obs["timestamp_robot"])[::args.robot_downsample]
    input_joint_r = np.asarray(input_obs[args.joint_key])[::args.robot_downsample]
    input_hand_r = np.asarray(input_obs[args.hand_key])[::args.robot_downsample]
    input_image = np.asarray(input_obs[args.image_key])[::args.robot_downsample]
    input_desired_pose = np.asarray(input_obs[args.virtual_key])[::args.robot_downsample].astype(np.float32)
    input_desired_pose[:, :3] *= args.virtual_position_scale

    input_timestamp_wrench = np.asarray(input_obs["timestamp_wrench"])[:]
    wrench_wrist_r = np.asarray(input_obs["wrench_wrist_R"])[:, :6]
    wrench_thumb_r = np.asarray(input_obs["wrench_thumb_R"])[:, 2:3]
    wrench_index_r = np.asarray(input_obs["wrench_index_R"])[:, 2:3]
    wrench_middle_r = np.asarray(input_obs["wrench_middle_R"])[:, 2:3]
    wrench_ring_r = np.asarray(input_obs["wrench_ring_R"])[:, 2:3]
    wrench_baby_r = np.asarray(input_obs["wrench_baby_R"])[:, 2:3]

    wrench_arrays = [
        wrench_wrist_r,
        wrench_thumb_r,
        wrench_index_r,
        wrench_middle_r,
        wrench_ring_r,
        wrench_baby_r,
    ]
    wrench_arrays = [
        subtract_offset(arr, args.wrench_offset_mean_number)
        for arr in wrench_arrays
    ]
    wrench_arrays = [
        arr[args.wrench_offset_mean_number:]
        for arr in wrench_arrays
    ]
    input_timestamp_wrench = input_timestamp_wrench[args.wrench_offset_mean_number:]
    wrench_arrays = [ema_filter(arr, args.wrench_ema_alpha) for arr in wrench_arrays]

    valid_robot_mask = (
        (input_timestamp_robot >= input_timestamp_wrench[0])
        & (input_timestamp_robot <= input_timestamp_wrench[-1])
    )
    input_timestamp_robot = input_timestamp_robot[valid_robot_mask]
    input_joint_r = input_joint_r[valid_robot_mask]
    input_hand_r = input_hand_r[valid_robot_mask]
    input_image = input_image[valid_robot_mask]
    input_desired_pose = input_desired_pose[valid_robot_mask]

    if len(input_timestamp_robot) == 0:
        return False

    nearest_wrench_indices = find_nearest_wrench_indices(
        input_timestamp_robot,
        input_timestamp_wrench,
    )
    valid_wrench_history_mask = nearest_wrench_indices >= (args.wrench_history_len - 1)
    input_timestamp_robot = input_timestamp_robot[valid_wrench_history_mask]
    input_joint_r = input_joint_r[valid_wrench_history_mask]
    input_hand_r = input_hand_r[valid_wrench_history_mask]
    input_image = input_image[valid_wrench_history_mask]
    input_desired_pose = input_desired_pose[valid_wrench_history_mask]
    nearest_wrench_indices = nearest_wrench_indices[valid_wrench_history_mask]

    if len(input_timestamp_robot) < 2:
        return False

    wrench_histories = [
        stack_wrench_history(arr, nearest_wrench_indices, args.wrench_history_len)
        for arr in wrench_arrays
    ]

    output_image = np.asarray([transform(img) for img in input_image])
    tcp_r = robot.fkine(input_joint_r)
    tcp_pose_r = tcp_r.t.astype(np.float32)
    tcp_quat_r = Rotation.from_matrix(tcp_r.R).as_quat().astype(np.float32)
    tcp_quat_r = np.asarray([-q if q[3] < 0 else q for q in tcp_quat_r], dtype=np.float32)
    hand_pose_r = input_hand_r[:, [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14]].astype(np.float32)

    actual_pose9 = np.hstack([tcp_pose_r, quat_to_6d(tcp_quat_r)]).astype(np.float32)
    virtual_pose9 = virtual_pose_to_pose9(
        input_desired_pose,
        args.virtual_rotation_format,
    ).astype(np.float32)
    actual_action_rel = abs_pose9_to_relative_pose9(
        actual_pose9[:-1],
        actual_pose9[1:],
    )
    residual_delta6_gt_actual_to_virtual = delta6_from_base_to_target(
        actual_pose9[1:],
        virtual_pose9[1:],
    )

    # obs[t] -> action[t] is the next actual pose, and virtual_target_abs[t]
    # is the next virtual target used later to build the fast residual.
    output_obs.create_dataset("robot_pose_R", data=tcp_pose_r[:-1])
    output_obs.create_dataset("robot_quat_R", data=tcp_quat_r[:-1])
    output_obs.create_dataset("hand_pose_R", data=hand_pose_r[:-1])
    output_obs.create_dataset(args.output_image_key, data=output_image[:-1])
    output_obs.create_dataset("wrench_wrist_R", data=wrench_histories[0][:-1])
    output_obs.create_dataset("wrench_thumb_R", data=wrench_histories[1][:-1])
    output_obs.create_dataset("wrench_index_R", data=wrench_histories[2][:-1])
    output_obs.create_dataset("wrench_middle_R", data=wrench_histories[3][:-1])
    output_obs.create_dataset("wrench_ring_R", data=wrench_histories[4][:-1])
    output_obs.create_dataset("wrench_baby_R", data=wrench_histories[5][:-1])
    output_obs.create_dataset("virtual_target_abs", data=virtual_pose9[1:])
    output_obs.create_dataset("actual_target_abs", data=actual_pose9[1:])
    output_obs.create_dataset("actual_action_rel", data=actual_action_rel)
    output_obs.create_dataset(
        "residual_delta6_gt_actual_to_virtual",
        data=residual_delta6_gt_actual_to_virtual,
    )

    output_demo.create_dataset("actions", data=actual_pose9[1:])
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert common robot data to a slow-policy diffusion HDF5. "
            "actions are next actual pose; obs/virtual_target_abs preserves the next virtual target "
            "for fast residual dataset generation. Also stores GT actual -> virtual residual."
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input common_data HDF5 file(s).")
    parser.add_argument("--output", required=True, help="Output slow diffusion HDF5.")
    parser.add_argument("--image-key", default="image_R")
    parser.add_argument("--output-image-key", default="image0")
    parser.add_argument("--joint-key", default="joint_R")
    parser.add_argument("--hand-key", default="hand_R")
    parser.add_argument("--virtual-key", default="desired_pose")
    parser.add_argument(
        "--virtual-rotation-format",
        default="rotvec_rad",
        choices=[
            "rotvec_rad",
            "rotvec_deg",
            "euler_ZYX_deg",
            "euler_zyx_deg",
            "euler_XYZ_deg",
            "euler_xyz_deg",
        ],
        help=(
            "Rotation representation for a 6D virtual target. "
            "common_data_height.hdf5 desired_pose uses euler_ZYX_deg."
        ),
    )
    parser.add_argument(
        "--virtual-position-scale",
        type=float,
        default=0.001,
        help=(
            "Scale applied to virtual target xyz before pose conversion. "
            "common_data desired_pose xyz is usually stored in mm, so the default converts it to meters."
        ),
    )
    parser.add_argument("--robot-downsample", type=int, default=2)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--output-width", type=int, default=224)
    parser.add_argument("--output-height", type=int, default=224)
    parser.add_argument("--wrench-history-len", type=int, default=32)
    parser.add_argument("--wrench-offset-mean-number", type=int, default=10)
    parser.add_argument("--wrench-ema-alpha", type=float, default=0.03)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = pathlib.Path.cwd() / output_path
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = get_image_transform(
        input_res=(args.image_width, args.image_height),
        output_res=(args.output_width, args.output_height),
        bgr_to_rgb=True,
    )

    output_demo_idx = 0
    with h5py.File(output_path, "w") as output_file:
        output_file.attrs["virtual_key"] = args.virtual_key
        output_file.attrs["virtual_position_scale"] = args.virtual_position_scale
        output_file.attrs["virtual_rotation_format"] = args.virtual_rotation_format
        output_file.attrs["robot_downsample"] = args.robot_downsample
        output_data = output_file.create_group("data")
        output_file.attrs["actions"] = "next_actual_pose9"
        output_file.attrs["obs/virtual_target_abs"] = "next_virtual_target_pose9"
        output_file.attrs["obs/residual_delta6_gt_actual_to_virtual"] = (
            "delta6_from_next_actual_pose9_to_next_virtual_target_pose9"
        )
        output_file.attrs["obs/actual_action_rel"] = (
            "relative_next_actual_pose9_from_current_actual_pose9"
        )
        for input_name in args.input:
            input_path = pathlib.Path(input_name).expanduser()
            if not input_path.is_absolute():
                input_path = pathlib.Path.cwd() / input_path
            with h5py.File(input_path, "r") as input_file:
                input_data = cast(h5py.Group, input_file["data"])
                demo_indices = sorted_demo_indices(input_data)
                if args.max_demos is not None:
                    remaining = args.max_demos - output_demo_idx
                    if remaining <= 0:
                        break
                    demo_indices = demo_indices[:remaining]
                print(input_path, "/ demo_len =", len(demo_indices))

                for demo_idx in tqdm.tqdm(demo_indices, desc="Converting common demos"):
                    input_demo = cast(h5py.Group, input_data[f"demo_{demo_idx}"])
                    output_demo_name = f"demo_{output_demo_idx}"
                    output_demo = output_data.create_group(output_demo_name)
                    ok = convert_one_demo(input_demo, output_demo, args, transform)
                    if not ok:
                        del output_data[output_demo_name]
                        continue
                    output_demo_idx += 1

    print("Slow dataset conversion completed / output_demo_len =", output_demo_idx)


if __name__ == "__main__":
    main()
