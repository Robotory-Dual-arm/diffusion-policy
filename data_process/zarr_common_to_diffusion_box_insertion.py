from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import tqdm
import zarr
from scipy.spatial.transform import Rotation as R


DEFAULT_INPUT_ROOT = Path("/data/baetae/260630_box_insertion_vr/20260630_195919")
DEFAULT_OUTPUT_DIR = Path("/data/baetae/260708_box_insertion_hand")
DEFAULT_VIRTUAL_OUTPUT = (
    "diffusion_data_box_insertion_405_ft_virtual_target_action.hdf5"
)
DEFAULT_ACTUAL_OUTPUT = (
    "diffusion_data_box_insertion_405_ft_actual_pose_action.hdf5"
)
DEFAULT_VIRTUAL_HAND_OUTPUT = (
    "diffusion_data_box_insertion_405_ft_virtual_target_hand_action.hdf5"
)
DEFAULT_ACTUAL_HAND_OUTPUT = (
    "diffusion_data_box_insertion_405_ft_actual_pose_hand_action.hdf5"
)
DEFAULT_HAND_INDICES = "0,1,2,4,5,7,8"
DEFAULT_WRENCH_HISTORY_DT = 0.004


def get_image_transform(
    input_res: tuple[int, int] = (640, 480),
    output_res: tuple[int, int] = (224, 224),
    bgr_to_rgb: bool = False,
):
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

    def transform_single(img: np.ndarray) -> np.ndarray:
        assert img.shape == (ih, iw, 3), (
            f"Unexpected image shape: {img.shape}, expected {(ih, iw, 3)}"
        )
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        return img[h_slice, w_slice, c_slice]

    return transform_single


def parse_int_list(text: str) -> list[int]:
    values = [value.strip() for value in text.split(",")]
    return [int(value) for value in values if value]


def nearest_indices(source_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    source_timestamps = np.asarray(source_timestamps, dtype=np.float64)
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    if len(source_timestamps) == 0:
        raise ValueError("source_timestamps is empty")
    if len(source_timestamps) == 1:
        return np.zeros(len(query_timestamps), dtype=np.int64)

    insert_idx = np.searchsorted(source_timestamps, query_timestamps, side="left")
    insert_idx = np.clip(insert_idx, 1, len(source_timestamps) - 1)

    left_idx = insert_idx - 1
    right_idx = insert_idx
    choose_right = (
        np.abs(source_timestamps[right_idx] - query_timestamps)
        < np.abs(query_timestamps - source_timestamps[left_idx])
    )
    return np.where(choose_right, right_idx, left_idx).astype(np.int64)


def subtract_offset(wrench_data: np.ndarray, mean_number: int) -> np.ndarray:
    if mean_number <= 0:
        return wrench_data
    mean_number = min(mean_number, len(wrench_data))
    offset = np.mean(wrench_data[:mean_number], axis=0)
    return wrench_data - offset


def ema_filter(wrench_data: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return wrench_data
    if not 0 < alpha <= 1:
        raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
    result = np.empty_like(wrench_data)
    result[0] = wrench_data[0]
    for idx in range(1, len(wrench_data)):
        result[idx] = alpha * wrench_data[idx] + (1.0 - alpha) * result[idx - 1]
    return result


def preprocess_wrench(
    wrench: np.ndarray,
    timestamps: np.ndarray,
    offset_samples: int,
    ema_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(wrench) == 0:
        raise ValueError("wrench data is empty")
    wrench = np.asarray(wrench, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    wrench = subtract_offset(wrench, offset_samples)
    if offset_samples > 0:
        drop_count = min(offset_samples, len(wrench))
        wrench = wrench[drop_count:]
        timestamps = timestamps[drop_count:]

    if len(wrench) == 0:
        raise ValueError("wrench data became empty after offset drop")

    wrench = ema_filter(wrench, ema_alpha)
    return wrench.astype(np.float32, copy=False), timestamps


def stack_wrench_history(
    wrench_data: np.ndarray,
    wrench_indices: np.ndarray,
    history_len: int,
) -> np.ndarray:
    return np.stack(
        [
            np.transpose(wrench_data[idx - history_len + 1 : idx + 1])
            for idx in wrench_indices
        ]
    ).astype(np.float32, copy=False)


def stack_wrench_history_at_times(
    wrench_data: np.ndarray,
    wrench_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    history_len: int,
    history_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(history_len - 1, -1, -1, dtype=np.float64) * history_dt
    history_timestamps = target_timestamps[:, None] - offsets[None, :]
    flat_indices = nearest_indices(wrench_timestamps, history_timestamps.reshape(-1))
    history_indices = flat_indices.reshape(len(target_timestamps), history_len)
    history = np.transpose(wrench_data[history_indices], (0, 2, 1))
    return history.astype(np.float32, copy=False), history_indices, history_timestamps


def matrices_to_pose_quat(mats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(mats[:, :3, 3], dtype=np.float64)
    quat = R.from_matrix(mats[:, :3, :3]).as_quat()
    quat = np.asarray([-q if q[3] < 0 else q for q in quat], dtype=np.float64)
    return pose, quat


def matrices_to_actions(mats: np.ndarray) -> np.ndarray:
    pose = np.asarray(mats[:, :3, 3], dtype=np.float64)
    rot = R.from_matrix(mats[:, :3, :3]).as_matrix()
    rot6d = np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=1)
    return np.hstack([pose, rot6d]).astype(np.float64, copy=False)


def reinterpret_rotation_order(
    mats: np.ndarray,
    source_order: str | None,
    target_order: str | None,
) -> np.ndarray:
    if not source_order or not target_order or source_order == target_order:
        return mats

    out = np.array(mats, dtype=np.float64, copy=True)
    euler = R.from_matrix(out[:, :3, :3]).as_euler(source_order, degrees=False)
    out[:, :3, :3] = R.from_euler(target_order, euler, degrees=False).as_matrix()
    return out


def read_zarr(path: Path):
    return zarr.open(str(path), mode="r")


def sorted_episode_dirs(input_root: Path) -> list[Path]:
    episodes = [path for path in input_root.glob("episode_*") if path.is_dir()]
    return sorted(episodes, key=lambda path: path.name)


def require_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_demo(
    data_group: h5py.Group,
    demo_name: str,
    action_data: np.ndarray,
    obs_data: dict[str, np.ndarray],
    source_episode: str,
) -> None:
    demo_group = data_group.create_group(demo_name)
    demo_group.attrs["source_episode"] = source_episode
    demo_group.create_dataset("actions", data=action_data)
    obs_group = demo_group.create_group("obs")
    for key, value in obs_data.items():
        obs_group.create_dataset(key, data=value)


def convert_episode(
    episode_dir: Path,
    camera_name: str,
    hand_indices: Iterable[int],
    include_hand_action: bool,
    target_hz: float,
    source_hz: float,
    wrench_offset_samples: int,
    wrench_ema_alpha: float,
    wrench_history_len: int,
    wrench_history_dt: float,
    virtual_rotation_source_order: str | None,
    virtual_rotation_target_order: str | None,
    image_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    if target_hz <= 0 or source_hz <= 0:
        raise ValueError("target_hz and source_hz must be positive")

    downsample = int(round(source_hz / target_hz))
    if downsample <= 0:
        raise ValueError(f"Invalid downsample value: {downsample}")

    camera_dir = episode_dir / camera_name
    rgb = read_zarr(camera_dir / "rgb.zarr")
    rgb_ts = np.asarray(read_zarr(camera_dir / "rgb_time_stamps.zarr"), dtype=np.float64)

    ee_pose = read_zarr(episode_dir / "robot" / "ee_pose_se3.zarr")
    ee_ts = np.asarray(
        read_zarr(episode_dir / "robot" / "ee_pose_time_stamps.zarr"),
        dtype=np.float64,
    )

    command_pose = read_zarr(episode_dir / "robot" / "command_pose_se3.zarr")
    command_ts = np.asarray(
        read_zarr(episode_dir / "robot" / "command_time_stamps.zarr"),
        dtype=np.float64,
    )

    hand = read_zarr(episode_dir / "robot" / "hand_joint.zarr")
    hand_ts = np.asarray(
        read_zarr(episode_dir / "robot" / "hand_joint_time_stamps.zarr"),
        dtype=np.float64,
    )

    raw_wrench = np.asarray(
        read_zarr(episode_dir / "ft" / "wrench_raw.zarr"),
        dtype=np.float32,
    )
    raw_wrench_ts = np.asarray(
        read_zarr(episode_dir / "ft" / "wrench_time_stamps.zarr"),
        dtype=np.float64,
    )
    wrench, wrench_ts = preprocess_wrench(
        raw_wrench,
        raw_wrench_ts,
        offset_samples=wrench_offset_samples,
        ema_alpha=wrench_ema_alpha,
    )

    target_indices = np.arange(len(ee_ts), dtype=np.int64)[::downsample]
    target_ts = ee_ts[target_indices]

    start_time = max(rgb_ts[0], ee_ts[0], hand_ts[0], command_ts[0], wrench_ts[0])
    end_time = min(rgb_ts[-1], ee_ts[-1], hand_ts[-1], command_ts[-1], wrench_ts[-1])
    valid_mask = (target_ts >= start_time) & (target_ts <= end_time)
    target_ts = target_ts[valid_mask]
    target_indices = target_indices[valid_mask]
    if len(target_ts) == 0:
        raise RuntimeError(f"{episode_dir.name}: no overlapping target timestamps")

    history_start_ts = target_ts - (wrench_history_len - 1) * wrench_history_dt
    history_mask = history_start_ts >= wrench_ts[0]
    target_ts = target_ts[history_mask]
    target_indices = target_indices[history_mask]
    if len(target_ts) < 2:
        raise RuntimeError(f"{episode_dir.name}: not enough samples after wrench history")

    rgb_indices = nearest_indices(rgb_ts, target_ts)
    hand_nearest = nearest_indices(hand_ts, target_ts)
    command_indices = nearest_indices(command_ts, target_ts)

    input_res = (int(rgb.shape[2]), int(rgb.shape[1]))
    transform = get_image_transform(
        input_res=input_res,
        output_res=image_size,
        bgr_to_rgb=False,
    )
    images = np.empty((len(rgb_indices), image_size[1], image_size[0], 3), dtype=np.uint8)
    for out_idx, rgb_idx in enumerate(rgb_indices):
        images[out_idx] = transform(np.asarray(rgb[int(rgb_idx)]))

    actual_mats = np.asarray(ee_pose.oindex[target_indices], dtype=np.float64)
    command_mats = np.asarray(command_pose.oindex[command_indices], dtype=np.float64)
    command_mats = reinterpret_rotation_order(
        command_mats,
        virtual_rotation_source_order,
        virtual_rotation_target_order,
    )
    robot_pose, robot_quat = matrices_to_pose_quat(actual_mats)

    hand_indices = list(hand_indices)
    hand_pose = np.asarray(hand.oindex[hand_nearest, hand_indices], dtype=np.float32)
    wrench_history, wrench_indices, history_timestamps = stack_wrench_history_at_times(
        wrench,
        wrench_ts,
        target_ts,
        wrench_history_len,
        wrench_history_dt,
    )

    obs_data = {
        "robot_pose_R": robot_pose[:-1],
        "robot_quat_R": robot_quat[:-1],
        "hand_pose_R": hand_pose[:-1],
        "image0": images[:-1],
        "wrench_wrist_R": wrench_history[:-1],
    }
    virtual_actions = matrices_to_actions(command_mats)
    actual_actions = matrices_to_actions(actual_mats)
    if include_hand_action:
        virtual_actions = np.hstack([virtual_actions, hand_pose])
        actual_actions = np.hstack([actual_actions, hand_pose])
    virtual_actions = virtual_actions[1:]
    actual_actions = actual_actions[1:]

    stats = {
        "num_samples": float(len(target_ts) - 1),
        "max_rgb_dt": float(np.max(np.abs(rgb_ts[rgb_indices] - target_ts))),
        "max_hand_dt": float(np.max(np.abs(hand_ts[hand_nearest] - target_ts))),
        "max_command_dt": float(np.max(np.abs(command_ts[command_indices] - target_ts))),
        "max_wrench_dt": float(
            np.max(np.abs(wrench_ts[wrench_indices] - history_timestamps))
        ),
        "max_current_wrench_dt": float(
            np.max(np.abs(wrench_ts[wrench_indices[:, -1]] - target_ts))
        ),
    }
    return obs_data, virtual_actions, actual_actions, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the 20260630 episode-wise Zarr common format into the "
            "robomimic-style diffusion HDF5 format."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--virtual-output", default=None)
    parser.add_argument("--actual-output", default=None)
    parser.add_argument("--camera", default="camera_0_D405")
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--source-hz", type=float, default=30.0)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--hand-indices", default=DEFAULT_HAND_INDICES)
    parser.add_argument("--include-hand-action", action="store_true")
    parser.add_argument("--wrench-offset-samples", type=int, default=10)
    parser.add_argument("--wrench-ema-alpha", type=float, default=0.03)
    parser.add_argument("--wrench-history-len", type=int, default=32)
    parser.add_argument("--wrench-history-dt", type=float, default=DEFAULT_WRENCH_HISTORY_DT)
    parser.add_argument("--virtual-rotation-source-order", default="zyz")
    parser.add_argument("--virtual-rotation-target-order", default="xyz")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.expanduser()
    output_dir = args.output_dir.expanduser()
    virtual_output_name = args.virtual_output
    actual_output_name = args.actual_output
    if virtual_output_name is None:
        virtual_output_name = (
            DEFAULT_VIRTUAL_HAND_OUTPUT
            if args.include_hand_action
            else DEFAULT_VIRTUAL_OUTPUT
        )
    if actual_output_name is None:
        actual_output_name = (
            DEFAULT_ACTUAL_HAND_OUTPUT if args.include_hand_action else DEFAULT_ACTUAL_OUTPUT
        )
    virtual_output = output_dir / virtual_output_name
    actual_output = output_dir / actual_output_name
    hand_indices = parse_int_list(args.hand_indices)
    image_size = (args.image_width, args.image_height)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    require_output_path(virtual_output, args.overwrite)
    require_output_path(actual_output, args.overwrite)

    episode_dirs = sorted_episode_dirs(input_root)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found under {input_root}")

    skipped: list[tuple[str, str]] = []
    stats_list: list[dict[str, float]] = []
    output_demo_idx = 0

    with h5py.File(virtual_output, "w") as virtual_file, h5py.File(
        actual_output, "w"
    ) as actual_file:
        for h5_file, action_type in [
            (virtual_file, "virtual_target"),
            (actual_file, "actual_pose"),
        ]:
            h5_file.attrs["source_root"] = str(input_root)
            h5_file.attrs["camera"] = args.camera
            h5_file.attrs["target_hz"] = args.target_hz
            h5_file.attrs["source_hz"] = args.source_hz
            h5_file.attrs["hand_indices"] = ",".join(str(i) for i in hand_indices)
            h5_file.attrs["include_hand_action"] = args.include_hand_action
            h5_file.attrs["hand_action_dim"] = (
                len(hand_indices) if args.include_hand_action else 0
            )
            h5_file.attrs["wrench_source"] = "ft/wrench_raw.zarr"
            h5_file.attrs["wrench_offset_samples"] = args.wrench_offset_samples
            h5_file.attrs["wrench_ema_alpha"] = args.wrench_ema_alpha
            h5_file.attrs["wrench_history_len"] = args.wrench_history_len
            h5_file.attrs["wrench_history_dt"] = args.wrench_history_dt
            h5_file.attrs["wrench_history_sampling"] = "nearest_0.004s_grid"
            h5_file.attrs["virtual_rotation_source_order"] = (
                args.virtual_rotation_source_order
            )
            h5_file.attrs["virtual_rotation_target_order"] = (
                args.virtual_rotation_target_order
            )
            h5_file.attrs["action_type"] = action_type

        virtual_data = virtual_file.create_group("data")
        actual_data = actual_file.create_group("data")

        for episode_dir in tqdm.tqdm(episode_dirs, desc="Converting episodes"):
            try:
                obs_data, virtual_actions, actual_actions, stats = convert_episode(
                    episode_dir=episode_dir,
                    camera_name=args.camera,
                    hand_indices=hand_indices,
                    include_hand_action=args.include_hand_action,
                    target_hz=args.target_hz,
                    source_hz=args.source_hz,
                    wrench_offset_samples=args.wrench_offset_samples,
                    wrench_ema_alpha=args.wrench_ema_alpha,
                    wrench_history_len=args.wrench_history_len,
                    wrench_history_dt=args.wrench_history_dt,
                    virtual_rotation_source_order=args.virtual_rotation_source_order,
                    virtual_rotation_target_order=args.virtual_rotation_target_order,
                    image_size=image_size,
                )
            except Exception as exc:
                skipped.append((episode_dir.name, str(exc)))
                continue

            demo_name = f"demo_{output_demo_idx}"
            write_demo(
                virtual_data,
                demo_name,
                virtual_actions,
                obs_data,
                episode_dir.name,
            )
            write_demo(
                actual_data,
                demo_name,
                actual_actions,
                obs_data,
                episode_dir.name,
            )
            stats_list.append(stats)
            output_demo_idx += 1

    if stats_list:
        sample_counts = [stat["num_samples"] for stat in stats_list]
        print(
            "Converted",
            output_demo_idx,
            "episodes / samples min/max/total =",
            int(min(sample_counts)),
            int(max(sample_counts)),
            int(sum(sample_counts)),
        )
        print(
            "Max timestamp deltas (sec) rgb/hand/command/wrench =",
            max(stat["max_rgb_dt"] for stat in stats_list),
            max(stat["max_hand_dt"] for stat in stats_list),
            max(stat["max_command_dt"] for stat in stats_list),
            max(stat["max_wrench_dt"] for stat in stats_list),
        )
        print(
            "Max current wrench delta (sec) =",
            max(stat["max_current_wrench_dt"] for stat in stats_list),
        )
    if skipped:
        print("Skipped episodes:")
        for episode_name, reason in skipped:
            print(f"  {episode_name}: {reason}")

    print("Virtual target action output:", virtual_output)
    print("Actual pose action output:", actual_output)


if __name__ == "__main__":
    main()
