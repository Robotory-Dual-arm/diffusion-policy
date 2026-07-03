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
import shutil

import dill
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.residual_policy.pose_util import (
    abs_pose9_to_relative_pose9,
    delta6_from_base_to_target,
    pose6_to_pose9,
    pose_like_to_pose9,
    relative_pose9_to_abs_pose9,
)


def load_policy_from_ckpt(ckpt_path, device, use_ema=True):
    ckpt_path = pathlib.Path(ckpt_path).expanduser()
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key], strict=False)
    policy.to(device)
    policy.eval()
    del payload
    return cfg, policy


def sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def window_indices(t, n_obs_steps):
    start = int(t) - int(n_obs_steps) + 1
    return np.asarray([max(0, start + i) for i in range(n_obs_steps)], dtype=np.int64)


def image_to_chw_float(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    return np.moveaxis(image, -1, 1)


def to_torch_obs(obs_np, device):
    return dict_apply(
        obs_np,
        lambda x: torch.from_numpy(np.asarray(x)).to(device),
    )


def current_pose9_for_step(obs_group, t, arm):
    pos = np.asarray(obs_group[f"robot_pose_{arm}"])[t]
    quat = np.asarray(obs_group[f"robot_quat_{arm}"])[t]
    rotvec = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate([pos, rotvec], axis=-1))


def build_policy_obs(policy, obs_group, t, device):
    n_obs_steps = int(getattr(policy, "n_obs_steps", 1))
    idxs = window_indices(t, n_obs_steps)
    obs_np = {}

    for key in getattr(policy, "rgb_keys", []):
        if key not in obs_group:
            continue
        obs_np[key] = image_to_chw_float(np.asarray(obs_group[key])[idxs])[None]

    for key in getattr(policy, "low_dim_keys", []):
        if key not in obs_group:
            continue
        obs_np[key] = np.asarray(obs_group[key])[idxs].astype(np.float32)[None]

    for key in getattr(policy, "wrench_keys", []):
        if key not in obs_group:
            continue
        # Slow force policy consumes one force-history window at the latest step.
        obs_np[key] = np.asarray(obs_group[key])[t:t + 1].astype(np.float32)[None]

    return to_torch_obs(obs_np, device)


def collate_policy_obs(obs_list):
    keys = obs_list[0].keys()
    return {
        key: torch.cat([obs[key] for obs in obs_list], dim=0)
        for key in keys
    }


def slow_action_to_abs(slow_action, current_pose9, action_pose_repr):
    slow_action = pose_like_to_pose9(slow_action)
    if action_pose_repr == "relative":
        return relative_pose9_to_abs_pose9(current_pose9, slow_action)
    if action_pose_repr in ("abs", "absolute"):
        return slow_action
    raise ValueError(f"Unsupported slow action_pose_repr: {action_pose_repr}")


def replace_dataset(group, name, data):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


@torch.no_grad()
def predict_demo_slow_base(
        demo_group,
        policy,
        device,
        arm,
        action_pose_repr,
        slow_action_index,
        target_shift,
        batch_size):
    obs_group = demo_group["obs"]
    length = len(demo_group["actions"])
    if length <= target_shift:
        raise ValueError(
            f"{demo_group.name} length={length} is not longer than target_shift={target_shift}"
        )

    actual_target_abs = pose_like_to_pose9(np.asarray(obs_group["actual_target_abs"]))
    virtual_target_abs = pose_like_to_pose9(np.asarray(obs_group["virtual_target_abs"]))
    pred_target_abs = actual_target_abs.copy()

    # Prefix rows are unused when target_shift > 0, but keep them meaningful for
    # inspection and for target_shift=0 debugging.
    prefix_count = max(1, target_shift)
    prefix_action_count = min(prefix_count, int(getattr(policy, "n_action_steps", prefix_count)))
    if prefix_action_count > 0:
        slow_obs = build_policy_obs(policy, obs_group, 0, device)
        slow_result = policy.predict_action(slow_obs)
        prefix_actions = slow_result["action"][0, :prefix_action_count].detach().cpu().numpy()
        current_pose9 = current_pose9_for_step(obs_group, 0, arm)
        for j, action in enumerate(prefix_actions):
            pred_target_abs[j] = slow_action_to_abs(action, current_pose9, action_pose_repr)

    context_indices = np.arange(0, length - target_shift, dtype=np.int64)
    target_indices = context_indices + target_shift
    target_action_index = int(slow_action_index) + int(target_shift)

    desc = demo_group.name.split("/")[-1]
    for start in tqdm(range(0, len(context_indices), batch_size), desc=f"Predicting {desc}"):
        batch_context = context_indices[start:start + batch_size]
        batch_target = target_indices[start:start + batch_size]
        slow_obs = collate_policy_obs([
            build_policy_obs(policy, obs_group, int(t), device)
            for t in batch_context
        ])
        slow_result = policy.predict_action(slow_obs)
        if target_action_index >= slow_result["action"].shape[1]:
            raise IndexError(
                f"slow_action_index + target_shift = {target_action_index}, "
                f"but slow action length is {slow_result['action'].shape[1]}"
            )
        slow_actions = slow_result["action"][:, target_action_index].detach().cpu().numpy()

        for context_t, target_t, slow_action in zip(batch_context, batch_target, slow_actions):
            current_pose9 = current_pose9_for_step(obs_group, int(context_t), arm)
            pred_target_abs[int(target_t)] = slow_action_to_abs(
                slow_action,
                current_pose9,
                action_pose_repr,
            )

    pred_action_rel = np.zeros_like(pred_target_abs, dtype=np.float32)
    for target_t in range(length):
        context_t = max(0, target_t - target_shift)
        current_pose9 = current_pose9_for_step(obs_group, context_t, arm)
        pred_action_rel[target_t] = abs_pose9_to_relative_pose9(
            current_pose9,
            pred_target_abs[target_t],
        )

    residual_delta6 = delta6_from_base_to_target(
        pred_target_abs,
        virtual_target_abs,
    )
    actual_error_delta6 = delta6_from_base_to_target(
        pred_target_abs,
        actual_target_abs,
    )
    return {
        "slow_pred_target_abs": pred_target_abs.astype(np.float32),
        "slow_pred_action_rel": pred_action_rel.astype(np.float32),
        "residual_delta6_slow_pred_to_virtual": residual_delta6.astype(np.float32),
        "residual_delta6_slow_pred_to_actual": actual_error_delta6.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy a residual-policy HDF5 and add slow-predicted base-action keys "
            "for predicted-base fast residual training."
        )
    )
    parser.add_argument("--input", required=True, help="Source residual HDF5.")
    parser.add_argument("--output", required=True, help="Output HDF5 with slow-pred keys.")
    parser.add_argument("--slow-ckpt", required=True, help="Slow policy checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-shift", type=int, default=1)
    parser.add_argument("--slow-action-index", type=int, default=0)
    parser.add_argument("--arm", default="R")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument(
        "--full-action-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set slow n_action_steps to horizon - n_obs_steps + 1, matching real inference.",
    )
    parser.add_argument("--demo-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).expanduser()
    output_path = pathlib.Path(args.output).expanduser()
    if not input_path.is_absolute():
        input_path = pathlib.Path.cwd() / input_path
    if not output_path.is_absolute():
        output_path = pathlib.Path.cwd() / output_path
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must be different from input.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(input_path, output_path)

    device = torch.device(args.device)
    cfg, policy = load_policy_from_ckpt(args.slow_ckpt, device=device, use_ema=args.use_ema)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
    if args.full_action_steps:
        policy.n_action_steps = int(policy.horizon) - int(policy.n_obs_steps) + 1
    action_pose_repr = getattr(
        policy,
        "action_pose_repr",
        OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default="relative"),
    )

    print("Input:", input_path)
    print("Output:", output_path)
    print("Slow ckpt:", args.slow_ckpt)
    print("Slow action_pose_repr:", action_pose_repr)
    print("Slow n_obs_steps:", getattr(policy, "n_obs_steps", None))
    print("Slow n_action_steps:", getattr(policy, "n_action_steps", None))
    print("Target shift:", args.target_shift)

    with h5py.File(output_path, "r+") as f:
        data_group = f["data"]
        demo_keys = sorted_demo_keys(data_group)
        if args.demo_limit is not None:
            demo_keys = demo_keys[:args.demo_limit]

        for demo_name in demo_keys:
            demo = data_group[demo_name]
            pred = predict_demo_slow_base(
                demo_group=demo,
                policy=policy,
                device=device,
                arm=args.arm,
                action_pose_repr=action_pose_repr,
                slow_action_index=args.slow_action_index,
                target_shift=args.target_shift,
                batch_size=args.batch_size,
            )
            obs = demo["obs"]
            for key, value in pred.items():
                replace_dataset(obs, key, value)

        f.attrs["obs/slow_pred_target_abs"] = (
            "slow-predicted next target pose9; row j>=target_shift is predicted from obs[j-target_shift]"
        )
        f.attrs["obs/slow_pred_action_rel"] = (
            "slow_pred_target_abs expressed relative to obs[j-target_shift] current pose"
        )
        f.attrs["obs/residual_delta6_slow_pred_to_virtual"] = (
            "delta6_from_slow_pred_target_abs_to_virtual_target_abs"
        )
        f.attrs["obs/residual_delta6_slow_pred_to_actual"] = (
            "delta6_from_slow_pred_target_abs_to_actual_target_abs; diagnostic only"
        )
        f.attrs["slow_pred_ckpt"] = str(args.slow_ckpt)
        f.attrs["slow_pred_target_shift"] = int(args.target_shift)
        f.attrs["slow_pred_action_index"] = int(args.slow_action_index)
        f.attrs["slow_pred_num_inference_steps"] = int(args.num_inference_steps)
        f.attrs["slow_pred_full_action_steps"] = bool(args.full_action_steps)

    print("Wrote predicted-base fast dataset:", output_path)


if __name__ == "__main__":
    main()
