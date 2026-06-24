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

import dill
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from diffusion_policy.residual_policy.pose_util import (
    delta6_from_base_to_target,
    pose6_to_pose9,
    pose_like_to_pose9,
    relative_pose9_to_abs_pose9,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_path(path):
    path = pathlib.Path(path).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    return path.resolve()


def sorted_demo_keys(data_group):
    def demo_idx(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(data_group.keys(), key=demo_idx)


def load_policy_from_ckpt(ckpt_path, device, use_ema=True):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key], strict=False)
    policy.to(device)
    policy.eval()
    del payload
    return cfg, policy


def read_demo_key(demo_group, key):
    if key == "action":
        key = "actions"
    if "/" in key:
        out = demo_group
        for part in key.split("/"):
            out = out[part]
        return np.asarray(out)
    if key in demo_group:
        return np.asarray(demo_group[key])
    if "obs" in demo_group and key in demo_group["obs"]:
        return np.asarray(demo_group["obs"][key])
    raise KeyError(f"Could not find key '{key}' in {demo_group.name}")


def window_indices(t, n_obs_steps):
    start = t - n_obs_steps + 1
    return np.asarray([max(0, start + i) for i in range(n_obs_steps)], dtype=np.int64)


def image_to_chw_float(images):
    images = np.asarray(images)
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    else:
        images = images.astype(np.float32)
    return np.moveaxis(images, -1, 1)


def current_pose9(obs_group, t, arm):
    pos = np.asarray(obs_group[f"robot_pose_{arm}"])[t]
    quat = np.asarray(obs_group[f"robot_quat_{arm}"])[t]
    rotvec = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate([pos, rotvec], axis=-1))


def build_slow_obs_batch(policy, obs_group, indices, device):
    n_obs_steps = int(getattr(policy, "n_obs_steps", 1))
    obs_np = {}

    for key in getattr(policy, "rgb_keys", []):
        if key not in obs_group:
            continue
        seq = [
            image_to_chw_float(np.asarray(obs_group[key])[window_indices(int(t), n_obs_steps)])
            for t in indices
        ]
        obs_np[key] = np.stack(seq, axis=0)

    for key in getattr(policy, "low_dim_keys", []):
        if key not in obs_group:
            continue
        seq = [
            np.asarray(obs_group[key])[window_indices(int(t), n_obs_steps)].astype(np.float32)
            for t in indices
        ]
        obs_np[key] = np.stack(seq, axis=0)

    for key in getattr(policy, "wrench_keys", []):
        if key not in obs_group:
            continue
        # Wrench encoder policies use one latest force-history token.
        seq = [
            np.asarray(obs_group[key])[int(t):int(t) + 1].astype(np.float32)
            for t in indices
        ]
        obs_np[key] = np.stack(seq, axis=0)

    return {
        key: torch.from_numpy(value).to(device=device, dtype=torch.float32)
        for key, value in obs_np.items()
    }


def copy_demo(src_demo, dst_demo):
    for key in src_demo:
        if key == "obs":
            continue
        dst_demo.create_dataset(key, data=np.asarray(src_demo[key]))
    src_obs = src_demo["obs"]
    dst_obs = dst_demo.create_group("obs")
    for key in src_obs:
        dst_obs.create_dataset(key, data=np.asarray(src_obs[key]))
    return dst_obs


def create_or_replace(group, name, data):
    if name in group:
        del group[name]
    return group.create_dataset(name, data=data)


def main():
    parser = argparse.ArgumentParser(
        description="Create fast residual targets from frozen slow predictions to virtual targets."
    )
    parser.add_argument("--input", required=True, help="Converted slow/residual HDF5 dataset.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--slow-ckpt", required=True, help="Trained slow policy checkpoint.")
    parser.add_argument("--virtual-key", default="obs/virtual_target_abs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--arm", default="R", choices=["L", "R"])
    parser.add_argument("--slow-action-index", type=int, default=0)
    parser.add_argument("--target-index-offset", type=int, default=0)
    parser.add_argument("--slow-use-ema", action="store_true", default=True)
    parser.add_argument("--no-slow-ema", dest="slow_use_ema", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    slow_ckpt = resolve_path(args.slow_ckpt)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    slow_cfg, slow_policy = load_policy_from_ckpt(slow_ckpt, device=device, use_ema=args.slow_use_ema)
    slow_action_pose_repr = OmegaConf.select(
        slow_cfg,
        "task.pose_repr.action_pose_repr",
        default=getattr(slow_policy, "action_pose_repr", "relative"),
    )
    if hasattr(slow_policy, "num_inference_steps"):
        slow_policy.num_inference_steps = 16
    if all(hasattr(slow_policy, key) for key in ("horizon", "n_obs_steps")):
        slow_policy.n_action_steps = slow_policy.horizon - slow_policy.n_obs_steps + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        dst.attrs.update(src.attrs)
        dst.attrs["residual_action"] = "delta6_from_slow_pred_abs_to_virtual_abs"
        dst.attrs["slow_ckpt"] = str(slow_ckpt)
        dst.attrs["virtual_key"] = args.virtual_key
        dst.attrs["slow_action_index"] = int(args.slow_action_index)
        dst.attrs["target_index_offset"] = int(args.target_index_offset)

        dst_data = dst.create_group("data")
        for demo_name in sorted_demo_keys(src["data"]):
            src_demo = src["data"][demo_name]
            src_obs = src_demo["obs"]
            dst_demo = dst_data.create_group(demo_name)
            dst_obs = copy_demo(src_demo, dst_demo)

            length = len(src_demo["actions"])
            slow_rel = np.zeros((length, 9), dtype=np.float32)
            slow_abs = np.zeros((length, 9), dtype=np.float32)
            virtual_abs = np.zeros((length, 9), dtype=np.float32)
            residual = np.zeros((length, 6), dtype=np.float32)
            virtual_all = read_demo_key(src_demo, args.virtual_key)

            indices = np.arange(length, dtype=np.int64)
            for start in tqdm(range(0, length, args.batch_size), desc=f"{demo_name} slow pred"):
                batch_indices = indices[start:start + args.batch_size]
                slow_obs = build_slow_obs_batch(
                    policy=slow_policy,
                    obs_group=src_obs,
                    indices=batch_indices,
                    device=device,
                )
                with torch.no_grad():
                    result = slow_policy.predict_action(slow_obs)
                    slow_action = result["action"][:, args.slow_action_index].detach().cpu().numpy()

                for offset, local_idx in enumerate(batch_indices):
                    local_idx = int(local_idx)
                    target_idx = int(np.clip(local_idx + args.target_index_offset, 0, length - 1))
                    this_current = current_pose9(src_obs, local_idx, args.arm)
                    this_slow_rel = pose_like_to_pose9(slow_action[offset])
                    if slow_action_pose_repr == "relative":
                        this_slow_abs = relative_pose9_to_abs_pose9(this_current, this_slow_rel)
                    elif slow_action_pose_repr == "abs":
                        this_slow_abs = this_slow_rel
                    else:
                        raise ValueError(f"Unsupported slow action_pose_repr: {slow_action_pose_repr}")

                    this_virtual = pose_like_to_pose9(virtual_all[target_idx])
                    slow_rel[local_idx] = this_slow_rel
                    slow_abs[local_idx] = this_slow_abs
                    virtual_abs[local_idx] = this_virtual
                    residual[local_idx] = delta6_from_base_to_target(this_slow_abs, this_virtual)

            create_or_replace(dst_obs, "slow_action_rel", slow_rel)
            create_or_replace(dst_obs, "slow_action_abs", slow_abs)
            create_or_replace(dst_obs, "virtual_target_abs", virtual_abs)
            create_or_replace(dst_obs, "residual_delta6_slow_pred_to_virtual", residual)
            if "actions" in dst_demo:
                del dst_demo["actions"]
            dst_demo.create_dataset("actions", data=residual)

    print(f"Wrote slow-pred residual dataset: {output_path}")


if __name__ == "__main__":
    main()
