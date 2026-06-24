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
import tqdm
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.residual_policy.pose_util import (
    current_obs_to_pose9,
    delta6_from_base_to_target,
    pose_like_to_pose9,
    relative_pose9_to_abs_pose9,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_path(path):
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        p = pathlib.Path.cwd() / p
    return p.resolve()


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


def make_dataset_from_slow_cfg(slow_cfg, dataset_path):
    dataset_cfg = OmegaConf.to_container(slow_cfg.task.dataset, resolve=True)
    dataset_cfg["dataset_path"] = str(dataset_path)
    dataset_cfg["use_cache"] = False
    dataset_cfg["val_ratio"] = 0.0
    dataset_cfg["seed"] = 42
    return hydra.utils.instantiate(OmegaConf.create(dataset_cfg))


def sample_current_global_idx(sampler_index, n_obs_steps):
    buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sampler_index
    pos_in_sample = n_obs_steps - 1
    if pos_in_sample < sample_start_idx:
        return int(buffer_start_idx)
    if pos_in_sample >= sample_end_idx:
        return int(buffer_end_idx - 1)
    return int(buffer_start_idx + pos_in_sample - sample_start_idx)


def global_to_demo_local(episode_ends, global_idx):
    demo_idx = int(np.searchsorted(episode_ends, global_idx, side="right"))
    start = 0 if demo_idx == 0 else int(episode_ends[demo_idx - 1])
    return demo_idx, int(global_idx - start)


def read_key(demo_group, key):
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


def current_env_obs_from_source(demo_group, local_idx):
    obs_group = demo_group["obs"]
    out = {}
    for key, dataset in obs_group.items():
        arr = np.asarray(dataset)
        if len(arr) <= local_idx:
            raise IndexError(f"{demo_group.name}/obs/{key} has len {len(arr)}, requested {local_idx}")
        out[key] = arr[local_idx:local_idx + 1]
    return out


def predict_slow_actions(dataset, policy, device, batch_size, slow_action_index):
    n = len(dataset)
    slow_action_rel = [None] * n
    global_indices = [None] * n
    for start in tqdm.trange(0, n, batch_size, desc="Predicting slow actions"):
        end = min(start + batch_size, n)
        samples = [dataset[i] for i in range(start, end)]
        obs_batch = {}
        for key in samples[0]["obs"]:
            obs_batch[key] = torch.stack([sample["obs"][key] for sample in samples], dim=0)
        obs_batch = dict_apply(obs_batch, lambda x: x.to(device))
        with torch.no_grad():
            result = policy.predict_action(obs_batch)
            action = result["action"][:, slow_action_index].detach().cpu().numpy()
        for offset, sample_idx in enumerate(range(start, end)):
            slow_action_rel[sample_idx] = action[offset].astype(np.float32)
            global_indices[sample_idx] = sample_current_global_idx(
                dataset.sampler.indices[sample_idx],
                dataset.n_obs_steps,
            )
    return global_indices, slow_action_rel


def copy_obs_group(src_demo, dst_demo):
    src_obs = src_demo["obs"]
    dst_obs = dst_demo.create_group("obs")
    for key in src_obs:
        dst_obs.create_dataset(key, data=np.asarray(src_obs[key]))
    return dst_obs


def main():
    parser = argparse.ArgumentParser(
        description="Create a stage-2 fast residual HDF5 from an existing diffusion HDF5 and a trained slow policy."
    )
    parser.add_argument("--input", required=True, help="Source diffusion HDF5.")
    parser.add_argument("--output", required=True, help="Output fast residual HDF5.")
    parser.add_argument("--slow-ckpt", required=True, help="Trained slow policy checkpoint.")
    parser.add_argument(
        "--virtual-key",
        required=True,
        help="Virtual target dataset path inside each demo, e.g. actions, obs/desired_pose, obs/virtual_target_pose.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--arm", default="R", choices=["L", "R"])
    parser.add_argument("--slow-use-ema", action="store_true", default=True)
    parser.add_argument("--no-slow-ema", dest="slow_use_ema", action="store_false")
    parser.add_argument("--slow-action-index", type=int, default=0)
    parser.add_argument(
        "--target-index-offset",
        type=int,
        default=0,
        help="Offset applied when reading virtual target. Use 1 if the target key is one step ahead.",
    )
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
    slow_action_pose_repr = slow_cfg.task.pose_repr.get("action_pose_repr", "abs")
    dataset = make_dataset_from_slow_cfg(slow_cfg, input_path)
    global_indices, slow_rel_by_sample = predict_slow_actions(
        dataset=dataset,
        policy=slow_policy,
        device=device,
        batch_size=args.batch_size,
        slow_action_index=args.slow_action_index,
    )

    episode_ends = dataset.replay_buffer.episode_ends[:]
    demo_names = None
    with h5py.File(input_path, "r") as src:
        demo_names = sorted_demo_keys(src["data"])

    slow_rel = [None for _ in demo_names]
    slow_abs = [None for _ in demo_names]
    virtual_abs = [None for _ in demo_names]
    residual_slow_pred_to_virtual = [None for _ in demo_names]
    seen = [None for _ in demo_names]

    with h5py.File(input_path, "r") as src:
        for demo_idx, demo_name in enumerate(demo_names):
            demo = src["data"][demo_name]
            length = len(demo["actions"])
            slow_rel[demo_idx] = np.zeros((length, 9), dtype=np.float32)
            slow_abs[demo_idx] = np.zeros((length, 9), dtype=np.float32)
            virtual_abs[demo_idx] = np.zeros((length, 9), dtype=np.float32)
            residual_slow_pred_to_virtual[demo_idx] = np.zeros((length, 6), dtype=np.float32)
            seen[demo_idx] = np.zeros((length,), dtype=bool)

        for sample_idx, global_idx in enumerate(tqdm.tqdm(global_indices, desc="Composing residual targets")):
            demo_idx, local_idx = global_to_demo_local(episode_ends, global_idx)
            demo = src["data"][demo_names[demo_idx]]
            length = len(demo["actions"])
            target_idx = int(np.clip(local_idx + args.target_index_offset, 0, length - 1))
            env_obs = current_env_obs_from_source(demo, local_idx)
            current_pose9 = current_obs_to_pose9(env_obs, arm=args.arm)
            this_slow_rel = pose_like_to_pose9(slow_rel_by_sample[sample_idx])
            if slow_action_pose_repr == "relative":
                this_slow_abs = relative_pose9_to_abs_pose9(current_pose9, this_slow_rel)
            elif slow_action_pose_repr == "abs":
                this_slow_abs = this_slow_rel
            else:
                raise ValueError(f"Unsupported slow action_pose_repr: {slow_action_pose_repr}")

            this_virtual_abs = pose_like_to_pose9(read_key(demo, args.virtual_key)[target_idx])
            this_residual_slow_pred_to_virtual = delta6_from_base_to_target(
                this_slow_abs,
                this_virtual_abs,
            )

            slow_rel[demo_idx][local_idx] = this_slow_rel
            slow_abs[demo_idx][local_idx] = this_slow_abs
            virtual_abs[demo_idx][local_idx] = this_virtual_abs
            residual_slow_pred_to_virtual[demo_idx][local_idx] = this_residual_slow_pred_to_virtual
            seen[demo_idx][local_idx] = True

    for demo_idx, demo_name in enumerate(demo_names):
        if not np.all(seen[demo_idx]):
            missing = np.flatnonzero(~seen[demo_idx])[:10]
            raise RuntimeError(f"Missing slow predictions for {demo_name}, first missing indices: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        dst.attrs.update(src.attrs)
        dst.attrs["residual_action"] = "delta6_from_slow_pred_abs_to_virtual_abs"
        dst.attrs["slow_ckpt"] = str(slow_ckpt)
        dst.attrs["virtual_key"] = args.virtual_key
        dst_data = dst.create_group("data")
        for demo_idx, demo_name in enumerate(tqdm.tqdm(demo_names, desc="Writing residual hdf5")):
            src_demo = src["data"][demo_name]
            dst_demo = dst_data.create_group(demo_name)
            dst_obs = copy_obs_group(src_demo, dst_demo)
            dst_obs.create_dataset("slow_action_rel", data=slow_rel[demo_idx])
            dst_obs.create_dataset("slow_action_abs", data=slow_abs[demo_idx])
            dst_obs.create_dataset("virtual_target_abs", data=virtual_abs[demo_idx])
            dst_obs.create_dataset(
                "residual_delta6_slow_pred_to_virtual",
                data=residual_slow_pred_to_virtual[demo_idx],
            )
            dst_demo.create_dataset("actions", data=residual_slow_pred_to_virtual[demo_idx])

    print(f"Wrote fast residual dataset: {output_path}")


if __name__ == "__main__":
    main()
