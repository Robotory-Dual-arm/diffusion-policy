#!/usr/bin/env python3
"""Compare training tensors for UNet and Transformer diffusion policies.

This follows the important parts of each policy's compute_loss path without
doing an optimizer step. It is meant for checking whether the dataset sample,
normalized action trajectory, diffusion target, and model inputs line up.
"""

import argparse
import copy
import pathlib
import random
import shutil
import sys
from typing import Any, Dict

import hydra
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    array_to_stats_for_wrench,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


def ensure_eval_resolver():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def compose_config(config_name: str, task: str, overrides):
    ensure_eval_resolver()
    config_path = pathlib.Path(__file__).resolve().parents[1] / "diffusion_policy" / "config"
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[f"task={task}"] + list(overrides),
        )
    OmegaConf.resolve(cfg)
    return cfg


def maybe_repoint_dataset_cache(cfg, cache_dir):
    if cache_dir is None:
        return
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_base = pathlib.Path(cfg.task.dataset.dataset_path)
    src_zip = pathlib.Path(str(src_base) + ".zarr.zip")
    dst_base = cache_dir / src_base.name
    dst_zip = pathlib.Path(str(dst_base) + ".zarr.zip")
    if not src_zip.exists():
        raise FileNotFoundError(f"Missing source zarr cache: {src_zip}")
    if (not dst_zip.exists()) or (dst_zip.stat().st_size != src_zip.stat().st_size):
        print(f"Copying zarr cache to {dst_zip}")
        shutil.copy2(src_zip, dst_zip)
    cfg.task.dataset.dataset_path = str(dst_base)
    cfg.task.dataset.use_cache = True


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    return dict_apply(batch, lambda x: x.clone() if torch.is_tensor(x) else copy.deepcopy(x))


def summarize_tensor(name: str, x: torch.Tensor, max_values: int = 8):
    x_cpu = x.detach().float().cpu()
    flat = x_cpu.reshape(-1)
    sample = flat[:max_values].numpy()
    print(
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={flat.min().item():.6g} max={flat.max().item():.6g} "
        f"mean={flat.mean().item():.6g} std={flat.std(unbiased=False).item():.6g}"
    )
    print(f"  first{len(sample)}={np.array2string(sample, precision=5, suppress_small=False)}")


def max_abs_diff(a: torch.Tensor, b: torch.Tensor):
    if tuple(a.shape) != tuple(b.shape):
        return f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a.detach().cpu() - b.detach().cpu()).abs().max())


def get_normalizer_single_process(dataset) -> LinearNormalizer:
    """Same intent as BaeRobomimicReplayDataset.get_normalizer, but no workers."""
    normalizer = LinearNormalizer()
    data_cache = {key: list() for key in dataset.lowdim_keys + dataset.wrench_keys + ["action"]}
    dataset.sampler.ignore_rgb(True)
    dataloader = DataLoader(dataset=dataset, batch_size=64, num_workers=0)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(f"normalizer scan batch {batch_idx}/{len(dataloader)}")
        for key in dataset.lowdim_keys:
            data_cache[key].append(copy.deepcopy(batch["obs"][key]))
        for key in dataset.wrench_keys:
            data_cache[key].append(copy.deepcopy(batch["obs"][key]))
        data_cache["action"].append(copy.deepcopy(batch["action"]))
    dataset.sampler.ignore_rgb(False)

    wrench_history = None
    for key in data_cache.keys():
        data_cache[key] = np.concatenate(data_cache[key])
        assert data_cache[key].shape[0] == len(dataset.sampler)
        if key in dataset.lowdim_keys or key == "action":
            b, t, d = data_cache[key].shape
            data_cache[key] = data_cache[key].reshape(b * t, d)
        elif key in dataset.wrench_keys:
            b, t, c, h = data_cache[key].shape
            wrench_history = h
            data_cache[key] = data_cache[key].reshape(b * t, c, h)

    action_normalizers = list()
    action_index = 0
    if dataset.use_left_arm:
        action_normalizers.append(
            get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., action_index:action_index + 3]))
        )
        action_normalizers.append(
            get_identity_normalizer_from_stat(array_to_stats(data_cache["action"][..., action_index + 3:action_index + 9]))
        )
        action_index += 9
    if dataset.use_right_arm:
        action_normalizers.append(
            get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., action_index:action_index + 3]))
        )
        action_normalizers.append(
            get_identity_normalizer_from_stat(array_to_stats(data_cache["action"][..., action_index + 3:action_index + 9]))
        )
        action_index += 9
    if dataset.use_left_hand or dataset.use_right_hand:
        action_normalizers.append(
            get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., action_index:]))
        )
    normalizer["action"] = concatenate_normalizer(action_normalizers)

    for key in dataset.lowdim_keys:
        stat = array_to_stats(data_cache[key])
        if key.endswith("pos") or "pose" in key:
            this_normalizer = get_range_normalizer_from_stat(stat)
        elif key.endswith("quat") or "quat" in key:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        elif key.endswith("qpos"):
            this_normalizer = get_range_normalizer_from_stat(stat)
        elif "wrench" in key or "force" in key or "torque" in key:
            this_normalizer = get_range_normalizer_from_stat(stat)
        else:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer[key] = this_normalizer

    for key in dataset.wrench_keys:
        stat = array_to_stats_for_wrench(data_cache[key], history=wrench_history)
        normalizer[key] = get_range_normalizer_from_stat(stat)

    for key in dataset.rgb_keys:
        normalizer[key] = get_image_range_normalizer()
    return normalizer


def apply_policy_aug(batch: Dict[str, Any], seed: int):
    # Same augmentation block as both policy compute_loss implementations.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    transform = T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        T.RandomGrayscale(p=0.005),
    ])
    num_image = len([key for key in batch["obs"].keys() if "image" in key])
    for i in range(num_image):
        batch["obs"][f"image{i}"] = transform(batch["obs"][f"image{i}"])


@torch.no_grad()
def trace_policy(policy, batch, kind: str, aug_seed: int, noise_seed: int):
    policy.train()
    batch = clone_batch(batch)
    apply_policy_aug(batch, aug_seed)

    nobs = policy.normalizer.normalize(batch["obs"])
    nactions = policy.normalizer["action"].normalize(batch["action"])
    batch_size = nactions.shape[0]
    horizon = nactions.shape[1]

    out = {
        "nobs": nobs,
        "nactions": nactions,
        "trajectory": nactions,
    }

    if kind == "unet":
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if policy.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = policy.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = policy.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        condition_mask = policy.mask_generator(trajectory.shape)
        model_kwargs = {"local_cond": local_cond, "global_cond": global_cond}
        model_call = lambda noisy, ts: policy.model(noisy, ts, **model_kwargs)
        out.update({
            "this_nobs": this_nobs,
            "nobs_features": nobs_features,
            "global_cond": global_cond,
            "cond": None,
            "cond_data": cond_data,
            "trajectory": trajectory,
            "condition_mask": condition_mask,
            "model_kwargs": model_kwargs,
        })

    elif kind == "transformer":
        cond = None
        trajectory = nactions
        if policy.obs_as_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = policy.obs_encoder(this_nobs)
            cond = nobs_features.reshape(batch_size, policy.n_obs_steps, -1)
            if policy.pred_action_steps_only:
                start = policy.n_obs_steps - 1
                end = start + policy.n_action_steps
                trajectory = nactions[:, start:end]
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = policy.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()
        if policy.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = policy.mask_generator(trajectory.shape)
        model_call = lambda noisy, ts: policy.model(noisy, ts, cond)
        out.update({
            "this_nobs": this_nobs,
            "nobs_features": nobs_features,
            "global_cond": None,
            "cond": cond,
            "cond_data": trajectory,
            "trajectory": trajectory,
            "condition_mask": condition_mask,
            "model_kwargs": {"cond": cond},
        })
    else:
        raise ValueError(kind)

    torch.manual_seed(noise_seed)
    noise = torch.randn(out["trajectory"].shape, device=out["trajectory"].device)
    timesteps = torch.randint(
        0,
        policy.noise_scheduler.config.num_train_timesteps,
        (out["trajectory"].shape[0],),
        device=out["trajectory"].device,
    ).long()
    noisy_trajectory = policy.noise_scheduler.add_noise(out["trajectory"], noise, timesteps)
    loss_mask = ~out["condition_mask"]
    noisy_trajectory[out["condition_mask"]] = out["cond_data"][out["condition_mask"]]
    pred = model_call(noisy_trajectory, timesteps)

    pred_type = policy.noise_scheduler.config.prediction_type
    if pred_type == "epsilon":
        target = noise
    elif pred_type == "sample":
        target = out["trajectory"]
    elif pred_type == "v_prediction":
        target = policy.noise_scheduler.get_velocity(out["trajectory"], noise, timesteps)
    else:
        raise ValueError(f"Unsupported prediction_type: {pred_type}")

    out.update({
        "noise": noise,
        "timesteps": timesteps,
        "noisy_trajectory": noisy_trajectory,
        "loss_mask": loss_mask,
        "pred": pred,
        "target": target,
    })
    return out


def print_trace(label: str, trace: Dict[str, Any]):
    print(f"\n## {label}")
    summarize_tensor("nactions", trace["nactions"])
    summarize_tensor("trajectory/model_input_clean", trace["trajectory"])
    summarize_tensor("noisy_trajectory/model_input_noisy", trace["noisy_trajectory"])
    summarize_tensor("target/model_target", trace["target"])
    summarize_tensor("pred/model_output", trace["pred"])
    print(f"timesteps={trace['timesteps'].detach().cpu().tolist()}")
    print(f"condition_mask true count={int(trace['condition_mask'].sum().item())}")
    print(f"loss_mask true count={int(trace['loss_mask'].sum().item())}")
    summarize_tensor("nobs_features", trace["nobs_features"])
    if trace["global_cond"] is not None:
        summarize_tensor("global_cond", trace["global_cond"])
    if trace["cond"] is not None:
        summarize_tensor("cond", trace["cond"])
    for key, value in trace["this_nobs"].items():
        summarize_tensor(f"this_nobs[{key}]", value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bbbae_dualarm_insert_plug_no_wrench")
    parser.add_argument("--unet-config", default="bae_train_diffusion_unet_real_hybrid_workspace")
    parser.add_argument("--transformer-config", default="bae_train_diffusion_transformer_real_hybrid_workspace")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--aug-seed", type=int, default=123)
    parser.add_argument("--noise-seed", type=int, default=456)
    parser.add_argument("--init-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-dir", default="/tmp/diffusion_policy_cache")
    parser.add_argument("--unet-override", action="append", default=[])
    parser.add_argument("--transformer-override", action="append", default=[])
    args = parser.parse_args()

    common_overrides = [
        f"task.dataset.use_cache={str(args.use_cache).lower()}",
        "dataloader.num_workers=0",
        "val_dataloader.num_workers=0",
    ]
    unet_cfg = compose_config(args.unet_config, args.task, common_overrides + args.unet_override)
    transformer_cfg = compose_config(
        args.transformer_config,
        args.task,
        common_overrides + args.transformer_override,
    )
    if args.use_cache:
        maybe_repoint_dataset_cache(unet_cfg, args.cache_dir)
        maybe_repoint_dataset_cache(transformer_cfg, args.cache_dir)

    print("# Config Summary")
    print(f"task={args.task}")
    print(f"unet crop_shape={unet_cfg.policy.crop_shape}, obs_as_global_cond={unet_cfg.policy.obs_as_global_cond}")
    print(f"transformer crop_shape={transformer_cfg.policy.crop_shape}, obs_as_cond={transformer_cfg.policy.obs_as_cond}")
    print(f"dataset pose_repr={OmegaConf.to_container(unet_cfg.task.pose_repr, resolve=True)}")

    dataset = hydra.utils.instantiate(unet_cfg.task.dataset)
    print(f"dataset_len={len(dataset)}")
    normalizer = get_normalizer_single_process(dataset)

    indices = list(range(args.start_index, args.start_index + args.batch_size))
    batch = default_collate([dataset[i] for i in indices])

    print("\n# Dataset Batch Before Normalizer")
    print(f"indices={indices}")
    summarize_tensor("batch[action]", batch["action"])
    for key, value in batch["obs"].items():
        summarize_tensor(f"batch[obs][{key}]", value)

    torch.manual_seed(args.init_seed)
    np.random.seed(args.init_seed)
    random.seed(args.init_seed)
    unet = hydra.utils.instantiate(unet_cfg.policy)

    torch.manual_seed(args.init_seed)
    np.random.seed(args.init_seed)
    random.seed(args.init_seed)
    transformer = hydra.utils.instantiate(transformer_cfg.policy)

    device = torch.device(args.device)
    unet.set_normalizer(normalizer)
    transformer.set_normalizer(normalizer)
    unet.to(device)
    transformer.to(device)
    batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)

    unet_trace = trace_policy(unet, batch, "unet", args.aug_seed, args.noise_seed)
    transformer_trace = trace_policy(transformer, batch, "transformer", args.aug_seed, args.noise_seed)

    print_trace("UNet Compute-Loss Tensors", unet_trace)
    print_trace("Transformer Compute-Loss Tensors", transformer_trace)

    print("\n# Direct Comparisons")
    for key in ["nactions", "trajectory", "noise", "timesteps", "noisy_trajectory", "target", "loss_mask"]:
        print(f"{key}: {max_abs_diff(unet_trace[key].float(), transformer_trace[key].float())}")
    for obs_key in unet_trace["this_nobs"].keys():
        print(
            f"this_nobs[{obs_key}]: "
            f"{max_abs_diff(unet_trace['this_nobs'][obs_key].float(), transformer_trace['this_nobs'][obs_key].float())}"
        )
    print(f"nobs_features: {max_abs_diff(unet_trace['nobs_features'].float(), transformer_trace['nobs_features'].float())}")
    print("UNet condition form: global_cond shape", None if unet_trace["global_cond"] is None else tuple(unet_trace["global_cond"].shape))
    print("Transformer condition form: cond shape", None if transformer_trace["cond"] is None else tuple(transformer_trace["cond"].shape))


if __name__ == "__main__":
    main()
