#!/home/vision/anaconda3/envs/robodiff/bin/python

"""
Residual slow-fast real-robot evaluation.

High-level structure
--------------------
This script executes a frozen slow policy plus a fast residual policy:

    slow policy:
        image / proprioception -> absolute target-pose chunk

    fast policy:
        latest obs + force + base_action_rel -> residual_delta6

    executed command:
        final_abs_target = slow_abs_target + fast_residual_delta6

The fast policy only schedules one command per 10 Hz control tick. This is
intentional: force/contact feedback should be refreshed every fast tick, not
predicted open-loop for a whole chunk.


Default timing mode: fixed-time chunks
--------------------------------------
The default is now --fixed_time_chunks. The goal is to keep slow chunks aligned
to a single episode-wide time grid instead of letting each new slow inference
choose its own obs timestamp as the anchor.

Default timing values:

    frequency = 10 Hz
    dt = 0.1 sec
    steps_per_inference = 6
    slow_tail_steps = 3
    action_target_shift = task.dataset.action_target_shift, usually 1

With these values, slow chunk starts are fixed to global input steps:

    chunk 0: input steps 0..5, targets 1..6
    chunk 1: input steps 6..11, targets 7..12
    chunk 2: input steps 12..17, targets 13..18

Visually:

    time grid      t0    .1    .2    .3    .4    .5    .6    .7
                   |-----|-----|-----|-----|-----|-----|-----|
    slow chunk 0         a1    a2    a3    a4    a5    a6
    fast residual        r0    r1    r2    r3    r4    r5
    command              a1+r0 a2+r1 a3+r2 a4+r3 a5+r4 a6+r5
    slow chunk 1                                             starts at .6

action_target_shift=1 means fast input step k predicts the residual for slow
target k+1. That is why the executable target range starts at a1 rather than a0.


Why previous-chunk tail fallback exists
---------------------------------------
Slow diffusion inference can be late. If the next slow chunk is not ready at
the nominal boundary, sending no command makes the robot hold the previous
target pose, which can look like:

    move -> hold -> jump -> hold -> jump

That is especially bad in contact tasks. In fixed-time mode this script requests
the next slow chunk in a background thread at the chunk boundary. While the new
chunk is still pending, the fast loop may keep using the previous slow chunk's
tail:

    nominal chunk use:  a1..a6
    fallback tail:     a7..a9   when slow_tail_steps=3

So the command stream can stay close to 0.1 sec spacing even if slow inference
finishes slightly late. The tail is bounded because old slow targets become
less trustworthy as contact state drifts.

If even the tail is exhausted before the next slow chunk is ready, this script
does not invent a new action. It logs a warning and leaves that grid tick
without a new command.


GRU fast policy behavior
------------------------
GRU hidden state is reset at each installed slow chunk. This is deliberate for
now: fast was trained on finite windows/chunks, not on full-episode recurrent
rollouts.

If latency causes some early target steps to be already stale, this script still
advances the fast policy through those missed steps without sending robot
commands. This keeps the GRU hidden state aligned with the local step index:

    missed step 0 -> predict for hidden update only
    missed step 1 -> predict for hidden update only
    fresh step 2  -> predict and send command if timestamp is still feasible


Event-driven fallback mode
--------------------------
Use --event_driven_chunks to return to the older behavior. In that mode, slow is
replanned when the current chunk is consumed, and each new chunk is anchored to
the latest observation timestamp. This is useful for comparison/debugging, but
can make neighboring chunks slightly misaligned in wall-clock time.


Useful commands
---------------
Basic fixed-time execution:

    python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
      --input data/outputs/YYYY.MM.DD/FAST_RUN/checkpoints/latest.ckpt \
      --slow_ckpt_path data/outputs/YYYY.MM.DD/SLOW_RUN/checkpoints/latest.ckpt \
      --output data/results/residual_insert_plug \
      --steps_per_inference 6 \
      --slow_tail_steps 3

Use PiGDM realtime chunking for the slow policy:

    python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
      --input data/outputs/YYYY.MM.DD/FAST_RUN/checkpoints/latest.ckpt \
      --slow_ckpt_path data/outputs/YYYY.MM.DD/SLOW_RUN/checkpoints/latest.ckpt \
      --output data/results/residual_insert_plug \
      --slow_use_pigdm \
      --steps_per_inference 6 \
      --slow_tail_steps 3

Compare with the older event-driven mode:

    python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
      --input data/outputs/YYYY.MM.DD/FAST_RUN/checkpoints/latest.ckpt \
      --output data/results/residual_insert_plug_event \
      --event_driven_chunks

Force/wrench frame override:

    --wrench_frame auto    # default, inferred from checkpoint metadata/path
    --wrench_frame sensor  # use sensor/EEF-frame wrench
    --wrench_frame world   # rotate wrench to world frame before fast policy


Expected startup log
--------------------
For no-force slow + force fast, typical keys are:

    slow obs keys: ['image0', 'robot_pose_R', 'robot_quat_R']
    fast obs keys: ... 'wrench_wrist_R', 'base_action_rel' ...
    env obs keys: ... 'wrench_wrist_R' ...
    env action shape: [9]
    fixed_time_chunks: True
    slow_tail_steps: 3

During execution, watch:

    Requested fixed slow chunk: ...
    Installed fixed slow seq: ...
    Submitted 1 residual step. global_step=..., chunk_start=..., tail=False/True


Control
-------
Press "s" in the OpenCV window to stop and save the episode.
"""

# %%
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from scipy.spatial.transform import Rotation

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import (
    get_abs_action_from_relative,
    get_real_obs_dict,
    get_real_obs_resolution,
    get_real_relative_obs_dict,
    get_relative_action_from_abs,
)
from diffusion_policy.residual_policy.pigdm_realtime_chunking import (
    make_realtime_chunking_pigdm,
)
from diffusion_policy.residual_policy.pose_util import apply_residual_action_to_pose9
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


SQRT2 = np.sqrt(2.0) / 2.0
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ],
    dtype=np.float32,
)

PREV_RESIDUAL_DELTA_KEY = "prev_residual_delta6"
PREV_RESIDUAL_VALID_KEY = "prev_residual_valid"
FAST_SYNTHETIC_OBS_KEYS = {
    "base_action_rel",
    "slow_action_rel",
    PREV_RESIDUAL_DELTA_KEY,
    PREV_RESIDUAL_VALID_KEY,
}


def _plain_cfg(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _strip_fast_only_obs(shape_meta):
    shape_meta = copy.deepcopy(_plain_cfg(shape_meta))
    for key in FAST_SYNTHETIC_OBS_KEYS:
        shape_meta["obs"].pop(key, None)
    return shape_meta


def _add_prev_residual_obs(
        obs_dict_np,
        shape_meta,
        prev_residual_action=None,
        prev_residual_valid=0.0):
    obs_shape_meta = _plain_cfg(shape_meta)["obs"]
    if PREV_RESIDUAL_DELTA_KEY in obs_shape_meta:
        shape = tuple(obs_shape_meta[PREV_RESIDUAL_DELTA_KEY]["shape"])
        if prev_residual_action is None:
            value = np.zeros(shape, dtype=np.float32)
        else:
            value = np.asarray(prev_residual_action, dtype=np.float32).reshape(shape)
        obs_dict_np[PREV_RESIDUAL_DELTA_KEY] = value[None]
    if PREV_RESIDUAL_VALID_KEY in obs_shape_meta:
        shape = tuple(obs_shape_meta[PREV_RESIDUAL_VALID_KEY]["shape"])
        value = np.full(shape, float(prev_residual_valid), dtype=np.float32)
        obs_dict_np[PREV_RESIDUAL_VALID_KEY] = value[None]
    return obs_dict_np


def _policy_shape_meta(policy, fallback_shape_meta):
    """Keep only obs keys the loaded policy actually consumes."""
    shape_meta = copy.deepcopy(_plain_cfg(fallback_shape_meta))
    obs_shape_meta = shape_meta["obs"]
    keep_keys = []
    for attr_name in ("rgb_keys", "low_dim_keys", "wrench_keys"):
        keep_keys.extend(getattr(policy, attr_name, []))

    filtered_obs = {}
    for key in keep_keys:
        if key in obs_shape_meta and key not in filtered_obs:
            filtered_obs[key] = obs_shape_meta[key]
    shape_meta["obs"] = filtered_obs
    return shape_meta


def _env_shape_meta(fast_shape_meta, slow_policy, base_action_key):
    """Collect fast obs from the robot, but keep the real executed action shape."""
    shape_meta = _strip_fast_only_obs(fast_shape_meta)
    env_action_dim = getattr(slow_policy, "action_dim", None)
    if env_action_dim is None:
        env_action_dim = _plain_cfg(fast_shape_meta)["obs"][base_action_key]["shape"][0]
    shape_meta["action"] = copy.deepcopy(shape_meta["action"])
    shape_meta["action"]["shape"] = [int(env_action_dim)]
    return shape_meta


def _fast_base_action_rel(abs_action, env_obs, fast_shape_meta, base_action_key):
    """Return only the action dimensions configured as fast-policy base input."""
    relative = get_relative_action_from_abs(action=abs_action, env_obs=env_obs)
    base_dim = int(_plain_cfg(fast_shape_meta)["obs"][base_action_key]["shape"][0])
    if relative.shape[-1] < base_dim:
        raise ValueError(
            f"Relative slow action has {relative.shape[-1]} dims, but fast base needs {base_dim}"
        )
    return relative[..., :base_dim]


def _infer_world_wrench(cfg):
    dataset_path = str(OmegaConf.select(cfg, "task.dataset_path", default=""))
    task_frame = str(OmegaConf.select(cfg, "task.wrench_frame", default="")).lower()
    return task_frame == "world" or "world_wrench" in dataset_path


def _override_slow_ckpt_path(cfg, slow_ckpt_path):
    if slow_ckpt_path is None:
        return
    with open_dict(cfg):
        cfg.slow_ckpt_path = slow_ckpt_path
        if "policy" in cfg:
            cfg.policy.slow_ckpt_path = slow_ckpt_path
        if "task" in cfg:
            cfg.task.slow_ckpt_path = slow_ckpt_path


def _rotate_wrench_to_world(obs_dict_np, env_obs, shape_meta):
    obs_shape_meta = _plain_cfg(shape_meta)["obs"]
    quat = np.asarray(env_obs["robot_quat_R"], dtype=np.float32)
    if quat.ndim == 1:
        quat = quat[None]

    for key, attr in obs_shape_meta.items():
        if attr.get("type", "low_dim") != "wrench" or key not in obs_dict_np:
            continue

        wrench = np.asarray(obs_dict_np[key], dtype=np.float32)
        squeeze_time = False
        if wrench.ndim == 2:
            wrench = wrench[None]
            squeeze_time = True
        if wrench.ndim != 3 or wrench.shape[1] != 6:
            continue

        step_quat = quat[-wrench.shape[0]:]
        if step_quat.shape[0] != wrench.shape[0]:
            raise ValueError(
                f"Cannot align {key} wrench length {wrench.shape[0]} "
                f"with robot_quat_R length {quat.shape[0]}"
            )

        robot_tcp = Rotation.from_quat(step_quat).as_matrix().astype(np.float32)
        world_tcp = np.einsum("ij,tjk->tik", RIGHT_ROBOT_TO_WORLD, robot_tcp)
        force_world = np.einsum("tij,tjh->tih", world_tcp, wrench[:, :3])
        torque_world = np.einsum("tij,tjh->tih", world_tcp, wrench[:, 3:6])
        wrench_world = np.concatenate([force_world, torque_world], axis=1).astype(np.float32)
        obs_dict_np[key] = wrench_world[0] if squeeze_time else wrench_world
    return obs_dict_np


def _ensure_wrench_time_dim(obs_dict_np, shape_meta):
    """Real env returns one wrench history window as (6, 32); policy wants T x 6 x 32."""
    obs_shape_meta = _plain_cfg(shape_meta)["obs"]
    for key, attr in obs_shape_meta.items():
        if attr.get("type", "low_dim") != "wrench" or key not in obs_dict_np:
            continue
        value = obs_dict_np[key]
        expected_shape = tuple(attr["shape"])
        if tuple(value.shape) == expected_shape:
            obs_dict_np[key] = value[None]
    return obs_dict_np


def _latest_obs_only(obs_dict_np, shape_meta):
    """Fast policy gets the latest image/pose/wrench plus one base_action_rel."""
    obs_shape_meta = _plain_cfg(shape_meta)["obs"]
    latest = dict()
    for key, value in obs_dict_np.items():
        attr = obs_shape_meta.get(key, {})
        obs_type = attr.get("type", "low_dim")
        value = np.asarray(value)
        if obs_type == "wrench":
            expected_shape = tuple(attr["shape"])
            if tuple(value.shape) == expected_shape:
                latest[key] = value[None]
            else:
                latest[key] = value[-1:]
        else:
            latest[key] = value[-1:]
    return latest


def _build_fixed_context_fast_obs(
        context_obs_dict_np,
        latest_obs_dict_np,
        base_action_rel,
        base_action_history,
        wrench_history,
        low_dim_history,
        fast_policy,
        max_steps):
    base_action_history.append(np.asarray(base_action_rel, dtype=np.float32))
    if len(base_action_history) > max_steps:
        del base_action_history[:-max_steps]

    out = copy.deepcopy(context_obs_dict_np)
    for key in getattr(fast_policy, "wrench_keys", []):
        if key not in latest_obs_dict_np:
            continue
        wrench_history.setdefault(key, []).append(
            np.asarray(latest_obs_dict_np[key][0], dtype=np.float32)
        )
        if len(wrench_history[key]) > max_steps:
            del wrench_history[key][:-max_steps]
        out[key] = np.stack(wrench_history[key], axis=0)

    if getattr(fast_policy, "include_step_low_dim", False):
        for key in getattr(fast_policy, "low_dim_keys", []):
            if key not in latest_obs_dict_np:
                continue
            low_dim_history.setdefault(key, []).append(
                np.asarray(latest_obs_dict_np[key][0], dtype=np.float32)
            )
            if len(low_dim_history[key]) > max_steps:
                del low_dim_history[key][:-max_steps]
            out[key] = np.stack(low_dim_history[key], axis=0)

    out[fast_policy.base_action_key] = np.stack(base_action_history, axis=0)
    return out


def _slow_step_for_timestamp(slow_anchor_timestamp, dt, min_timestamp, min_step_idx):
    step_idx = int(np.ceil(((min_timestamp - slow_anchor_timestamp) / dt) - 1e-9))
    return max(int(min_step_idx), step_idx)


def _get_policy_obs_dict(env_obs, shape_meta, obs_pose_repr, world_wrench=False):
    real_shape_meta = _strip_fast_only_obs(shape_meta)
    if obs_pose_repr == "relative":
        obs_dict_np = get_real_relative_obs_dict(
            env_obs=env_obs,
            shape_meta=real_shape_meta,
        )
    else:
        obs_dict_np = get_real_obs_dict(
            env_obs=env_obs,
            shape_meta=real_shape_meta,
        )
    if world_wrench:
        obs_dict_np = _rotate_wrench_to_world(obs_dict_np, env_obs, real_shape_meta)
    return _ensure_wrench_time_dim(obs_dict_np, real_shape_meta)


def _to_torch_obs(obs_dict_np, device):
    return dict_apply(
        obs_dict_np,
        lambda x: torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device),
    )


def _init_temporal_fast_hidden(fast_policy, fast_obs_dict):
    if not hasattr(fast_policy, "_build_sequence_inputs"):
        return None
    initial_hidden, _ = fast_policy._build_sequence_inputs(
        fast_obs_dict,
        need_initial_hidden=True,
    )
    return initial_hidden


def _predict_slow_action(slow_policy, slow_obs_dict, env_obs, slow_pigdm=None):
    if slow_pigdm is not None:
        return slow_pigdm.predict_action(slow_obs_dict, env_obs)
    return slow_policy.predict_action(slow_obs_dict)


def _predict_slow_abs_action_seq(
        slow_policy,
        env_obs,
        slow_shape_meta,
        slow_obs_pose_repr,
        slow_action_pose_repr,
        device,
        slow_pigdm=None):
    with torch.no_grad():
        slow_obs_dict_np = _get_policy_obs_dict(
            env_obs=env_obs,
            shape_meta=slow_shape_meta,
            obs_pose_repr=slow_obs_pose_repr,
            world_wrench=False,
        )
        slow_obs_dict = _to_torch_obs(slow_obs_dict_np, device)
        slow_result = _predict_slow_action(
            slow_policy,
            slow_obs_dict,
            env_obs,
            slow_pigdm=slow_pigdm,
        )
        slow_action = slow_result["action"][0].detach().to("cpu").numpy()
    if slow_action_pose_repr == "relative":
        slow_abs_action = get_abs_action_from_relative(
            action=slow_action,
            env_obs=env_obs,
        )
    else:
        slow_abs_action = slow_action
    return np.asarray(slow_abs_action, dtype=np.float32)


@click.command()
@click.option("--input", "-i", required=True, help="Path to fast residual checkpoint")
@click.option("--slow_ckpt_path", default=None, help="Optional override for the slow checkpoint path stored in the fast checkpoint cfg.")
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--robot_ip", "-ri", default="192.168.111.50", help="Robot IP placeholder kept for compatibility.")
@click.option("--match_dataset", "-m", default=None, help="Dataset used to overlay and adjust initial condition")
@click.option("--match_episode", "-me", default=None, type=int, help="Match specific episode from the match dataset")
@click.option("--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize.")
@click.option("--init_joints", "-j", is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option("--steps_per_inference", "-si", default=6, type=int, help="Slow chunk length before replanning.")
@click.option(
    "--slow_action_start_offset",
    default=1,
    type=int,
    help="Earliest slow target step to execute after each replan. With action_target_shift=1, fast step 0 predicts the residual for a_1.",
)
@click.option("--max_duration", "-md", default=60, help="Max duration for each epoch in seconds.")
@click.option("--frequency", "-f", default=10, type=float, help="Control frequency in Hz.")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Latency between receiving command and executing on robot in sec.")
@click.option("--device", default="cuda:0", help="Torch device for slow/fast inference, e.g. cuda:0 or cpu.")
@click.option(
    "--wrench_frame",
    default="auto",
    type=click.Choice(["auto", "sensor", "world"]),
    help="Frame expected by the fast wrench encoder. auto uses the checkpoint dataset/task metadata.",
)
@click.option("--slow_use_pigdm", is_flag=True, default=False, help="Use residual_policy-local PiGDM realtime chunking for the slow policy.")
@click.option(
    "--fixed_time_chunks/--event_driven_chunks",
    default=True,
    help="Align slow chunks to the episode 0.1s grid and use previous chunk tail while the next slow inference is late.",
)
@click.option("--slow_tail_steps", default=3, type=int, help="Extra previous-chunk fast steps allowed after the nominal slow chunk length.")
@click.option("--pigdm_executed_steps", default=None, type=int, help="How many slow steps are executed before the next slow replan. Defaults to steps_per_inference.")
@click.option("--pigdm_overlap_steps", default=None, type=int, help="Number of previous-chunk tail steps to guide. Defaults to horizon - executed_steps.")
@click.option("--pigdm_hard_steps", default=3, type=int, help="Number of overlap steps with full PiGDM weight before tapering.")
@click.option("--pigdm_guidance_scale", default=5.0, type=float, help="PiGDM guidance scale.")
@click.option(
    "--pigdm_weight_mode",
    default="exp_ramp",
    type=click.Choice(["exp_ramp", "linear", "uniform"]),
    help="PiGDM overlap weight schedule.",
)
@click.option("--pigdm_condition_start", default=0, type=int, help="Destination start index for previous-chunk overlap guidance.")
def main(input, slow_ckpt_path, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints,
    steps_per_inference, slow_action_start_offset, max_duration,
    frequency, command_latency, device, wrench_frame,
    slow_use_pigdm, fixed_time_chunks, slow_tail_steps,
    pigdm_executed_steps, pigdm_overlap_steps, pigdm_hard_steps,
    pigdm_guidance_scale, pigdm_weight_mode, pigdm_condition_start):

    # load checkpoint; checkpoint의 cfg 및 파라미터들 다 가져옴
    ckpt_path = input
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but torch.cuda.is_available() is False.")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    _override_slow_ckpt_path(cfg, slow_ckpt_path)

    # Head = 242422304502, Front = 336222070518, Left = 218622276386, Right = 126122270712
    # serial_numbers = ['126122270712', '151222078010'] # right, table
    serial_numbers = ["126122270712"] # right

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # =============== residual policy ==================
    # 여기서 policy는 fast residual policy이고, 그 안에 frozen slow_policy가 들어있음.
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    fast_policy = policy
    slow_policy = fast_policy.slow_policy

    fast_policy.eval().to(device)
    slow_policy.eval().to(device)

    # slow는 기존 diffusion policy와 동일하게 DDIM step / action slice 설정.
    slow_policy.num_inference_steps = 16
    slow_policy.n_action_steps = slow_policy.horizon - slow_policy.n_obs_steps + 1
    slow_pigdm = None
    if slow_use_pigdm:
        if pigdm_executed_steps is None:
            pigdm_executed_steps = steps_per_inference
        slow_pigdm = make_realtime_chunking_pigdm(
            slow_policy,
            executed_steps=pigdm_executed_steps,
            overlap_steps=pigdm_overlap_steps,
            hard_steps=pigdm_hard_steps,
            guidance_scale=pigdm_guidance_scale,
            weight_mode=pigdm_weight_mode,
            condition_start=pigdm_condition_start,
        )

    base_action_key = getattr(fast_policy, "base_action_key", "base_action_rel")
    fast_has_gru_step = hasattr(fast_policy, "predict_step")
    fast_uses_fixed_context = bool(getattr(fast_policy, "uses_fixed_context_sequence", False))
    fast_action_target_shift = int(OmegaConf.select(
        cfg,
        "task.dataset.action_target_shift",
        default=0,
    ))
    if fast_action_target_shift < 0:
        raise ValueError(f"action_target_shift must be >= 0, got {fast_action_target_shift}")
    # ===================================================

    # setup experiment
    dt = 1 / frequency
    if int(steps_per_inference) <= 0:
        raise ValueError(f"steps_per_inference must be > 0, got {steps_per_inference}")
    if int(slow_tail_steps) < 0:
        raise ValueError(f"slow_tail_steps must be >= 0, got {slow_tail_steps}")

    fast_shape_meta = cfg.task.shape_meta
    slow_shape_meta = _policy_shape_meta(
        slow_policy,
        _strip_fast_only_obs(fast_shape_meta),
    )
    env_shape_meta = _env_shape_meta(fast_shape_meta, slow_policy, base_action_key)
    use_world_wrench = (
        wrench_frame == "world"
        or (wrench_frame == "auto" and _infer_world_wrench(cfg))
    )

    obs_res = get_real_obs_resolution(slow_shape_meta)
    n_obs_steps = slow_policy.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("fixed_time_chunks:", fixed_time_chunks)
    print("slow_tail_steps:", slow_tail_steps)
    print("fast_has_gru_step:", fast_has_gru_step)
    print("fast_uses_fixed_context:", fast_uses_fixed_context)
    print("fast_action_target_shift:", fast_action_target_shift)
    print("slow obs keys:", list(_plain_cfg(slow_shape_meta)["obs"].keys()))
    print("fast obs keys:", list(_plain_cfg(fast_shape_meta)["obs"].keys()))
    print("env obs keys:", list(_plain_cfg(env_shape_meta)["obs"].keys()))
    print("env action shape:", _plain_cfg(env_shape_meta)["action"]["shape"])
    print("slow ckpt path:", OmegaConf.select(cfg, "policy.slow_ckpt_path", default=OmegaConf.select(cfg, "slow_ckpt_path", default=None)))
    print("fast wrench frame:", "world" if use_world_wrench else "sensor")
    print("device:", device)
    print("slow PiGDM:", "on" if slow_pigdm is not None else "off")
    if slow_pigdm is not None:
        print("PiGDM config:", slow_pigdm.config)

    # =============== relative ==================
    # fast cfg의 action_pose_repr는 residual_delta6/pose9라 real env action 변환용이 아님.
    # slow policy의 obs/action repr를 기준으로 slow action을 abs target으로 복원함.
    slow_obs_pose_repr = getattr(slow_policy, "obs_pose_repr", OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default="abs"))
    slow_action_pose_repr = getattr(slow_policy, "action_pose_repr", "relative")
    fast_obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=slow_obs_pose_repr)
    print("slow obs/action repr:", slow_obs_pose_repr, slow_action_pose_repr)
    print("fast obs repr:", fast_obs_pose_repr)
    # ===========================================

    # sharedmemory에 데이터들 쌓기; 같은 공유 공간 사용
    from diffusion_policy.real_world.bae_real_env_rightarm_hand_insert_plug import DualarmRealEnv
    with SharedMemoryManager() as shm_manager:
        with DualarmRealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            camera_serial_numbers=serial_numbers,
            n_obs_steps=n_obs_steps,
            shape_meta=env_shape_meta,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up residual policy inference")

            # obs 받아오기
            obs = env.get_obs()

            with torch.no_grad():
                slow_policy.reset()
                if slow_pigdm is not None:
                    slow_pigdm.reset()

                # slow obs: 기존 policy와 동일하게 n_obs_steps 전체 사용
                slow_obs_dict_np = _get_policy_obs_dict(
                    env_obs=obs,
                    shape_meta=slow_shape_meta,
                    obs_pose_repr=slow_obs_pose_repr,
                    world_wrench=False,
                )

                for key in slow_obs_dict_np.keys():
                    print(f"{key}: {slow_obs_dict_np[key].shape}, {slow_obs_dict_np[key].dtype}")

                slow_obs_dict = _to_torch_obs(slow_obs_dict_np, device)
                slow_result = _predict_slow_action(
                    slow_policy,
                    slow_obs_dict,
                    obs,
                    slow_pigdm=slow_pigdm,
                )
                slow_action = slow_result["action"][0].detach().to("cpu").numpy()
                if slow_action_pose_repr == "relative":
                    slow_abs_action = get_abs_action_from_relative(
                        action=slow_action,
                        env_obs=obs,
                    )
                else:
                    slow_abs_action = slow_action

                # fast warmup: feed the target base action that receives the residual.
                warmup_target_step_idx = min(fast_action_target_shift, len(slow_abs_action) - 1)
                base_action_rel = _fast_base_action_rel(
                    action=slow_abs_action[[warmup_target_step_idx]],
                    env_obs=obs,
                    fast_shape_meta=fast_shape_meta,
                    base_action_key=base_action_key,
                )[0]

                fast_obs_dict_np = _get_policy_obs_dict(
                    env_obs=obs,
                    shape_meta=fast_shape_meta,
                    obs_pose_repr=fast_obs_pose_repr,
                    world_wrench=use_world_wrench,
                )
                fast_obs_dict_np = _latest_obs_only(fast_obs_dict_np, fast_shape_meta)
                fast_obs_dict_np = _add_prev_residual_obs(
                    fast_obs_dict_np,
                    fast_shape_meta,
                    prev_residual_action=None,
                    prev_residual_valid=0.0,
                )
                fast_obs_dict_np[base_action_key] = base_action_rel[None].astype(np.float32)
                fast_obs_dict = _to_torch_obs(fast_obs_dict_np, device)

                if fast_has_gru_step:
                    fast_result = fast_policy.predict_step(fast_obs_dict, hidden=None)
                else:
                    fast_result = fast_policy.predict_action(fast_obs_dict)
                del slow_result, fast_result

            slow_executor = ThreadPoolExecutor(max_workers=1) if fixed_time_chunks else None
            np.set_printoptions(suppress=True, floatmode="fixed", precision=11)
            print("Ready!")
            while True:

                # ========== policy control loop ==============
                try:
                    # start episode
                    slow_policy.reset()
                    if slow_pigdm is not None:
                        slow_pigdm.reset()
                    fast_policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay

                    env.start_episode(eval_t_start)
                    frame_latency = 1 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    iter_idx = 0
                    slow_abs_action_seq = None
                    slow_action_anchor_timestamp = None
                    slow_chunk_start_input_step = None
                    slow_anchor_env_obs = None
                    slow_min_target_step_idx = 0
                    slow_min_step_idx = 0
                    slow_max_step_idx = -1
                    slow_nominal_max_step_idx = -1
                    fast_state_next_slow_step_idx = 0
                    fast_hidden = None
                    fast_anchor_obs_np = None
                    fast_context_obs_np = None
                    fast_context_base_actions = []
                    fast_context_wrench_history = {}
                    fast_context_low_dim_history = {}
                    prev_residual_action = None
                    prev_residual_valid = 0.0
                    pending_slow_future = None
                    pending_slow_chunk_start_step = None
                    pending_slow_anchor_env_obs = None
                    pending_slow_request_time = None
                    last_requested_chunk_start_step = None

                    def install_slow_chunk(slow_abs_action, anchor_env_obs, chunk_start_step, infer_latency):
                        nonlocal slow_abs_action_seq
                        nonlocal slow_action_anchor_timestamp
                        nonlocal slow_chunk_start_input_step
                        nonlocal slow_anchor_env_obs
                        nonlocal slow_min_target_step_idx
                        nonlocal slow_min_step_idx
                        nonlocal slow_max_step_idx
                        nonlocal slow_nominal_max_step_idx
                        nonlocal fast_state_next_slow_step_idx
                        nonlocal fast_hidden
                        nonlocal fast_anchor_obs_np
                        nonlocal fast_context_obs_np
                        nonlocal fast_context_base_actions
                        nonlocal fast_context_wrench_history
                        nonlocal fast_context_low_dim_history

                        fast_hidden = None
                        fast_anchor_obs_np = None
                        fast_context_obs_np = None
                        fast_context_base_actions = []
                        fast_context_wrench_history = {}
                        fast_context_low_dim_history = {}

                        slow_abs_action_seq = slow_abs_action
                        slow_chunk_start_input_step = int(chunk_start_step)
                        slow_action_anchor_timestamp = eval_t_start + slow_chunk_start_input_step * dt
                        slow_anchor_env_obs = anchor_env_obs
                        slow_min_target_step_idx = max(0, int(slow_action_start_offset))
                        slow_min_step_idx = max(
                            0,
                            slow_min_target_step_idx - fast_action_target_shift,
                        )
                        slow_nominal_max_step_idx = min(
                            len(slow_abs_action_seq) - fast_action_target_shift,
                            slow_min_step_idx + int(steps_per_inference),
                        ) - 1
                        slow_max_step_idx = min(
                            len(slow_abs_action_seq) - fast_action_target_shift,
                            slow_min_step_idx + int(steps_per_inference) + max(0, int(slow_tail_steps)),
                        ) - 1
                        if slow_max_step_idx < slow_min_step_idx:
                            raise RuntimeError("Slow policy returned no executable action.")

                        fast_anchor_obs_np = _get_policy_obs_dict(
                            env_obs=anchor_env_obs,
                            shape_meta=fast_shape_meta,
                            obs_pose_repr=fast_obs_pose_repr,
                            world_wrench=use_world_wrench,
                        )
                        fast_anchor_obs_np = _latest_obs_only(
                            fast_anchor_obs_np,
                            fast_shape_meta,
                        )
                        fast_anchor_obs_np = _add_prev_residual_obs(
                            fast_anchor_obs_np,
                            fast_shape_meta,
                            prev_residual_action=prev_residual_action,
                            prev_residual_valid=prev_residual_valid,
                        )

                        if fast_uses_fixed_context:
                            fast_context_obs_np = fast_anchor_obs_np
                        elif fast_has_gru_step:
                            first_target_step_idx = slow_min_step_idx + fast_action_target_shift
                            first_base_action_rel = _fast_base_action_rel(
                                action=slow_abs_action_seq[[first_target_step_idx]],
                                env_obs=anchor_env_obs,
                                fast_shape_meta=fast_shape_meta,
                                base_action_key=base_action_key,
                            )[0]
                            fast_init_obs_dict_np = copy.deepcopy(fast_anchor_obs_np)
                            fast_init_obs_dict_np[base_action_key] = first_base_action_rel[None].astype(np.float32)
                            fast_init_obs_dict = _to_torch_obs(fast_init_obs_dict_np, device)
                            fast_hidden = _init_temporal_fast_hidden(
                                fast_policy,
                                fast_init_obs_dict,
                            )

                        fast_state_next_slow_step_idx = slow_min_step_idx
                        print("Slow inference latency:", infer_latency)
                        print(
                            "Installed fixed slow seq:",
                            slow_abs_action_seq.shape,
                            "chunk_start:",
                            slow_chunk_start_input_step,
                            "anchor_time:",
                            slow_action_anchor_timestamp,
                            "nominal_input_steps:",
                            (slow_min_step_idx, slow_nominal_max_step_idx),
                            "tail_input_steps:",
                            (
                                slow_nominal_max_step_idx + 1,
                                slow_max_step_idx,
                            ) if slow_max_step_idx > slow_nominal_max_step_idx else None,
                            "target_steps:",
                            (
                                slow_min_step_idx + fast_action_target_shift,
                                slow_max_step_idx + fast_action_target_shift,
                            ),
                        )

                    def submit_slow_request(chunk_start_step, anchor_env_obs):
                        nonlocal pending_slow_future
                        nonlocal pending_slow_chunk_start_step
                        nonlocal pending_slow_anchor_env_obs
                        nonlocal pending_slow_request_time
                        nonlocal last_requested_chunk_start_step

                        if slow_executor is None:
                            raise RuntimeError("fixed_time_chunks requires a slow executor.")
                        anchor_env_obs = copy.deepcopy(anchor_env_obs)
                        pending_slow_chunk_start_step = int(chunk_start_step)
                        pending_slow_anchor_env_obs = anchor_env_obs
                        pending_slow_request_time = time.time()
                        last_requested_chunk_start_step = int(chunk_start_step)
                        pending_slow_future = slow_executor.submit(
                            _predict_slow_abs_action_seq,
                            slow_policy,
                            anchor_env_obs,
                            slow_shape_meta,
                            slow_obs_pose_repr,
                            slow_action_pose_repr,
                            device,
                            slow_pigdm,
                        )
                        print(
                            "Requested fixed slow chunk:",
                            pending_slow_chunk_start_step,
                            "request_time:",
                            pending_slow_request_time,
                        )

                    while True:
                        # residual은 fast가 최신 force를 봐야 하므로 action은 한 tick에 하나만 스케줄함.
                        if fixed_time_chunks:
                            wall_input_step = int(np.floor(max(0.0, time.time() - eval_t_start) / dt))
                            current_global_input_step = max(int(iter_idx), wall_input_step)
                            iter_idx = current_global_input_step
                        else:
                            current_global_input_step = int(iter_idx)
                        t_cycle_end = t_start + (current_global_input_step + 1) * dt
                        obs = None
                        obs_timestamps = None

                        if fixed_time_chunks:
                            obs = env.get_obs()
                            obs_timestamps = obs["timestamp"]
                            print(f"Obs latency {time.time() - obs_timestamps[-1]}")

                            if pending_slow_future is not None and pending_slow_future.done():
                                slow_abs_action = pending_slow_future.result()
                                install_slow_chunk(
                                    slow_abs_action,
                                    pending_slow_anchor_env_obs,
                                    pending_slow_chunk_start_step,
                                    time.time() - pending_slow_request_time,
                                )
                                pending_slow_future = None
                                pending_slow_chunk_start_step = None
                                pending_slow_anchor_env_obs = None
                                pending_slow_request_time = None

                            current_nominal_chunk_start = (
                                current_global_input_step // int(steps_per_inference)
                            ) * int(steps_per_inference)
                            if slow_abs_action_seq is None and pending_slow_future is None:
                                submit_slow_request(current_nominal_chunk_start, obs)
                            elif (
                                slow_abs_action_seq is not None
                                and current_global_input_step >= (
                                    slow_chunk_start_input_step + int(steps_per_inference)
                                )
                                and pending_slow_future is None
                            ):
                                next_chunk_start = (
                                    (current_global_input_step // int(steps_per_inference))
                                    * int(steps_per_inference)
                                )
                                if (
                                    next_chunk_start > slow_chunk_start_input_step
                                    and next_chunk_start != last_requested_chunk_start_step
                                ):
                                    submit_slow_request(next_chunk_start, obs)

                            # Initial episode startup has no previous chunk tail to fall back to.
                            if slow_abs_action_seq is None and pending_slow_future is not None:
                                slow_abs_action = pending_slow_future.result()
                                install_slow_chunk(
                                    slow_abs_action,
                                    pending_slow_anchor_env_obs,
                                    pending_slow_chunk_start_step,
                                    time.time() - pending_slow_request_time,
                                )
                                pending_slow_future = None
                                pending_slow_chunk_start_step = None
                                pending_slow_anchor_env_obs = None
                                pending_slow_request_time = None

                            wall_input_step = int(np.floor(max(0.0, time.time() - eval_t_start) / dt))
                            if wall_input_step > current_global_input_step:
                                current_global_input_step = max(int(iter_idx), wall_input_step)
                                iter_idx = current_global_input_step
                                t_cycle_end = t_start + (current_global_input_step + 1) * dt

                        # slow chunk가 끝나면 새로 16-step 예측하고 앞 steps_per_inference개 target만 실행함.
                        if (not fixed_time_chunks) and (
                            slow_abs_action_seq is None
                            or fast_state_next_slow_step_idx > slow_max_step_idx
                        ):
                            fast_hidden = None # GRU hidden은 slow replan마다 reset
                            fast_anchor_obs_np = None
                            fast_context_obs_np = None
                            fast_context_base_actions = []
                            fast_context_wrench_history = {}
                            fast_context_low_dim_history = {}
                            fast_state_next_slow_step_idx = 0

                            obs = env.get_obs()
                            obs_timestamps = obs["timestamp"]
                            slow_anchor_env_obs = obs
                            slow_action_anchor_timestamp = obs_timestamps[-1]
                            print(f"Obs latency before slow {time.time() - obs_timestamps[-1]}")

                            with torch.no_grad():
                                s = time.time()
                                slow_obs_dict_np = _get_policy_obs_dict(
                                    env_obs=obs,
                                    shape_meta=slow_shape_meta,
                                    obs_pose_repr=slow_obs_pose_repr,
                                    world_wrench=False,
                                )
                                slow_obs_dict = _to_torch_obs(slow_obs_dict_np, device)
                                slow_result = _predict_slow_action(
                                    slow_policy,
                                    slow_obs_dict,
                                    obs,
                                    slow_pigdm=slow_pigdm,
                                )
                                slow_action = slow_result["action"][0].detach().to("cpu").numpy()

                                if slow_action_pose_repr == "relative":
                                    slow_abs_action = get_abs_action_from_relative(
                                        action=slow_action,
                                        env_obs=obs,
                                    )
                                else:
                                    slow_abs_action = slow_action

                                slow_abs_action_seq = slow_abs_action
                                slow_chunk_start_input_step = 0
                                slow_min_target_step_idx = max(0, int(slow_action_start_offset))
                                slow_min_step_idx = max(
                                    0,
                                    slow_min_target_step_idx - fast_action_target_shift,
                                )
                                slow_max_step_idx = min(
                                    len(slow_abs_action_seq) - fast_action_target_shift,
                                    slow_min_step_idx + int(steps_per_inference),
                                ) - 1
                                slow_nominal_max_step_idx = slow_max_step_idx
                                if slow_max_step_idx < slow_min_step_idx:
                                    raise RuntimeError("Slow policy returned no executable action.")

                                fast_anchor_obs_np = _get_policy_obs_dict(
                                    env_obs=obs,
                                    shape_meta=fast_shape_meta,
                                    obs_pose_repr=fast_obs_pose_repr,
                                    world_wrench=use_world_wrench,
                                )
                                fast_anchor_obs_np = _latest_obs_only(
                                    fast_anchor_obs_np,
                                    fast_shape_meta,
                                )
                                fast_anchor_obs_np = _add_prev_residual_obs(
                                    fast_anchor_obs_np,
                                    fast_shape_meta,
                                    prev_residual_action=prev_residual_action,
                                    prev_residual_valid=prev_residual_valid,
                                )

                                if fast_uses_fixed_context:
                                    fast_context_obs_np = fast_anchor_obs_np
                                elif fast_has_gru_step:
                                    first_target_step_idx = slow_min_step_idx + fast_action_target_shift
                                    first_base_action_rel = _fast_base_action_rel(
                                        action=slow_abs_action_seq[[first_target_step_idx]],
                                        env_obs=obs,
                                        fast_shape_meta=fast_shape_meta,
                                        base_action_key=base_action_key,
                                    )[0]
                                    fast_init_obs_dict_np = copy.deepcopy(fast_anchor_obs_np)
                                    fast_init_obs_dict_np[base_action_key] = first_base_action_rel[None].astype(np.float32)
                                    fast_init_obs_dict = _to_torch_obs(fast_init_obs_dict_np, device)
                                    fast_hidden = _init_temporal_fast_hidden(
                                        fast_policy,
                                        fast_init_obs_dict,
                                    )

                                del slow_result
                                print("Slow inference latency:", time.time() - s)
                                print(
                                    "New slow seq:",
                                    slow_abs_action_seq.shape,
                                    "input_steps:",
                                    (slow_min_step_idx, slow_max_step_idx),
                                    "target_steps:",
                                    (
                                        slow_min_step_idx + fast_action_target_shift,
                                        slow_max_step_idx + fast_action_target_shift,
                                    ),
                                )

                        # 매 fast step마다 obs를 새로 받음. 여기서 force/image/pose가 최신값.
                        if not fixed_time_chunks:
                            obs = env.get_obs()
                            obs_timestamps = obs["timestamp"]
                            print(f"Obs latency {time.time() - obs_timestamps[-1]}")

                        action_ready = False
                        need_replan = False
                        input_slow_step_idx = None
                        output_slow_step_idx = None
                        action_timestamp = None

                        with torch.no_grad():
                            for fast_attempt in range(3):
                                s = time.time()
                                if fixed_time_chunks:
                                    nominal_action_timestamp = (
                                        eval_t_start
                                        + (current_global_input_step + fast_action_target_shift) * dt
                                    )
                                    min_action_timestamp = max(
                                        nominal_action_timestamp,
                                        obs_timestamps[-1] + dt,
                                        time.time() + command_latency,
                                    )
                                else:
                                    min_action_timestamp = max(
                                        obs_timestamps[-1] + dt,
                                        time.time() + command_latency,
                                    )
                                output_slow_step_idx = _slow_step_for_timestamp(
                                    slow_anchor_timestamp=slow_action_anchor_timestamp,
                                    dt=dt,
                                    min_timestamp=min_action_timestamp,
                                    min_step_idx=slow_min_target_step_idx,
                                )
                                input_slow_step_idx = max(
                                    output_slow_step_idx - fast_action_target_shift,
                                    fast_state_next_slow_step_idx,
                                    slow_min_step_idx,
                                )
                                output_slow_step_idx = (
                                    input_slow_step_idx + fast_action_target_shift
                                )
                                if input_slow_step_idx > slow_max_step_idx:
                                    if fixed_time_chunks:
                                        print(
                                            "[WARNING] Fixed-time slow tail exhausted; "
                                            "waiting for the next slow chunk.",
                                            "global_input_step:",
                                            current_global_input_step,
                                            "chunk_start:",
                                            slow_chunk_start_input_step,
                                            "input_step:",
                                            input_slow_step_idx,
                                            "max_step:",
                                            slow_max_step_idx,
                                        )
                                        break
                                    print(
                                        "[INFO] Slow chunk exhausted before next feasible "
                                        "timestamp; replanning."
                                    )
                                    slow_abs_action_seq = None
                                    need_replan = True
                                    break

                                action_timestamp = (
                                    slow_action_anchor_timestamp + output_slow_step_idx * dt
                                )
                                fast_latest_obs_dict_np = _get_policy_obs_dict(
                                    env_obs=obs,
                                    shape_meta=fast_shape_meta,
                                    obs_pose_repr=fast_obs_pose_repr,
                                    world_wrench=use_world_wrench,
                                )
                                fast_latest_obs_dict_np = _latest_obs_only(
                                    fast_latest_obs_dict_np,
                                    fast_shape_meta,
                                )
                                fast_latest_obs_dict_np = _add_prev_residual_obs(
                                    fast_latest_obs_dict_np,
                                    fast_shape_meta,
                                    prev_residual_action=prev_residual_action,
                                    prev_residual_valid=prev_residual_valid,
                                )

                                while fast_state_next_slow_step_idx < input_slow_step_idx:
                                    skipped_target_step_idx = (
                                        fast_state_next_slow_step_idx
                                        + fast_action_target_shift
                                    )
                                    skipped_abs_target = slow_abs_action_seq[skipped_target_step_idx]
                                    skipped_env_obs = (
                                        slow_anchor_env_obs
                                        if fast_state_next_slow_step_idx == 0
                                        else obs
                                    )
                                    skipped_obs_dict_np = (
                                        fast_anchor_obs_np
                                        if fast_state_next_slow_step_idx == 0
                                        else fast_latest_obs_dict_np
                                    )
                                    skipped_obs_dict_np = copy.deepcopy(skipped_obs_dict_np)
                                    skipped_obs_dict_np = _add_prev_residual_obs(
                                        skipped_obs_dict_np,
                                        fast_shape_meta,
                                        prev_residual_action=prev_residual_action,
                                        prev_residual_valid=prev_residual_valid,
                                    )
                                    skipped_base_action_rel = _fast_base_action_rel(
                                        action=skipped_abs_target[None],
                                        env_obs=skipped_env_obs,
                                        fast_shape_meta=fast_shape_meta,
                                        base_action_key=base_action_key,
                                    )[0]
                                    if fast_uses_fixed_context:
                                        if fast_context_obs_np is None:
                                            raise RuntimeError("Missing fixed context obs for context-step fast policy.")
                                        _build_fixed_context_fast_obs(
                                            context_obs_dict_np=fast_context_obs_np,
                                            latest_obs_dict_np=skipped_obs_dict_np,
                                            base_action_rel=skipped_base_action_rel,
                                            base_action_history=fast_context_base_actions,
                                            wrench_history=fast_context_wrench_history,
                                            low_dim_history=fast_context_low_dim_history,
                                            fast_policy=fast_policy,
                                            max_steps=getattr(fast_policy, "n_obs_steps", steps_per_inference),
                                        )
                                    elif fast_has_gru_step:
                                        fast_warm_obs_dict_np = copy.deepcopy(skipped_obs_dict_np)
                                        fast_warm_obs_dict_np[base_action_key] = (
                                            skipped_base_action_rel[None].astype(np.float32)
                                        )
                                        fast_warm_obs_dict = _to_torch_obs(fast_warm_obs_dict_np, device)
                                        fast_warm_result = fast_policy.predict_step(
                                            fast_warm_obs_dict,
                                            hidden=fast_hidden,
                                        )
                                        fast_hidden = fast_warm_result["hidden"]
                                        if PREV_RESIDUAL_DELTA_KEY in _plain_cfg(fast_shape_meta)["obs"]:
                                            prev_residual_action = (
                                                fast_warm_result["action"][0, 0]
                                                .detach()
                                                .to("cpu")
                                                .numpy()
                                            )
                                            prev_residual_valid = 1.0
                                        del fast_warm_result
                                    fast_state_next_slow_step_idx += 1

                                slow_abs_target = slow_abs_action_seq[output_slow_step_idx]
                                input_env_obs = (
                                    slow_anchor_env_obs if input_slow_step_idx == 0 else obs
                                )
                                input_obs_dict_np = (
                                    fast_anchor_obs_np
                                    if input_slow_step_idx == 0
                                    else fast_latest_obs_dict_np
                                )
                                input_obs_dict_np = copy.deepcopy(input_obs_dict_np)
                                input_obs_dict_np = _add_prev_residual_obs(
                                    input_obs_dict_np,
                                    fast_shape_meta,
                                    prev_residual_action=prev_residual_action,
                                    prev_residual_valid=prev_residual_valid,
                                )
                                base_action_rel = _fast_base_action_rel(
                                    action=slow_abs_target[None],
                                    env_obs=input_env_obs,
                                    fast_shape_meta=fast_shape_meta,
                                    base_action_key=base_action_key,
                                )[0]
                                if fast_uses_fixed_context:
                                    if fast_context_obs_np is None:
                                        raise RuntimeError("Missing fixed context obs for context-step fast policy.")
                                    fast_obs_dict_np = _build_fixed_context_fast_obs(
                                        context_obs_dict_np=fast_context_obs_np,
                                        latest_obs_dict_np=input_obs_dict_np,
                                        base_action_rel=base_action_rel,
                                        base_action_history=fast_context_base_actions,
                                        wrench_history=fast_context_wrench_history,
                                        low_dim_history=fast_context_low_dim_history,
                                        fast_policy=fast_policy,
                                        max_steps=getattr(fast_policy, "n_obs_steps", steps_per_inference),
                                    )
                                else:
                                    fast_obs_dict_np = copy.deepcopy(input_obs_dict_np)
                                    fast_obs_dict_np[base_action_key] = base_action_rel[None].astype(np.float32)
                                fast_obs_dict = _to_torch_obs(fast_obs_dict_np, device)

                                if fast_has_gru_step:
                                    fast_result = fast_policy.predict_step(
                                        fast_obs_dict,
                                        hidden=fast_hidden,
                                    )
                                    fast_hidden = fast_result["hidden"]
                                else:
                                    fast_result = fast_policy.predict_action(fast_obs_dict)
                                fast_state_next_slow_step_idx = input_slow_step_idx + 1

                                residual_action = fast_result["action"][0, 0].detach().to("cpu").numpy()
                                final_abs_action = apply_residual_action_to_pose9(
                                    slow_abs_target,
                                    residual_action,
                                )
                                del fast_result

                                curr_time = time.time()
                                if action_timestamp > (curr_time + command_latency):
                                    action_ready = True
                                    if PREV_RESIDUAL_DELTA_KEY in _plain_cfg(fast_shape_meta)["obs"]:
                                        prev_residual_action = residual_action
                                        prev_residual_valid = 1.0
                                    print(
                                        "Fast inference latency:",
                                        curr_time - s,
                                        "global_input_step:",
                                        current_global_input_step,
                                        "chunk_start:",
                                        slow_chunk_start_input_step,
                                        "input_step:",
                                        input_slow_step_idx,
                                        "target_step:",
                                        output_slow_step_idx,
                                        "tail:",
                                        input_slow_step_idx > slow_nominal_max_step_idx,
                                        "cmd_lead:",
                                        action_timestamp - curr_time,
                                    )
                                    break

                                print(
                                    "[WARNING] Residual action is outdated; "
                                    "advancing fast state to the next slow step.",
                                    "global_input_step:",
                                    current_global_input_step,
                                    "chunk_start:",
                                    slow_chunk_start_input_step,
                                    "input_step:",
                                    input_slow_step_idx,
                                    "target_step:",
                                    output_slow_step_idx,
                                    "lateness:",
                                    curr_time + command_latency - action_timestamp,
                                )

                        if need_replan:
                            continue

                        if not action_ready:
                            if fixed_time_chunks:
                                print(
                                    "[WARNING] No fresh residual action on fixed grid; "
                                    "leaving this tick without a new command.",
                                    "global_input_step:",
                                    current_global_input_step,
                                )
                                precise_wait(t_cycle_end - frame_latency)
                                iter_idx += 1
                                continue
                            print("[WARNING] Could not produce a fresh residual action; replanning slow.")
                            slow_abs_action_seq = None
                            continue

                        this_target_poses = np.zeros((1, final_abs_action.shape[-1]), dtype=np.float64)
                        this_target_poses[0, :final_abs_action.shape[-1]] = final_abs_action

                        action_timestamps = np.array([action_timestamp], dtype=np.float64)

                        # execute actions; 실제 action 실행부분
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                        )
                        print(
                            f"Submitted 1 residual step. "
                            f"global_step={current_global_input_step}, "
                            f"chunk_start={slow_chunk_start_input_step}, "
                            f"input_step={input_slow_step_idx}/{slow_max_step_idx}, "
                            f"target_step={output_slow_step_idx}, "
                            f"tail={input_slow_step_idx > slow_nominal_max_step_idx}, "
                            f"cmd_lead={action_timestamps[0] - time.time():.4f}s"
                        )

                        # visualize
                        # episode_id = env.replay_buffer.n_episodes
                        # vis_img = obs[f'image{vis_camera_idx}'][-1]
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        # cv2.putText(
                        #     vis_img,
                        #     text,
                        #     (10,20),
                        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=0.5,
                        #     thickness=1,
                        #     color=(255,255,255)
                        # )
                        # cv2.imshow('default', vis_img[...,::-1])

                        # 's' 누르면 종료
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord("s"):
                            env.end_episode()
                            print("Stopped.")
                            break

                        # auto termination; 한계시간 지나면 종료
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print("Terminated by the timeout!")

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution; residual은 한 tick씩 진행
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()

                print("Stopped.")


# %%
if __name__ == "__main__":
    main()
