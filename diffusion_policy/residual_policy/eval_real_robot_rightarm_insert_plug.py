#!/home/vision/anaconda3/envs/robodiff/bin/python

"""
Residual slow-fast policy version of bae_eval_real_robot_rightarm_insert_plug.py.

Usage:
    # 기본 실행. ckpt cfg의 dataset_path에 world_wrench가 있으면 wrench를 world frame으로 자동 변환함.
    python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
      --input data/outputs/YYYY.MM.DD/HH.MM.SS_NAME/checkpoints/latest.ckpt \
      --output data/results/residual_insert_plug \
      --steps_per_inference 8 \
      --wrench_frame auto

    # slow ckpt 파일 이름/위치를 바꾼 경우 fast ckpt 안의 slow_ckpt_path를 덮어쓰기
    python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
      --input data/outputs/YYYY.MM.DD/HH.MM.SS_NAME/checkpoints/latest.ckpt \
      --slow_ckpt_path data/outputs/2026.06.18/slow/20260528_unet_no_force_slow_epoch900.ckpt \
      --output data/results/residual_insert_plug \
      --steps_per_inference 8 \
      --wrench_frame auto

    # 센서/EEF frame wrench로 학습한 fast ckpt를 강제로 실행할 때
    python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
      --input data/outputs/YYYY.MM.DD/HH.MM.SS_NAME/checkpoints/latest.ckpt \
      --output data/results/residual_insert_plug \
      --steps_per_inference 8 \
      --wrench_frame sensor

    # world frame wrench 데이터셋으로 학습한 fast ckpt를 강제로 실행할 때
    python -m diffusion_policy.residual_policy.eval_real_robot_rightarm_insert_plug \
      --input data/outputs/YYYY.MM.DD/HH.MM.SS_NAME/checkpoints/latest.ckpt \
      --output data/results/residual_insert_plug \
      --steps_per_inference 8 \
      --wrench_frame world

Expected startup log for no-force slow + force fast:
    slow obs keys: ['image0', 'robot_pose_R', 'robot_quat_R']
    fast obs keys: ... 'wrench_wrist_R', 'base_action_rel' ...
    env obs keys: ... 'wrench_wrist_R' ...
    env action shape: [9]

Control:
    Press "s" in the OpenCV window to stop and save the episode.

The overall real-robot loop intentionally follows the original eval script.
The only conceptual change is:
    slow policy predicts a short absolute target chunk,
    fast policy corrects one target at a time with the latest force/obs,
    one final absolute pose is scheduled per control tick.
"""

# %%
import copy
import time
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
from diffusion_policy.real_world.bae_real_env_rightarm_hand_insert_plug import DualarmRealEnv
from diffusion_policy.real_world.real_inference_util import (
    get_abs_action_from_relative,
    get_real_obs_dict,
    get_real_obs_resolution,
    get_real_relative_obs_dict,
    get_relative_action_from_abs,
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


def _plain_cfg(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _strip_fast_only_obs(shape_meta):
    shape_meta = copy.deepcopy(_plain_cfg(shape_meta))
    shape_meta["obs"].pop("base_action_rel", None)
    shape_meta["obs"].pop("slow_action_rel", None)
    return shape_meta


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

    out[fast_policy.base_action_key] = np.stack(base_action_history, axis=0)
    return out


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


@click.command()
@click.option("--input", "-i", required=True, help="Path to fast residual checkpoint")
@click.option("--slow_ckpt_path", default=None, help="Optional override for the slow checkpoint path stored in the fast checkpoint cfg.")
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--robot_ip", "-ri", default="192.168.111.50", required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option("--match_dataset", "-m", default=None, help="Dataset used to overlay and adjust initial condition")
@click.option("--match_episode", "-me", default=None, type=int, help="Match specific episode from the match dataset")
@click.option("--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize.")
@click.option("--init_joints", "-j", is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option("--steps_per_inference", "-si", default=8, type=int, help="Slow chunk length before replanning.")
@click.option("--max_duration", "-md", default=60, help="Max duration for each epoch in seconds.")
@click.option("--frequency", "-f", default=10, type=float, help="Control frequency in Hz.")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Latency between receiving command and executing on robot in sec.")
@click.option(
    "--wrench_frame",
    default="auto",
    type=click.Choice(["auto", "sensor", "world"]),
    help="Frame expected by the fast wrench encoder. auto uses the checkpoint dataset/task metadata.",
)
def main(input, slow_ckpt_path, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints,
    steps_per_inference, max_duration,
    frequency, command_latency, wrench_frame):

    # load checkpoint; checkpoint의 cfg 및 파라미터들 다 가져옴
    ckpt_path = input
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
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

    device = torch.device("cuda")
    fast_policy.eval().to(device)
    slow_policy.eval().to(device)

    # slow는 기존 diffusion policy와 동일하게 DDIM step / action slice 설정.
    slow_policy.num_inference_steps = 16
    slow_policy.n_action_steps = slow_policy.horizon - slow_policy.n_obs_steps + 1

    base_action_key = getattr(fast_policy, "base_action_key", "base_action_rel")
    fast_has_gru_step = hasattr(fast_policy, "predict_step")
    fast_uses_fixed_context = bool(getattr(fast_policy, "uses_fixed_context_sequence", False))
    # ===================================================

    # setup experiment
    dt = 1 / frequency

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
    print("fast_has_gru_step:", fast_has_gru_step)
    print("fast_uses_fixed_context:", fast_uses_fixed_context)
    print("slow obs keys:", list(_plain_cfg(slow_shape_meta)["obs"].keys()))
    print("fast obs keys:", list(_plain_cfg(fast_shape_meta)["obs"].keys()))
    print("env obs keys:", list(_plain_cfg(env_shape_meta)["obs"].keys()))
    print("env action shape:", _plain_cfg(env_shape_meta)["action"]["shape"])
    print("slow ckpt path:", OmegaConf.select(cfg, "policy.slow_ckpt_path", default=OmegaConf.select(cfg, "slow_ckpt_path", default=None)))
    print("fast wrench frame:", "world" if use_world_wrench else "sensor")

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
                slow_result = slow_policy.predict_action(slow_obs_dict)
                slow_action = slow_result["action"][0].detach().to("cpu").numpy()
                if slow_action_pose_repr == "relative":
                    slow_abs_action = get_abs_action_from_relative(
                        action=slow_action,
                        env_obs=obs,
                    )
                else:
                    slow_abs_action = slow_action

                # fast warmup: slow 첫 target을 현재 pose 기준 relative로 다시 넣음
                base_action_rel = get_relative_action_from_abs(
                    action=slow_abs_action[[0]],
                    env_obs=obs,
                )[0]

                fast_obs_dict_np = _get_policy_obs_dict(
                    env_obs=obs,
                    shape_meta=fast_shape_meta,
                    obs_pose_repr=fast_obs_pose_repr,
                    world_wrench=use_world_wrench,
                )
                fast_obs_dict_np = _latest_obs_only(fast_obs_dict_np, fast_shape_meta)
                fast_obs_dict_np[base_action_key] = base_action_rel[None].astype(np.float32)
                fast_obs_dict = _to_torch_obs(fast_obs_dict_np, device)

                if fast_has_gru_step:
                    fast_result = fast_policy.predict_step(fast_obs_dict, hidden=None)
                else:
                    fast_result = fast_policy.predict_action(fast_obs_dict)
                del slow_result, fast_result

            np.set_printoptions(suppress=True, floatmode="fixed", precision=11)
            print("Ready!")
            while True:

                # ========== policy control loop ==============
                try:
                    # start episode
                    slow_policy.reset()
                    fast_policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay

                    env.start_episode(eval_t_start)
                    frame_latency = 1 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    iter_idx = 0
                    slow_abs_action_chunk = None
                    slow_chunk_idx = 0
                    fast_hidden = None
                    fast_context_obs_np = None
                    fast_context_base_actions = []
                    fast_context_wrench_history = {}

                    while True:
                        # residual은 fast가 최신 force를 봐야 하므로 action은 한 tick에 하나만 스케줄함.
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # slow chunk가 끝나면 새로 16-step 예측하고 앞 steps_per_inference개만 실행함.
                        if slow_abs_action_chunk is None or slow_chunk_idx >= len(slow_abs_action_chunk):
                            fast_hidden = None # GRU hidden은 slow replan마다 reset
                            fast_context_obs_np = None
                            fast_context_base_actions = []
                            fast_context_wrench_history = {}
                            slow_chunk_idx = 0

                            obs = env.get_obs()
                            obs_timestamps = obs["timestamp"]
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
                                slow_result = slow_policy.predict_action(slow_obs_dict)
                                slow_action = slow_result["action"][0].detach().to("cpu").numpy()

                                if slow_action_pose_repr == "relative":
                                    slow_abs_action = get_abs_action_from_relative(
                                        action=slow_action,
                                        env_obs=obs,
                                    )
                                else:
                                    slow_abs_action = slow_action

                                slow_abs_action_chunk = slow_abs_action[:steps_per_inference]
                                if len(slow_abs_action_chunk) == 0:
                                    raise RuntimeError("Slow policy returned no executable action.")

                                if fast_uses_fixed_context:
                                    fast_context_obs_np = _get_policy_obs_dict(
                                        env_obs=obs,
                                        shape_meta=fast_shape_meta,
                                        obs_pose_repr=fast_obs_pose_repr,
                                        world_wrench=use_world_wrench,
                                    )
                                    fast_context_obs_np = _latest_obs_only(
                                        fast_context_obs_np,
                                        fast_shape_meta,
                                    )
                                elif fast_has_gru_step:
                                    first_base_action_rel = get_relative_action_from_abs(
                                        action=slow_abs_action_chunk[[0]],
                                        env_obs=obs,
                                    )[0]
                                    fast_init_obs_dict_np = _get_policy_obs_dict(
                                        env_obs=obs,
                                        shape_meta=fast_shape_meta,
                                        obs_pose_repr=fast_obs_pose_repr,
                                        world_wrench=use_world_wrench,
                                    )
                                    fast_init_obs_dict_np = _latest_obs_only(
                                        fast_init_obs_dict_np,
                                        fast_shape_meta,
                                    )
                                    fast_init_obs_dict_np[base_action_key] = first_base_action_rel[None].astype(np.float32)
                                    fast_init_obs_dict = _to_torch_obs(fast_init_obs_dict_np, device)
                                    fast_hidden = _init_temporal_fast_hidden(
                                        fast_policy,
                                        fast_init_obs_dict,
                                    )

                                del slow_result
                                print("Slow inference latency:", time.time() - s)
                                print("New slow chunk:", slow_abs_action_chunk.shape)

                        # 매 fast step마다 obs를 새로 받음. 여기서 force/image/pose가 최신값.
                        obs = env.get_obs()
                        obs_timestamps = obs["timestamp"]
                        print(f"Obs latency {time.time() - obs_timestamps[-1]}")

                        with torch.no_grad():
                            s = time.time()

                            slow_abs_target = slow_abs_action_chunk[slow_chunk_idx]
                            base_action_rel = get_relative_action_from_abs(
                                action=slow_abs_target[None],
                                env_obs=obs,
                            )[0]

                            fast_obs_dict_np = _get_policy_obs_dict(
                                env_obs=obs,
                                shape_meta=fast_shape_meta,
                                obs_pose_repr=fast_obs_pose_repr,
                                world_wrench=use_world_wrench,
                            )
                            fast_obs_dict_np = _latest_obs_only(fast_obs_dict_np, fast_shape_meta)
                            if fast_uses_fixed_context:
                                if fast_context_obs_np is None:
                                    raise RuntimeError("Missing fixed context obs for context-step fast policy.")
                                fast_obs_dict_np = _build_fixed_context_fast_obs(
                                    context_obs_dict_np=fast_context_obs_np,
                                    latest_obs_dict_np=fast_obs_dict_np,
                                    base_action_rel=base_action_rel,
                                    base_action_history=fast_context_base_actions,
                                    wrench_history=fast_context_wrench_history,
                                    fast_policy=fast_policy,
                                    max_steps=getattr(fast_policy, "n_obs_steps", steps_per_inference),
                                )
                            else:
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

                            residual_action = fast_result["action"][0, 0].detach().to("cpu").numpy()
                            final_abs_action = apply_residual_action_to_pose9(
                                slow_abs_target,
                                residual_action,
                            )
                            del fast_result

                            print("Fast inference latency:", time.time() - s)

                        this_target_poses = np.zeros((1, final_abs_action.shape[-1]), dtype=np.float64)
                        this_target_poses[0, :final_abs_action.shape[-1]] = final_abs_action

                        # deal with timing
                        # 기존 eval의 timestamp 방식과 비슷하게 obs timestamp 기준 다음 tick을 목표로 함.
                        action_timestamps = np.array([obs_timestamps[-1] + dt], dtype=np.float64)
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + command_latency)

                        if np.sum(is_new) == 0:
                            print("[WARNING] Residual action is outdated!")
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt)) + 1
                            action_timestamp = eval_t_start + next_step_idx * dt
                            print("Over budget", action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp], dtype=np.float64)

                        # execute actions; 실제 action 실행부분
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                        )
                        print(
                            f"Submitted 1 residual step. "
                            f"slow_chunk={slow_chunk_idx + 1}/{len(slow_abs_action_chunk)}, "
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
                        slow_chunk_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()

                print("Stopped.")


# %%
if __name__ == "__main__":
    main()
