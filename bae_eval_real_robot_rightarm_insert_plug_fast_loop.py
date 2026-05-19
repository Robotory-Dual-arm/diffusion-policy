#!/home/vision/anaconda3/envs/robodiff/bin/python

# 실행코드
# python bae_eval_real_robot_rightarm_hand_with_wrench_encoder.py --input data/outputs/260330_erase_and_trash/checkpoints/epoch=0900-train_loss=0.001.ckpt --output data/results
"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.bae_real_env_rightarm_hand_insert_plug import DualarmRealEnv   # 새로 만듬
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict,
    get_real_relative_obs_dict,
    get_abs_action_from_relative,
    get_relative_action_from_abs)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')   # checkpoint
@click.option('--output', '-o', required=True, help='Directory to save recording')   
@click.option('--robot_ip', '-ri', default="192.168.111.50", required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')   
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--full_action_refresh_steps', '-si', default=6, type=int, help="How many executed steps before sampling a new full action.")
@click.option('--remaining_update_steps', default=2, type=int, help="How many control steps to wait before re-denoising the remaining action.")
@click.option('--remaining_noise_ratio', default=0.5, type=float, help="Noise ratio for the remaining action. 0.5 means the middle diffusion timestep.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")  
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    full_action_refresh_steps, remaining_update_steps, remaining_noise_ratio, max_duration,
    frequency, command_latency):

    # load checkpoint; checkpoint의 cfg 및 파라미터들 다 가져옴
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']   # yaml에 있던 변수들 설정값
    
    # Head = 242422304502, Front = 336222070518, Left = 218622276386, Right = 126122270712
    # serial_numbers = ['126122270712', '151222078010'] # right, table
    serial_numbers = ['126122270712'] # right
   
    cls = hydra.utils.get_class(cfg._target_)   # WorkSpace 설정
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # 여기서 workspace.model에 cfg.policy가 들어감


    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False  
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model   # state_dicts의 model을 가져옴 (가중치 값들)
        if cfg.training.use_ema:
            policy = workspace.ema_model   # ema_model 가져옴 (가중치 값들)
        device = torch.device('cuda')
        policy.eval().to(device)
        
        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations; 노이즈 제거 step 수
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1   # 과거부터 horizon 뽑고, obs만큼 빼고, 1 더하기 (16 - 2 + 1 = 15)

    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)


    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # obs의 image 해상도 (width, height)
    n_obs_steps = cfg.n_obs_steps   # obs 관측 step 수
    print("n_obs_steps: ", n_obs_steps)   # obs 관측 step수 (2)
    print("full_action_refresh_steps:", full_action_refresh_steps)   # full action을 새로 뽑는 주기 (6)
    print("remaining_update_steps:", remaining_update_steps)   # 남은 action 갱신 주기 (2)
    print("remaining_noise_ratio:", remaining_noise_ratio)   # 남은 action에 다시 넣을 noise 수준 (0.5)
    print("action_offset:", action_offset)   # action 지연 실행 (0)


    # =============== relative ==================
    # 있으면 'relative' or 'abs' / 없으면 None
    obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=None)
    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default=None)

    # ===========================================
    if n_obs_steps != 2:
        raise ValueError("Reactive remaining-action update assumes n_obs_steps == 2.")
    if full_action_refresh_steps <= 0:
        raise ValueError("full_action_refresh_steps must be positive.")
    if remaining_update_steps <= 0:
        raise ValueError("remaining_update_steps must be positive.")
    if remaining_noise_ratio < 0 or remaining_noise_ratio > 1:
        raise ValueError("remaining_noise_ratio must be between 0 and 1.")


    # sharedmemory에 데이터들 쌓기; 같은 공유 공간 사용
    with SharedMemoryManager() as shm_manager:
        with DualarmRealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,   
            camera_serial_numbers=serial_numbers,
            n_obs_steps=n_obs_steps,   
            shape_meta=cfg.task.shape_meta,
            obs_image_resolution=obs_res, # (224,224)
            obs_float32=True,   
            init_joints=init_joints,   # False
            enable_multi_cam_vis=True,   # 실시간 시각화 
            record_raw_video=False,   # 원본 화질 영상 저장 
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)


            # Realsense-viewer에서 설정
            # Should be the same as demo
            # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            
            # obs 받아오기
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()

                # 받은 obs에서 image 정규화 및 다듬기, pose 다듬기
                # obs: relative or abs
                if obs_pose_repr == 'relative':
                    obs_dict_np = get_real_relative_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                else:
                    obs_dict_np = get_real_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)

                for key in obs_dict_np.keys():
                    print(f"{key}: {obs_dict_np[key].shape}, {obs_dict_np[key].dtype}")

                # shape_meta 계층구조는 유지하면서 np --> tensor로 변환, 텐서 배치차원 추가
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                # obs로 action 예측 
                result = policy.predict_action(obs_dict)


                # 실제 실행할 action trajectory
                action = result['action'][0].detach().to('cpu').numpy()   # [0]은 배치차원 제거, tensor --> np
                assert action.shape[-1] == cfg.task.shape_meta.action.shape[0]   # action 차원에 맞게 바꿔주기
                del result

            np.set_printoptions(suppress=True, floatmode="fixed", precision=11)
            print('Ready!')
            while True:
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay   # 시스템시간, 영상 로그용
                    t_start = time.monotonic() + start_delay   # 로봇 제어 시간
                    # print("[TIME] t_start: ", t_start%100)

                    env.start_episode(eval_t_start)   # 영상 저장 시작
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency; 카메라 프레임 잘 받아오도록
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    step_idx = 0   # episode에서 지난 control step 수
                    steps_after_full_action = full_action_refresh_steps # 6
                    remaining_action_abs = None
                    while True:  
                        next_update_time = t_start + (step_idx + remaining_update_steps) * dt

                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference; action 예측
                        with torch.no_grad():
                            s = time.time()

                            # obs: relative or abs
                            if obs_pose_repr == 'relative':
                                obs_dict_np = get_real_relative_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            else:
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)

                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                            need_full_action = (
                                remaining_action_abs is None
                                or len(remaining_action_abs) <= 1
                                or steps_after_full_action >= full_action_refresh_steps
                            )

                            if need_full_action:
                                result = policy.predict_action(obs_dict)

                                policy_action = result['action'][0].detach().to('cpu').numpy()

                                # action: relative or abs
                                if action_pose_repr == 'relative':
                                    action = get_abs_action_from_relative(action=policy_action, env_obs=obs)
                                else:
                                    action = policy_action

                                # predict_action() already drops the first action because n_obs_steps=2.
                                # After running two actions, action[1] becomes the front action for
                                # the next obs window, so keep action[1:] for remaining-action denoising.
                                next_remaining_action_abs = action.copy()
                                trim_steps = max(remaining_update_steps - 1, 0)
                                steps_after_full_action = 0
                                action_mode = "full_action"
                                remaining_action_len = None
                                noise_timestep = None
                            else:
                                remaining_action_len = len(remaining_action_abs)

                                # remaining action: abs -> policy action repr
                                remaining_action = np.asarray(remaining_action_abs, dtype=np.float32)
                                if action_pose_repr == 'relative':
                                    remaining_action = get_relative_action_from_abs(
                                        action=remaining_action, env_obs=obs)
                                start_action = torch.from_numpy(
                                    remaining_action.astype(np.float32)).unsqueeze(0).to(device)

                                result = policy.predict_action_horizon(
                                    obs_dict=obs_dict,
                                    start_action=start_action, # 남은 action
                                    noise_ratio=remaining_noise_ratio, # 노이즈 추가 수준
                                )

                                policy_action = result['action'][0].detach().to('cpu').numpy()
                                policy_action_pred = result['action_pred'][0].detach().to('cpu').numpy()

                                # action: relative or abs
                                if action_pose_repr == 'relative':
                                    action = get_abs_action_from_relative(action=policy_action, env_obs=obs)
                                    next_remaining_action_abs = get_abs_action_from_relative(
                                        action=policy_action_pred, env_obs=obs)
                                else:
                                    action = policy_action
                                    next_remaining_action_abs = policy_action_pred

                                trim_steps = remaining_update_steps
                                action_mode = "remaining_action"
                                noise_timestep = result.get('noise_timestep')

                            print('Inference latency:', time.time() - s)
                            if noise_timestep is None:
                                print(f"Reactive action mode: {action_mode}, schedule_len: {len(action)}")
                            else:
                                print(f"Reactive action mode: {action_mode}, remaining_len: {remaining_action_len}, schedule_len: {len(action)}, noise_t: {noise_timestep}")

                        this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                        this_target_poses[:, :action.shape[-1]] = action

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)   # 현재시점 이후 action만 실행
                        
                        
                        if np.sum(is_new) == 0:   # 전부 지나버림
                            print('[WARNING] All actions are outdated!')
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]   # 마지막 action이라도 실행
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])

                        else:   # is_new = 1 인것만 실행
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # execute actions; 실제 action 실행부분; 
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")
                       

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
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break


                        # auto termination; 한계시간 지나면 종료
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            env.end_episode()
                            break


                        # wait for execution; 로봇이 action 여러개 실행할동안 기다림
                        precise_wait(next_update_time - frame_latency)
                        if next_remaining_action_abs is not None:
                            slice_start = min(trim_steps, len(next_remaining_action_abs))
                            remaining_action_abs = next_remaining_action_abs[slice_start:].copy()
                        steps_after_full_action += remaining_update_steps
                        step_idx += remaining_update_steps
                        # print("[TIME] cycle 끝 시간: ", time.monotonic()%100, time.time()%10)
                        # time.sleep(1)

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
