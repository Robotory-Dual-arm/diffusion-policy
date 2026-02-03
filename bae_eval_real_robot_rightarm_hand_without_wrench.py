#!/home/vision/anaconda3/envs/robodiff/bin/python

# 실행코드
# python bae_eval_real_robot_rightarm_hand_without_wrench.py --input data/outputs/251225_erase_board/checkpoints/epoch=1000-train_loss=0.001.ckpt --output data/results
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
from diffusion_policy.real_world.bae_real_env_rightarm_hand_without_wrench import DualarmRealEnv   # 새로 만듬
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict,
    get_real_relative_obs_dict,
    get_abs_action_from_relative)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')   # checkpoint
@click.option('--output', '-o', required=True, help='Directory to save recording')   
@click.option('--robot_ip', '-ri', default="192.168.111.50", required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')   
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")   # 몇개의 action 실행할건지
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")  
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):

    # load checkpoint; checkpoint의 cfg 및 파라미터들 다 가져옴
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']   # yaml에 있던 변수들 설정값
    
    # Head = 242422304502, Front = 336222070518, Left = 218622276386, Right = 126122270712
    serial_numbers = ['242422304502', '336222070518'] # head, front
    use_pigdm = False
    if use_pigdm == True: # relative에서는 다시 확인해봐라 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cfg._target_ = 'diffusion_policy.workspace.bae_train_diffusion_unet_hybrid_pigdm_workspace.TrainDiffusionUnetHybridPigdmWorkspace'
        cfg.policy._target_ = "diffusion_policy.policy.bae_diffusion_unet_hybrid_image_policy_pigdm.DiffusionUnetHybridImagePigdmPolicy"
        cfg.policy.noise_scheduler._target_ = "bae_scheduling_ddim_pigdm.DDIMPIGDMScheduler"

    cls = hydra.utils.get_class(cfg._target_)  
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)


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
        
        #추가됨
        # policy.obs_encoder.to(device)
        # print("policy device setting")
        
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
    print("steps_per_inference:", steps_per_inference)   # 예측한 action sequence에서 몇개의 action 실행할건지 (6)
    print("action_offset:", action_offset)   # action 지연 실행 (0)


    # =============== relative ==================
    # 있으면 'relative' or 'abs' / 없으면 None
    obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=None)
    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default=None)

    # ===========================================


    # sharedmemory에 데이터들 쌓기; 같은 공유 공간 사용
    with SharedMemoryManager() as shm_manager:
        with DualarmRealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,   
            camera_serial_numbers=serial_numbers,
            n_obs_steps=n_obs_steps,   
            obs_image_resolution=obs_res, 
            obs_float32=True,   
            init_joints=init_joints,   # False
            enable_multi_cam_vis=True,   # 실시간 시각화 
            record_raw_video=False,   # 영상 저장 
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)


            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            
            # obs 받아오기
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()

                # 받은 obs에서 image 정규화 및 다듬기, pose 다듬기
                
                # Obs: relative or abs 
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
                # action 예측 
                if use_pigdm == True:
                    abs_obs = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action_pigdm(obs_dict, obs)
                else:
                    result = policy.predict_action(obs_dict)

                # 실제 실행할 action trajectory
                action = result['action'][0].detach().to('cpu').numpy()   # [0]은 배치차원 제거, tensor --> np
                assert action.shape[-1] == cfg.task.shape_meta.action.shape[0]
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

                    env.start_episode(eval_t_start)   # 영상 저장 시작
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency; 카메라 프레임 잘 받아오도록
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0   # trajectory 실행 개수
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:  
                        # calculate timing; 실행할 action 만큼 기다릴 시간
                        # print("[TIME] current time: ", time.monotonic()%100)
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
                        # print("[TIME] t_cycle_end: ", t_cycle_end%100)

                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference; action 예측
                        with torch.no_grad():
                            s = time.time()

                            # Obs: relative or abs
                            if obs_pose_repr == 'relative':
                                obs_dict_np = get_real_relative_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            else:
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)

                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            
                            # action 예측 
                            if use_pigdm == True:
                                result = policy.predict_action_pigdm(obs_dict, obs)
                            else:
                                result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()   # 실행할 action[Horizon, Action_Dim]
                            
                            # Action: relative or abs
                            if action_pose_repr == 'relative':
                                action = get_abs_action_from_relative(action=action,env_obs=obs)

                            print('Inference latency:', time.time() - s)

                        # convert policy action to env actions
                        if delta_action:   # False
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[[0,1]] += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)

                        else:   # len(action): Horizon / len(target_pose): 9
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            this_target_poses[:, :action.shape[-1]] = action

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)   # 현재시점 이후 action만 실행


                        ############################################ timestamp
                        # while문은 6 * 0.1 주기로 무조건 돌음
                        # 현재시간 t / 이용하는 obs_timestamp = t - obs_latency
                        #   1           2              3      4      5      6      7         8         9     10    11    12    13    14    15 16
                        # -0.1    obs_timestamp      +0.1   +0.2   +0.3   +0.4   +0.5      +0.6
                        #                                                        -0.1  obs_timestamp  +0.1  +0.2  +0.3  +0.4  +0.5  +0.6
                        # print("Current time:", curr_time)
                        # print("Action timestamps:", action_timestamps)
                        ############################################
                        
                        if np.sum(is_new) == 0:   # 전부 지나버림
                            print('[WARNING] All actions are outdated!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]   # 마지막 action이라도 실행
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])

                        else:   # is_new = 1 인것만 실행
                            this_target_poses = this_target_poses[is_new]#[:6]
                            action_timestamps = action_timestamps[is_new]#[:6]

                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")


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
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference
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
