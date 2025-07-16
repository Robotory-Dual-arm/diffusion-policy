

import rospy
import spatialmath.base as smb
from std_msgs.msg import Int32, Float64, String
from diffusion_policy_test.msg import OnRobotRGOutput, OnRobotRGInput
from diffusion_policy.rb10_api.cobot import * 
from diffusion_policy.rb import *
from scipy.spatial.transform import Rotation as R

import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

# from rtde_control import RTDEControlInterface
# from rtde_receive import RTDEReceiveInterface

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

# 추가
def gripper_callback(msg):
    global latest_gripper_qpos
    latest_gripper_qpos = [msg.gGWD]
    # print("[DEBUG] gripper callback received:", latest_gripper_qpos[0])

def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
   
    a1 = rot6d[:3]
    a2 = rot6d[3:]

    b1 = a1 / np.linalg.norm(a1)

    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2)

    b3 = np.cross(b1, b2)

    R_mat = np.stack((b1, b2, b3), axis=1)  

    rot = R.from_matrix(R_mat)
    rot_vec = rot.as_rotvec()  

    return rot_vec

# current_joint: rad / target_pose: m, rad
def servoL_rb(robot, current_joint, target_pose, dt, acc_limit=40.0):   # target_pose : rot_vec
    
    current_pose = robot.fkine(current_joint)
    pos     = np.array(target_pose[:3])   # m
    rot_vec = np.array(target_pose[3:])   # rad         
    rotm    = R.from_rotvec(rot_vec).as_matrix()   
    T = np.eye(4)
    T[:3, :3] = rotm
    T[:3,  3] = pos

    target_pose = SE3(T)                   
    
    # 디버깅
    # print("[DEBUG] current_pose:", current_pose)
    # print("[DEBUG] target_pose:", target_pose)

    pose_error = current_pose.inv() * target_pose

    err_pos = target_pose.t - current_pose.t
    err_rot_ee = smb.tr2rpy(pose_error.R, unit='rad')
    err_rot_base = current_pose.R @ err_rot_ee
    err_6d = np.concatenate((err_pos, err_rot_base))

    J = robot.jacob0(current_joint)
    dq = np.linalg.pinv(J) @ err_6d

    if np.linalg.norm(dq[:3]) > acc_limit:
        dq[:3] *= acc_limit / np.linalg.norm(dq[:3])
    
    next_joint = current_joint + dq * 0.15
    ServoJ(next_joint * 180 / np.pi, time1=dt)

# times1: 제어시간, time2: lookahead time, gain: p-gain, lpf_gain: low pass filter gain
# time2 -> 클수록 smooth, but 반응 느림 (0.02 < time2 < 0.2)
# gain -> 클수록 빠르게 반응, but 진동 발생 (p > 0)
# lpf_gain -> 클수록 진동 감소, but 반응 느림 (0 < lpf_gain < 1)
def ServoJ(joint_deg, time1=0.002, time2=0.1, gain=0.02, lpf_gain=0.2):
    msg = f"move_servo_j(jnt[{','.join(f'{j:.3f}' for j in joint_deg)}],{time1},{time2},{gain},{lpf_gain})"
    SendCOMMAND(msg, CMD_TYPE.MOVE)
    


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class RTDEInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,   # 30
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:   # None
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:   # None
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:   # None
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:   # None
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController") 
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue; 예시 구조로부터 Queue 생성
        example = {
            'cmd': Command.SERVOL.value,
            # 'target_pose': np.zeros((6,), dtype=np.float64),
            'target_pose': np.zeros((9,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(   # 액션 담아놓을 메모리
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        # if receive_keys is None:
        #     receive_keys = [   # 여기에 내가 받을 로봇 Pose로 바꾸기
        #         'ActualTCPPose',   # GetCurrentTCP
        #         'ActualTCPSpeed',   # 없음
        #         'ActualQ',   # GetCurrentJoint
        #         'ActualQd',   # 없음

        #         'TargetTCPPose',
        #         'TargetTCPSpeed',
        #         'TargetQ',
        #         'TargetQd'
        #     ]
        if receive_keys is None:
            receive_keys = [
                'robot_eef_pos',   
                'robot_eef_quat',
                'robot_gripper_qpos'
            ]

        # rtde_r = RTDEReceiveInterface(hostname=robot_ip)   
        # ToCB(ip=robot_ip)
        # rb10 = RB10()   # 여기서는 굳이 필요없을듯

        example = dict()
        
        # for key in receive_keys:           
        #     example[key] = np.array(getattr(rtde_r, 'get'+key)())   
        for key in receive_keys:
            if key == 'robot_eef_pos':
                example[key] = np.zeros((3,), dtype=np.float64)
            elif key == 'robot_eef_quat':
                example[key] = np.zeros((4,), dtype=np.float64)
            # elif key == 'robot_gripper_qpos':
            #     example[key] = np.zeros((1,), dtype=np.float64)

        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(   # state 담아놓을 메모리
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()  
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        self.gripper_pub = None   # 그리퍼 제어용 퍼블리셔

    # ========= launch method ===========
    def start(self, wait=True):   
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):  
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):   # 안씀
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)   

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)


    def gripper_control(self, target_gripper_qpos):
        if self.gripper_pub is None:
            self.gripper_pub = rospy.Publisher('/OnRobotRGOutput', OnRobotRGOutput, queue_size=10)

        cmd = OnRobotRGOutput()
        cmd.rGWD = int(target_gripper_qpos[0])   
        cmd.rGFR = 400
        cmd.rCTR = 16

        self.gripper_pub.publish(cmd)


    def schedule_waypoint(self, pose, target_time):   # 이거 사용
        assert target_time > time.time()
        pose = np.array(pose)
        # assert pose.shape == (6,)
        assert pose.shape == (9,)   

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip

        # rtde_c = RTDEControlInterface(hostname=robot_ip)   # 바꾸기!
        # rtde_r = RTDEReceiveInterface(hostname=robot_ip)   # 바꾸기!
        ToCB(ip=robot_ip)
        rb10 = RB10()
        CobotInit()

        # Real or Simulation
        # SetProgramMode(PG_MODE.REAL)
        SetProgramMode(PG_MODE.SIMULATION)


        # global latest_gripper_qpos
        # latest_gripper_qpos = [1100]
        # rospy.init_node('gripper', anonymous=True)   
        # rospy.Subscriber('/OnRobotRGInput', OnRobotRGInput, gripper_callback, queue_size=1)  

        try:
            if self.verbose:   # False
                print(f"[RTDEPositionalController] Connect to robot: {robot_ip}")

            # set parameters
            # if self.tcp_offset_pose is not None:   # None
            #     rtde_c.setTcp(self.tcp_offset_pose)
            # if self.payload_mass is not None:   # None
            #     if self.payload_cog is not None:
            #         assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
            #     else:
            #         assert rtde_c.setPayload(self.payload_mass)
            
            # init pose
            if self.joints_init is not None:   # None
                # assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)
                MoveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1. / self.frequency

            # curr_pose = rtde_r.getActualTCPPose()   # 현재 pose 가져오기; 바꾸기!
            p = GetCurrentTCP()
            curr_pose = [p.x, p.y, p.z, p.rx, p.ry, p.rz]   # mm, deg
            # curr_pose.append(latest_gripper_qpos[0])   
            curr_pose = np.array(curr_pose)   
            curr_pose[0:3] = curr_pose[0:3] / 1000.0   # mm -> m
            curr_pose[3:6] = curr_pose[3:6] * np.pi / 180.0   # deg -> rad
            
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(   # abs
                times=[curr_t],     # [ time ]
                poses=[curr_pose]   # [ [x,y,z,rx,ry,rz] ] ---> [ [x,y,z,rx,ry,rz,gripper] ]
            )
            
            iter_idx = 0
            keep_running = True

            t_start = time.monotonic()   # 수동 제어 주기 맞추기

            while keep_running:   # 루프 시작
                # start control iteration
                # t_start = rtde_c.initPeriod()   # 바꾸기! 

                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now)   # 보간 해놓고 현재 시간의 목표 pose 가져옴
                # vel = 0.5
                # acc = 0.5
                # assert rtde_c.servoL(pose_command,   # 로봇 제어 부분! 바꾸기!
                #     vel, acc, # dummy, not used by ur5
                #     dt, 
                #     self.lookahead_time, 
                #     self.gain)

                
                # print("[DEBUG] curr_pose: ", curr_pose)
                print("[DEBUG] pose_command: ", pose_command)


                # RB10 제어              
                j = GetCurrentJoint()
                current_joint = np.array([j.j0, j.j1, j.j2, j.j3, j.j4, j.j5]) * np.pi / 180   # rad


                # 매니퓰레이터 및 그리퍼 제어                
                servoL_rb(rb10, current_joint, pose_command[:6], dt)
                # self.gripper_control(pose_command[6:])


                # 현재 State 저장
                # update robot state; ringbuffer에 state 저장
                state = dict()
                # for key in self.receive_keys:
                #     state[key] = np.array(getattr(rtde_r, 'get'+key)())
                
                p = GetCurrentTCP()
                curr_pose = [p.x, p.y, p.z, p.rx, p.ry, p.rz]   # mm, deg
                # curr_pose.append(latest_gripper_qpos[0])
                curr_pose = np.array(curr_pose)
                curr_pose[0:3] = curr_pose[0:3] / 1000.0   # mm -> m
                curr_pose[3:6] = curr_pose[3:6] * np.pi / 180.0   # deg -> rad
                # print('[DEBUG] current pose:', curr_pose)
                for key in self.receive_keys:
                    if key == 'robot_eef_pos':
                        state[key] = np.array(curr_pose[:3])   # 현재 pose; meter
                    elif key == 'robot_eef_quat':
                        # state[key] = np.array(smb.r2q(curr_pose[3:6]))   # 현재 quat
                        rot_vec = np.array(curr_pose[3:6])
                        quat = R.from_rotvec(rot_vec).as_quat()   # rot_vec --> quat
                        state[key] = np.array(quat)
                    # elif key == 'robot_gripper_qpos':
                    #     state[key] = np.array(curr_pose[6:])   # 현재 그리퍼 pose

                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)   


                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()   # command 긁어옴
                    # print('[DEBUG] commands', commands)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):   # 가져온 cmd 수만큼 실행
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:   # STOP: 그만두기
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:   # SERVOL: target_pose 실행 
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']  
                        duration = float(command['duration']) 
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                            
                    # 이걸로 제어 (n_cmd 1개씩 제어)
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']   # abs; 3d pose, 6d rotation

                        # print('[DEBUG] target_pose_before', target_pose)

                        # target_pose[:3] = target_pose[:3] * 1000.0   # m -> mm
                        
                        target_position = target_pose[:3]   # 3d position, meter
                        target_rotvec = rot6d_to_rotvec(target_pose[3:])   # 6d rotation -> rot_vec
                        target_pose = np.concatenate([target_position, target_rotvec])   # 3d position, rot_vec

                        print('[DEBUG] target_pose', target_pose)
                        
                        # print('[DEBUG] current rot_vec', curr_pose[3:6])
                        # target_pose[:3] = curr_pose[:3] + target_pose[:3] * MAX_TRANS   # pose, meter
                        # target_pose[3:6] = (R.from_rotvec(target_pose[3:6] * MAX_ROT) * R.from_rotvec(curr_pose[3:6])).as_rotvec()   # rotation, rad
                        # target_pose[6] = curr_pose[6] + target_pose[6] * MAX_GRIP    # gripper

                        # print('[DEBUG] target rot_vec', target_pose[3:6])

                        target_time = float(command['target_time'])   # time.time 기준
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time   # time.monotonic 기준
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(   # 여기서 pose_interp 갱신
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break
                
                # regulate frequency
                # rtde_c.waitPeriod(t_start)
                t_elapsed = time.monotonic() - t_start
                sleep_time = dt - t_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)   # 수동 제어 주기

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()   # 준비완료; wait하던거 실행됨
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            # rtde_c.servoStop()   # 끝내기; 바꾸기!

            # # terminate
            # rtde_c.stopScript()
            # rtde_c.disconnect()
            # rtde_r.disconnect()
            MotionHalt()
            DisConnectToCB()

            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
