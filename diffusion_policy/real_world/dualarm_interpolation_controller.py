
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from spatialmath import SE3
import spatialmath.base as smb
from std_msgs.msg import Int32, Float64, String
from sensor_msgs.msg import JointState

import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R

import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator



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
    

def se3_to_pos_rotvec(T: SE3):
    """
    T : spatialmath.SE3 or 4x4 homogeneous np.array
    return: (pos[3], rotvec[3])  # rotvec in rad
    """

    if isinstance(T, SE3):
        M = T.A  # 4x4 numpy array
    else:
        M = np.asarray(T)
        assert M.shape == (4,4), "T must be 4x4"

    pos = M[:3, 3].copy()
    rotvec = R.from_matrix(M[:3, :3]).as_rotvec()
    return np.hstack((pos, rotvec))
    

# current_joint: rad / target_pose: m, rad
def servoJ(robot, current_joint, target_pose, acc_pos_limit=40.0, acc_rot_limit=5.0):   # target_pose : rot_vec
    
    current_pose = robot.fkine(current_joint)   # SE3

    pos     = np.array(target_pose[:3])   # m
    rot_vec = np.array(target_pose[3:])   # rad         
    rotm    = R.from_rotvec(rot_vec).as_matrix()   
   
    T = np.eye(4)
    T[:3, :3] = rotm
    T[:3,  3] = pos
    target_pose = SE3(T)                   
    
    current_pose_rotvec = R.from_matrix(current_pose.R).as_rotvec()

    pose_error = current_pose.inv() * target_pose

    err_pos = target_pose.t - current_pose.t
    err_rot_ee = smb.tr2rpy(pose_error.R, unit='rad')
    err_rot_base = current_pose.R @ err_rot_ee
    err_6d = np.concatenate((err_pos, err_rot_base))

    J = robot.jacob0(current_joint)
    dq = np.linalg.pinv(J) @ err_6d

    # print("[DEBUG] acc pos:", np.linalg.norm(dq[:3]))
    # print("[DEBUG] acc rot:", np.linalg.norm(dq[3:]))

    if np.linalg.norm(dq[:3]) > acc_pos_limit:
        dq[:3] *= acc_pos_limit / np.linalg.norm(dq[:3])
    
    if np.linalg.norm(dq[3:]) > acc_rot_limit:
        dq[3:] *= acc_rot_limit / np.linalg.norm(dq[3:])
    
    next_joint = current_joint + dq * 0.2
    return next_joint   # rad 이거 맞나


class Dualarm(Node):
    def __init__(self):
        super().__init__('dualarm_node')
        
        self.joint_name = [f"left_joint_{i}" for i in range(1,7)] + \
                            [f"right_joint_{i}" for i in range(1,7)]
        
        self.joint_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.joint_command_publisher_L = self.create_publisher(
            JointState,
            '/left_dsr_joint_controller/joint_state_command99999999',
            10
        )
        self.joint_command_publisher_R = self.create_publisher(
            JointState,
            '/right_dsr_joint_controller/joint_state_command99999999',
            10
        )

    def joint_callback(self, msg):
        global latest_joint_L, latest_joint_R
    
        joint_mapping = {n: p for n, p in zip(msg.name, msg.position)}
        joint_position = [joint_mapping.get(j) for j in self.joint_name]

        latest_joint_L = joint_position[:6]
        latest_joint_R = joint_position[6:]

    def joint_command_publish_L(self, joint_position):
        msg = JointState()
        msg.name = self.joint_name[:6]
        joint_position = [float(x) for x in joint_position]
        msg.position = joint_position
        self.joint_command_publisher_L.publish(msg)
        
    def joint_command_publish_R(self, joint_position):
        msg = JointState()
        msg.name = self.joint_name[6:]
        joint_position = [float(x) for x in joint_position]
        msg.position = joint_position
        self.joint_command_publisher_R.publish(msg)

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class DualarmInterpolationController(mp.Process):
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

        # build input queue; action 담아놓을 메모리
        example = {
            'cmd': Command.SERVOL.value,
            # 'target_pose': np.zeros((6,), dtype=np.float64),
            'target_pose': np.zeros((18,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples( 
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer; state 담아놓을 메모리
        if receive_keys is None:
            receive_keys = [
                'robot_pose_L',   
                'robot_pose_R',
                'robot_quat_L',
                'robot_quat_R',
            ]
        
        example = dict()
        for key in receive_keys:
            if key == 'robot_pose_L':
                example[key] = np.zeros((3,), dtype=np.float64)
            elif key == 'robot_pose_R':
                example[key] = np.zeros((3,), dtype=np.float64)
            elif key == 'robot_quat_L':
                example[key] = np.zeros((4,), dtype=np.float64)
            elif key == 'robot_quat_R':
                example[key] = np.zeros((4,), dtype=np.float64)

        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(   
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
        print("[DEBUG] Robot Controller initialized")

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
    # def servoL(self, pose, duration=0.1):   # 안씀
    #     """
    #     duration: desired time to reach pose
    #     """
    #     assert self.is_alive()
    #     assert(duration >= (1/self.frequency))
    #     pose = np.array(pose)
    #     assert pose.shape == (6,)   

    #     message = {
    #         'cmd': Command.SERVOL.value,
    #         'target_pose': pose,
    #         'duration': duration
    #     }
    #     self.input_queue.put(message)


    def schedule_waypoint(self, pose, target_time):   # 이거 사용
        assert target_time > time.time()
        pose = np.array(pose)
        # assert pose.shape == (6,)
        assert pose.shape == (18,)   

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
        print("[DEBUG] Robot Running")
        # start rtde
        robot_ip = self.robot_ip

        urdf_path = "/home/vision/dualarm_ws/src/doosan-robot2/dsr_description2/urdf/m0609.white.urdf"
        doosan_robot = rtb.ERobot.URDF(urdf_path)   

        global latest_joint_L, latest_joint_R
        latest_joint_L = None
        latest_joint_R = None
        rclpy.init(args=None)
        node = Dualarm()
        # exec = SingleThreadedExecutor()
        # exec.add_node(node)

        try:
            rclpy.spin_once(node)
            # if self.verbose:   # False
            #     print(f"[RTDEPositionalController] Connect to robot: {robot_ip}")

            # init pose
            # if self.joints_init is not None:   # None
            #     # assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)
            #     MoveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1. / self.frequency

            # current state; m, rad
            # curr_pose = rtde_r.getActualTCPPose()   # 현재 pose 가져오기; 바꾸기!
            # j = GetCurrentJoint()
            # current_joint = np.array([j.j0, j.j1, j.j2, j.j3, j.j4, j.j5]) * np.pi / 180   # rad
            # curr_se3 = rb10.fkine(current_joint)   # m, rad (SE3)
            # curr_pose = se3_to_pos_rotvec(curr_se3)
            curr_joint_L = latest_joint_L
            curr_joint_R = latest_joint_R

            curr_tcp_L = doosan_robot.fkine(curr_joint_L)
            curr_tcp_R = doosan_robot.fkine(curr_joint_R)

            curr_tcp_pose_L = curr_tcp_L.t
            curr_tcp_pose_R = curr_tcp_R.t
            curr_tcp_rotmat_L = curr_tcp_L.R
            curr_tcp_rotmat_R = curr_tcp_R.R
                
            curr_tcp_quat_L = R.from_matrix(curr_tcp_rotmat_L).as_quat()
            curr_tcp_quat_R = R.from_matrix(curr_tcp_rotmat_R).as_quat()

            if curr_tcp_quat_L[3] < 0:
                curr_tcp_quat_L = -curr_tcp_quat_L
            if curr_tcp_quat_R[3] < 0:
                curr_tcp_quat_R = -curr_tcp_quat_R

            curr_tcp_rotvec_L = R.from_quat(curr_tcp_quat_L).as_rotvec()
            curr_tcp_rotvec_R = R.from_quat(curr_tcp_quat_R).as_rotvec()

            curr_pose = np.concatenate([curr_tcp_pose_L, curr_tcp_rotvec_L, curr_tcp_pose_R, curr_tcp_rotvec_R])

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(  
                times=[curr_t],     # [ time ]
                poses=[curr_pose]   # [ [x,y,z,rx,ry,rz] ]
            )

            iter_idx = 0
            keep_running = True

            t_start = time.monotonic()   # 수동 제어 주기 맞추기

            while keep_running:   # 루프 시작
                rclpy.spin_once(node)
                # start control iteration
                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now)   # 보간 해놓고 현재 시간의 목표 pose 가져옴 (pose_L, rotvec_L, pose_R, rotvec_R)
            
                # 두산 로봇 제어                
                # pose_command 해체 
                target_pose_L = pose_command[:3]
                target_rotvec_L = pose_command[3:6]
                target_pose_R = pose_command[6:9]
                target_rotvec_R = pose_command[9:12]
                target_quat_L = R.from_rotvec(target_rotvec_L).as_quat()
                target_quat_R = R.from_rotvec(target_rotvec_R).as_quat()
                
                # 전처리
                target_L = np.concatenate([target_pose_L, target_rotvec_L])
                target_R = np.concatenate([target_pose_R, target_rotvec_R])
                target_joint_L = servoJ(doosan_robot, latest_joint_L, target_L)
                target_joint_R = servoJ(doosan_robot, latest_joint_R, target_R)

                # 토픽발사
                # print("[DEBUG] target_joint:",target_joint_L*180/np.pi, target_joint_R*180/np.pi)
                node.joint_command_publish_L(target_joint_L)
                node.joint_command_publish_R(target_joint_R)

                # current state
                curr_joint_L = latest_joint_L
                curr_joint_R = latest_joint_R
                curr_tcp_L = doosan_robot.fkine(curr_joint_L)
                curr_tcp_R = doosan_robot.fkine(curr_joint_R)

                curr_tcp_pose_L = curr_tcp_L.t
                curr_tcp_pose_R = curr_tcp_R.t
                curr_tcp_rotmat_L = curr_tcp_L.R
                curr_tcp_rotmat_R = curr_tcp_R.R
                    
                curr_tcp_quat_L = R.from_matrix(curr_tcp_rotmat_L).as_quat()
                curr_tcp_quat_R = R.from_matrix(curr_tcp_rotmat_R).as_quat()
                    
                if curr_tcp_quat_L[3] < 0:
                    curr_tcp_quat_L = -curr_tcp_quat_L
                if curr_tcp_quat_R[3] < 0:
                    curr_tcp_quat_R = -curr_tcp_quat_R
                
                curr_tcp_rotvec_L = R.from_quat(curr_tcp_quat_L).as_rotvec()
                curr_tcp_rotvec_R = R.from_quat(curr_tcp_quat_R).as_rotvec()
                # print("[DEBUG] curr_tcp_quat_L:", curr_tcp_rotvec_L)
                # print("[DEBUG] curr_tcp_quat_R:", curr_tcp_rotvec_R)
                # curr_pose = np.concatenate([curr_tcp_pose_L, curr_tcp_rotvec_L, curr_tcp_pose_R, curr_tcp_rotvec_R])

                # 현재 State 저장
                # update robot state; ringbuffer에 state 저장
                state = dict()

                for key in self.receive_keys:
                    if key == 'robot_pose_L':
                        state[key] = np.array(curr_tcp_pose_L)
                    elif key == 'robot_pose_R':
                        state[key] = np.array(curr_tcp_pose_R)
                    elif key == 'robot_quat_L':
                        state[key] = np.array(curr_tcp_quat_L)
                    elif key == 'robot_quat_R':
                        state[key] = np.array(curr_tcp_quat_R)
                        
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)   


                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()   # command 긁어옴
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
                    # elif cmd == Command.SERVOL.value:   # SERVOL: target_pose 실행 
                    #     # since curr_pose always lag behind curr_target_pose
                    #     # if we start the next interpolation with curr_pose
                    #     # the command robot receive will have discontinouity 
                    #     # and cause jittery robot behavior.
                    #     target_pose = command['target_pose']  
                    #     duration = float(command['duration']) 
                    #     curr_time = t_now + dt
                    #     t_insert = curr_time + duration
                    #     pose_interp = pose_interp.drive_to_waypoint(
                    #         pose=target_pose,
                    #         time=t_insert,
                    #         curr_time=curr_time,
                    #         max_pos_speed=self.max_pos_speed,
                    #         max_rot_speed=self.max_rot_speed
                    #     )
                    #     last_waypoint_time = t_insert
                    #     if self.verbose:
                    #         print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                    #             target_pose, duration))
                            
                    # 이걸로 제어 (n_cmd 1개씩 제어)
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']   # abs; (pose_L, rot6d_L, pose_R, rot6d_R)

                        target_position_L = target_pose[:3]   # 3d position, m
                        target_position_R = target_pose[9:12]   # 3d position, m
                        target_rotvec_L = rot6d_to_rotvec(target_pose[3:9])   # 6d rotation -> rot_vec
                        target_rotvec_R = rot6d_to_rotvec(target_pose[12:18])   # 6d rotation -> rot_vec
                        print('[DEBUG] target_6d_L:', target_pose[3:9])
                        print('[DEBUG] target_6d_R:', target_pose[12:18])
                        target_pose = np.concatenate([target_position_L, target_rotvec_L, target_position_R, target_rotvec_R])   
                        

                        # print('[DEBUG] target_pose', target_pose)
                        
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
            # terminate
            node.destroy_node()
            rclpy.shutdown()

            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
