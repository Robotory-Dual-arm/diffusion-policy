
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


from spatialmath import SE3
import spatialmath.base as smb
from std_msgs.msg import Int32, Float64, Float64MultiArray
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState, MultiDOFJointState

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
from collections import deque
import threading

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


# True: impedance controller (/right_dsr_controller/task_space_command)
# False: position controller (/right_dsr_joint_controller/joint_state_command)
USE_IMPEDANCE_CONTROLLER = False


FINGER_WRENCH_KEYS = {
    'wrench_thumb_R': 0,
    'wrench_index_R': 1,
    'wrench_middle_R': 2,
    'wrench_ring_R': 3,
    'wrench_baby_R': 4,
}


def get_requested_wrench_keys(shape_meta):
    if shape_meta is None:
        return []
    return [
        key for key in shape_meta.get('obs', {}).keys()
        if key.startswith('wrench_')
    ]


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

    if np.linalg.norm(dq[:3]) > acc_pos_limit:
        dq[:3] *= acc_pos_limit / np.linalg.norm(dq[:3])
    
    if np.linalg.norm(dq[3:]) > acc_rot_limit:
        dq[3:] *= acc_rot_limit / np.linalg.norm(dq[3:])

    next_joint = current_joint + dq * 0.5
    return next_joint   # rad


class Dualarm(Node):
    def __init__(self, shape_meta=None):
        super().__init__('dualarm_node')
        self.callback_group = ReentrantCallbackGroup()
        self.shape_meta = shape_meta
        self.required_wrench_keys = get_requested_wrench_keys(shape_meta)
        self.requires_wrist_wrench = 'wrench_wrist_R' in self.required_wrench_keys
        self.required_finger_indices = [
            FINGER_WRENCH_KEYS[key]
            for key in self.required_wrench_keys
            if key in FINGER_WRENCH_KEYS
        ]

        self.joint_name = [f"left_joint_{i}" for i in range(1,7)] + \
                            [f"right_joint_{i}" for i in range(1,7)]
        self.right_joint_name = [f"right_joint_{i}" for i in range(1,7)]
        self.dsr_joint_name = [f"joint_{i}" for i in range(1,7)]
        self.joint_name_debug_printed = False
        
        # self.hand_name = [f"left_thumb_joint{i}" for i in range(1,4)] + \
        #                  [f"left_index_joint{i}" for i in range(1,4)] + \
        #                  [f"left_middle_joint{i}" for i in range(1,4)] + \
        #                  [f"left_ring_joint{i}" for i in range(1,4)] + \
        #                  [f"left_baby_joint{i}" for i in range(1,4)] + \
        #                  [f"right_thumb_joint{i}" for i in range(1,4)] + \
        #                  [f"right_index_joint{i}" for i in range(1,4)] + \
        #                  [f"right_middle_joint{i}" for i in range(1,4)] + \
        #                  [f"right_ring_joint{i}" for i in range(1,4)] + \
        #                  [f"right_baby_joint{i}" for i in range(1,4)]

        # self.use_left_hand_index = [0,1,2,4,7,10] 0-14
        # self.use_right_hand_index = [15,16,17,19,20,22,23,25,26,28,29] # 15-29

        self.joint_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10,
            callback_group=self.callback_group
        )

        # 오른손 wrench wrist
        self.wrench_wrist_R_subscriber = self.create_subscription(
            WrenchStamped,
            '/aft_sensor2/wrench',
            self.wrench_wrist_R_callback,
            10,
            callback_group=self.callback_group
        )
        # 오른손 wrench finger
        # self.wrench_hand_R_subscriber = self.create_subscription(
        #     MultiDOFJointState,
        #     '/right_ft_sensor_broadcaster/wrench',
        #     self.wrench_hand_R_callback,
        #     10,
        #     callback_group=self.callback_group
        # )

        # ===== Wrench EMA 설정 =====
        self.WRENCH_EMA_ALPHA = 0.03  # 0~1, 작을수록 smooth
        self.raw_wrench_wrist_R = None
        self.raw_wrench_fingers_R = [None] * 5  # thumb, index, middle, ring, baby

        # ===== Wrench Offset Calibration =====
        self.WRENCH_CALIB_COUNT = 10  # 초기 몇 번의 값을 모을지
        self.wrench_calib_samples_wrist_R = []
        self.wrench_calib_samples_fingers_R = [[] for _ in range(5)]  # thumb, index, middle, ring, baby
        self.wrench_calibrated = len(self.required_wrench_keys) == 0
        self.wrench_offset_wrist_R = None
        self.wrench_offset_fingers_R = [None] * 5

        # 250Hz 타이머로 EMA 갱신
        self.wrench_ema_timer = self.create_timer(
            1.0 / 250.0,  # 250Hz
            self.wrench_ema_update,
            callback_group=self.callback_group
        )

        # ===== 32-frame wrench history buffer (250Hz) =====
        self.wrench_lock = threading.Lock()
        self.wrench_hist_32_wrist_R = deque(maxlen=32)   
        self.wrench_hist_32_fingers_R = [
            deque(maxlen=32) for _ in range(5)  # thumb, index, middle, ring, baby
        ]


        # self.joint_command_publisher_L = self.create_publisher(
        #     JointState,
        #     '/left_dsr_joint_controller/joint_state_command',
        #     10
        # )
        self.joint_command_publisher_R = self.create_publisher(
            JointState,
            '/right_dsr_joint_controller/joint_state_command',
            10
        )
        self.task_space_command_publisher_R = self.create_publisher(
            PoseStamped,
            '/right_dsr_controller/task_space_command',
            10
        )
        # self.hand_command_publisher = self.create_publisher(
        #     JointState,
        #     '/aidin_dualarm_joint_controller/joint_state_command',
        #     10
        # )
        
        # trajectory 확인용
        self.tcp_publisher_R = self.create_publisher(
            PoseStamped,
            '/TCP_target_pose_R',
            10
        )
        control_name = 'impedance' if USE_IMPEDANCE_CONTROLLER else 'position'
        print(f"[Control] right arm control mode: {control_name}")

    def joint_callback(self, msg):
        global latest_joint_R, latest_hand_R
    
        joint_mapping = {n: p for n, p in zip(msg.name, msg.position)}
        joint_position = [joint_mapping.get(j) for j in self.right_joint_name]
        if any(x is None for x in joint_position):
            joint_position = [joint_mapping.get(j) for j in self.dsr_joint_name]

        if any(x is None for x in joint_position):
            if len(msg.position) == 6:
                joint_position = list(msg.position)
            else:
                if not self.joint_name_debug_printed:
                    print(f"[WARN] Cannot map right arm joints from /joint_states names: {list(msg.name)}")
                    self.joint_name_debug_printed = True
                return

        joint_position = np.asarray(joint_position, dtype=np.float64)
        if joint_position.shape != (6,) or not np.all(np.isfinite(joint_position)):
            if not self.joint_name_debug_printed:
                print(f"[WARN] Invalid right arm joint positions: {joint_position}")
                self.joint_name_debug_printed = True
            return

        latest_joint_R = joint_position
        # hand_position = [joint_mapping.get(j) for j in self.hand_name]
        # latest_joint_L = joint_position[:6]
        # latest_hand_L = hand_position[0:3] + hand_position[4:6] + hand_position[7:9]
        # latest_hand_R = np.array([hand_position[i] for i in self.use_right_hand_index])
    
    # def wrench_wrist_L_callback(self, msg):
    #     global latest_wrench_wrist_L
    #     latest_wrench_wrist_L = np.array([
    #         msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
    #         msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
    #     ])
    def wrench_wrist_R_callback(self, msg):
        self.raw_wrench_wrist_R = np.array([
            msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    # def wrench_hand_R_callback(self, msg):
    #     self.raw_wrench_fingers_R = [
    #         np.array([msg.wrench[i].force.x,  msg.wrench[i].force.y,  msg.wrench[i].force.z,
    #                   msg.wrench[i].torque.x, msg.wrench[i].torque.y, msg.wrench[i].torque.z])
    #         for i in range(5)
    #     ]

    def _has_required_wrench_samples(self):
        if len(self.required_wrench_keys) == 0:
            return False
        if self.requires_wrist_wrench and self.raw_wrench_wrist_R is None:
            return False
        for finger_idx in self.required_finger_indices:
            if self.raw_wrench_fingers_R[finger_idx] is None:
                return False
        return True

    def wrench_ema_update(self):

        global latest_wrench_wrist_R, latest_wrench_fingers_R
        global latest_wrench_thumb_R, latest_wrench_index_R, latest_wrench_middle_R, latest_wrench_ring_R, latest_wrench_baby_R
        a = self.WRENCH_EMA_ALPHA

        # ===== Calibration Phase: 초기 N번 값을 모아서 offset 계산 =====
        if not self.wrench_calibrated:
            if self._has_required_wrench_samples():
                if self.requires_wrist_wrench:
                    self.wrench_calib_samples_wrist_R.append(self.raw_wrench_wrist_R.copy())
                for i in self.required_finger_indices:
                    self.wrench_calib_samples_fingers_R[i].append(self.raw_wrench_fingers_R[i].copy())

                n_calib_samples = 0
                if self.requires_wrist_wrench:
                    n_calib_samples = len(self.wrench_calib_samples_wrist_R)
                elif self.required_finger_indices:
                    first_finger_idx = self.required_finger_indices[0]
                    n_calib_samples = len(self.wrench_calib_samples_fingers_R[first_finger_idx])

                if n_calib_samples >= self.WRENCH_CALIB_COUNT:

                    if self.requires_wrist_wrench:
                        self.wrench_offset_wrist_R = np.mean(self.wrench_calib_samples_wrist_R, axis=0)
                    for i in self.required_finger_indices:
                        self.wrench_offset_fingers_R[i] = np.mean(self.wrench_calib_samples_fingers_R[i], axis=0)
                    self.wrench_calibrated = True
                    print(f"[Wrench] Calibration done! ({self.WRENCH_CALIB_COUNT} samples)")
                    if self.requires_wrist_wrench:
                        print(f"[Wrench] Offset wrist: {self.wrench_offset_wrist_R}")
                    if self.required_finger_indices:
                        print(f"[Wrench] Offset fingers: {[o.tolist() if o is not None else None for o in self.wrench_offset_fingers_R]}")
            return  

        # ===== EMA Phase: offset 적용 후 EMA 갱신 =====
        # wrist
        if self.requires_wrist_wrench and self.raw_wrench_wrist_R is not None:
            corrected_wrist_R = self.raw_wrench_wrist_R - self.wrench_offset_wrist_R
            if latest_wrench_wrist_R is None:
                latest_wrench_wrist_R = corrected_wrist_R.copy()
            else:
                latest_wrench_wrist_R = a * corrected_wrist_R + (1 - a) * latest_wrench_wrist_R
        # fingers
        if self.required_finger_indices:
            corrected_fingers_R = [None] * 5
            for i in self.required_finger_indices:
                corrected_fingers_R[i] = self.raw_wrench_fingers_R[i] - self.wrench_offset_fingers_R[i]
            if latest_wrench_fingers_R is None:
                latest_wrench_fingers_R = [
                    corrected_fingers_R[i].copy() if corrected_fingers_R[i] is not None else None
                    for i in range(5)
                ]
              
            else:
                latest_wrench_fingers_R = [
                    a * corrected_fingers_R[i] + (1-a) * latest_wrench_fingers_R[i]
                    if corrected_fingers_R[i] is not None else latest_wrench_fingers_R[i]
                    for i in range(5)
                ]
            
            # for usage
            if 0 in self.required_finger_indices:
                latest_wrench_thumb_R = latest_wrench_fingers_R[0][2:3] # fz
            if 1 in self.required_finger_indices:
                latest_wrench_index_R = latest_wrench_fingers_R[1][2:3] # fz
            if 2 in self.required_finger_indices:
                latest_wrench_middle_R = latest_wrench_fingers_R[2][2:3] # fz
            if 3 in self.required_finger_indices:
                latest_wrench_ring_R = latest_wrench_fingers_R[3][2:3] # fz
            if 4 in self.required_finger_indices:
                latest_wrench_baby_R = latest_wrench_fingers_R[4][2:3] # fz

        # ===== Append to 32-frame history (thread-safe) =====
        if self.wrench_calibrated:
            with self.wrench_lock:
                if latest_wrench_wrist_R is not None:
                    self.wrench_hist_32_wrist_R.append(latest_wrench_wrist_R.copy())
                if latest_wrench_thumb_R is not None:
                    self.wrench_hist_32_fingers_R[0].append(latest_wrench_thumb_R.copy())
                if latest_wrench_index_R is not None:
                    self.wrench_hist_32_fingers_R[1].append(latest_wrench_index_R.copy())
                if latest_wrench_middle_R is not None:
                    self.wrench_hist_32_fingers_R[2].append(latest_wrench_middle_R.copy())
                if latest_wrench_ring_R is not None:
                    self.wrench_hist_32_fingers_R[3].append(latest_wrench_ring_R.copy())
                if latest_wrench_baby_R is not None:
                    self.wrench_hist_32_fingers_R[4].append(latest_wrench_baby_R.copy())

    # def joint_command_publish_L(self, joint_position):
    #     msg = JointState()
    #     msg.name = self.joint_name[:6]
    #     joint_position = [float(x) for x in joint_position]
    #     msg.position = joint_position
    #     self.joint_command_publisher_L.publish(msg)
    def joint_command_publish_R(self, joint_position):
        msg = JointState()
        msg.name = self.right_joint_name
        joint_position = [float(x) for x in joint_position]
        msg.position = joint_position
        self.joint_command_publisher_R.publish(msg)

    def task_space_command_publish_R(self, tcp_pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
        quat = R.from_rotvec(tcp_pose[3:6]).as_quat()

        msg.pose.position.x = float(tcp_pose[0] * 1000.0)
        msg.pose.position.y = float(tcp_pose[1] * 1000.0)
        msg.pose.position.z = float(tcp_pose[2] * 1000.0)
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.task_space_command_publisher_R.publish(msg)

    # def hand_command_publish(self, hand_position):
    #     msg = JointState()
    #     msg.name = self.hand_name
    #     hand_data = np.zeros(30, dtype=float)
    #     hand_data[self.use_right_hand_index] = [float(x) for x in hand_position[:]]
    #     msg.position = hand_data.tolist()
    #     self.hand_command_publisher.publish(msg)

    # trajectory 확인용
    def tcp_pose_publish_R(self, tcp_pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(tcp_pose[0])
        msg.pose.position.y = float(tcp_pose[1])
        msg.pose.position.z = float(tcp_pose[2])
        
        self.tcp_publisher_R.publish(msg)

    def get_wrench_hist_32(self, shape_meta=None):
        """
        Get recent 32-frame wrench history for encoder-style wrench obs.
        Only obs entries with type='wrench' should use this output.
        """
        result = {}
        
        obs_meta = shape_meta.get('obs', {}) if shape_meta is not None else {}
        target_keys = [
            key for key, attr in obs_meta.items()
            if key.startswith('wrench_') and attr.get('type', 'low_dim') == 'wrench'
        ]
        
        with self.wrench_lock:
            # Wrist
            if target_keys is None or 'wrench_wrist_R' in target_keys:
                arr_wrist = np.array(list(self.wrench_hist_32_wrist_R), dtype=np.float32)  # (L, 6)
                if arr_wrist.shape[0] == 0:
                    ch = shape_meta['obs']['wrench_wrist_R']['shape'][0]
                    arr_wrist = np.zeros((1, ch), dtype=np.float32)
                if arr_wrist.shape[0] < 32:
                    pad = np.repeat(arr_wrist[:1], 32 - arr_wrist.shape[0], axis=0)
                    arr_wrist = np.concatenate([pad, arr_wrist], axis=0)
                result['wrench_wrist_R'] = arr_wrist[-32:].T  # (ch, 32)
            
            # Fingers
            finger_names = ['thumb', 'index', 'middle', 'ring', 'baby']
            for i, fname in enumerate(finger_names):
                key_name = f'wrench_{fname}_R'
                if target_keys is None or key_name in target_keys:
                    arr_f = np.array(list(self.wrench_hist_32_fingers_R[i]), dtype=np.float32)
                    if arr_f.shape[0] == 0:
                        ch = shape_meta['obs'][key_name]['shape'][0] 
                        arr_f = np.zeros((1, ch), dtype=np.float32)
                    if arr_f.shape[0] < 32:
                        pad = np.repeat(arr_f[:1], 32 - arr_f.shape[0], axis=0)
                        arr_f = np.concatenate([pad, arr_f], axis=0)
                    result[key_name] = arr_f[-32:].T  # (ch, 32)
        
        return result

    def get_wrench_low_dim(self, shape_meta=None):
        """
        Get latest 6-axis/1-axis wrench values for low_dim-style wrench obs.
        Only obs entries with type='low_dim' should use this output.
        """
        obs_meta = shape_meta.get('obs', {}) if shape_meta is not None else {}
        target_keys = [
            key for key, attr in obs_meta.items()
            if key.startswith('wrench_') and attr.get('type', 'low_dim') == 'low_dim'
        ]

        result = {}
        if 'wrench_wrist_R' in target_keys:
            if latest_wrench_wrist_R is None:
                ch = obs_meta['wrench_wrist_R']['shape'][0]
                result['wrench_wrist_R'] = np.zeros((ch,), dtype=np.float32)
            else:
                result['wrench_wrist_R'] = latest_wrench_wrist_R.astype(np.float32)

        finger_values = {
            'wrench_thumb_R': latest_wrench_thumb_R,
            'wrench_index_R': latest_wrench_index_R,
            'wrench_middle_R': latest_wrench_middle_R,
            'wrench_ring_R': latest_wrench_ring_R,
            'wrench_baby_R': latest_wrench_baby_R,
        }
        for key, value in finger_values.items():
            if key not in target_keys:
                continue
            if value is None:
                ch = obs_meta[key]['shape'][0]
                result[key] = np.zeros((ch,), dtype=np.float32)
            else:
                result[key] = value.astype(np.float32)

        return result

    def get_wrench_state(self, shape_meta=None):
        """
        Return wrench obs in the exact format requested by shape_meta.
        - type='low_dim' -> latest wrench vector, e.g. (6,)
        - type='wrench'  -> recent wrench history, e.g. (6, 32)
        """
        result = {}
        result.update(self.get_wrench_low_dim(shape_meta=shape_meta))
        result.update(self.get_wrench_hist_32(shape_meta=shape_meta))
        return result

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
            shape_meta=None,
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
        self.shape_meta = shape_meta

        # build input queue; action 담아놓을 메모리
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((shape_meta['action']['shape'][0],), dtype=np.float64),
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
            receive_keys = []
            if shape_meta is not None:
                for key, obs in shape_meta.get('obs', {}).items():
                    obs_type = obs.get('type', 'low_dim')
                    if obs_type in ('low_dim', 'wrench'):
                        receive_keys.append(key)

            if len(receive_keys) == 0:
                receive_keys = [
                    # 'robot_pose_L',
                    # 'robot_quat_L',
                    'robot_pose_R',
                    'robot_quat_R',
                    # 'hand_pose_L',
                    # 'hand_pose_R',
                    'wrench_wrist_R',
                    'wrench_thumb_R',
                    'wrench_index_R',
                    'wrench_middle_R',
                    'wrench_ring_R',
                    'wrench_baby_R'
                ]
        
        example = dict()
        default_obs_shapes = {
            'robot_pose_R': (3,),
            'robot_quat_R': (4,),
            # 'hand_pose_R': (11,),
            'wrench_wrist_R': (6, 32),
            'wrench_thumb_R': (1, 32),
            'wrench_index_R': (1, 32),
            'wrench_middle_R': (1, 32),
            'wrench_ring_R': (1, 32),
            'wrench_baby_R': (1, 32),
        }
        for key in receive_keys:
            shape = None

            # shape_meta에서 각 키의 shape 정보를 읽어오기
            if shape_meta is not None and key in shape_meta.get('obs', {}):
                obs = shape_meta['obs'][key]
                obs_type = obs.get('type', 'low_dim')  # 기본값은 'low_dim'

                if obs_type == 'low_dim':
                    obs_shape = obs.get('shape', None)
                    if obs_shape is not None:
                        shape = (obs_shape[0],)
                elif obs_type == 'wrench':
                    obs_shape = obs.get('shape', None)
                    if obs_shape is not None:
                        shape = tuple(obs_shape)

            if shape is None and key in default_obs_shapes:
                shape = default_obs_shapes[key]

            if shape is not None:
                example[key] = np.zeros(shape, dtype=np.float64)
                       
                         
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


    def schedule_waypoint(self, pose, target_time):   # 이거 사용
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (self.shape_meta['action']['shape'][0],)

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

        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "m0609.white.urdf"))
        doosan_robot = rtb.ERobot.URDF(urdf_path)   

        # global latest_joint_R, latest_hand_R
        # latest_joint_R, latest_hand_R = None, None
        global latest_joint_R
        latest_joint_R = None

        global latest_wrench_wrist_R, latest_wrench_fingers_R
        global latest_wrench_thumb_R, latest_wrench_index_R, latest_wrench_middle_R, latest_wrench_ring_R, latest_wrench_baby_R
        
        latest_wrench_wrist_R, latest_wrench_fingers_R = None, None
        latest_wrench_thumb_R, latest_wrench_index_R, latest_wrench_middle_R, latest_wrench_ring_R, latest_wrench_baby_R = None, None, None, None, None

        rclpy.init(args=None)
        node = Dualarm(shape_meta=self.shape_meta)
        self.dualarm_node = node  # Store reference for accessing wrench history

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)

        try:
            print("[DEBUG] Waiting for initial data...")
            while latest_joint_R is None:
                executor.spin_once(timeout_sec=0.01)
            print("[DEBUG] All initial data received!")
         
            # main loop
            dt = 1. / self.frequency

            # curr_joint_L = latest_joint_L
            curr_joint_R = latest_joint_R
            # curr_hand_L = latest_hand_L
            # curr_hand_R = latest_hand_R

            # curr_tcp_L = doosan_robot.fkine(curr_joint_L)
            curr_tcp_R = doosan_robot.fkine(curr_joint_R)

            # curr_tcp_pose_L = curr_tcp_L.t
            curr_tcp_pose_R = curr_tcp_R.t
            # curr_tcp_rotmat_L = curr_tcp_L.R
            curr_tcp_rotmat_R = curr_tcp_R.R
                
            # curr_tcp_quat_L = R.from_matrix(curr_tcp_rotmat_L).as_quat()
            curr_tcp_quat_R = R.from_matrix(curr_tcp_rotmat_R).as_quat()

            # if curr_tcp_quat_L[3] < 0:
            #     curr_tcp_quat_L = -curr_tcp_quat_L
            if curr_tcp_quat_R[3] < 0:
                curr_tcp_quat_R = -curr_tcp_quat_R

            # curr_tcp_rotvec_L = R.from_quat(curr_tcp_quat_L).as_rotvec()
            curr_tcp_rotvec_R = R.from_quat(curr_tcp_quat_R).as_rotvec()

            # curr_pose = np.concatenate([curr_tcp_pose_L, curr_tcp_rotvec_L, curr_tcp_pose_R, curr_tcp_rotvec_R, curr_hand_L, curr_hand_R])
            curr_pose = np.concatenate([curr_tcp_pose_R, curr_tcp_rotvec_R])
            print("[DEBUG] curr_pose:", curr_pose)
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(  
                times=[curr_t],     # [ time ]
                poses=[curr_pose],  # [ [x,y,z,rx,ry,rz] ]
                action_type='rightarm'
            )

            iter_idx = 0
            keep_running = True

            t_start = time.monotonic()   # 수동 제어 주기 맞추기

            # trajectory 확인용
            cmd_index = 0

            while keep_running:   # 루프 시작
                
                # start control iteration
                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                executor.spin_once(timeout_sec=0.001)
                pose_command = pose_interp(t_now)   # 보간 해놓고 현재 시간의 목표 pose 가져옴
            
                # 두산 로봇 제어                
                # pose_command 해체 
                target_pose_R = pose_command[0:3]
                target_rotvec_R = pose_command[3:6]
                target_hand_R = pose_command[6:]
                
                target_R = np.concatenate([target_pose_R, target_rotvec_R])

                # 토픽발사
                if USE_IMPEDANCE_CONTROLLER:
                    node.task_space_command_publish_R(target_R)
                else:
                    target_joint_R = servoJ(doosan_robot, latest_joint_R, target_R)
                    node.joint_command_publish_R(target_joint_R)
                # node.hand_command_publish(np.concatenate([target_hand_R]))   

                # trajectory 확인용
                # node.tcp_pose_publish_R(target_pose_R)

                # current state
                # curr_joint_L = latest_joint_L
                curr_joint_R = latest_joint_R
                # curr_hand_L = latest_hand_L
                # curr_hand_R = latest_hand_R

                # curr_tcp_L = doosan_robot.fkine(curr_joint_L)
                curr_tcp_R = doosan_robot.fkine(curr_joint_R)

                # curr_tcp_pose_L = curr_tcp_L.t
                curr_tcp_pose_R = curr_tcp_R.t
                # curr_tcp_rotmat_L = curr_tcp_L.R
                curr_tcp_rotmat_R = curr_tcp_R.R
                    
                # curr_tcp_quat_L = R.from_matrix(curr_tcp_rotmat_L).as_quat()
                curr_tcp_quat_R = R.from_matrix(curr_tcp_rotmat_R).as_quat()
                    
                # if curr_tcp_quat_L[3] < 0:
                #     curr_tcp_quat_L = -curr_tcp_quat_L
                if curr_tcp_quat_R[3] < 0:
                    curr_tcp_quat_R = -curr_tcp_quat_R
                
                # curr_tcp_rotvec_L = R.from_quat(curr_tcp_quat_L).as_rotvec()
                curr_tcp_rotvec_R = R.from_quat(curr_tcp_quat_R).as_rotvec()
                # curr_pose = np.concatenate([curr_tcp_pose_L, curr_tcp_rotvec_L, curr_tcp_pose_R, curr_tcp_rotvec_R])

                wrench_state = node.get_wrench_state(shape_meta=self.shape_meta)
                
                # 현재 State 저장
                # update robot state; ringbuffer에 state 저장
                state = dict()

                for key in self.receive_keys:
                    # if key == 'robot_pose_L':
                    #     state[key] = np.array(curr_tcp_pose_L)
                    # elif key == 'robot_quat_L':
                    #     state[key] = np.array(curr_tcp_quat_L)
                    if key == 'robot_pose_R':
                        state[key] = np.array(curr_tcp_pose_R)
                    elif key == 'robot_quat_R':
                        state[key] = np.array(curr_tcp_quat_R)
                    # elif key == 'hand_pose_L':
                    #     state[key] = np.array(curr_hand_L)
                    # elif key == 'hand_pose_R':
                    #     state[key] = np.array(curr_hand_R)
                    elif key in wrench_state:
                        state[key] = np.array(wrench_state[key], dtype=np.float64)
                    
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)   

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()   # command 긁어옴
                    n_cmd = len(commands['cmd'])
                    print("[DEBUG] received n_cmd:", n_cmd)
                                        
                    
                except Empty:
                    n_cmd = 0

                # execute commands
                # action 한번에 참
                
                for i in range(n_cmd):   # 가져온 cmd 수만큼 실행
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:   # STOP: 그만두기
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                            
                    # 이걸로 제어 (n_cmd 1개씩 제어)
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']   # abs; (pose_R, rot6d_R, hand_R)

                        target_position_R = target_pose[0:3]   # 3d position, m
                        target_rotvec_R = rot6d_to_rotvec(target_pose[3:9])   # 6d rotation -> rot_vec
                        # target_hand_R = target_pose[9:]

                        target_pose = np.concatenate([target_position_R, target_rotvec_R])   # (15,)
                        print("[DEBUG] target_pose: ", target_pose)

                        if cmd_index < 12:
                            node.tcp_pose_publish_R(target_position_R)
                        cmd_index += 1
                        if cmd_index == 14:
                            cmd_index = 0

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
                            last_waypoint_time=last_waypoint_time,
                            action_type='rightarm'
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break
                                
                # regulate frequency
                t_elapsed = time.monotonic() - t_now
                sleep_time = dt - t_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)   # 수동 제어 주기

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()   # 준비완료; wait하던거 실행됨
                iter_idx += 1


        finally:
            # terminate
            node.destroy_node()
            rclpy.shutdown()

            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
