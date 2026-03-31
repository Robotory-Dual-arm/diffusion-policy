import os
import sys
import h5py
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import cv2
import time
from datetime import datetime
import argparse
import multiprocessing
from multiprocessing import Process, Queue, Event
import queue

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState, MultiDOFJointState
from geometry_msgs.msg import WrenchStamped
from rclpy.logging import get_logger
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import threading

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import UInt8
from rclpy.duration import Duration

# Import ForceVisualizer
from force_visualizer import ForceVisualizer


sys.path.append('/home/vision/dualarm_ws/src/doosan-robot2/dsr_common2/imp')
# from DSR_ROBOT2 import (
#             get_current_posj, get_current_posx, servoj, 
#             set_robot_mode, ROBOT_MODE_AUTONOMOUS,get_current_tool_flange_posx,
#             DR_SERVO_OVERRIDE, fkin, ikin
#         )

_logger = get_logger('Logger')

# Add the directory containing DSR_ROBOT2.py to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# 데이터 저장 경로
today = datetime.now().strftime('%y%m%d_%H%M')
data_dir = f'/data/{today}'

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)



def init_buffer():   
    return {
        'observations': {
            ## Robot and Image data (20Hz)
            'joint_L': [], # abs, rad
            'joint_R': [], # abs, rad
            
            'hand_L': [], # abs, rad, [thumb(3), index(3), middle(3), ring(3), baby(3)]
            'hand_R': [], # abs, rad, [thumb(3), index(3), middle(3), ring(3), baby(3)]

            'image_H': [], # D455  (640x480) (BGR) (Head)
            'image_T': [], # D435  (640x480) (BGR) (Table)
            # 'image_L': [], # D405  (640x480) (BGR) (Left)
            'image_R': [], # D405  (640x480) (BGR) (Right)

            'timestamp_robot': [],


            ## Wrench data (250Hz)
            'wrench_wrist_L': [], # [fx, fy, fz, tx, ty, tz]
            'wrench_thumb_L': [],  # [fx, fy, fz, tx, ty, tz]
            'wrench_index_L': [],  # [fx, fy, fz, tx, ty, tz]
            'wrench_middle_L': [], # [fx, fy, fz, tx, ty, tz]
            'wrench_ring_L': [],   # [fx, fy, fz, tx, ty, tz]
            'wrench_baby_L': [],   # [fx, fy, fz, tx, ty, tz]

            'wrench_wrist_R': [], # [fx, fy, fz, tx, ty, tz]
            'wrench_thumb_R': [],  # [fx, fy, fz, tx, ty, tz]
            'wrench_index_R': [],  # [fx, fy, fz, tx, ty, tz]
            'wrench_middle_R': [], # [fx, fy, fz, tx, ty, tz]
            'wrench_ring_R': [],   # [fx, fy, fz, tx, ty, tz]
            'wrench_baby_R': [],   # [fx, fy, fz, tx, ty, tz]

            'joint_torque_L': [], 
            'joint_torque_R': [],

            'timestamp_wrench': []
        }
    }   # image, pose : 20Hz / wrench : 250Hz 

def save_wrench_data(buffer, last_save_time):
    
    # Left arm wrench data
    buffer['observations']['wrench_wrist_L'].append(latest_wrench_aft_L)
    buffer['observations']['wrench_thumb_L'].append(latest_wrench_thumb_L)
    buffer['observations']['wrench_index_L'].append(latest_wrench_index_L)
    buffer['observations']['wrench_middle_L'].append(latest_wrench_middle_L)
    buffer['observations']['wrench_ring_L'].append(latest_wrench_ring_L)
    buffer['observations']['wrench_baby_L'].append(latest_wrench_baby_L)
    buffer['observations']['joint_torque_L'].append(latest_joint_torque_L)
    
    # Right arm wrench data
    buffer['observations']['wrench_wrist_R'].append(latest_wrench_aft_R)
    buffer['observations']['wrench_thumb_R'].append(latest_wrench_thumb_R)
    buffer['observations']['wrench_index_R'].append(latest_wrench_index_R)
    buffer['observations']['wrench_middle_R'].append(latest_wrench_middle_R)
    buffer['observations']['wrench_ring_R'].append(latest_wrench_ring_R)
    buffer['observations']['wrench_baby_R'].append(latest_wrench_baby_R)
    buffer['observations']['joint_torque_R'].append(latest_joint_torque_R)
    
    # Timestamp
    buffer['observations']['timestamp_wrench'].append(last_save_time)


def save_robot_data(buffer, images, last_save_time):
    
    # Joint positions
    buffer['observations']['joint_L'].append(latest_joint_L)
    buffer['observations']['joint_R'].append(latest_joint_R)
    
    # Hand positions
    buffer['observations']['hand_L'].append(latest_hand_L)
    buffer['observations']['hand_R'].append(latest_hand_R)
    
    # Images
    buffer['observations']['image_H'].append(images[0].copy())
    buffer['observations']['image_T'].append(images[1].copy())
    buffer['observations']['image_R'].append(images[2].copy())
    
    # Timestamp
    buffer['observations']['timestamp_robot'].append(last_save_time)
    

def get_device_serials():
    ctx = rs.context()
    serials = []
    for device in ctx.query_devices():
        serials.append(device.get_info(rs.camera_info.serial_number))
    return serials

def on_press(key):
    global recording, terminal, teleop_controller, homming
    try:
        if key.char == 's':
            if not recording:  # 녹화 시작
                recording = True
                print("Start recording")
                print("homming:", homming)
                if homming:
                    teleop_controller.send_teleop_command(1)
            
        elif key.char == 'q':
            if recording:  # 녹화 중지
                recording = False
                print("Stop recording")
                if homming:
                    teleop_controller.send_teleop_command(3)
                
        elif key.char == 't':
            terminal = True
    except AttributeError:
        pass
    
def make_demo_n(buffer):
    n = len(data.keys())
    demo_n = data.create_group(f'demo_{n}')
    obs = demo_n.create_group('observations')

    for name, values in buffer['observations'].items():
        
        # Convert to numpy array and handle potential None values
        arr = np.array(values)
        
        # If the array has object dtype, it likely contains None values or inconsistent shapes
        if arr.dtype == np.object_:
            # Filter out None values and convert to proper numeric array
            filtered_values = [v for v in values if v is not None]
            if len(filtered_values) > 0:
                arr = np.array(filtered_values)
            else:
                print(f"Warning: {name} contains only None values, skipping...")
                continue
        
        obs.create_dataset(name, data=arr)

    # Flush data to disk immediately after saving demo
    f.flush()
    print(f"stored_demo_{n}, {len(buffer['observations']['joint_R'])} robot steps, {len(buffer['observations']['wrench_wrist_R'])} wrench steps")
    print(f"Data flushed to disk: {f.filename}")
    


class Pipeline:
    def __init__(self, serial, exposure):
        self.serial = serial
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)

        camera_model = self.get_camera_model(serial)

        # color_sensor = profile.get_device().query_sensors()[1]
        # if exposure is not None:
        #     color_sensor.set_option(rs.option.enable_auto_exposure, False)
        #     color_sensor.set_option(rs.option.exposure, exposure)
        #     print(f"Realsense camera {camera_model[-4:]}[{serial}] Exposure set to [{exposure}]")
        # else:
        #     color_sensor.set_option(rs.option.enable_auto_exposure, True)
        #     print(f"Realsense camera {camera_model[-4:]}[{serial}] Exposure set to [Auto]")

        for i in range(3):
            try: 
                self.pipeline.wait_for_frames(timeout_ms=1000)
                # print(f"Realsense camera {serial} initialized. try {i}")
            except:
                self.pipeline.stop()
                profile = self.pipeline.start(self.config)
                self.pipeline.wait_for_frames()
                # print(f"Realsense camera {serial} re-initialized. except {i}")

        # Get camera intrinsics
        self.profile = profile
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        print(f"Realsense camera {camera_model[-4:]}[{serial}] initialized.")

    def get_frame(self):

        try:
            frame = self.pipeline.wait_for_frames(timeout_ms=50)

        except RuntimeError as e:
            _logger.warning(f"Realsense timeout: {e}")
            return None

        color_frame = frame.get_color_frame()

        if not color_frame:
            _logger.warning("No color frame from the camera")
            return None
        
        color_image = np.asanyarray(color_frame.get_data())

        return color_image
    
    def get_camera_matrix(self):
        """Get camera intrinsic matrix and distortion coefficients"""
        camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]])
        dist_coeffs = np.array(self.intrinsics.coeffs)
        return camera_matrix, dist_coeffs

    def get_camera_model(self, serial):
        ctx = rs.context()
        for device in ctx.query_devices():
            device_serial = device.get_info(rs.camera_info.serial_number)
            if device_serial == serial:
                model_name = device.get_info(rs.camera_info.name)
                return model_name
        return None
    
class JointSubscriber(Node):
    def __init__(self):
        super().__init__('joint_node') 

        # Create callback group for parallel processing
        # ReentrantCallbackGroup allows multiple callbacks to execute simultaneously
        self.callback_group = ReentrantCallbackGroup()

        self.joint_name = [f"left_joint_{i}" for i in range(1,7)] + \
                          [f"right_joint_{i}" for i in range(1,7)]
        
        self.hand_name = [f"left_thumb_joint{i}" for i in range(1,4)] + \
                         [f"left_index_joint{i}" for i in range(1,4)] + \
                         [f"left_middle_joint{i}" for i in range(1,4)] + \
                         [f"left_ring_joint{i}" for i in range(1,4)] + \
                         [f"left_baby_joint{i}" for i in range(1,4)] + \
                         [f"right_thumb_joint{i}" for i in range(1,4)] + \
                         [f"right_index_joint{i}" for i in range(1,4)] + \
                         [f"right_middle_joint{i}" for i in range(1,4)] + \
                         [f"right_ring_joint{i}" for i in range(1,4)] + \
                         [f"right_baby_joint{i}" for i in range(1,4)]
        
        self.hand_id = ['thumb', 'index', 'middle', 'ring', 'baby']

        # Thread-safe lock for data access
        self.data_lock = threading.Lock()

        self.joint_subscriber = self.create_subscription(
            JointState,                
            '/joint_states',             
            self.joint_callback, 
            10,
            callback_group=self.callback_group)

        self.left_aft_sensor_sub = self.create_subscription(
            WrenchStamped,
            '/aft_sensor1/wrench', # 이거 왼쪽 맞나요???
            self.left_aft_sensor_callback,
            10,
            callback_group=self.callback_group)
        
        self.right_aft_sensor_sub = self.create_subscription(
            WrenchStamped,
            '/aft_sensor2/wrench',
            self.right_aft_sensor_callback,
            10,
            callback_group=self.callback_group)
        
        # Subscribe to each finger's FT sensor with callback group
        self.left_finger_wrench_sub = self.create_subscription(
            MultiDOFJointState,
            '/left_ft_sensor_broadcaster/wrench',
            self.left_finger_wrench_callback,
            10,
            callback_group=self.callback_group)
        
        self.right_finger_wrench_sub = self.create_subscription(
            MultiDOFJointState,
            '/right_ft_sensor_broadcaster/wrench',
            self.right_finger_wrench_callback,
            10,
            callback_group=self.callback_group)



    def joint_callback(self, msg):
        global latest_joint_L, latest_hand_L, latest_joint_torque_L, \
               latest_joint_R, latest_hand_R, latest_joint_torque_R
        
        joint_mapping = {n: p for n, p in zip(msg.name, msg.position)}
        
        joint_position = [joint_mapping.get(j) for j in self.joint_name]
        latest_joint_L = joint_position[0:6]
        latest_joint_R = joint_position[6:12] 

        # hand 추가
        hand_position = [joint_mapping.get(j) for j in self.hand_name]
        latest_hand_L = hand_position[0:15]   # thumb(3), index(3), middle(3), ring(3), baby(3)
        latest_hand_R = hand_position[15:30]   # thumb(3), index(3), middle(3), ring(3), baby(3)

        joint_torque_mapping = {n: t for n, t in zip(msg.name, msg.effort)}
        joint_torque = [joint_torque_mapping.get(j) for j in self.joint_name]
        latest_joint_torque_L = joint_torque[0:6]
        latest_joint_torque_R = joint_torque[6:12]

    def left_aft_sensor_callback(self, msg):
        """Callback for AFT sensor data"""
        global latest_wrench_aft_L
        latest_wrench_aft_L = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])
    def right_aft_sensor_callback(self, msg):
        """Callback for AFT sensor data"""
        global latest_wrench_aft_R
        latest_wrench_aft_R = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])    

    def left_finger_wrench_callback(self, msg):
        """Callback for left finger FT sensor"""
        global latest_wrench_thumb_L, latest_wrench_index_L, latest_wrench_middle_L, latest_wrench_ring_L, latest_wrench_baby_L
        
        # Assuming the message contains data for all fingers in a predefined order
        # Here we split the wrench data into 5 segments for each finger
        forces = [np.array([msg.wrench[i].force.x, msg.wrench[i].force.y, msg.wrench[i].force.z,
                            msg.wrench[i].torque.x, msg.wrench[i].torque.y, msg.wrench[i].torque.z]) 
                            for i in range(5)]

        # Assign to respective fingers (this is a placeholder; actual implementation may vary)
        latest_wrench_thumb_L = forces[0]
        latest_wrench_index_L = forces[1]
        latest_wrench_middle_L = forces[2]
        latest_wrench_ring_L = forces[3]
        latest_wrench_baby_L = forces[4]
    def right_finger_wrench_callback(self, msg):
        """Callback for right finger FT sensor"""
        global latest_wrench_thumb_R, latest_wrench_index_R, latest_wrench_middle_R, latest_wrench_ring_R, latest_wrench_baby_R
        
        # Assuming the message contains data for all fingers in a predefined order
        # Here we split the wrench data into 5 segments for each finger
        forces = [np.array([msg.wrench[i].force.x, msg.wrench[i].force.y, msg.wrench[i].force.z,
                            msg.wrench[i].torque.x, msg.wrench[i].torque.y, msg.wrench[i].torque.z]) 
                            for i in range(5)]

        # Assign to respective fingers (this is a placeholder; actual implementation may vary)
        latest_wrench_thumb_R = forces[0]
        latest_wrench_index_R = forces[1]
        latest_wrench_middle_R = forces[2]
        latest_wrench_ring_R = forces[3]
        latest_wrench_baby_R = forces[4]


class TeleopController(Node):
    def __init__(self):
        super().__init__('teleop_controller')
        self.teleop_control_pub = self.create_publisher(
            UInt8,
            '/teleop_control',
            10)
    
    def send_teleop_command(self, command):
        msg = UInt8()
        msg.data = command
        self.teleop_control_pub.publish(msg)
        
        command_names = {0: "DISABLE_VR", 1: "ENABLE_VR", 2: "HOMMING"}
        command_name = command_names.get(command, f"UNKNOWN({command})")
        print(f"Teleop command sent: {command_name} ({command})")

class RandomController(Node):
    def __init__(self):
        super().__init__('random_controller')
        self.random_control_pub = self.create_publisher(
            UInt8,
            '/random_control',
            10)
    
    def send_random_command(self, command):
        msg = UInt8()
        msg.data = command
        self.random_control_pub.publish(msg)
        
        command_names = {0: "DISABLE_RANDOM", 1: "ENABLE_RANDOM"}
        command_name = command_names.get(command, f"UNKNOWN({command})")
        print(f"Random command sent: {command_name} ({command})")

# --- Display process function for separate GUI handling ---------------------
def display_process_func(image_queue, window_names, force_visualizer_params, force_viz_camera_idx, stop_event):
    """완전히 별도 프로세스에서 화면 표시"""
    
    # 이 프로세스 내에서 ForceVisualizer 재초기화
    force_visualizer = None
    if force_visualizer_params is not None:
        try:
            force_visualizer = ForceVisualizer(
                camera_matrix=force_visualizer_params['camera_matrix'],
                dist_coeffs=force_visualizer_params['dist_coeffs'],
                T_base2cam=force_visualizer_params['T_base2cam'],
                force_scale=force_visualizer_params['force_scale']
            )
        except Exception as e:
            print(f"Display process: ForceVisualizer init failed: {e}")
    
    # OpenCV 창 생성
    for w in window_names:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w, 800, 600)
    
    while not stop_event.is_set():
        try:
            # 큐에서 데이터 가져오기
            data = image_queue.get(timeout=0.1)
            images, recording, force_viz_joints, force_viz_vector, force_finger_bar = data
            
            show_images(images, window_names, recording=recording,
                       force_visualizer=force_visualizer,
                       joint_angles=force_viz_joints,
                       force_vector=force_viz_vector,
                       force_finger_bar=force_finger_bar,
                       force_viz_camera_idx=force_viz_camera_idx,)
        except queue.Empty:
            cv2.waitKey(1)  # GUI 이벤트만 처리
        except Exception as e:
            print(f"Display process error: {e}")
    
    cv2.destroyAllWindows()
    print("Display process terminated")

# --- ADDED: Real-time camera preview helper ---------------------------------
def show_images(images, window_names, recording=False, force_visualizer=None, 
                joint_angles=None, force_vector=None, force_finger_bar=None, force_viz_camera_idx=None):
    """
    OpenCV 창에 각 카메라 프레임을 실시간 표시.
    images: [np.ndarray or None, ...]
    window_names: [str, ...]
    recording: 녹화 상태 표시용
    force_visualizer: ForceVisualizer instance (optional)
    joint_angles: Current joint angles in radians (optional)
    force_vector: Current force vector [fx, fy, fz] (optional)
    force_viz_camera_idx: Index of camera to apply force visualization (optional)
    """
    for idx, (name, img) in enumerate(zip(window_names, images)):
        if img is None:
            continue
        # 상태 오버레이(REC/LIVE)
        overlay = img.copy()
        
        # Apply force visualization only to the specified camera
        if (idx == force_viz_camera_idx and 
            force_visualizer is not None and 
            joint_angles is not None and 
            force_vector is not None):
            try:
                overlay = force_visualizer.visualize_force_on_image(
                    overlay, joint_angles, force_vector, show_magnitude=True
                )
            except Exception as e:
                # If force visualization fails, continue with normal display
                print(f"Force visualization error: {e}")
                pass
        
        # === 오른손 손가락 Force 막대 그래프 추가 (왼쪽 하단) ===
        if idx == force_viz_camera_idx and force_finger_bar is not None:
            overlay = visualize_finger_force_bar_on_image(overlay, force_finger_bar)
        
        # Add recording status (position adjusted to not overlap with force info on force viz camera)
        status_y = 30 if idx != force_viz_camera_idx else overlay.shape[0] - 20
        cv2.putText(
            overlay,
            'REC' if recording else 'LIVE',
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if recording else (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        cv2.imshow(name, overlay)
    # 창 이벤트 처리
    
    cv2.waitKey(1)

def visualize_finger_force_bar_on_image(image, finger_forces_R):
    
    h, w = image.shape[:2]
    
    # 막대 그래프 설정
    bar_width = 15
    bar_spacing = 30
    max_bar_height = 100
    max_force = 50.0  # N (최대 힘 스케일)
    start_x = w - 20 - (bar_width + bar_spacing) * 3
    start_y = h - 20

    finger_labels = ['thumb', 'index', 'middle', 'ring', 'baby']
    color = (0, 0, 255)  # red bar

    # 제목
    cv2.putText(image, 'Right Hand Force (N)', (start_x-3, start_y - max_bar_height - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    
    # 오른손 손가락 그래프
    for i, (fz, label) in enumerate(zip(finger_forces_R, finger_labels)):
        if fz is None:
            continue

        fz *= -1.0
        if fz <= 0:
            fz_bar = 0
        else:
            fz_bar = min(fz, max_force)
        
        # 막대 높이 계산
        bar_height = int((fz_bar / max_force) * max_bar_height)
        bar_height = min(bar_height, max_bar_height)
        
        # 막대 위치
        x = start_x + i * bar_spacing
        y_top = start_y - bar_height
        
        # 막대 그리기
        cv2.rectangle(image, (x, start_y), (x + bar_width, y_top), color, -1)
        cv2.rectangle(image, (x, start_y), (x + bar_width, y_top), (255, 255, 255), 1)
        
        # 손가락 라벨 (막대 아래)
        cv2.putText(image, label, (x-2, start_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        # 힘 값 표시 (막대 위에)
        if bar_height > 10:
            cv2.putText(image, f'{fz:.1f}', (x-2, y_top-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return image
# -----------------------------------------------------------------------------


def main(args=None):
    global terminal, recording, teleop_controller, homming, random_mode, \
           latest_joint_L, latest_joint_R, latest_hand_L, latest_hand_R, \
           latest_wrench_aft_L, latest_wrench_thumb_L, latest_wrench_index_L, \
           latest_wrench_middle_L, latest_wrench_ring_L, latest_wrench_baby_L, \
           latest_wrench_aft_R, latest_wrench_thumb_R, latest_wrench_index_R, \
           latest_wrench_middle_R, latest_wrench_ring_R, latest_wrench_baby_R, \
           latest_joint_torque_L, latest_joint_torque_R, \
           data, f
    
    recording = False
    terminal = False
    latest_joint_L, latest_joint_R = None, None
    latest_hand_L, latest_hand_R = None, None
    latest_wrench_aft_L, latest_wrench_aft_R = None, None
    latest_wrench_thumb_L, latest_wrench_thumb_R = None, None
    latest_wrench_index_L, latest_wrench_index_R = None, None
    latest_wrench_middle_L, latest_wrench_middle_R = None, None
    latest_wrench_ring_L, latest_wrench_ring_R = None, None
    latest_wrench_baby_L, latest_wrench_baby_R = None, None
    latest_joint_torque_L, latest_joint_torque_R = None, None

    # Open HDF5 file in append mode to allow incremental saving
    hdf5_filepath = os.path.join(data_dir, 'common_data.hdf5')
    
    # Check if file exists
    if os.path.exists(hdf5_filepath):
        print(f"HDF5 file already exists: {hdf5_filepath}")
        overwrite = input("Overwrite existing file? (y/n): ").strip().lower()
        if overwrite == 'y':
            f = h5py.File(hdf5_filepath, 'w')
            data = f.create_group('data')
            print("Existing file overwritten.")
        else:
            # Open in append mode
            f = h5py.File(hdf5_filepath, 'a')
            if 'data' in f:
                data = f['data']
                print(f"Appending to existing file. Current demos: {len(data.keys())}")
            else:
                data = f.create_group('data')
                print("Created new 'data' group in existing file.")
    else:
        f = h5py.File(hdf5_filepath, 'w')
        data = f.create_group('data')
        print(f"Created new HDF5 file: {hdf5_filepath}")
    
    connected_serials = get_device_serials()
    print("Connected serials: ", connected_serials)


    ### 카메라 Serial number 설정 
    # [  Head(D455),     Front(D435),    Left(D405),     Right(D405)    Table(D435)]
    # ['242422304502', '336222070518', '218622276386', '126122270712', '151222078010']   
    serials = [ '242422304502', '151222078010', '126122270712']   # Head, Table, Left

    ### 카메라 Exposure 설정
    exposures = [40, 50, 15000] # Head, Table, Left


    if serials == None:
        serials = get_device_serials()
    print("Selected serials: ", serials)

    assert all(serial in connected_serials for serial in serials), "Selected serials not connected"
    pipelines = [Pipeline(serial, exposure) for serial, exposure in zip(serials, exposures)]

    # --- ADDED: OpenCV windows for live preview ---
    window_names = [f'Cam{i+1} ({s})' for i, s in enumerate(serials)]
    # window_names = [f'Head ({serials[0]})', f'Front ({serials[1]})', f'Right ({serials[2]})']
    # ----------------------------------------------

    # Initialize ForceVisualizer for specific camera (serial: 242422304502)
    force_visualizer_params = None
    force_viz_camera_idx = None
    FORCE_VIZ_SERIAL = '242422304502'  # Head camera serial number
    
    # Find the index of the camera with the target serial number
    for idx, serial in enumerate(serials):
        if serial == FORCE_VIZ_SERIAL:
            force_viz_camera_idx = idx
            break
    
    if force_viz_camera_idx is not None:
        head_camera_matrix, head_dist_coeffs = pipelines[force_viz_camera_idx].get_camera_matrix()
        
        # Transformation from robot base to head camera (calibrate this for your setup)
        T_base2cam = np.array(
            [[-0.01304348, -0.86555326,  0.500647  ,  0.0721608 ],
             [ 0.82407923, -0.29288214, -0.48488501, -0.04730888],
             [ 0.56632437,  0.40624821,  0.71710467, -0.25345054],
             [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
        force_visualizer_params = {
            'camera_matrix': head_camera_matrix,
            'dist_coeffs': head_dist_coeffs,
            'T_base2cam': T_base2cam,
            'force_scale': 0.01
        }
        print(f"ForceVisualizer params prepared for camera {FORCE_VIZ_SERIAL} (index {force_viz_camera_idx})")
    else:
        print(f"Warning: Camera with serial {FORCE_VIZ_SERIAL} not found. Force visualization disabled.")

    assert len(pipelines) > 0, "No cameras found"

    # ROS2 초기화
    rclpy.init(args=args)
    joint_node = JointSubscriber()
    teleop_controller = TeleopController()  # TeleopController 초기화
    random_controller = RandomController()  # RandomController 초기화
    # tcp_node = TCPSubscriber()

    # Create MultiThreadedExecutor for parallel callback processing
    # num_threads: 8개의 독립적인 스레드로 콜백 처리 (joint, wrist, 5개 손가락 센서)
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(joint_node)
    executor.add_node(teleop_controller)
    executor.add_node(random_controller)

    # Start executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # 별도 프로세스로 GUI 실행
    image_queue = Queue(maxsize=1)
    stop_event = Event()
    
    display_process = Process(
        target=display_process_func,
        args=(image_queue, window_names, force_visualizer_params, force_viz_camera_idx, stop_event)
    )
    display_process.start()

    # 키보드 리스너
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 저장 주기
    robot_hz = 20
    robot_rate = 1.0 / robot_hz
    wrench_hz = 250
    wrench_rate = 1.0 / wrench_hz

    buffer = init_buffer()
    episode_start_time = None
    

    print('s: start, q: stop, t: terminate')
    print("끝나면 무조건 t로 종료하세요!!!, 저장파일 이름도 잘 바꾸기")
    print("Starting demo session...")

    if homming:
        print("Moving robots to preset position...")
        if random_mode:
            print("Random mode enabled - sending random signal to servo controllers...")
            random_controller.send_random_command(1)  # Enable random
        teleop_controller.send_teleop_command(2)
        
        # 5초 대기
        print("Waiting 5 seconds for robots to reach preset position...")
        time.sleep(5.0)

        print("Ready for demonstration! Press 's' to start recording.")
    else:
        print("Manual mode - homming should be operated manually.")
        print("Ready for demonstration! Press 's' to start recording.")

    images = [pipeline.get_frame() for pipeline in pipelines]
    time.sleep(0.035)

    try: 
        while rclpy.ok():
            # Note: executor.spin() is running in background thread
            # No need to call rclpy.spin_once() here anymore
            # Just process the main loop
            
            loop_start_time = time.monotonic()

            if terminal:
                f.close()
                print("Terminating.")
                break

            # Prepare force visualization parameters
            force_viz_joints = latest_joint_R if latest_joint_R is not None else None
            force_viz_vector = latest_wrench_aft_R if latest_wrench_aft_R is not None else None
            force_finger_bar = [latest_wrench_thumb_R[2], 
                                latest_wrench_index_R[2],
                                latest_wrench_middle_R[2], 
                                latest_wrench_ring_R[2], 
                                latest_wrench_baby_R[2]] if latest_wrench_thumb_R is not None else None
            

            # GUI 업데이트는 별도 프로세스로 전송 (논블로킹)
            try:
                image_queue.put_nowait((images, recording, force_viz_joints, force_viz_vector, force_finger_bar))
            except queue.Full:
                # 큐가 꽉 찼으면 오래된 것 버리고 새로운 것 넣기
                try:
                    image_queue.get_nowait()
                    image_queue.put_nowait((images, recording, force_viz_joints, force_viz_vector, force_finger_bar))
                except:
                    pass
            # -------------------------------------------------------------

            if recording: # data recording (250Hz)

                if episode_start_time is None:
                    
                    episode_start_time = time.monotonic()
                    last_robot_save_time = 0.0
                    last_wrench_save_time = 0.0

                    images = [pipeline.get_frame() for pipeline in pipelines] # 최대 0.0015s
                    save_wrench_data(buffer, last_wrench_save_time)
                    save_robot_data(buffer, images, last_robot_save_time)
                   

                relative_time = time.monotonic() - episode_start_time
                time_until_wrench_save = (last_wrench_save_time + wrench_rate) - relative_time
                if time_until_wrench_save > 0:
                    time.sleep(time_until_wrench_save)

                last_wrench_save_time += wrench_rate
                save_wrench_data(buffer, last_wrench_save_time)

                if last_wrench_save_time - last_robot_save_time >= robot_rate:
                    
                    last_robot_save_time += robot_rate
                    images = [pipeline.get_frame() for pipeline in pipelines]
                    save_robot_data(buffer, images, last_robot_save_time)
                    

            elif not recording and len(buffer['observations']['joint_R']) > 0:

                episode_start_time = None
                
                # Ask user whether to save or discard the demo
                while True:
                    data_store = input("Save this demo? (y/n): ").strip().lower()
                    if data_store == 'y':
                        print("Saving demo...")
                        make_demo_n(buffer)
                        print("Demo saved. Press 's' to start a new recording.")
                        break
                    elif data_store == 'n':
                        print("Demo discarded. Press 's' to start a new recording.")
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                
                buffer = init_buffer()


            else: # data not recording (20Hz)
                images = [pipeline.get_frame() for pipeline in pipelines] # for visualize
                elapsed_time = time.monotonic() - loop_start_time
                reward_time = robot_rate - elapsed_time
                if reward_time > 0:
                    time.sleep(reward_time)
            


    except KeyboardInterrupt:
        pass
    
    finally:
        listener.stop()
        
        # Display 프로세스 종료
        stop_event.set()
        display_process.join(timeout=5.0)
        if display_process.is_alive():
            display_process.terminate()

        # Shutdown executor and ROS2
        executor.shutdown()
        rclpy.shutdown()
        executor_thread.join(timeout=5.0)  # Wait for executor thread to finish
        
        for pipeline in pipelines:
            pipeline.pipeline.stop()
            print(f'pipeline stopped (serial number: {pipeline.serial})')


if __name__ == "__main__":
    # multiprocessing 설정 (Linux에서는 'spawn' 사용 권장)
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Dual-arm data generation with optional auto homming control')
    parser.add_argument('-m', '--homming', action='store_true', 
                       help='Enable automatic homming mode switching (default: disabled)')
    parser.add_argument('-r', '--random', action='store_true',
                       help='Enable random homming positions (default: disabled)')
    args = parser.parse_args()
    
    homming = args.homming
    random_mode = args.random

    main()
  