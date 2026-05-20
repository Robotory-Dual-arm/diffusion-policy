#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.node import Node
import os
import time
import cv2
import signal
import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS2 messages
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import UInt8

# Spatial math
from spatialmath import SE3
import spatialmath.base as smb
import roboticstoolbox as rtb

# Global variables
running = True
print_counter = 0  # For controlling print frequency
teleop_enabled = False
moving_to_preset = False
preset_target_joints = None
preset_target_pose = None
preset_command_pose = None
task_linear_speed = 0.05  # TCP position speed (m/s)
task_angular_speed = np.deg2rad(5.0)  # TCP rotation speed (rad/s)
task_position_tolerance = 0.002  # m
task_rotation_tolerance = np.deg2rad(1.0)  # rad
last_preset_plan_time = None

# Right arm variables
T_tracker_init_right = None
T_station2track_right = None
current_joints_rad_right = None
current_pose_se3_right = None
robot_pose_init_right = None

robot = None
node = None

# VR tracker target frame ID
baseline = 0.7
TARGET_FRAME_RIGHT = "vive_tracker_LHR_4FF4BABB"

# Calibration matrix - transformation from robot base to tracking station
T_robot2station_left = SE3.CopyFrom(
    np.array([[-0.13449103,  0.02376124, -0.99062988, -2.73413664],
              [ 0.79634854,  0.59752308, -0.0937826 ,  0.09727939],
              [ 0.58969583, -0.80149958, -0.09928373, -1.56317163],
              [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    
    check=False
)

T_R2L = SE3.CopyFrom(np.array([[-1, 0,  0, 0],
       [0,          0,          -1,         baseline/np.sqrt(2)],
       [0,          -1,         0,          baseline/np.sqrt(2)],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    check=False
)

T_robot2station_right = T_R2L * T_robot2station_left

# Control parameters
Kp_pos = 1.0  # Position gain
Kp_rot = 0.8  # Rotation gain
step_gain = 0.5   # Step size for joint updates
acc_limit = 30.0  # Acceleration limit

# 사전 정의된 조인트 위치 (라디안) - right arm home pose
PRESET_JOINT_POSITIONS = np.array([
    5.5, 52.0, 112.0, 28.0, -107.0, -35.0
])
PRESET_JOINT_POSITIONS = np.deg2rad(PRESET_JOINT_POSITIONS)

def signal_handler(sig, frame):
    global running
    print('\nRight Arm: Program termination requested.')
    running = False

signal.signal(signal.SIGINT, signal_handler)

def normalize_rotation_matrix(rotation_matrix):
    """Project a numerically drifted 3x3 matrix back onto SO(3)."""
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3) or not np.all(np.isfinite(rotation_matrix)):
        raise ValueError("invalid rotation matrix")

    u, _, vh = np.linalg.svd(rotation_matrix)
    normalized = u @ vh
    if np.linalg.det(normalized) < 0.0:
        u[:, -1] *= -1.0
        normalized = u @ vh
    return normalized

def teleop_control_callback(msg):
    """Callback for external teleop control signal.

    0: stop/hold current TCP pose
    1: enable VR teleoperation
    2 or 3: move to home TCP pose calculated from PRESET_JOINT_POSITIONS
    """
    global teleop_enabled, T_tracker_init_right, robot_pose_init_right
    global T_station2track_right, current_pose_se3_right
    global moving_to_preset, preset_target_joints, preset_target_pose, preset_command_pose, last_preset_plan_time

    prev_teleop_enabled = teleop_enabled

    if msg.data == 1:
        teleop_enabled = True
        moving_to_preset = False
        preset_target_pose = None
        preset_command_pose = None
        last_preset_plan_time = None
        print("Right Arm: Teleoperation ENABLED")

        if not prev_teleop_enabled:
            if current_pose_se3_right is not None and T_station2track_right is not None:
                robot_pose_init_right = current_pose_se3_right.copy()
                T_tracker_init_right = T_station2track_right.copy()
                print("Right Arm: Re-initialized reference poses for teleoperation")
                print(f"Right Arm: New initial tracker pose: {T_tracker_init_right.t}")
                print(f"Right Arm: New initial robot pose: {robot_pose_init_right.t}")
            else:
                print("Right Arm: Warning - Cannot re-initialize poses (missing current data)")

    elif msg.data == 0:
        teleop_enabled = False
        moving_to_preset = False
        preset_target_pose = None
        preset_command_pose = None
        last_preset_plan_time = None
        print("Right Arm: Teleoperation DISABLED")

    elif msg.data == 2 or msg.data == 3:
        teleop_enabled = False
        preset_target_joints = PRESET_JOINT_POSITIONS.copy()
        preset_target_pose = None
        preset_command_pose = current_pose_se3_right.copy() if current_pose_se3_right is not None else None
        last_preset_plan_time = None
        print("Right Arm: Moving to HOME TCP pose")

        preset_target_pose = calculate_fk_pose_right(preset_target_joints)
        if preset_target_pose is None:
            moving_to_preset = False
            print("Right Arm: Failed to calculate HOME TCP pose from FK")
        else:
            moving_to_preset = True
            print(f"Right Arm: HOME TCP target: {preset_target_pose.t}, Rotation (rpy): {smb.tr2rpy(preset_target_pose.R, unit='deg')}")

    else:
        print(f"Right Arm: Invalid control signal: {msg.data} (expected 0, 1, 2, or 3)")

def vr_pose_callback(msg):
    """Callback function for VR tracker pose messages"""
    global T_station2track_right, T_tracker_init_right, robot_pose_init_right, current_pose_se3_right
    
    for transform in msg.transforms:
        # Right tracker
        if transform.child_frame_id == TARGET_FRAME_RIGHT:
            trans = transform.transform.translation
            rot = transform.transform.rotation
            pos = np.array([trans.x, trans.y, trans.z])
            quat = [rot.w, rot.x, rot.y, rot.z]
            R_matrix = smb.q2r(quat)
            T_station2track_right = SE3.Rt(R_matrix, pos)
            
            # Store initial tracker pose and robot pose for right arm
            if T_tracker_init_right is None and current_pose_se3_right is not None:
                T_tracker_init_right = T_station2track_right.copy()
                robot_pose_init_right = current_pose_se3_right.copy()
                print("Right Arm: Initial VR tracker and robot poses stored.")
                print(f"Right Arm: Initial tracker pose: {T_tracker_init_right.t}")
                print(f"Right Arm: Initial robot pose: {robot_pose_init_right.t}")

def joint_state_callback(msg):
    """Callback for joint state updates"""
    global current_joints_rad_right, current_pose_se3_right, robot
    
    if not msg.name or not msg.position:
        return
    
    
    right_joint_positions = [0.0] * 6  # Initialize with zeros for 6 joints  
    right_joint_names = []
    
    # Filter and sort joints for right arm
    for i, name in enumerate(msg.name):
        if i < len(msg.position) and not np.isnan(msg.position[i]):
            # Right arm joints
            if name.startswith('right_joint') or name.startswith('joint'):
                try:
                    joint_num = int(name.split('_')[-1])
                    if 1 <= joint_num <= 6:
                        right_joint_positions[joint_num - 1] = msg.position[i]
                        right_joint_names.append(name)
                except (ValueError, IndexError):
                    continue
    
    # Update right arm joint positions and pose
    if len(right_joint_names) >= 6:
        current_joints_rad_right = np.array(right_joint_positions)
        
        # Calculate right arm current pose using forward kinematics
        if robot is not None:
            try:
                current_pose_se3_right = robot.fkine(current_joints_rad_right, end=robot.links[7])
            except Exception as e:
                try:
                    current_pose_se3_right = robot.fkine(current_joints_rad_right, end=robot.links[5])
                except Exception as e2:
                    print(f"Right Arm: Forward kinematics failed: {e2}")

def calculate_target_pose_right():
    """Calculate target pose for RIGHT arm based on VR tracker movement"""
    global T_tracker_init_right, T_station2track_right, T_robot2station_right, robot_pose_init_right, print_counter
    
    if T_tracker_init_right is None or T_station2track_right is None or robot_pose_init_right is None:
        return None
    
    # Calculate tracker movement from initial position
    T_tracker_delta = T_tracker_init_right.inv() * T_station2track_right
    delta_vr_pos = T_tracker_delta.t  # Relative position in meters
    delta_vr_rot = smb.tr2rpy(T_tracker_delta.R, unit='rad')  # Relative rotation in radians
    
    # Transform deltas using proper frame transformations
    R_base2station = T_robot2station_right.R
    R_tracker_init = T_tracker_init_right.R
    R_end2base_init = robot_pose_init_right.inv().R

    print_counter += 1
    
    # Transform deltas to end-effector frame
    delta_pos_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_pos
    delta_rot_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_rot
    
    # Apply scaling
    scaled_delta_pos = Kp_pos * delta_pos_end
    scaled_delta_rot = Kp_rot * delta_rot_end
    
    # Calculate target pose
    target_pose = robot_pose_init_right * SE3(scaled_delta_pos) * SE3.RPY(*scaled_delta_rot, unit='rad')
    
    return target_pose

def calculate_joint_delta_right(target_pose):
    """Calculate joint delta for RIGHT arm using Jacobian pseudoinverse"""
    global robot, current_joints_rad_right, current_pose_se3_right
    
    if current_pose_se3_right is None or robot is None:
        return None
    
    # Calculate pose error
    pose_error = current_pose_se3_right.inv() * target_pose
    
    # Extract position and rotation errors
    err_pos = target_pose.t - current_pose_se3_right.t
    err_rot_ee = smb.tr2rpy(pose_error.A, unit='rad')
    err_rot_base = current_pose_se3_right.R @ err_rot_ee
    print(f"Right Arm: Position error: {err_pos}, Rotation error: {err_rot_base}")
    
    # Combine into 6D error vector
    err_6d = np.concatenate((err_pos, err_rot_base))
    
    # Compute Jacobian and pseudo-inverse solution
    try:
        J = robot.jacob0(current_joints_rad_right, end=robot.links[7])
    except:
        try:
            J = robot.jacob0(current_joints_rad_right, end=robot.links[5])
        except Exception as e:
            print(f"Right Arm: Jacobian calculation failed: {e}")
            return None
    
    # Calculate joint delta using pseudoinverse
    dq = np.linalg.pinv(J) @ err_6d
    
    # Apply acceleration limits for all joints
    if np.linalg.norm(dq) > acc_limit:
        dq *= acc_limit / np.linalg.norm(dq)
    
    # Additional safety: limit individual joint velocities
    max_joint_vel = np.deg2rad(20.0)  # 20 deg/s max per joint
    dq = np.clip(dq, -max_joint_vel, max_joint_vel)
    
    return dq

def calculate_fk_pose_right(joint_angles_rad):
    """Convert joint angles to the right-arm TCP pose with FK."""
    global robot

    if robot is None:
        return None

    try:
        return robot.fkine(joint_angles_rad, end=robot.links[7])
    except Exception:
        try:
            return robot.fkine(joint_angles_rad, end=robot.links[5])
        except Exception as e:
            print(f"Right Arm: FK calculation failed: {e}")
            return None

def calculate_preset_task_movement_right():
    """Plan one small TCP-space step toward the home pose."""
    global current_pose_se3_right, preset_target_pose, preset_command_pose
    global moving_to_preset, last_preset_plan_time

    if current_pose_se3_right is None or preset_target_pose is None:
        return None

    if preset_command_pose is None:
        preset_command_pose = current_pose_se3_right.copy()

    now = time.monotonic()
    if last_preset_plan_time is None:
        dt = 0.005
    else:
        dt = max(0.001, min(now - last_preset_plan_time, 0.05))
    last_preset_plan_time = now

    current_pos = preset_command_pose.t
    target_pos = preset_target_pose.t
    pos_diff = target_pos - current_pos
    pos_dist = np.linalg.norm(pos_diff)
    max_pos_step = task_linear_speed * dt

    pose_error = preset_command_pose.inv() * preset_target_pose
    rot_error = smb.tr2rpy(pose_error.A, unit='rad')
    rot_dist = np.linalg.norm(rot_error)
    max_rot_step = task_angular_speed * dt

    if pos_dist < task_position_tolerance and rot_dist < task_rotation_tolerance:
        preset_command_pose = preset_target_pose.copy()

        feedback_pos_dist = np.linalg.norm(preset_target_pose.t - current_pose_se3_right.t)
        feedback_error = current_pose_se3_right.inv() * preset_target_pose
        feedback_rot_dist = np.linalg.norm(smb.tr2rpy(feedback_error.A, unit='rad'))
        if feedback_pos_dist < task_position_tolerance and feedback_rot_dist < task_rotation_tolerance:
            moving_to_preset = False
            preset_command_pose = None
            last_preset_plan_time = None
            print("Right Arm: Reached HOME TCP pose")

        return preset_target_pose

    pos_ratio = 1.0 if pos_dist < 1e-9 else min(1.0, max_pos_step / pos_dist)
    rot_ratio = 1.0 if rot_dist < 1e-9 else min(1.0, max_rot_step / rot_dist)
    ratio = min(pos_ratio, rot_ratio)

    next_pos = current_pos + pos_diff * ratio
    next_rot = preset_command_pose.R @ SE3.RPY(*(rot_error * ratio), unit='rad').R
    next_rot = normalize_rotation_matrix(next_rot)
    preset_command_pose = SE3.Rt(next_rot, next_pos)
    return preset_command_pose

def publish_task_command_right(target_pose_se3):
    """Publish task-space pose command"""
    global node, print_counter

    # 50번에 한 번만 출력
    if print_counter % 50 == 0:
        print(f"Right Arm: Target Pose Position: {target_pose_se3.t}, Rotation (rpy): {smb.tr2rpy(target_pose_se3.R, unit='deg')}")
    print_counter += 1

    pose_msg = PoseStamped()
    pose_msg.header.stamp = node.get_clock().now().to_msg()
    pose_msg.header.frame_id = "base_link"

    # Position (m to mm)
    pose_msg.pose.position.x = target_pose_se3.t[0] * 1000.0  # mm
    pose_msg.pose.position.y = target_pose_se3.t[1] * 1000.0
    pose_msg.pose.position.z = target_pose_se3.t[2] * 1000.0

    # Orientation (rotation matrix to quaternion)
    quat = smb.r2q(target_pose_se3.R)  # [w, x, y, z]
    pose_msg.pose.orientation.w = quat[0]
    pose_msg.pose.orientation.x = quat[1]
    pose_msg.pose.orientation.y = quat[2]
    pose_msg.pose.orientation.z = quat[3]

    node.task_command_pub.publish(pose_msg)

class RightArmVRServoNode(Node):
    def __init__(self):
        super().__init__('right_arm_vr_servo_node')

        # Task space command publisher
        self.task_command_pub = self.create_publisher(
            PoseStamped,
            '/right_dsr_controller/task_space_command',
            10
        )
        
        # Subscribers
        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            vr_pose_callback,
            10
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/dsr01/joint_states',
            joint_state_callback,
            10
        )

        self.teleop_control_sub = self.create_subscription(
            UInt8,
            '/teleop_control',
            teleop_control_callback,
            10
        )
        
        print("Right Arm VR Servo Node initialized")
        print("Right Arm: Listening for teleop control signals on /teleop_control topic")

def main(args=None):
    global running, robot, node
    global current_joints_rad_right
    global T_tracker_init_right, T_station2track_right, robot_pose_init_right
    
    # Load robot model
    try:
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "m0609.white.urdf"))
        robot = rtb.ERobot.URDF(urdf_path)
        print("Right Arm: Robot model loaded successfully")
        print(f"Right Arm: Robot DOF: {robot.n}")
    except Exception as e:
        print(f"Right Arm: Error: Could not load robot model: {e}")
        return
    
    # Initialize ROS2
    rclpy.init(args=args)
    node = RightArmVRServoNode()
    
    # Wait for joint states to be available for right arm
    print("Right Arm: Waiting for joint states...")
    while current_joints_rad_right is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

        
    if current_joints_rad_right is None:
        print("Right Arm: Error: No joint states received")
        return
        
    print("Right Arm: Joint states received, starting VR servo control...")

    print("Right Arm: VR Servo Control Started!")
    print("Right Arm: Calibration matrices loaded from previous calibration results")
    print("Right Arm: Waiting for control signals:")
    print("Right Arm:   - Publish 1 to /teleop_control to enable VR control")
    print("Right Arm:   - Publish 2 to /teleop_control to move to HOME TCP pose")
    print("Right Arm:   - Publish 0 to /teleop_control to stop/hold current TCP pose")

    try:
        while running and rclpy.ok():
            # Process ROS messages
            rclpy.spin_once(node, timeout_sec=0.005)
            
            # Initialize TCP command
            next_pose_right = None
            
            # Control RIGHT arm
            if (teleop_enabled and
                T_station2track_right is not None and 
                T_tracker_init_right is not None and 
                current_joints_rad_right is not None):
                
                # Calculate target pose from RIGHT VR tracker movement
                target_pose_right = calculate_target_pose_right()
                if target_pose_right is not None:
                    next_pose_right = target_pose_right

            # HOME TCP pose로 이동 중인 경우
            elif moving_to_preset and current_pose_se3_right is not None:
                next_pose_right = calculate_preset_task_movement_right()

            # Send TCP command for right arm
            if next_pose_right is not None:
                publish_task_command_right(next_pose_right)
            elif current_pose_se3_right is not None:
                # Hold current TCP pose
                publish_task_command_right(current_pose_se3_right)
                
            # Small delay to prevent excessive CPU usage
            # time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nRight Arm: Keyboard interrupt received. Stopping...")
    
    finally:
        running = False
        print("Right Arm: VR Servo Control terminated")
        
        # Shutdown ROS2
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
