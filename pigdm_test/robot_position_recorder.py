#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import h5py
import os
import time
from datetime import datetime


class RobotPositionRecorder(Node):
    def __init__(self, recording_frequency=50.0):
        super().__init__('robot_position_recorder')
        
        # Recording frequency
        self.recording_frequency = recording_frequency
        
        # Subscribe to joint states
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # Data storage
        self.joint_L = []
        self.joint_R = []
        self.hand_L = []
        self.hand_R = []
        self.timestamps = []
        
        # Latest data
        self.latest_joint_L = None
        self.latest_joint_R = None
        self.latest_hand_L = None
        self.latest_hand_R = None
        
        # Joint names (필요시 수정)
        self.joint_name_L = [f"left_joint_{i}" for i in range(1, 7)]
        self.joint_name_R = [f"right_joint_{i}" for i in range(1, 7)]
        
        # Hand joint names
        self.hand_name = ['left_thumb_joint1', 'left_thumb_joint2', 'left_thumb_joint3',
                          'left_index_joint1', 'left_index_joint2', 'left_index_joint3',
                          'left_middle_joint1', 'left_middle_joint2', 'left_middle_joint3',
                          'left_ring_joint1', 'left_ring_joint2', 'left_ring_joint3',
                          'left_baby_joint1', 'left_baby_joint2', 'left_baby_joint3',
                          'right_thumb_joint1', 'right_thumb_joint2', 'right_thumb_joint3',
                          'right_index_joint1', 'right_index_joint2', 'right_index_joint3',
                          'right_middle_joint1', 'right_middle_joint2', 'right_middle_joint3',
                          'right_ring_joint1', 'right_ring_joint2', 'right_ring_joint3',
                          'right_baby_joint1', 'right_baby_joint2', 'right_baby_joint3']

        # Timing
        self.start_time = None  # Will be set on first recording
        
        # Save directory
        self.save_dir = 'data/joint_recordings'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create timer for recording
        timer_period = 1.0 / self.recording_frequency
        self.record_timer = self.create_timer(timer_period, self.record_callback)
        
        print(f'Recording started at {self.recording_frequency}Hz... Press Ctrl+C to stop and save')
    
    def joint_callback(self, msg: JointState):
        """Joint state callback - just update latest values"""
        try:
            joint_mapping = {n: p for n, p in zip(msg.name, msg.position)}
            
            # Extract arm joints
            joint_L = [joint_mapping.get(name) for name in self.joint_name_L]
            joint_R = [joint_mapping.get(name) for name in self.joint_name_R]
            
            # Extract hand joints
            hand_position = [joint_mapping.get(name) for name in self.hand_name]
            
            # Update if all values exist
            if None not in joint_L:
                self.latest_joint_L = np.array(joint_L)
            if None not in joint_R:
                self.latest_joint_R = np.array(joint_R)
            if None not in hand_position:
                self.latest_hand_L = np.array(hand_position[:15])  # Left hand all joints
                self.latest_hand_R = np.array(hand_position[15:])  # Right hand all joints
        
        except Exception as e:
            self.get_logger().error(f'Error in joint_callback: {e}')
    
    def record_callback(self):
        """Timer callback at specified frequency"""
        if (self.latest_joint_L is not None and self.latest_joint_R is not None and
            self.latest_hand_L is not None and self.latest_hand_R is not None):
            current_time = time.monotonic()
            
            # Set start time on first recording
            if self.start_time is None:
                self.start_time = current_time
                relative_time = 0.0
            else:
                relative_time = current_time - self.start_time
            
            self.joint_L.append(self.latest_joint_L.copy())
            self.joint_R.append(self.latest_joint_R.copy())
            self.hand_L.append(self.latest_hand_L.copy())
            self.hand_R.append(self.latest_hand_R.copy())
            self.timestamps.append(relative_time)
            
            # Print every second worth of samples
            if len(self.timestamps) % int(self.recording_frequency) == 0:
                print(f'[{relative_time:.2f}s] {len(self.timestamps)} samples recorded')
    
    def save_data(self):
        """Save to HDF5"""
        if len(self.timestamps) == 0:
            print('No data to save')
            return
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M')
        filename = os.path.join(self.save_dir, f'joint_{timestamp_str}.hdf5')
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('joint_L', data=np.array(self.joint_L))
            f.create_dataset('joint_R', data=np.array(self.joint_R))
            f.create_dataset('hand_L', data=np.array(self.hand_L))
            f.create_dataset('hand_R', data=np.array(self.hand_R))
            f.create_dataset('timestamp', data=np.array(self.timestamps))
            f.attrs['duration'] = self.timestamps[-1]
            f.attrs['num_samples'] = len(self.timestamps)
            f.attrs['sample_rate'] = self.recording_frequency
        
        print(f'\nData saved to: {filename}')
        print(f'Total samples: {len(self.timestamps)}')
        print(f'Duration: {self.timestamps[-1]:.2f}s')
        print(f'Joint L shape: {np.array(self.joint_L).shape}')
        print(f'Joint R shape: {np.array(self.joint_R).shape}')
        print(f'Hand L shape: {np.array(self.hand_L).shape}')
        print(f'Hand R shape: {np.array(self.hand_R).shape}')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Record robot joint positions')
    parser.add_argument('-f', '--frequency', type=float, default=50.0,
                        help='Recording frequency in Hz (default: 50.0)')
    args = parser.parse_args()
    
    rclpy.init()
    recorder = RobotPositionRecorder(recording_frequency=args.frequency)
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print('\n\nStopping...')
        recorder.save_data()
    finally:
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()