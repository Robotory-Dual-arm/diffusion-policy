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
    def __init__(self):
        super().__init__('robot_position_recorder')
        
        # Subscribe to joint states
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # Data storage
        self.joint_L = []
        self.joint_R = []
        self.timestamps = []
        
        # Latest data
        self.latest_joint_L = None
        self.latest_joint_R = None
        
        # Joint names (필요시 수정)
        self.joint_name_L = [f"left_joint_{i}" for i in range(1, 7)]
        self.joint_name_R = [f"right_joint_{i}" for i in range(1, 7)]
        
        # Timing
        self.start_time = time.monotonic()
        self.last_save_time = 0.0
        self.save_rate = 0.02  # 50Hz
        
        # Save directory
        self.save_dir = 'data/joint_recordings'
        os.makedirs(self.save_dir, exist_ok=True)
        
        print('Recording started... Press Ctrl+C to stop and save')
    
    def joint_callback(self, msg: JointState):
        """Joint state callback"""
        try:
            joint_names = list(msg.name)
            
            # Extract left arm joints
            joint_L = []
            for name in self.joint_name_L:
                if name in joint_names:
                    idx = joint_names.index(name)
                    joint_L.append(msg.position[idx])
            
            # Extract right arm joints
            joint_R = []
            for name in self.joint_name_R:
                if name in joint_names:
                    idx = joint_names.index(name)
                    joint_R.append(msg.position[idx])
            
            if len(joint_L) == 6:
                self.latest_joint_L = np.array(joint_L)
            if len(joint_R) == 6:
                self.latest_joint_R = np.array(joint_R)
            
            # Record at 50Hz
            current_time = time.monotonic()
            relative_time = current_time - self.start_time
            
            if relative_time - self.last_save_time >= self.save_rate:
                if self.latest_joint_L is not None and self.latest_joint_R is not None:
                    self.joint_L.append(self.latest_joint_L.copy())
                    self.joint_R.append(self.latest_joint_R.copy())
                    self.timestamps.append(relative_time)
                    self.last_save_time = relative_time
                    
                    if len(self.timestamps) % 50 == 0:
                        print(f'[{relative_time:.1f}s] {len(self.timestamps)} samples recorded')
        
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
    
    def save_data(self):
        """Save to HDF5"""
        if len(self.timestamps) == 0:
            print('No data to save')
            return
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.save_dir, f'joint_record_{timestamp_str}.hdf5')
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('joint_L', data=np.array(self.joint_L))
            f.create_dataset('joint_R', data=np.array(self.joint_R))
            f.create_dataset('timestamp', data=np.array(self.timestamps))
            f.attrs['duration'] = self.timestamps[-1]
            f.attrs['num_samples'] = len(self.timestamps)
            f.attrs['sample_rate'] = 50.0


def main():
    rclpy.init()
    recorder = RobotPositionRecorder()
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print('\n\nStopping...')
        recorder.save_data()
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
