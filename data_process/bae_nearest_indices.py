import numpy as np

def find_nearest_wrench_indices(robot_timestamps, wrench_timestamps):
                        nearest_indices = []
                        wrench_idx = 0
                        for robot_time in robot_timestamps:
                            idx = np.searchsorted(wrench_timestamps, robot_time, side='left')
                            if np.abs(wrench_timestamps[idx]-robot_time) < np.abs(robot_time-wrench_timestamps[idx-1]):
                                nearest_indices.append(idx)
                            else:
                                nearest_indices.append(idx-1)
                            
                        return nearest_indices

def main():
    robot_timestamps = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    wrench_timestamps = np.array([0.01, 0.04, 0.06, 0.07, 0.090000001, 0.11, 0.12, 0.16, 0.19, 0.21])
    
    nearest_indices = find_nearest_wrench_indices(robot_timestamps, wrench_timestamps)
    print(nearest_indices)

if __name__ == "__main__":
    main()