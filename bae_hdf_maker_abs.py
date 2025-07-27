#!/home/vision/anaconda3/envs/robodiff/bin/python
import sys
sys.path.append('/home/vision/catkin_ws/src/diffusion_policy/diffusion-policy/diffusion_policy')
import h5py
import rospy
import numpy as np
from pynput import keyboard
from rb10_api.cobot import *
from rb import *
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import cv2
import time
# 저장할때마다 저장 파일 이름 바꾸기!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!주의!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
f = h5py.File('/home/vision/catkin_ws/src/diffusion_policy/diffusion-policy/data/baetae/bae_push_image_abs_test.hdf5', 'w')
data = f.create_group('data')
i = 0

def init_buffer():
    return {
        'demo_0': {'actions': []},
        'obs': {
            'robot_eef_pos': [],
            'robot_eef_quat': [],
            'image0': [],   # D405
            'image1': []    # D435
        }
    }

def make_demo_n(buffer):
    demo_n = data.create_group(f'demo_{len(data.keys())}')
    obs = demo_n.create_group('obs')

    for name, values in buffer['demo_0'].items():
        demo_n.create_dataset(name, data=np.array(values))
    for name, values in buffer['obs'].items():
        obs.create_dataset(name, data=np.array(values))

def on_press(key):
    global recording, terminal
    try:
        if key.char == 's':
            recording = True
            print("Start recording")
        elif key.char == 'q':
            recording = False
            print("Stop recording")
        elif key.char == 't':
            terminal = True
    except AttributeError:
        pass

def get_device_serials():
    ctx = rs.context()
    serials = []
    for device in ctx.query_devices():
        serials.append(device.get_info(rs.camera_info.serial_number))
    if len(serials) < 2:
        raise RuntimeError("2개 이상의 Realsense 카메라가 연결되어 있어야 합니다.")
    print("Detected serials:", serials)
    return serials

def get_robot_state(robot):
    current_jnt = np.array(GetCurrentSplitedJoint()) * np.pi / 180.0   # rad
    current_pose = robot.fkine(current_jnt)   # m, rad
    T = np.array(current_pose)

    robot_pos = list(T[:3, 3])   
    rot = R.from_matrix(T[:3, :3])
    x, y, z, w = rot.as_quat()   # pos
    robot_quat = [w, x, y, z]    # quat                
    robot_6d_rot = np.concatenate([     # 6d rotation     
        T[:3, 0],   # column 0
        T[:3, 1],   # column 1
    ])
    return robot_pos, robot_quat, robot_6d_rot

def get_image(pipeline0, pipeline1):
    CROP_SIZE = 480   
    try:
        frame0 = pipeline0.wait_for_frames(timeout_ms=2000)
        frame1 = pipeline1.wait_for_frames(timeout_ms=2000)
    except RuntimeError as e:
        rospy.logwarn(f"Realsense timeout: {e}")
        return None

    color_frame0 = frame0.get_color_frame()
    color_frame1 = frame1.get_color_frame()
    if not color_frame0 or not color_frame1:
        rospy.logwarn("No color frame from one of the cameras")
        return None

    color_image0 = np.asanyarray(color_frame0.get_data())
    color_image1 = np.asanyarray(color_frame1.get_data())

    # cv2.imshow('image0', color_image0)
    # cv2.imshow('image1', color_image1)

    h, w = color_image0.shape[:2]   # resize, crop 고민 흠
    # print(f"Image shape: {h}x{w}")
    start_x = (w - CROP_SIZE) // 2
    image0 = color_image0[:, start_x:start_x+CROP_SIZE]   # CROP_SIZE = 480
    image1 = color_image1[:, start_x:start_x+CROP_SIZE]
    image0 = cv2.resize(image0, (84, 84))
    image1 = cv2.resize(image1, (84, 84))
    
    # cv2.imshow('image0', image0)
    # cv2.imshow('image1', image1)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        return None
    
    # image0 = cv2.cvtColor(image0.copy(), cv2.COLOR_BGR2RGB)
    # image1 = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB)

    cv2.imshow('image0', image0)
    cv2.imshow('image1', image1)

    image = [image0, image1]

    return image


def main():
    global i
    
    serials = ['126122270795', '117322071192']   # D405, D435
    # serials = None
    if serials == None:
        serials = get_device_serials()
    serial_d405 = serials[0]
    serial_d435 = serials[1]

    pipeline0 = rs.pipeline()
    config0 = rs.config()
    config0.enable_device(serial_d405)
    config0.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline0.start(config0)

    pipeline1 = rs.pipeline()
    config1 = rs.config()
    config1.enable_device(serial_d435)
    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline1.start(config1)

    cv2.namedWindow('image0',  cv2.WINDOW_NORMAL)
    cv2.namedWindow('image1',  cv2.WINDOW_NORMAL)

    cv2.resizeWindow('image0',  400, 400)  
    cv2.resizeWindow('image1',  400, 400)

    
    ToCB("192.168.111.50")
    robot = RB10()
    CobotInit()

    global terminal, recording
    recording = False
    terminal = False

    rospy.init_node("hdf_maker", anonymous=True)
    # rospy.loginfo("HDF5 Maker Node Initialized")
    # rospy.Subscriber("/OnRobotRGInput", OnRobotRGInput, gripper_callback, queue_size=1)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    buffer = init_buffer()
    rate = rospy.Rate(10)   # 10 Hz

    print('s: start, q: stop, t: terminate')
    print("끝나면 무조건 t로 종료하세요!!!!!!!, 저장파일 이름도 잘 바꾸기")


    while not rospy.is_shutdown():
        if terminal:
            f.close()
            print("Terminating.")
            break
        

        if recording:
            
            robot_pos, robot_quat, robot_6d_rot = get_robot_state(robot)
            image = get_image(pipeline0, pipeline1)
            if image is None:
                continue
            image0, image1 = image

            action = np.concatenate([robot_pos, robot_6d_rot])   # 현재 pose(m), 6d rotation

            buffer['demo_0']['actions'].append(action)
            buffer['obs']['robot_eef_pos'].append(prev_robot_pos)   # meter
            buffer['obs']['robot_eef_quat'].append(prev_robot_quat)   
            buffer['obs']['image0'].append(prev_image0.copy())
            buffer['obs']['image1'].append(prev_image1.copy())

            prev_robot_pos = robot_pos
            prev_robot_quat = robot_quat
            prev_robot_6d_rot = robot_6d_rot
            prev_image0 = image0.copy()
            prev_image1 = image1.copy()


        elif not recording and len(buffer['demo_0']['actions']) > 0:
            while True:
                data_store = input("Store demo data? (y/n): ").strip().lower()
                if data_store == 'y':
                    make_demo_n(buffer)
                    print(f"'demo_{i}' Data stored.")
                    i += 1
                    break
                elif data_store == 'n':
                    print("Data discarded.")
                    break
                else:
                    print("Invalid input.")

            buffer = init_buffer()


        else:
            prev_robot_pos, prev_robot_quat, prev_robot_6d_rot = get_robot_state(robot)
            prev_image = get_image(pipeline0, pipeline1)
            if prev_image is None:
                continue
            prev_image0, prev_image1 = prev_image


        rate.sleep()

    pipeline0.stop()
    pipeline1.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
