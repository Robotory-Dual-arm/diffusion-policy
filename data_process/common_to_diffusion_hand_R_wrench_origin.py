from typing import Tuple
import math
import cv2
import h5py
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb
from roboticstoolbox import ERobot
from spatialmath import SE3, UnitQuaternion
import os

# urdf_path = "/home/vision/dualarm_ws/src/doosan-robot2/dsr_description2/urdf/m0609.white.urdf"
# urdf_path = "../m0609.white.urdf"
urdf_path = os.path.abspath("../m0609.white.urdf")
robot = rtb.ERobot.URDF(urdf_path)   


""" common data, 20Hz
data
    demo_0
        observations
            joint_R   # rad, len=6
            hand_R    # rad, len=15 (thumb3, index3, middle3, ring3, baby3)
            image_F   # (640, 480) or (320, 240) ; (front cam)  
            image_H   # (640, 480) or (320, 240) ; (head cam)
            image_T   # (640, 480) or (320, 240) ; (table cam)
            wrench_wrist_R  # N, len=6 (fx,fy,fz,tx,ty,tz)
            wrench_thumb_R  # N, len=6 (fx,fy,fz,tx,ty,tz)
            wrench_index_R  # N, len=6 (fx,fy,fz,tx,ty,tz)
            wrench_middle_R # N, len=6 (fx,fy,fz,tx,ty,tz)
            wrench_ring_R   # N, len=6 (fx,fy,fz,tx,ty,tz)
            wrench_baby_R   # N, len=6 (fx,fy,fz,tx,ty,tz)
"""

""" diffusion data, 10Hz
data
    demo_0
        actions (robot_pose_R(3), robot_6d_R(6), hand_pose_R(6))
        obs
            robot_pose_R   # m, len=3 (x,y,z)
            robot_quat_R   # len=4 (x,y,z,w)
            hand_pose_R    # rad, len=11 (thumb3, index2, middle2, ring2, baby2)
            image_H   # (224, 224)
            image_T   # (224, 224)
            wrench_wrist_R  # N, len=(6, 32) (fx,fy,fz,tx,ty,tz)
            wrench_thumb_R  # N, len=(1, 32) (fz)
            wrench_index_R  # N, len=(1, 32) (fz)
            wrench_middle_R # N, len=(1, 32) (fz)
            wrench_ring_R   # N, len=(1, 32) (fz)
            wrench_baby_R   # N, len=(1, 32) (fz)
"""

# 데이터 초기에 wrench 값 10개 평균을 0set으로 맞추기
# 초반에 wrench가 24개 먼저 시작 25번째에 image랑 sync -> wrench값 32개씩 []로 묶기
#         img         img         img
# [f f f f f] [f f f f f] [f f f f f]

### Diffusion image 변환인데, 참고해서 활용하기!!!
# def get_image_transform(
#         input_res: Tuple[int,int]=(640,480), 
#         output_res: Tuple[int,int]=(320,240), 
#         bgr_to_rgb: bool=False):

#     iw, ih = input_res
#     ow, oh = output_res
#     rw, rh = None, None
#     interp_method = cv2.INTER_AREA

#     if (iw/ih) >= (ow/oh):
#         # input is wider
#         rh = oh
#         rw = math.ceil(rh / ih * iw)
#         if oh > ih:
#             interp_method = cv2.INTER_LINEAR
#     else:
#         rw = ow
#         rh = math.ceil(rw / iw * ih)
#         if ow > iw:
#             interp_method = cv2.INTER_LINEAR
    
#     w_slice_start = (rw - ow) // 2
#     w_slice = slice(w_slice_start, w_slice_start + ow)
#     h_slice_start = (rh - oh) // 2
#     h_slice = slice(h_slice_start, h_slice_start + oh)
#     c_slice = slice(None)
#     if bgr_to_rgb:
#         c_slice = slice(None, None, -1)

#     def transform(img: np.ndarray):
#         assert img.shape == ((ih,iw,3))
#         # resize
#         img = cv2.resize(img, (rw, rh), interpolation=interp_method)
#         # crop
#         img = img[h_slice, w_slice, c_slice]
#         return img
#     return transform
def get_image_transform(
    input_res: Tuple[int, int] = (640, 480),
    output_res: Tuple[int, int] = (224, 224),
    bgr_to_rgb: bool = True,
):
    iw, ih = input_res
    ow, oh = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw / ih) >= (ow / oh):
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR

    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None, None, -1) if bgr_to_rgb else slice(None)

    def transform(imgs: np.ndarray):
        """이미지 배열 변환 (N, H, W, C) or (H, W, C) 지원"""
        if imgs.ndim == 4:  # (N, H, W, C) - 여러 이미지
            return np.array([transform_single(img, ih, iw, rw, rh, h_slice, w_slice, c_slice, interp_method) 
                           for img in imgs])
        else:  # (H, W, C) - 단일 이미지
            return transform_single(imgs, ih, iw, rw, rh, h_slice, w_slice, c_slice, interp_method)
    
    def transform_single(img, ih, iw, rw, rh, h_slice, w_slice, c_slice, interp_method):
        assert img.shape == (ih, iw, 3), f"Unexpected image shape: {img.shape}, expected {(ih, iw, 3)}"
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        img = img[h_slice, w_slice, c_slice]
        return img

    return transform

def quat_to_6d(quats):
    """
    quats: [[x,y,z,w], [x,y,z,w], ...]  (x,y,z,w 순서)
    return: [[r11,r21,r31,r12,r22,r32], ...] (각각 6D 회전 표현)
    """
    quats = np.asarray(quats)
    rotation_matrix = R.from_quat(quats).as_matrix()  # (N, 3, 3)
    
    # 열 단위로 뽑기
    r1 = rotation_matrix[:, :, 0]  # 첫 번째 column → (N, 3)
    r2 = rotation_matrix[:, :, 1]  # 두 번째 column → (N, 3)
    
    # [r1, r2] 붙이기
    rotation_6d = np.concatenate([r1, r2], axis=1)  # (N, 6)
    return rotation_6d


def resize_images(image_list, size=(320, 240)):
    """
    image_list : [img1, img2, ...] (각 img는 numpy array, shape (480,640,3))
    size       : (width, height)
    return     : [resized_img1, resized_img2, ...] (모두 (240,320,3))
    """
    return [cv2.resize(img, size) for img in image_list]

# 앞에 n개 데이터 평균을 offset으로 빼기
def subtract_offset(wrench_data, mean_number):
    offset = np.mean(wrench_data[:mean_number], axis=0)
    return wrench_data - offset

def ema_filter(wrench_data, alpha):
    result = [wrench_data[0]]
    for i in range(1, len(wrench_data)):
        result.append(alpha * wrench_data[i] + (1 - alpha) * result[i-1])
    return result

def main():
    input_filenames = ['/home/baetae/Downloads/common_data_erase_board.hdf5']
    output_filename = '/data/baetae/260412/diffusion_data_erase_board_no_image.hdf5'
    output_demo_idx = 0
    transform = get_image_transform(input_res=(640,480), output_res=(224,224), bgr_to_rgb=True)

    with h5py.File(output_filename, 'w') as output_file:

        output_data = output_file.create_group('data')

        for input_filename in input_filenames:
            with h5py.File(input_filename, 'r') as input_file:
                input_data: dict = input_file['data']
                demo_len = len(input_data)
                print(input_filename, '/ demo_len =', demo_len)

                for demo_idx in tqdm.tqdm(range(demo_len), desc=f"Processing file"):
                     
                    # n번째 demo 생성
                    input_demo_name = f'demo_{demo_idx}'
                    output_demo_name = f'demo_{output_demo_idx}'
                    
                    input_demo_n = input_data[input_demo_name]
                    output_demo_n = output_data.create_group(output_demo_name)

                    # observations
                    input_obs = input_demo_n['observations']
                    output_obs = output_demo_n.create_group('obs')
                    
                    
                    # image, joint: 20Hz -> 10Hz
                    input_timestamp_robot = input_obs['timestamp_robot'][::2] # [0.0, 0.1, 0.2, 0.3...]

                    # input_joint_L = input_obs['joint_L'][::2]
                    input_joint_R = input_obs['joint_R'][::2]
                    # input_hand_pose_L = input_obs['hand_L'][::2]
                    input_hand_pose_R = input_obs['hand_R'][::2]
                    # input_image_H = input_obs['image_H'][::2]
                    # input_image_F = input_obs['image_F'][::2]
                    # input_image_L = input_obs['image_L'][::2]
                    # input_image_R = input_obs['image_R'][::2]
                    # input_image_T = input_obs['image_T'][::2]

                    # wrench: 250Hz
                    input_timestamp_wrench = input_obs['timestamp_wrench'][:] # [0.0, 0.004, 0.008, 0.012, ...]

                    input_wrench_wrist_R = input_obs['wrench_wrist_R']
                    input_wrench_thumb_R = input_obs['wrench_thumb_R']
                    input_wrench_index_R = input_obs['wrench_index_R']
                    input_wrench_middle_R = input_obs['wrench_middle_R']
                    input_wrench_ring_R = input_obs['wrench_ring_R']
                    input_wrench_baby_R = input_obs['wrench_baby_R']

                    # only wrench usage
                    output_wrench_wrist_R = input_wrench_wrist_R[:, :6]
                    output_wrench_thumb_R = input_wrench_thumb_R[:, 2:3]   # fz
                    output_wrench_index_R = input_wrench_index_R[:, 2:3]   # fz
                    output_wrench_middle_R = input_wrench_middle_R[:, 2:3] # fz
                    output_wrench_ring_R = input_wrench_ring_R[:, 2:3]     # fz
                    output_wrench_baby_R = input_wrench_baby_R[:, 2:3]     # fz


                    # 앞에 n개 평균으로 0set 맞추기
                    wrench_offset_mean_number = 10
                    output_wrench_wrist_R = subtract_offset(output_wrench_wrist_R, wrench_offset_mean_number)
                    output_wrench_thumb_R = subtract_offset(output_wrench_thumb_R, wrench_offset_mean_number)
                    output_wrench_index_R = subtract_offset(output_wrench_index_R, wrench_offset_mean_number)
                    output_wrench_middle_R = subtract_offset(output_wrench_middle_R, wrench_offset_mean_number)
                    output_wrench_ring_R = subtract_offset(output_wrench_ring_R, wrench_offset_mean_number)
                    output_wrench_baby_R = subtract_offset(output_wrench_baby_R, wrench_offset_mean_number)
                    # 앞에 n개 버리기
                    output_wrench_wrist_R = output_wrench_wrist_R[wrench_offset_mean_number:]
                    output_wrench_thumb_R = output_wrench_thumb_R[wrench_offset_mean_number:]
                    output_wrench_index_R = output_wrench_index_R[wrench_offset_mean_number:]
                    output_wrench_middle_R = output_wrench_middle_R[wrench_offset_mean_number:]
                    output_wrench_ring_R = output_wrench_ring_R[wrench_offset_mean_number:]
                    output_wrench_baby_R = output_wrench_baby_R[wrench_offset_mean_number:]

                    input_timestamp_wrench = input_timestamp_wrench[wrench_offset_mean_number:]


                    # EMA 필터링 
                    alpha = 0.03
                    output_wrench_wrist_R = ema_filter(output_wrench_wrist_R, alpha)
                    output_wrench_thumb_R = ema_filter(output_wrench_thumb_R, alpha)
                    output_wrench_index_R = ema_filter(output_wrench_index_R, alpha)
                    output_wrench_middle_R = ema_filter(output_wrench_middle_R, alpha)
                    output_wrench_ring_R = ema_filter(output_wrench_ring_R, alpha)
                    output_wrench_baby_R = ema_filter(output_wrench_baby_R, alpha)

                    
                    # robot_timestamp로 가까운 wrench_timestamp 찾기 -> 기점으로 이전 32개 데이터 묶기
                    def find_nearest_wrench_indices(robot_timestamps, wrench_timestamps):
                        nearest_indices = []
                        for robot_time in robot_timestamps:
                            idx = np.searchsorted(wrench_timestamps, robot_time, side='left')
                            if np.abs(wrench_timestamps[idx]-robot_time) < np.abs(robot_time-wrench_timestamps[idx-1]):
                                nearest_indices.append(idx)
                            else:
                                nearest_indices.append(idx-1)
                            
                        return nearest_indices
                    
                    nearest_wrench_indices = find_nearest_wrench_indices(input_timestamp_robot, input_timestamp_wrench)

                    for i in range(len(input_timestamp_robot)):
                        wrench_idx = nearest_wrench_indices[i]
                        if wrench_idx < 31:
                            robot_idx = i
                        else:
                            break

                    # print(f"before robot idx {robot_idx+1}, it can't get 32 wrench data")

                    robot_start_idx = robot_idx + 1

                    input_timestamp_robot = input_timestamp_robot[robot_start_idx:]

                    # input_image_R = input_image_R[robot_start_idx:]
                    # input_image_T = input_image_T[robot_start_idx:]
                    
                    input_joint_R = input_joint_R[robot_start_idx:]
                    input_hand_pose_R = input_hand_pose_R[robot_start_idx:]

                    
                    # wrench 32개씩 묶기
                    output_wrench_wrist_R_32hist = []
                    output_wrench_thumb_R_32hist = []
                    output_wrench_index_R_32hist = []
                    output_wrench_middle_R_32hist = []
                    output_wrench_ring_R_32hist = []
                    output_wrench_baby_R_32hist = []
                    for i in range(robot_start_idx, robot_start_idx + len(input_timestamp_robot)):
                        wrench_idx = nearest_wrench_indices[i]
                        output_wrench_wrist_R_32hist.append(np.transpose(output_wrench_wrist_R[wrench_idx-31:wrench_idx+1]))  
                        output_wrench_thumb_R_32hist.append(np.transpose(output_wrench_thumb_R[wrench_idx-31:wrench_idx+1]))
                        output_wrench_index_R_32hist.append(np.transpose(output_wrench_index_R[wrench_idx-31:wrench_idx+1]))
                        output_wrench_middle_R_32hist.append(np.transpose(output_wrench_middle_R[wrench_idx-31:wrench_idx+1]))
                        output_wrench_ring_R_32hist.append(np.transpose(output_wrench_ring_R[wrench_idx-31:wrench_idx+1]))
                        output_wrench_baby_R_32hist.append(np.transpose(output_wrench_baby_R[wrench_idx-31:wrench_idx+1]))
                        

        

                    ### 이미지 변환 (배치 처리)
                    # output_image_R = np.array([transform(img) for img in input_image_R])
                    # output_image_T = np.array([transform(img) for img in input_image_T])


                    # joint -> pose, quat
                    # output_TCP_L = robot.fkine(input_joint_L)
                    output_TCP_R = robot.fkine(input_joint_R)

                    # output_TCP_pose_L = output_TCP_L.t
                    output_TCP_pose_R = output_TCP_R.t
                    # output_TCP_rotmat_L = output_TCP_L.R
                    output_TCP_rotmat_R = output_TCP_R.R
                    
                    # output_TCP_quat_L = R.from_matrix(output_TCP_rotmat_L).as_quat()
                    output_TCP_quat_R = R.from_matrix(output_TCP_rotmat_R).as_quat()
                    
                    # quaternion w가 양수가 되도록 변경
                    # output_TCP_quat_L = np.array([-q if q[3] < 0 else q for q in output_TCP_quat_L])
                    output_TCP_quat_R = np.array([-q if q[3] < 0 else q for q in output_TCP_quat_R])

                    # hand 15 -> 7 (thumb3, index2, middle2)
                    # output_hand_pose_L = input_hand_pose_L[:, [0,1,2, 4,5, 7,8]]
                    output_hand_pose_R = input_hand_pose_R[:, [0,1,2, 4,5, 7,8, 10,11, 13,14]]


                    # output_obs에 데이터 저장
                    # output_obs.create_dataset('robot_pose_L', data=output_TCP_pose_L[:-1])
                    # output_obs.create_dataset('robot_quat_L', data=output_TCP_quat_L[:-1])
                    output_obs.create_dataset('robot_pose_R', data=output_TCP_pose_R[:-1])
                    output_obs.create_dataset('robot_quat_R', data=output_TCP_quat_R[:-1])
                    # output_obs.create_dataset('hand_pose_L', data=output_hand_pose_L[:-1])
                    output_obs.create_dataset('hand_pose_R', data=output_hand_pose_R[:-1])

                    # output_obs.create_dataset('image0', data=output_image_H[:-1])
                    # output_obs.create_dataset('image1', data=output_image_F[:-1])
                    # output_obs.create_dataset('imageX', data=output_image_L[:-1])
                    # output_obs.create_dataset('image0', data=output_image_R[:-1])
                    # output_obs.create_dataset('image0', data=output_image_T[:-1])

                    output_obs.create_dataset('wrench_wrist_R', data=output_wrench_wrist_R_32hist[:-1])
                    output_obs.create_dataset('wrench_thumb_R', data=output_wrench_thumb_R_32hist[:-1])
                    output_obs.create_dataset('wrench_index_R', data=output_wrench_index_R_32hist[:-1])
                    output_obs.create_dataset('wrench_middle_R', data=output_wrench_middle_R_32hist[:-1])
                    output_obs.create_dataset('wrench_ring_R', data=output_wrench_ring_R_32hist[:-1])
                    output_obs.create_dataset('wrench_baby_R', data=output_wrench_baby_R_32hist[:-1])
                    

                    # actions 저장
                    # quat -> 6d rotation
                    # output_6d_rotation_L = quat_to_6d(output_TCP_quat_L)
                    output_6d_rotation_R = quat_to_6d(output_TCP_quat_R)

                    output_actions = np.hstack([output_TCP_pose_R, output_6d_rotation_R, output_hand_pose_R]).tolist()

                    output_demo_n.create_dataset('actions', data=output_actions[1:])

                    output_demo_idx += 1
        print("Data conversion completed / output_demo_lem =", output_demo_idx)
        

if __name__ == "__main__":
    main()
