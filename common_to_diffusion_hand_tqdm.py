import cv2
import h5py
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb
from roboticstoolbox import ERobot
from spatialmath import SE3, UnitQuaternion

urdf_path = "/home/vision/dualarm_ws/src/doosan-robot2/dsr_description2/urdf/m0609.white.urdf"
robot = rtb.ERobot.URDF(urdf_path)   


""" common data, 20Hz
data
    demo_0
        observations
            joint_L   # rad, len=6
            joint_R   # rad, len=6
            hand_L    # rad, len=15 (thumb3, index3, middle3, ring3, baby3)
            hand_R    # rad, len=15 (thumb3, index3, middle3, ring3, baby3)
            image_F   # (640, 480)   
            image_H   # (640, 480)
            image_L   # (640, 480)
            image_R   # (640, 480)
"""

""" diffusion data, 10Hz
data
    demo_0
        actions (robot_pose_L, robot_6d_L, robot_pose_R, robot_6d_R)
        obs
            robot_pose_L   # m, len=3 (x,y,z)
            robot_pose_R   # m, len=3 (x,y,z)
            robot_quat_L   # len=4 (x,y,z,w)
            robot_quat_R   # len=4 (x,y,z,w)
            hand_pose_L    # rad, len=7 (thumb3, index2, middle2)
            hand_pose_R    # rad, len=7 (thumb3, index2, middle2)
            image_F   # (320, 240)
            image_H   # (320, 240)
            # image_L   # (320, 240)
            # image_R   # (320, 240)
"""

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


    return rotation_6d

def resize_images(image_list, size=(320, 240)):
    """
    image_list : [img1, img2, ...] (각 img는 numpy array, shape (480,640,3))
    size       : (width, height)
    return     : [resized_img1, resized_img2, ...] (모두 (240,320,3))
    """
    return [cv2.resize(img, size) for img in image_list]



def main():
    input_filenames = ['1009_2011/common_data.hdf5', '1009_2110/common_data.hdf5', '1009_2203/common_data.hdf5']
    output_filename = 'diffusion_merong.hdf5'
    output_demo_idx = 0

    with h5py.File(output_filename, 'w') as output_file:

        output_data = output_file.create_group('data')

        for input_filename in input_filenames:
            with h5py.File(input_filename, 'r') as input_file:
                input_data = input_file['data']
                demo_len = len(input_data)
                print(input_filename, '/ demo_len =', demo_len)
                # input_file_tqdm = tqdm(range(demo_len))
                # demo_keys = sorted(list(input_data.keys()))

                for demo_idx in tqdm.tqdm(range(demo_len), desc=f"Processing {input_filename}"):
                    
                    # n번째 demo 생성
                    input_demo_name = f'demo_{demo_idx}'
                    output_demo_name = f'demo_{output_demo_idx}'
                    
                    input_demo_n = input_data[input_demo_name]
                    output_demo_n = output_data.create_group(output_demo_name)

                    # observations
                    input_obs = input_demo_n['observations']
                    output_obs = output_demo_n.create_group('obs')

                    # input_obs에서 데이터 꺼내기, 20Hz -> 10Hz
                    input_joint_L = input_obs['joint_L'][::2]
                    input_joint_R = input_obs['joint_R'][::2]
                    input_hand_pose_L = input_obs['hand_L'][::2]
                    input_hand_pose_R = input_obs['hand_R'][::2]
                    input_image_H = input_obs['image_H'][::2]
                    input_image_F = input_obs['image_F'][::2]
                    input_image_L = input_obs['image_L'][::2]
                    input_image_R = input_obs['image_R'][::2]

                    # image 해상도 조정
                    output_image_H = resize_images(input_image_H, (320, 240))
                    output_image_F = resize_images(input_image_F, (320, 240))
                    output_image_L = resize_images(input_image_L, (320, 240))
                    output_image_R = resize_images(input_image_R, (320, 240))

                    # bgr -> rgb
                    output_image_H = np.array(output_image_H)[..., ::-1]
                    output_image_F = np.array(output_image_F)[..., ::-1]
                    output_image_L = np.array(output_image_L)[..., ::-1]
                    output_image_R = np.array(output_image_R)[..., ::-1]
                    output_image_H = list(output_image_H)
                    output_image_F = list(output_image_F)
                    output_image_L = list(output_image_L)
                    output_image_R = list(output_image_R)

                    # joint -> pose, quat
                    output_TCP_L = robot.fkine(input_joint_L)
                    output_TCP_R = robot.fkine(input_joint_R)

                    output_TCP_pose_L = output_TCP_L.t
                    output_TCP_pose_R = output_TCP_R.t
                    output_TCP_rotmat_L = output_TCP_L.R
                    output_TCP_rotmat_R = output_TCP_R.R
                    
                    output_TCP_quat_L = R.from_matrix(output_TCP_rotmat_L).as_quat()
                    output_TCP_quat_R = R.from_matrix(output_TCP_rotmat_R).as_quat()
                    
                    # quaternion w가 양수가 되도록 변경
                    output_TCP_quat_L = np.array([-q if q[3] < 0 else q for q in output_TCP_quat_L])
                    output_TCP_quat_R = np.array([-q if q[3] < 0 else q for q in output_TCP_quat_R])

                    # hand 15 -> 7 (thumb3, index2, middle2)
                    output_hand_pose_L = input_hand_pose_L[:, [0,1,2, 4,5, 7,8]]
                    output_hand_pose_R = input_hand_pose_R[:, [0,1,2, 4,5, 7,8]]

                    # output_obs에 데이터 저장
                    output_obs.create_dataset('robot_pose_L', data=output_TCP_pose_L[:-1])
                    output_obs.create_dataset('robot_pose_R', data=output_TCP_pose_R[:-1])
                    output_obs.create_dataset('robot_quat_L', data=output_TCP_quat_L[:-1])
                    output_obs.create_dataset('robot_quat_R', data=output_TCP_quat_R[:-1])
                    output_obs.create_dataset('hand_pose_L', data=output_hand_pose_L[:-1])
                    output_obs.create_dataset('hand_pose_R', data=output_hand_pose_R[:-1])
                    output_obs.create_dataset('image0', data=output_image_H[:-1])
                    output_obs.create_dataset('image1', data=output_image_F[:-1])
                    # output_obs.create_dataset('image2', data=output_image_L[:-1])
                    # output_obs.create_dataset('image3', data=output_image_R[:-1])

                    # actions 저장
                    # quat -> 6d rotation
                    output_6d_rotation_L = quat_to_6d(output_TCP_quat_L)
                    output_6d_rotation_R = quat_to_6d(output_TCP_quat_R)

                    output_actions = np.hstack([output_TCP_pose_L, output_6d_rotation_L, 
                                        output_TCP_pose_R, output_6d_rotation_R,
                                        output_hand_pose_L, output_hand_pose_R]).tolist()

                    output_demo_n.create_dataset('actions', data=output_actions[1:])

                    output_demo_idx += 1
        print("Data conversion completed / output_demo_lem =", output_demo_idx)
        

if __name__ == "__main__":
    main()
