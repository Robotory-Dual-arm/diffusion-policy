from typing import Dict, Callable, Tuple
import numpy as np
import copy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import (
    convert_pose_mat_rep, compute_relative_pose, compute_hand_relative_pose)
from diffusion_policy.model.common.rotation_transformer_rel import RotationTransformer

# abs일때 
def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


# obs에서 image의 해상도 출력 (width, height)
def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res



# ============== relative 추가 ================
# relative일때
def get_real_relative_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        rot_quat2mat=None,
        rot_mat2target: dict=None,
        ) -> Dict[str, np.ndarray]:
    """
    Compute relative poses for real robot observations similar to son_robomimic_replay_dataset.
    
    Args:
        env_obs: Environment observations
        shape_meta: Shape metadata
        rot_quat2mat: Rotation transformer from quaternion to matrix
        rot_mat2target: Dictionary mapping robot keys to target rotation transformers
    
    Returns:
        Dictionary with relative pose observations
    """
    
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    
    assert rot_quat2mat is not None
    assert rot_mat2target is not None
    
    # relative pose 계산
    # Process non-pose observations first
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim' and 'wrt' not in key:
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    

    # Handle bimanual relative pose computation
    use_left_arm = 'robot_pose_L' in env_obs 
    use_right_arm = 'robot_pose_R' in env_obs
    use_left_hand = 'hand_pose_L' in env_obs
    use_right_hand = 'hand_pose_R' in env_obs


    if use_left_arm:
        current_pose_L = copy.copy(env_obs['robot_pose_L'][-1])
        current_rot_mat_L = copy.copy(rot_quat2mat.forward(env_obs['robot_quat_L'][-1]))

        obs_dict_np['robot_pose_L'], obs_dict_np['robot_quat_L'] = compute_relative_pose(
            pos=env_obs['robot_pose_L'],
            rot=env_obs['robot_quat_L'],
            base_pos=current_pose_L,
            base_rot_mat=current_rot_mat_L,
            rot_transformer_to_mat=rot_quat2mat,
            rot_transformer_to_target=rot_mat2target.get('robot_quat_L', rot_quat2mat))

    if use_right_arm:
        current_pose_R = copy.copy(env_obs['robot_pose_R'][-1])
        current_rot_mat_R = copy.copy(rot_quat2mat.forward(env_obs['robot_quat_R'][-1]))
    
        obs_dict_np['robot_pose_R'], obs_dict_np['robot_quat_R'] = compute_relative_pose(
            pos=env_obs['robot_pose_R'],
            rot=env_obs['robot_quat_R'],
            base_pos=current_pose_R,
            base_rot_mat=current_rot_mat_R,
            rot_transformer_to_mat=rot_quat2mat,
            rot_transformer_to_target=rot_mat2target.get('robot_quat_R', rot_quat2mat))

    if use_left_hand:
        current_hand_pose_L = copy.copy(env_obs['hand_pose_L'][-1])
        obs_dict_np['hand_pose_L'] = compute_hand_relative_pose(
            pos=env_obs['hand_pose_L'],
            base_pos=current_hand_pose_L)
        
    if use_right_hand:
        current_hand_pose_R = copy.copy(env_obs['hand_pose_R'][-1])
        obs_dict_np['hand_pose_R'] = compute_hand_relative_pose(
            pos=env_obs['hand_pose_R'],
            base_pos=current_hand_pose_R)
    
    return obs_dict_np


def action_to_pose(action: np.ndarray, rot_quat2mat=None):
    
    assert action.shape[-1] == 9

    rot_sd2mat = RotationTransformer('rotation_6d', 'matrix')
    
    pose = np.zeros((action.shape[0], 4, 4), dtype=np.float32)
    pose[..., :3, 3] = action[..., :3]
    ## RotationTransformer로 6D -> Matrix 변환 시 rot[:2, :] 사용
    ## 실제 데이터셋 & 학습에는 rot[:, :2]를 사용했으니 RotationTransformer의 결과에 *transpose* 필요
    # pose[..., :3, :3] = rot_sd2mat.forward(action[..., 3:]).transpose(0, 2, 1)
    ## 수정: transpose 제거 -> 아래 mat_to_action에서 어차피 필요한 action의 6D 성분을 적절히 가져와서 6D rot으로 만들어버림
    # print("rot_sd2mat.forward(action[..., 3:]):", rot_sd2mat.forward(action[..., 3:]).transpose(0, 2, 1))
    pose[..., :3, :3] = rot_sd2mat.forward(action[..., 3:])#.transpose(0, 2, 1)
    pose[..., 3, 3] = 1.0
    return pose

def mat_to_action(pose: np.ndarray, rot_mat2target=None):
    # SE3로 표현된 action set을 9D로 변환
    assert pose.shape[-2:] == (4,4)
    assert rot_mat2target is not None
    action = np.zeros((pose.shape[0], 9), dtype=np.float32)
    action[..., :3] = pose[..., :3, 3]
    action[..., 3:] = rot_mat2target.forward(pose[..., :3, :3])
    return action

def get_real_relative_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='relative',
        rot_quat2mat=None,
        rot_mat2target=None,
    ):

    assert action.shape[-1] == 32
    assert rot_quat2mat is not None

    print(action[-1])

    env_action = list()
    robot_idx_map = {'L': 0, 'R': 1}
    for robot_idx in ('L', 'R'):
        base_pose_mat = np.eye(4, dtype=np.float32)

        if robot_idx == 'L':
            base_pos = env_obs['robot_pose_L'][-1]
            base_rot_quat = env_obs['robot_quat_L'][-1]
        else:
            base_pos = env_obs['robot_pose_R'][-1]
            base_rot_quat = env_obs['robot_quat_R'][-1]
        ## debug
        # print(f"robot_idx: {robot_idx}")
        # print(f"base_rot_quat: {base_rot_quat}")
        # print(f"base_rot_quat shape: {base_rot_quat.shape if hasattr(base_rot_quat, 'shape') else 'No shape'}")
        # print(f"base_rot_quat dtype: {base_rot_quat.dtype if hasattr(base_rot_quat, 'dtype') else 'No dtype'}")
        # print(f"rot_quat2mat: {rot_quat2mat}")
        ## debug
        base_rot_mat = rot_quat2mat.forward(base_rot_quat)
        base_pose_mat[:3,3] = base_pos
        base_pose_mat[:3,:3] = base_rot_mat
        # print("robot_idx:", robot_idx)
        # print(f"base_pose_mat: {base_pose_mat}")
        # print("action_to_pose:", action_to_pose(action[..., robot_idx_map[robot_idx] * 9: robot_idx_map[robot_idx] * 9 + 9], rot_quat2mat))
        action_mat = convert_pose_mat_rep(
            pose_mat=action_to_pose(action[..., robot_idx_map[robot_idx] * 16: robot_idx_map[robot_idx] * 16 + 9], rot_quat2mat),
            base_pose_mat=base_pose_mat,
            pose_rep=action_pose_repr,
            backward=True)
        action_pose = mat_to_action(action_mat, rot_mat2target['robot_quat_'+robot_idx])
        hand_action = action[..., robot_idx_map[robot_idx] * 16 + 9: robot_idx_map[robot_idx] * 16 + 16]
        print(f"{robot_idx}_hand_action: {hand_action}")
        # print()
        # print(f"ACTION_POSE!!!: {action_pose[:5]}")
        env_action.append(action_pose)
        env_action.append(hand_action)
    env_action = np.concatenate(env_action, axis=-1)
    assert env_action.shape[-1] == 32
    # print(f"ENV_ACTION!!!: {env_action[:2]}")
    return env_action