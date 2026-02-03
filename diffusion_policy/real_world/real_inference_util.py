import scipy.spatial.transform as st
from typing import Dict, Callable, Tuple
import numpy as np
import copy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import (
    convert_pose_mat_rep, compute_hand_relative_pose)
from diffusion_policy.model.common.pose_util import pose10d_to_mat, pose_to_mat, mat_to_pose10d
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
        shape_meta: dict) -> Dict[str, np.ndarray]:
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
        obs_pose_mat_L = pose_to_mat(np.concatenate([
            obs_dict_np['robot_pose_L'],
            st.Rotation.from_quat(obs_dict_np['robot_quat_L']).as_rotvec()
        ], axis=-1)) 

        obs_relative_pose_mat_L = convert_pose_mat_rep(
            pose_mat=obs_pose_mat_L,
            base_pose_mat=obs_pose_mat_L[-1],
            pose_rep='relative',
            backward=False)
            
        obs_relative_pose_L = mat_to_pose10d(obs_relative_pose_mat_L) # mat -> pos + rot6d (9)
        obs_dict_np['robot_pose_L'] = obs_relative_pose_L[..., :3].astype(np.float32)
        obs_dict_np['robot_quat_L'] = obs_relative_pose_L[..., 3:].astype(np.float32)

    if use_right_arm:
        obs_pose_mat_R = pose_to_mat(np.concatenate([
            obs_dict_np['robot_pose_R'],
            st.Rotation.from_quat(obs_dict_np['robot_quat_R']).as_rotvec()
        ], axis=-1))

        obs_relative_pose_mat_R = convert_pose_mat_rep(
            pose_mat=obs_pose_mat_R,
            base_pose_mat=obs_pose_mat_R[-1],
            pose_rep='relative',
            backward=False)
        
        obs_relative_pose_R = mat_to_pose10d(obs_relative_pose_mat_R) # mat -> pos + rot6d (9)
        obs_dict_np['robot_pose_R'] = obs_relative_pose_R[..., :3].astype(np.float32)
        obs_dict_np['robot_quat_R'] = obs_relative_pose_R[..., 3:].astype(np.float32)
        
    if use_left_hand:
        obs_dict_np['hand_pose_L'] = compute_hand_relative_pose(
            pos=obs_dict_np['hand_pose_L'],
            base_pos=obs_dict_np['hand_pose_L'][-1]).astype(np.float32)
        
    if use_right_hand:
        obs_dict_np['hand_pose_R'] = compute_hand_relative_pose(
            pos=obs_dict_np['hand_pose_R'],
            base_pos=obs_dict_np['hand_pose_R'][-1]).astype(np.float32)
    
    
    return obs_dict_np


def get_abs_action_from_relative(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray]
    ):
    
    env_action = list()

    use_left_arm = 'robot_pose_L' in env_obs
    use_right_arm = 'robot_pose_R' in env_obs
    use_left_hand = 'hand_pose_L' in env_obs
    use_right_hand = 'hand_pose_R' in env_obs
    action_index = 0

    if use_left_arm:
        obs_pose_mat_L = pose_to_mat(np.concatenate([
            env_obs['robot_pose_L'],
            st.Rotation.from_quat(env_obs['robot_quat_L']).as_rotvec()
        ], axis=-1))
        action_relative_pose_mat_L = pose10d_to_mat(action[..., action_index:action_index+9])
        action_pose_mat_L = convert_pose_mat_rep(
            pose_mat=action_relative_pose_mat_L,
            base_pose_mat=obs_pose_mat_L[-1],
            pose_rep='relative',
            backward=True)
        action_pose_L = mat_to_pose10d(action_pose_mat_L)
        env_action.append(action_pose_L) # pos + rot6d (9)
        action_index += 9
        
    if use_right_arm:
        obs_pose_mat_R = pose_to_mat(np.concatenate([
            env_obs['robot_pose_R'],
            st.Rotation.from_quat(env_obs['robot_quat_R']).as_rotvec()
        ], axis=-1))
        action_relative_pose_mat_R = pose10d_to_mat(action[..., action_index:action_index+9])
        action_pose_mat_R = convert_pose_mat_rep(
            pose_mat=action_relative_pose_mat_R,
            base_pose_mat=obs_pose_mat_R[-1],
            pose_rep='relative',
            backward=True)
        action_pose_R = mat_to_pose10d(action_pose_mat_R)
        env_action.append(action_pose_R) # pos + rot6d (9)
        action_index += 9
    
    hand_length = action.shape[-1] - action_index
    if hand_length > 0:      
        if use_left_hand:
            if use_right_hand:
                # both hand
                action_hand_L = compute_hand_relative_pose(
                    pos=action[..., action_index:action_index + hand_length//2],
                    base_pos=env_obs['hand_pose_L'][-1],
                    backward=True)
                action_hand_R = compute_hand_relative_pose(
                    pos=action[..., action_index + hand_length//2:],
                    base_pos=env_obs['hand_pose_R'][-1],
                    backward=True)
                env_action.append(action_hand_L)
                env_action.append(action_hand_R)
            else:
                # left_hand
                action_hand_L = compute_hand_relative_pose(
                    pos=action[..., action_index:],
                    base_pos=env_obs['hand_pose_L'][-1],
                    backward=True)
                env_action.append(action_hand_L)
        else:
            # right_hand
            action_hand_R = compute_hand_relative_pose(
                pos=action[..., action_index:],
                base_pos=env_obs['hand_pose_R'][-1],
                backward=True)
            env_action.append(action_hand_R)   

    env_action = np.concatenate(env_action, axis=-1)
    return env_action

# For relative PIGDM
def get_relative_action_from_abs(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray]
    ):
    
    env_action = list()

    use_left_arm = 'robot_pose_L' in env_obs
    use_right_arm = 'robot_pose_R' in env_obs
    use_left_hand = 'hand_pose_L' in env_obs
    use_right_hand = 'hand_pose_R' in env_obs
    action_index = 0

    if use_left_arm:
        obs_pose_mat_L = pose_to_mat(np.concatenate([
            env_obs['robot_pose_L'],
            st.Rotation.from_quat(env_obs['robot_quat_L']).as_rotvec()
        ], axis=-1))
        action_abs_pose_mat_L = pose10d_to_mat(action[..., action_index:action_index+9])
        action_pose_mat_L = convert_pose_mat_rep(
            pose_mat=action_abs_pose_mat_L,
            base_pose_mat=obs_pose_mat_L[-1],
            pose_rep='relative',
            backward=False)
        action_pose_L = mat_to_pose10d(action_pose_mat_L)
        env_action.append(action_pose_L) # pos + rot6d (9)
        action_index += 9
        
    if use_right_arm:
        obs_pose_mat_R = pose_to_mat(np.concatenate([
            env_obs['robot_pose_R'],
            st.Rotation.from_quat(env_obs['robot_quat_R']).as_rotvec()
        ], axis=-1))
        action_abs_pose_mat_R = pose10d_to_mat(action[..., action_index:action_index+9])
        action_pose_mat_R = convert_pose_mat_rep(
            pose_mat=action_abs_pose_mat_R,
            base_pose_mat=obs_pose_mat_R[-1],
            pose_rep='relative',
            backward=False)
        action_pose_R = mat_to_pose10d(action_pose_mat_R)
        env_action.append(action_pose_R) # pos + rot6d (9)
        action_index += 9
    
    hand_length = action.shape[-1] - action_index
    if hand_length > 0:      
        if use_left_hand:
            if use_right_hand:
                # both hand
                action_hand_L = compute_hand_relative_pose(
                    pos=action[..., action_index:action_index + hand_length//2],
                    base_pos=env_obs['hand_pose_L'][-1],
                    backward=False)
                action_hand_R = compute_hand_relative_pose(
                    pos=action[..., action_index + hand_length//2:],
                    base_pos=env_obs['hand_pose_R'][-1],
                    backward=False)
                env_action.append(action_hand_L)
                env_action.append(action_hand_R)
            else:
                # left_hand
                action_hand_L = compute_hand_relative_pose(
                    pos=action[..., action_index:],
                    base_pos=env_obs['hand_pose_L'][-1],
                    backward=False)
                env_action.append(action_hand_L)
        else:
            # right_hand
            action_hand_R = compute_hand_relative_pose(
                pos=action[..., action_index:],
                base_pos=env_obs['hand_pose_R'][-1],
                backward=False)
            env_action.append(action_hand_R)   

    env_action = np.concatenate(env_action, axis=-1)
    return env_action