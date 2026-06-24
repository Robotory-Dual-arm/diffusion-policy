import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.model.common.pose_util import (
    mat_to_pose10d,
    pose10d_to_mat,
    pose_to_mat,
)


def _as_array(x):
    return np.asarray(x, dtype=np.float32)


def pose6_to_pose9(pose6):
    pose6 = _as_array(pose6)
    return mat_to_pose10d(pose_to_mat(pose6)).astype(np.float32)


def pose9_to_mat(pose9):
    pose9 = _as_array(pose9)
    if pose9.shape[-1] != 9:
        raise ValueError(f"Expected pose9 (..., 9), got {pose9.shape}")
    return pose10d_to_mat(pose9)


def mat_to_pose9(mat):
    return mat_to_pose10d(_as_array(mat)).astype(np.float32)


def pose_like_to_pose9(pose):
    pose = _as_array(pose)
    if pose.shape[-1] >= 9:
        return pose[..., :9].astype(np.float32)
    if pose.shape[-1] == 6:
        return pose6_to_pose9(pose)
    raise ValueError(f"Expected pose with 6 or >=9 dims, got {pose.shape}")


def current_obs_to_pose9(obs_dict, arm="R", step=-1):
    pos_key = f"robot_pose_{arm}"
    quat_key = f"robot_quat_{arm}"
    pos = _as_array(obs_dict[pos_key])[step]
    quat = _as_array(obs_dict[quat_key])[step]
    rotvec = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate([pos, rotvec], axis=-1))


def relative_pose9_to_abs_pose9(base_pose9, relative_pose9):
    base_mat = pose9_to_mat(base_pose9)
    rel_mat = pose9_to_mat(relative_pose9)
    return mat_to_pose9(base_mat @ rel_mat)


def abs_pose9_to_relative_pose9(base_pose9, abs_pose9):
    base_mat = pose9_to_mat(base_pose9)
    abs_mat = pose9_to_mat(abs_pose9)
    return mat_to_pose9(np.linalg.inv(base_mat) @ abs_mat)


def delta6_from_base_to_target(base_pose9, target_pose9):
    base_mat = pose9_to_mat(base_pose9)
    target_mat = pose9_to_mat(target_pose9)
    delta_mat = np.linalg.inv(base_mat) @ target_mat
    delta_pos = delta_mat[..., :3, 3]
    delta_rot = Rotation.from_matrix(delta_mat[..., :3, :3]).as_rotvec()
    return np.concatenate([delta_pos, delta_rot], axis=-1).astype(np.float32)


def residual_pose9_from_base_to_target(base_pose9, target_pose9):
    base_mat = pose9_to_mat(base_pose9)
    target_mat = pose9_to_mat(target_pose9)
    delta_mat = np.linalg.inv(base_mat) @ target_mat
    return mat_to_pose9(delta_mat)


def delta6_to_mat(delta6):
    delta6 = _as_array(delta6)
    if delta6.shape[-1] != 6:
        raise ValueError(f"Expected delta6 (..., 6), got {delta6.shape}")
    mat = np.zeros(delta6.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, :3] = Rotation.from_rotvec(delta6[..., 3:]).as_matrix()
    mat[..., :3, 3] = delta6[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def apply_delta6_to_pose9(base_pose9, delta6):
    base_mat = pose9_to_mat(base_pose9)
    delta_mat = delta6_to_mat(delta6)
    return mat_to_pose9(base_mat @ delta_mat)


def apply_residual_action_to_pose9(base_pose9, residual_action):
    residual_action = _as_array(residual_action)
    if residual_action.shape[-1] == 6:
        return apply_delta6_to_pose9(base_pose9, residual_action)
    if residual_action.shape[-1] == 9:
        return relative_pose9_to_abs_pose9(base_pose9, residual_action)
    raise ValueError(f"Expected residual action with 6 or 9 dims, got {residual_action.shape}")
