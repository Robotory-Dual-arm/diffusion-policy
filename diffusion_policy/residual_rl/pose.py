"""SE(3) pose utilities used by residual RL.

This module intentionally owns its pose math instead of importing from
``diffusion_policy.residual_policy``.  A pose9 is encoded as XYZ followed by
the first two columns of a rotation matrix (rotation-6D).  A residual delta6
is XYZ translation followed by a rotation vector.  Residuals are composed on
the right, i.e. ``target = base @ delta``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


def _as_float32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def rot6d_to_mat(rotation6d) -> np.ndarray:
    """Convert first-two-column rotation-6D vectors to rotation matrices."""
    rotation6d = _as_float32(rotation6d)
    if rotation6d.shape[-1] != 6:
        raise ValueError(
            f"Expected rotation6d with shape (..., 6), got {rotation6d.shape}"
        )
    first = rotation6d[..., :3]
    second = rotation6d[..., 3:]
    basis1 = _normalize(first)
    basis2 = second - np.sum(
        basis1 * second, axis=-1, keepdims=True
    ) * basis1
    basis2 = _normalize(basis2)
    basis3 = np.cross(basis1, basis2, axis=-1)
    return np.stack((basis1, basis2, basis3), axis=-1).astype(np.float32)


def mat_to_rot6d(matrix) -> np.ndarray:
    """Convert rotation matrices to the repository's rotation-6D layout."""
    matrix = _as_float32(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrix with shape (..., 3, 3), got {matrix.shape}"
        )
    batch_shape = matrix.shape[:-2]
    return (
        matrix[..., :, :2]
        .swapaxes(-1, -2)
        .copy()
        .reshape(batch_shape + (6,))
        .astype(np.float32)
    )


def pose6_to_mat(pose6) -> np.ndarray:
    """Convert XYZ + rotation-vector pose6 to a homogeneous matrix."""
    pose6 = _as_float32(pose6)
    if pose6.shape[-1] != 6:
        raise ValueError(f"Expected pose6 with shape (..., 6), got {pose6.shape}")
    matrix = np.zeros(pose6.shape[:-1] + (4, 4), dtype=np.float32)
    matrix[..., :3, :3] = Rotation.from_rotvec(
        pose6[..., 3:]
    ).as_matrix().astype(np.float32)
    matrix[..., :3, 3] = pose6[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def pose6_to_pose9(pose6) -> np.ndarray:
    return mat_to_pose9(pose6_to_mat(pose6))


def pose9_to_mat(pose9) -> np.ndarray:
    """Convert XYZ + rotation-6D pose9 to a homogeneous matrix."""
    pose9 = _as_float32(pose9)
    if pose9.shape[-1] != 9:
        raise ValueError(f"Expected pose9 with shape (..., 9), got {pose9.shape}")
    matrix = np.zeros(pose9.shape[:-1] + (4, 4), dtype=np.float32)
    matrix[..., :3, :3] = rot6d_to_mat(pose9[..., 3:])
    matrix[..., :3, 3] = pose9[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def mat_to_pose9(matrix) -> np.ndarray:
    matrix = _as_float32(matrix)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected homogeneous matrix with shape (..., 4, 4), got {matrix.shape}"
        )
    position = matrix[..., :3, 3]
    rotation6d = mat_to_rot6d(matrix[..., :3, :3])
    return np.concatenate((position, rotation6d), axis=-1).astype(np.float32)


def pose_like_to_pose9(pose) -> np.ndarray:
    """Return the leading pose9, converting pose6 inputs when necessary."""
    pose = _as_float32(pose)
    if pose.shape[-1] >= 9:
        return pose[..., :9].astype(np.float32)
    if pose.shape[-1] == 6:
        return pose6_to_pose9(pose)
    raise ValueError(f"Expected pose with 6 or at least 9 dims, got {pose.shape}")


def current_obs_to_pose9(
    obs_dict: Mapping[str, np.ndarray], arm: str = "R", step: int = -1
) -> np.ndarray:
    """Build pose9 from a robot position and XYZW quaternion observation."""
    position = _as_float32(obs_dict[f"robot_pose_{arm}"])
    quaternion = _as_float32(obs_dict[f"robot_quat_{arm}"])
    if position.shape[-1] != 3 or quaternion.shape[-1] != 4:
        raise ValueError(
            "Expected robot position (..., 3) and quaternion (..., 4), got "
            f"{position.shape} and {quaternion.shape}"
        )
    if position.ndim > 1:
        position = position[step]
    if quaternion.ndim > 1:
        quaternion = quaternion[step]
    rotvec = Rotation.from_quat(quaternion).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate((position, rotvec), axis=-1))


def relative_pose9_to_abs_pose9(base_pose9, relative_pose9) -> np.ndarray:
    return mat_to_pose9(pose9_to_mat(base_pose9) @ pose9_to_mat(relative_pose9))


def abs_pose9_to_relative_pose9(base_pose9, abs_pose9) -> np.ndarray:
    base_matrix = pose9_to_mat(base_pose9)
    abs_matrix = pose9_to_mat(abs_pose9)
    return mat_to_pose9(np.linalg.inv(base_matrix) @ abs_matrix)


def delta6_from_base_to_target(base_pose9, target_pose9) -> np.ndarray:
    """Compute local delta6 satisfying ``target = base @ delta``."""
    base_matrix = pose9_to_mat(base_pose9)
    target_matrix = pose9_to_mat(target_pose9)
    delta_matrix = np.linalg.inv(base_matrix) @ target_matrix
    delta_position = delta_matrix[..., :3, 3]
    delta_rotvec = Rotation.from_matrix(
        delta_matrix[..., :3, :3]
    ).as_rotvec().astype(np.float32)
    return np.concatenate((delta_position, delta_rotvec), axis=-1).astype(
        np.float32
    )


def residual_pose9_from_base_to_target(base_pose9, target_pose9) -> np.ndarray:
    base_matrix = pose9_to_mat(base_pose9)
    target_matrix = pose9_to_mat(target_pose9)
    return mat_to_pose9(np.linalg.inv(base_matrix) @ target_matrix)


def delta6_to_mat(delta6) -> np.ndarray:
    delta6 = _as_float32(delta6)
    if delta6.shape[-1] != 6:
        raise ValueError(
            f"Expected delta6 with shape (..., 6), got {delta6.shape}"
        )
    matrix = np.zeros(delta6.shape[:-1] + (4, 4), dtype=np.float32)
    matrix[..., :3, :3] = Rotation.from_rotvec(
        delta6[..., 3:]
    ).as_matrix().astype(np.float32)
    matrix[..., :3, 3] = delta6[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def apply_delta6_to_pose9(base_pose9, delta6) -> np.ndarray:
    return mat_to_pose9(pose9_to_mat(base_pose9) @ delta6_to_mat(delta6))


def apply_residual_action_to_pose9(base_action, residual_action) -> np.ndarray:
    """Correct the leading EE pose9 and preserve the slow-owned action tail."""
    base_action = _as_float32(base_action)
    if base_action.shape[-1] < 9:
        raise ValueError(
            f"Expected base action with at least pose9, got {base_action.shape}"
        )
    residual_action = _as_float32(residual_action)
    if residual_action.shape[-1] == 6:
        corrected_pose9 = apply_delta6_to_pose9(
            base_action[..., :9], residual_action
        )
    elif residual_action.shape[-1] == 9:
        corrected_pose9 = relative_pose9_to_abs_pose9(
            base_action[..., :9], residual_action
        )
    else:
        raise ValueError(
            "Expected residual action with 6 or 9 dims, got "
            f"{residual_action.shape}"
        )
    if base_action.shape[-1] == 9:
        return corrected_pose9
    return np.concatenate(
        (corrected_pose9, base_action[..., 9:]), axis=-1
    ).astype(np.float32)


__all__ = [
    "abs_pose9_to_relative_pose9",
    "apply_delta6_to_pose9",
    "apply_residual_action_to_pose9",
    "current_obs_to_pose9",
    "delta6_from_base_to_target",
    "delta6_to_mat",
    "mat_to_pose9",
    "mat_to_rot6d",
    "pose6_to_mat",
    "pose6_to_pose9",
    "pose9_to_mat",
    "pose_like_to_pose9",
    "relative_pose9_to_abs_pose9",
    "residual_pose9_from_base_to_target",
    "rot6d_to_mat",
]
