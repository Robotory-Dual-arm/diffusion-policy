import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from diffusion_policy.residual_rl.pose import (
    abs_pose9_to_relative_pose9,
    apply_delta6_to_pose9,
    apply_residual_action_to_pose9,
    current_obs_to_pose9,
    delta6_from_base_to_target,
    delta6_to_mat,
    mat_to_pose9,
    pose6_to_pose9,
    pose9_to_mat,
    relative_pose9_to_abs_pose9,
)


def _pose6(position, rotvec):
    return np.asarray([*position, *rotvec], dtype=np.float32)


def test_pose9_matrix_round_trip_batched():
    pose6 = np.stack(
        [
            _pose6([0.1, -0.2, 0.3], [0.2, -0.1, 0.05]),
            _pose6([-0.3, 0.4, 0.2], [-0.3, 0.2, 0.1]),
        ]
    )
    pose9 = pose6_to_pose9(pose6)
    reconstructed = mat_to_pose9(pose9_to_mat(pose9))

    assert pose9.shape == (2, 9)
    np.testing.assert_allclose(
        pose9_to_mat(reconstructed), pose9_to_mat(pose9), atol=1e-6
    )


def test_delta6_is_right_composed_in_base_local_frame():
    base = pose6_to_pose9(
        _pose6([0.4, -0.2, 0.1], [0.0, 0.0, np.pi / 2])
    )
    delta = np.asarray(
        [0.1, 0.0, -0.02, 0.1, -0.2, 0.05], dtype=np.float32
    )

    target = apply_delta6_to_pose9(base, delta)
    expected_matrix = pose9_to_mat(base) @ delta6_to_mat(delta)
    recovered_delta = delta6_from_base_to_target(base, target)

    np.testing.assert_allclose(pose9_to_mat(target), expected_matrix, atol=1e-6)
    np.testing.assert_allclose(recovered_delta, delta, atol=1e-6)
    # A local +X displacement is world +Y under the base's +90 degree yaw.
    np.testing.assert_allclose(
        target[:3], np.asarray([0.4, -0.1, 0.08]), atol=1e-6
    )


def test_relative_and_absolute_pose9_are_inverses():
    base = pose6_to_pose9(_pose6([0.2, 0.3, -0.1], [0.2, 0.1, -0.3]))
    absolute = pose6_to_pose9(
        _pose6([-0.1, 0.5, 0.25], [-0.2, 0.4, 0.15])
    )

    relative = abs_pose9_to_relative_pose9(base, absolute)
    reconstructed = relative_pose9_to_abs_pose9(base, relative)

    np.testing.assert_allclose(
        pose9_to_mat(reconstructed), pose9_to_mat(absolute), atol=1e-6
    )


def test_apply_residual_preserves_hand_tail():
    pose = pose6_to_pose9(_pose6([0.1, 0.2, 0.3], [0.0, 0.1, 0.0]))
    hand = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
    base_action = np.concatenate((pose, hand))
    delta = np.asarray(
        [0.01, -0.02, 0.03, 0.04, 0.0, -0.02], dtype=np.float32
    )

    result = apply_residual_action_to_pose9(base_action, delta)

    assert result.shape == (16,)
    np.testing.assert_array_equal(result[9:], hand)
    np.testing.assert_allclose(
        pose9_to_mat(result[:9]),
        pose9_to_mat(pose) @ delta6_to_mat(delta),
        atol=1e-6,
    )


def test_current_obs_to_pose9_accepts_single_step_and_history():
    quaternion = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_quat().astype(
        np.float32
    )
    single = {
        "robot_pose_R": np.asarray([0.2, -0.1, 0.4], dtype=np.float32),
        "robot_quat_R": quaternion,
    }
    history = {
        "robot_pose_R": np.stack(
            [np.zeros(3, dtype=np.float32), single["robot_pose_R"]]
        ),
        "robot_quat_R": np.stack(
            [np.asarray([0, 0, 0, 1], dtype=np.float32), quaternion]
        ),
    }

    np.testing.assert_allclose(
        pose9_to_mat(current_obs_to_pose9(single)),
        pose9_to_mat(current_obs_to_pose9(history, step=-1)),
        atol=1e-6,
    )


def test_invalid_pose_and_residual_shapes_raise():
    with pytest.raises(ValueError, match="pose9"):
        pose9_to_mat(np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="delta6"):
        delta6_to_mat(np.zeros(7, dtype=np.float32))

