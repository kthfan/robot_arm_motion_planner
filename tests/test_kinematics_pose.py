import numpy as np
import pytest

from motion_planner.pose import matrix_to_pose, pose_distance, pose_to_matrix


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_pose_to_matrix_uses_tm_fixed_xyz_convention():
    pose = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    T = pose_to_matrix(pose)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(T[:3, :3], _rz(0.6) @ _ry(-0.5) @ _rx(0.4), atol=1e-12)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_matrix_to_pose_round_trips():
    pose = [0.5, 0.0, -0.1, -1.0, 0.7, 2.5]
    np.testing.assert_allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-12)


def test_matrix_to_pose_returns_plain_list_of_six_floats():
    out = matrix_to_pose(np.eye(4))
    assert isinstance(out, list) and len(out) == 6
    assert all(isinstance(v, float) for v in out)


def test_pose_distance_reports_translation_and_rotation_angle():
    a = pose_to_matrix([0, 0, 0, 0, 0, 0])
    b = pose_to_matrix([0.3, 0.4, 0, 0, 0, 0.25])
    lin, ang = pose_distance(a, b)
    assert lin == pytest.approx(0.5)
    assert ang == pytest.approx(0.25)
