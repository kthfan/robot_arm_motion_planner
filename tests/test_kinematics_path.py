import numpy as np
import pytest

from motion_planner.path import CartesianPath, LinearPath
from motion_planner.pose import pose_distance, pose_to_matrix


def test_from_poses_returns_linear_path_hitting_both_endpoints():
    start = pose_to_matrix([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    goal = pose_to_matrix([0.4, -0.2, 0.5, 0.3, -0.2, 1.0])
    path = CartesianPath.from_poses(start, goal)
    assert isinstance(path, LinearPath)
    np.testing.assert_allclose(path.pose_at(0.0), start, atol=1e-12)
    np.testing.assert_allclose(path.pose_at(1.0), goal, atol=1e-12)


def test_translation_only_moves_along_straight_line_with_fixed_orientation():
    start = pose_to_matrix([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
    goal = pose_to_matrix([0.2, 0.4, -0.4, 0.1, 0.2, 0.3])
    path = CartesianPath.from_poses(start, goal)
    assert path.translation_length == pytest.approx(0.6)
    assert path.rotation_angle == pytest.approx(0.0, abs=1e-12)
    mid = path.pose_at(0.25)
    np.testing.assert_allclose(mid[:3, 3], [0.05, 0.1, -0.1], atol=1e-12)
    np.testing.assert_allclose(mid[:3, :3], start[:3, :3], atol=1e-12)


def test_rotation_only_rotates_about_fixed_axis_at_constant_rate():
    start = pose_to_matrix([0.3, 0.1, 0.2, 0.0, 0.0, 0.0])
    goal = pose_to_matrix([0.3, 0.1, 0.2, 0.0, 0.0, 1.2])
    path = CartesianPath.from_poses(start, goal)
    assert path.translation_length == pytest.approx(0.0, abs=1e-12)
    assert path.rotation_angle == pytest.approx(1.2)
    lin, ang = pose_distance(start, path.pose_at(0.5))
    assert lin == pytest.approx(0.0, abs=1e-12)
    assert ang == pytest.approx(0.6)
    np.testing.assert_allclose(
        path.pose_at(0.5), pose_to_matrix([0.3, 0.1, 0.2, 0.0, 0.0, 0.6]), atol=1e-12
    )


def test_rotation_takes_shortest_way():
    start = pose_to_matrix([0, 0, 0, 0, 0, np.radians(170)])
    goal = pose_to_matrix([0, 0, 0, 0, 0, np.radians(-170)])
    path = CartesianPath.from_poses(start, goal)
    assert path.rotation_angle == pytest.approx(np.radians(20))


def test_pose_at_rejects_parameter_outside_unit_interval():
    path = CartesianPath.from_poses(np.eye(4), pose_to_matrix([0.1, 0, 0, 0, 0, 0]))
    with pytest.raises(ValueError):
        path.pose_at(1.5)
