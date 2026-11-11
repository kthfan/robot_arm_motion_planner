"""Public API of the package and the default planner factory."""
import dataclasses

import numpy as np
import pytest

import motion_planner
from motion_planner import JointWaypoint, KinematicMotionPlanner, default_planner


def test_public_names_are_exported():
    for name in motion_planner.__all__:
        assert hasattr(motion_planner, name), name
    assert isinstance(motion_planner.__version__, str)


def test_joint_waypoint_is_a_plain_dataclass():
    wp = JointWaypoint([0.0] * 6, 50.0, 0.2, 100, False, False)
    assert dataclasses.asdict(wp) == {
        "positions": [0.0] * 6,
        "velocity": 50.0,
        "acc_time": 0.2,
        "blend_percentage": 100,
        "fine_goal": False,
        "blocked": False,
    }


def test_default_planner_rejects_unknown_names():
    with pytest.raises(ValueError, match="no default planner"):
        default_planner("no-such-library")


def test_default_planner_is_shared():
    pytest.importorskip("pinocchio")
    planner = default_planner("pinocchio")
    assert isinstance(planner, KinematicMotionPlanner)
    assert default_planner("pinocchio") is planner
    q = np.array([0.0, 0.3, 1.2, 0.2, 1.5, 0.1])
    assert planner.fk(q).shape == (4, 4)
