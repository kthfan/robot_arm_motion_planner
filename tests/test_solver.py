import math

import numpy as np
import pytest

from motion_planner.self_motion import SelfMotion
from motion_planner.solver import KinematicsSolver


class _Limits(KinematicsSolver):
    """Solver stub with given joint limits (no kinematics)."""

    def __init__(self, lower, upper):
        self._lo, self._hi = np.asarray(lower, float), np.asarray(upper, float)

    @property
    def lower_limits(self):
        return self._lo

    @property
    def upper_limits(self):
        return self._hi

    @property
    def velocity_limits(self):
        return np.ones_like(self._lo)

    def fk(self, q):
        return np.eye(4)

    def ik(self, target, seed):
        return None


def _self_motion(solver):
    return SelfMotion(solver, 1e-3, math.radians(0.5), 0.2, 3e-3, math.radians(3.0))


def test_turn_joints_are_the_joints_whose_range_exceeds_a_full_turn():
    # TM5S limits: J2 is +-pi rounded up (range just above 2 pi), which is not a full turn
    big, pi_up = 6.083185307179586, 3.1415927
    solver = _Limits([-big, -pi_up, -2.76, -big, -big, -big], [big, pi_up, 2.76, big, big, big])
    assert solver.turn_joints == (0, 3, 4, 5)
    assert _self_motion(solver).turn_joints == [0, 3, 4, 5]


def test_turn_joints_of_an_arm_without_full_turns():
    assert _Limits([-math.pi] * 5, [math.pi] * 5).turn_joints == ()


def test_family_uses_the_turn_joints():
    solver = _Limits([-7.0, -3.0, -7.0], [7.0, 3.0, 7.0])
    solver.branch = lambda q: (1,)
    sm = _self_motion(solver)
    q_ref = np.zeros(3)
    assert sm.family(np.array([2 * math.pi, 0.0, -2 * math.pi]), q_ref) == (1, 1, -1)


def test_self_motion_rejects_redundant_arms():
    with pytest.raises(NotImplementedError, match="redundant"):
        _self_motion(_Limits([-math.pi] * 7, [math.pi] * 7))
