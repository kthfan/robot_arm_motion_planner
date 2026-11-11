import math

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from motion_planner.pinocchio_solver import PinocchioSolver
from motion_planner.pose import pose_distance
from motion_planner.self_motion import SelfMotion

Q_NEAR_WRIST = np.array([0.3, -0.4, 1.2, 0.5, 0.005, -0.2])


@pytest.fixture(scope="module")
def solver():
    return PinocchioSolver()


@pytest.fixture(scope="module")
def sm(solver):
    return SelfMotion(
        solver,
        linear_tolerance=1e-3,
        angular_tolerance=math.radians(0.5),
        max_joint_step=0.2,
        max_deviation=3e-3,
        max_angle=math.radians(3.0),
    )


def test_sigma_min_vanishes_at_the_wrist_singularity(sm):
    regular = Q_NEAR_WRIST.copy()
    regular[4] = 1.0
    assert sm.sigma_min(Q_NEAR_WRIST) < 0.02 * sm.sigma_min(regular)


def test_self_motion_at_the_wrist_singularity_reaches_the_other_branches_holding_the_tcp(
    sm, solver
):
    target = solver.fk(Q_NEAR_WRIST)

    runs = sm.runs(Q_NEAR_WRIST, target)

    # UR-like wrist: the self-motion loops through both elbow and wrist branches
    assert {r.family[:3] for r in runs} >= {(1, -1, 1), (1, 1, -1), (1, -1, -1)}
    for run in runs:
        assert all(solver.branch(q) == run.family[:3] for q in run.points)
        points = [Q_NEAR_WRIST] + run.prefix + run.points
        for a, b in zip(points, points[1:]):
            assert np.max(np.abs(b - a)) <= 0.2 + 1e-12
            for t in np.linspace(0, 1, 9):
                lin, ang = pose_distance(solver.fk(a + t * (b - a)), target)
                assert lin <= 3e-3 and ang <= math.radians(3.0)
        assert 0.0 <= max(run.deviation) <= 3e-3


def test_no_self_motion_away_from_singularities(sm, solver):
    q = Q_NEAR_WRIST.copy()
    q[4] = 0.8
    assert sm.runs(q, solver.fk(q)) == []
