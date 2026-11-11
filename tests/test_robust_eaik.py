"""RobustIK (numerical layer on the analytical EAIK solver)."""
import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("eaik")

from motion_planner.pinocchio_solver import PinocchioSolver
from motion_planner.pose import pose_distance


@pytest.fixture(scope="module")
def ik():
    return PinocchioSolver().analytical_ik


def _random_q(solver, rng):
    return rng.uniform(0.8 * solver.lower_limits, 0.8 * solver.upper_limits)


def test_every_solution_reaches_the_target_within_the_limits(ik):
    solver = PinocchioSolver()
    rng = np.random.default_rng(3)
    for _ in range(10):
        T = solver.fk(_random_q(solver, rng))
        res = ik.ik(T)
        assert len(res) > 0
        assert len(res.errors) == len(res.Q)
        for q, (dp, dr) in zip(res.Q, res.errors):
            q = np.asarray(q)
            assert np.all(q >= solver.lower_limits - 1e-6)
            assert np.all(q <= solver.upper_limits + 1e-6)
            lin, ang = pose_distance(solver.fk(q), T)
            assert lin < 1e-6 and ang < 1e-6
            assert dp < 1e-6 and dr < 1e-6


def test_the_seed_configuration_is_among_the_solutions(ik):
    solver = PinocchioSolver()
    rng = np.random.default_rng(4)
    for _ in range(10):
        q = _random_q(solver, rng)
        Q = np.asarray(ik.ik(solver.fk(q)).Q)
        assert np.min(np.max(np.abs(Q - q), axis=1)) < 1e-6


def test_batched_ik_equals_one_by_one(ik):
    solver = PinocchioSolver()
    rng = np.random.default_rng(5)
    Ts = [solver.fk(_random_q(solver, rng)) for _ in range(6)]
    batched = ik.ik_batch(Ts, num_threads=2)
    assert len(batched) == len(Ts)
    for T, res in zip(Ts, batched):
        single = ik.ik(T)
        a, b = np.asarray(res.Q), np.asarray(single.Q)
        assert a.shape == b.shape
        # same set of solutions (order may differ)
        for q in a:
            assert np.min(np.max(np.abs(b - q), axis=1)) < 1e-9


def test_fk_of_the_original_model_matches_pinocchio(ik):
    solver = PinocchioSolver()
    rng = np.random.default_rng(6)
    for _ in range(5):
        q = _random_q(solver, rng)
        np.testing.assert_allclose(ik.fk(q), solver.fk(q), atol=1e-9)
