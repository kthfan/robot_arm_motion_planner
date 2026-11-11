import numpy as np
import pytest

pytest.importorskip("pinocchio")

from motion_planner.pinocchio_solver import PinocchioSolver
from motion_planner.pose import pose_distance, pose_to_matrix
from motion_planner.solver import KinematicsSolver, closest_to_reference

Q_HOME = np.array([0.0, 0.3, 1.2, 0.2, 1.5, 0.1])


@pytest.fixture(scope="module")
def solver():
    return PinocchioSolver()


def _random_q(solver, rng):
    # Stay away from the limits so IK has room on every joint.
    return rng.uniform(0.8 * solver.lower_limits, 0.8 * solver.upper_limits)


def test_is_a_kinematics_solver_with_six_joints(solver):
    assert isinstance(solver, KinematicsSolver)
    assert solver.nq == 6
    assert solver.lower_limits.shape == (6,)
    assert np.all(solver.velocity_limits > 0)


def test_fk_returns_homogeneous_transform(solver):
    T = solver.fk(Q_HOME)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)


def test_ik_seeded_near_solution_recovers_that_solution(solver):
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = _random_q(solver, rng)
        q_ik = solver.ik(solver.fk(q), seed=q + rng.normal(0, 0.05, 6))
        assert q_ik is not None
        np.testing.assert_allclose(q_ik, q, atol=1e-4)


def test_ik_result_reaches_target_and_respects_limits(solver):
    rng = np.random.default_rng(2)
    target = solver.fk(_random_q(solver, rng))
    q_ik = solver.ik(target, seed=Q_HOME)
    assert q_ik is not None
    lin, ang = pose_distance(solver.fk(q_ik), target)
    assert lin < 1e-5 and ang < 1e-5
    assert np.all(q_ik >= solver.lower_limits) and np.all(q_ik <= solver.upper_limits)


def test_ik_returns_none_for_unreachable_pose(solver):
    far = pose_to_matrix([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert solver.ik(far, seed=Q_HOME) is None


def test_ik_all_finds_multiple_distinct_valid_solutions(solver):
    target = solver.fk(Q_HOME)
    sols = solver.ik_all(target, seeds=[Q_HOME])
    assert len(sols) >= 2
    for q in sols:
        lin, ang = pose_distance(solver.fk(q), target)
        assert lin < 1e-5 and ang < 1e-5
    for i in range(len(sols)):
        for j in range(i + 1, len(sols)):
            assert np.max(np.abs(sols[i] - sols[j])) > 1e-3


def test_closest_to_reference_picks_nearest_solution(solver):
    target = solver.fk(Q_HOME)
    sols = solver.ik_all(target, seeds=[Q_HOME])
    np.testing.assert_allclose(closest_to_reference(sols, Q_HOME), Q_HOME, atol=1e-4)


def test_tcp_offset_shifts_end_effector(solver):
    offset = pose_to_matrix([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    tool = PinocchioSolver(tcp_offset=offset)
    np.testing.assert_allclose(tool.fk(Q_HOME), solver.fk(Q_HOME) @ offset, atol=1e-12)
    q_ik = tool.ik(tool.fk(Q_HOME), seed=Q_HOME + 0.02)
    np.testing.assert_allclose(q_ik, Q_HOME, atol=1e-4)


# ---------------------------------------------------------------- ik_candidates (EAIK)

Q_USER = np.array([2.44735982, 1.13695154, 2.05527087, -3.22704178, -0.10781322, -0.0331056])
Q_WRIST_SINGULAR = np.array([2.44735982, 1.13695154, 2.05527087, -3.22704178, 0.0, -0.0331056])


class _NoNumericIK(PinocchioSolver):
    def ik_all(self, target, seeds=(), **kwargs):
        return []


class _MarkedNumericIK(PinocchioSolver):
    MARK = np.full(6, 0.123)

    def ik_all(self, target, seeds=(), **kwargs):
        return [self.MARK.copy()]


def _assert_exact_distinct_within_limits(s, sols, target):
    for q in sols:
        lin, ang = pose_distance(s.fk(q), target)
        assert lin < 1e-8 and ang < 1e-8
        assert np.all(q >= s.lower_limits) and np.all(q <= s.upper_limits)
    for i in range(len(sols)):
        for j in range(i + 1, len(sols)):
            assert np.max(np.abs(sols[i] - sols[j])) > 1e-3


def test_base_solver_ik_candidates_are_ik_all():
    class Fake(KinematicsSolver):
        lower_limits = upper_limits = velocity_limits = np.zeros(1)
        fk = ik = None

        def ik_all(self, target, seeds=(), **kwargs):
            return ["all", list(seeds)]

    assert Fake().ik_candidates("T", seeds=[1]) == ["all", [1]]


def test_base_solver_ik_analytical_is_ik_all():
    class Fake(KinematicsSolver):
        lower_limits = upper_limits = velocity_limits = np.zeros(1)
        fk = ik = None

        def ik_all(self, target, seeds=(), **kwargs):
            return ["all", target]

    assert Fake().ik_analytical("T") == ["all", "T"]


def test_ik_analytical_is_only_the_analytical_solutions():
    s = _MarkedNumericIK()
    target = s.fk(Q_HOME)
    sols = s.ik_analytical(target)
    assert not any(np.array_equal(q, s.MARK) for q in sols)
    assert any(np.max(np.abs(q - Q_HOME)) < 1e-6 for q in sols)
    _assert_exact_distinct_within_limits(s, sols, target)


def test_ik_candidates_come_from_the_analytical_solver(solver):
    s = _NoNumericIK()
    target = s.fk(Q_HOME)
    sols = s.ik_candidates(target, seeds=[Q_HOME])
    assert len(sols) >= len(solver.ik_all(target, seeds=[Q_HOME])) >= 2
    assert any(np.max(np.abs(q - Q_HOME)) < 1e-6 for q in sols)
    _assert_exact_distinct_within_limits(s, sols, target)


def test_ik_candidates_include_the_numerical_solutions():
    s = _MarkedNumericIK()
    sols = s.ik_candidates(s.fk(Q_HOME), seeds=[Q_HOME])
    assert sum(np.array_equal(q, s.MARK) for q in sols) == 1
    assert len(sols) >= 3


def test_ik_candidates_respect_the_tcp_offset():
    offset = pose_to_matrix([0.01, -0.02, 0.1, 0.1, 0.0, 0.3])
    s = PinocchioSolver(tcp_offset=offset)
    target = s.fk(Q_HOME)
    sols = s.ik_candidates(target)
    assert any(np.max(np.abs(q - Q_HOME)) < 1e-6 for q in sols)
    _assert_exact_distinct_within_limits(s, sols, target)


def _calibrated():
    from motion_planner.pinocchio_solver import DEFAULT_CALIBRATION

    if not DEFAULT_CALIBRATION.exists():
        pytest.skip("needs the robot calibration")
    return DEFAULT_CALIBRATION


@pytest.mark.parametrize("q", [Q_USER, Q_WRIST_SINGULAR], ids=["user", "wrist-singular"])
def test_ik_candidates_on_the_calibrated_model_find_more_than_ik_all(q):
    s = PinocchioSolver(calibration=_calibrated())
    target = s.fk(q)
    sols = s.ik_candidates(target, seeds=[q])
    assert any(np.max(np.abs(x - q)) < 1e-6 for x in sols)
    assert len(sols) > len(s.ik_all(target, seeds=[q]))
    _assert_exact_distinct_within_limits(s, sols, target)


def _fd_jacobian(s, q, h=1e-7):
    from scipy.spatial.transform import Rotation

    T = s.fk(q)
    J = np.empty((6, len(q)))
    for j in range(len(q)):
        x = np.array(q, dtype=float)
        x[j] += h
        Tx = s.fk(x)
        J[:3, j] = (Tx[:3, 3] - T[:3, 3]) / h
        J[3:, j] = Rotation.from_matrix(Tx[:3, :3] @ T[:3, :3].T).as_rotvec() / h
    return J


def test_jacobian_is_the_world_aligned_tcp_twist_jacobian():
    offset = pose_to_matrix([0.01, -0.02, 0.1, 0.1, 0.0, 0.3])
    s = PinocchioSolver(tcp_offset=offset)
    rng = np.random.default_rng(5)
    for _ in range(5):
        q = _random_q(s, rng)
        np.testing.assert_allclose(s.jacobian(q), _fd_jacobian(s, q), atol=1e-5)


def test_base_solver_jacobian_defaults_to_finite_differences(solver):
    class Wrapped(KinematicsSolver):
        lower_limits = solver.lower_limits
        upper_limits = solver.upper_limits
        velocity_limits = solver.velocity_limits
        fk = staticmethod(solver.fk)
        ik = staticmethod(solver.ik)

    np.testing.assert_allclose(Wrapped().jacobian(Q_HOME), solver.jacobian(Q_HOME), atol=1e-5)


def test_branch_tells_the_analytical_ik_solutions_apart():
    # Each (shoulder, elbow, wrist) branch holds one solution up to 2 pi turns of q1/q4/q5/q6.
    s = PinocchioSolver(calibration=_calibrated())
    rng = np.random.default_rng(6)
    for _ in range(20):
        q = _random_q(s, rng)
        groups = {}
        for x in s.analytical_ik.ik(s.fk(q)).Q:
            x = np.asarray(x)
            b = s.branch(x)
            assert set(b) <= {-1, 1}
            groups.setdefault(b, []).append(x)
        assert s.branch(q) in groups
        for members in groups.values():
            for x in members[1:]:
                d = x - members[0]
                np.testing.assert_allclose(d[[1, 2]], 0.0, atol=1e-6)
                turns = d[[0, 3, 4, 5]] / (2 * np.pi)
                np.testing.assert_allclose(turns, np.round(turns), atol=1e-6)


def test_branch_signs_follow_the_elbow_and_wrist_joints(solver):
    q = Q_HOME.copy()
    shoulder, elbow, wrist = solver.branch(q)
    assert (elbow, wrist) == (1, 1)
    q[2], q[4] = -q[2], -q[4]
    assert solver.branch(q)[1:] == (-1, -1)


def test_wrist_branch_is_the_same_for_q5_one_turn_apart(solver):
    # q5 and q5 -+ 2 pi are the same wrist posture (J5 range exceeds a full turn)
    q = Q_HOME.copy()
    for q5 in (0.5, 2.5, -0.5, -2.5):
        q[4] = q5
        turned = q.copy()
        turned[4] -= np.sign(q5) * 2 * np.pi
        assert solver.branch(turned) == solver.branch(q)
        np.testing.assert_allclose(solver.fk(turned), solver.fk(q), atol=1e-12)


def test_analytical_ik_raises_when_eaik_cannot_decompose_the_model():
    # calibrated axes 3/4 are ~3 mrad off parallel
    s = PinocchioSolver(calibration=_calibrated(), eaik_parallel_tol=1e-3)
    with pytest.raises(ValueError, match="EAIK"):
        s.analytical_ik
