import math
import re
import warnings

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from motion_planner.path import CartesianPath
from motion_planner.pinocchio_solver import PinocchioSolver
from motion_planner.planner import (
    KinematicMotionPlanner,
    MotionPlanner,
    PathDiscontinuityError,
)
from motion_planner.pose import matrix_to_pose, pose_distance, pose_to_matrix
from motion_planner.solver import IKError

Q_HOME = np.array([0.0, 0.3, 1.2, 0.2, 1.5, 0.1])


@pytest.fixture(scope="module")
def solver():
    return PinocchioSolver()


@pytest.fixture(scope="module")
def planner(solver):
    return KinematicMotionPlanner(solver)


@pytest.fixture(scope="module")
def robot_planner():
    """The robot's (calibrated) planner, which the line-motion experiment cases use."""
    from motion_planner import default_planner
    from motion_planner.pinocchio_solver import DEFAULT_CALIBRATION

    if not DEFAULT_CALIBRATION.exists():
        pytest.skip("needs the robot calibration")
    return default_planner("pinocchio")


def _shifted_goal(solver, q, dxyz=(0.0, 0.0, 0.0), drz=0.0):
    T = solver.fk(q)
    pose = matrix_to_pose(T)
    pose = [pose[0] + dxyz[0], pose[1] + dxyz[1], pose[2] + dxyz[2], pose[3], pose[4], pose[5]]
    G = pose_to_matrix(pose)
    if drz:
        c, s = np.cos(drz), np.sin(drz)
        G[:3, :3] = G[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return G


def _segment_durations(solver, q0, actions):
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    out = []
    for qa, qb, a in zip(qs[:-1], qs[1:], actions):
        out.append(np.max(np.abs(qb - qa) / (a.velocity / 100.0 * solver.velocity_limits)))
    return np.array(out)


# ---------------------------------------------------------------- PTP_T


def test_planner_is_motion_planner(planner):
    assert isinstance(planner, MotionPlanner)


def test_planner_fk_matches_solver(planner, solver):
    np.testing.assert_allclose(planner.fk(Q_HOME), solver.fk(Q_HOME))


def test_ptp_t_returns_single_ptp_j_action_reaching_goal(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (0.05, -0.05, 0.02), drz=0.3)
    actions = planner.plan_ptp_t(
        Q_HOME, goal, velocity=30, acc_time=0.2, blend_percentage=10,
        fine_goal=True, blocked=False,
    )
    assert len(actions) == 1
    a = actions[0]
    assert (a.velocity, a.acc_time, a.blend_percentage, a.fine_goal, a.blocked) == (
        30, 0.2, 10, True, False,
    )
    lin, ang = pose_distance(solver.fk(a.positions), goal)
    assert lin < 1e-5 and ang < 1e-5


def test_ptp_t_picks_solution_closest_to_q0(planner, solver):
    q0 = Q_HOME + 0.05
    actions = planner.plan_ptp_t(q0, solver.fk(Q_HOME), velocity=50, acc_time=0.1)
    np.testing.assert_allclose(actions[0].positions, Q_HOME, atol=1e-4)


def test_ptp_t_uses_custom_selector(solver):
    seen = {}

    def pick_last(solutions, q_ref):
        seen["n"] = len(solutions)
        return solutions[-1]

    planner = KinematicMotionPlanner(solver, selector=pick_last)
    actions = planner.plan_ptp_t(Q_HOME, solver.fk(Q_HOME), velocity=50, acc_time=0.1)
    assert seen["n"] >= 2
    lin, ang = pose_distance(solver.fk(actions[0].positions), solver.fk(Q_HOME))
    assert lin < 1e-5 and ang < 1e-5


def test_ptp_t_unreachable_goal_raises_ik_error(planner):
    with pytest.raises(IKError):
        planner.plan_ptp_t(Q_HOME, pose_to_matrix([5, 0, 0, 0, 0, 0]), velocity=50, acc_time=0.1)


# ---------------------------------------------------------------- LINE_T


def test_line_t_tcp_stays_on_straight_line(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.1, -0.05))
    actions = planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2)
    assert len(actions) > 1

    path = CartesianPath.from_poses(solver.fk(Q_HOME), goal)
    p0, p1 = path.pose_at(0)[:3, 3], path.pose_at(1)[:3, 3]
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    qs = [Q_HOME] + [np.asarray(a.positions) for a in actions]
    # check waypoints and joint-space midpoints (what the robot interpolates)
    probes = qs + [(qa + qb) / 2 for qa, qb in zip(qs[:-1], qs[1:])]
    for q in probes:
        T = solver.fk(q)
        r = T[:3, 3] - p0
        assert np.linalg.norm(r - np.dot(r, d) * d) <= planner.linear_tolerance + 1e-9
        _, ang = pose_distance(T, path.pose_at(0))
        assert ang <= planner.angular_tolerance + 1e-9

    lin, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert lin < 1e-5 and ang < 1e-5


def test_line_t_rotation_only_keeps_tcp_position(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, drz=0.8)
    actions = planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2)
    assert len(actions) > 1
    start = solver.fk(Q_HOME)
    for a in actions:
        lin, _ = pose_distance(solver.fk(a.positions), start)
        assert lin <= planner.linear_tolerance + 1e-9
    _, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert ang < 1e-5


def test_line_t_waypoint_flags(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.0, 0.0))
    actions = planner.plan_line_t(
        Q_HOME, goal, velocity=10, acc_time=0.2, blend_percentage=20,
        fine_goal=True, blocked=True,
    )
    for a in actions[:-1]:
        assert (a.blend_percentage, a.fine_goal, a.blocked) == (100, False, False)
    last = actions[-1]
    assert (last.blend_percentage, last.fine_goal, last.blocked) == (20, True, True)


def test_line_t_timing_follows_trapezoidal_tcp_speed(planner, solver):
    length, pct, ta = 0.15, 10, 0.2
    v = pct / 100 * planner.max_linear_velocity
    assert planner.max_linear_velocity == 1.0
    goal = _shifted_goal(solver, Q_HOME, (length, 0.0, 0.0))
    actions = planner.plan_line_t(Q_HOME, goal, velocity=pct, acc_time=ta)
    assert all(0 < a.velocity <= 100 for a in actions)

    dts = _segment_durations(solver, Q_HOME, actions)
    assert dts.sum() == pytest.approx(length / v + ta, rel=1e-3)

    # cruise phase: TCP speed equals the requested velocity
    qs = [Q_HOME] + [np.asarray(a.positions) for a in actions]
    t = np.concatenate([[0.0], np.cumsum(dts)])
    i = int(np.searchsorted(t, t[-1] / 2))
    seg = np.linalg.norm(solver.fk(qs[i])[:3, 3] - solver.fk(qs[i - 1])[:3, 3])
    assert seg / dts[i - 1] == pytest.approx(v, rel=1e-2)
    assert all(a.acc_time <= ta + 1e-12 for a in actions)


def test_line_t_same_start_and_goal_returns_no_actions(planner, solver):
    assert planner.plan_line_t(Q_HOME, solver.fk(Q_HOME), velocity=10, acc_time=0.2) == []


def test_line_t_through_unreachable_region_raises_ik_error(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (3.0, 0.0, 0.0))
    with pytest.raises(IKError):
        planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2)


def test_line_t_velocity_is_percentage_of_max_linear_velocity(solver):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.1, -0.05))
    a = KinematicMotionPlanner(solver).plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2)
    b = KinematicMotionPlanner(solver, max_linear_velocity=0.5).plan_line_t(
        Q_HOME, goal, velocity=20, acc_time=0.2
    )
    assert a == b


@pytest.mark.parametrize("bad", [0.0, -5.0, 100.5])
def test_line_t_velocity_must_be_a_speed_percentage(planner, solver, bad):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="velocity"):
        planner.plan_line_t(Q_HOME, goal, velocity=bad, acc_time=0.2)


def test_max_linear_velocity_must_be_positive(solver):
    with pytest.raises(ValueError, match="max_linear_velocity"):
        KinematicMotionPlanner(solver, max_linear_velocity=0.0)


def test_line_t_acc_time_never_exceeds_segment_duration(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.0, 0.0))
    for pct in (0.1, 10, 100):
        actions = planner.plan_line_t(Q_HOME, goal, velocity=pct, acc_time=0.2)
        dts = _segment_durations(solver, Q_HOME, actions)
        assert all(a.acc_time <= dt + 1e-9 for a, dt in zip(actions, dts))


# ------------------------------------------------- LINE_T: singularities, speed

Q_WRIST_SINGULAR = np.array([0.0, 0.3, 1.2, 0.2, 0.0, 0.1])  # q5 = 0


def _wrist_line(solver, start_dy, goal_dy):
    """Constant-orientation line along y, which crosses the q5 = 0 surface at dy = 0."""
    Ts = solver.fk(Q_WRIST_SINGULAR)
    start, goal = Ts.copy(), Ts.copy()
    start[1, 3] += start_dy
    goal[1, 3] += goal_dy
    q0 = solver.ik(start, Q_WRIST_SINGULAR + np.array([0, 0, 0, 0, 0.3, 0]))
    assert q0 is not None and q0[4] > 0.05
    return q0, goal


def _joint_profile(solver, q0, actions):
    """Constant joint velocity per segment, starting and ending at rest."""
    qs = np.array([q0] + [a.positions for a in actions])
    dts = _segment_durations(solver, q0, actions)
    vel = np.diff(qs, axis=0) / dts[:, None]
    padded = np.vstack([np.zeros(6), vel, np.zeros(6)])
    spans = np.concatenate([[dts[0]], 0.5 * (dts[:-1] + dts[1:]), [dts[-1]]])
    acc = np.abs(np.diff(padded, axis=0)) / spans[:, None]
    return qs, dts, vel, acc


def _max_line_deviation(solver, qs, p0, p1, samples=41):
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    worst = 0.0
    for qa, qb in zip(qs[:-1], qs[1:]):
        for t in np.linspace(0.0, 1.0, samples):
            r = solver.fk((1 - t) * qa + t * qb)[:3, 3] - p0
            worst = max(worst, np.linalg.norm(r - np.dot(r, d) * d))
    return worst


def test_line_t_crossing_singularity_raises_path_discontinuity(planner, solver):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    assert solver.ik_all(goal, seeds=[q0])  # goal itself is reachable
    with pytest.raises(PathDiscontinuityError, match="singular"):
        planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2)


def test_line_t_overspeed_slows_whole_motion_uniformly(solver):
    planner = KinematicMotionPlanner(solver, max_linear_velocity=3.0)
    goal = _shifted_goal(solver, Q_HOME, (-0.1, 0.1, -0.1))
    with pytest.warns(UserWarning, match="slowed"):
        actions = planner.plan_line_t(Q_HOME, goal, velocity=100, acc_time=0.0)
    assert all(a.velocity <= 100.0 for a in actions)
    qs = [Q_HOME] + [np.asarray(a.positions) for a in actions]
    dts = _segment_durations(solver, Q_HOME, actions)
    speeds = [
        np.linalg.norm(solver.fk(qb)[:3, 3] - solver.fk(qa)[:3, 3]) / dt
        for qa, qb, dt in zip(qs[:-1], qs[1:], dts)
    ]
    # constant-speed request stays constant speed, just slower
    assert max(speeds) == pytest.approx(min(speeds), rel=2e-2)
    assert max(speeds) < 3.0
    _, _, _, acc = _joint_profile(solver, Q_HOME, actions)
    assert np.all(acc <= planner.max_joint_acceleration * (1 + 1e-6))


def test_line_t_near_singularity_keeps_joint_motion_within_limits(planner, solver):
    q0, goal = _wrist_line(solver, -0.1, -0.0005)  # stop 0.5 mm before q5 = 0
    with pytest.warns(UserWarning, match="slowed"):
        actions = planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2)
    qs, dts, vel, acc = _joint_profile(solver, q0, actions)
    assert np.all(np.abs(vel) <= solver.velocity_limits * (1 + 1e-6))
    assert np.all(acc <= planner.max_joint_acceleration * (1 + 1e-6))
    dev = _max_line_deviation(solver, qs, solver.fk(q0)[:3, 3], goal[:3, 3])
    assert dev <= planner.linear_tolerance + 1e-9
    lin, ang = pose_distance(solver.fk(qs[-1]), goal)
    assert lin < 1e-5 and ang < 1e-5


def test_line_t_nominal_motion_is_not_slowed(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (-0.1, 0.1, -0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actions = planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2)
    qs, _, _, acc = _joint_profile(solver, Q_HOME, actions)
    assert np.all(acc <= planner.max_joint_acceleration)
    dev = _max_line_deviation(solver, qs, solver.fk(Q_HOME)[:3, 3], goal[:3, 3])
    assert dev <= planner.linear_tolerance + 1e-9


# ---------------------------------------------------------------- unsafe LINE_T


def _on_line(solver, q, p0, p1, tol):
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    r = solver.fk(q)[:3, 3] - p0
    return np.linalg.norm(r - np.dot(r, d) * d) <= tol


def _unsafe_warnings(record):
    return [str(w.message) for w in record if "left the straight line" in str(w.message)]


@pytest.fixture(scope="module")
def direct_planner(solver):
    """Unsafe joint jumps are never routed: the TCP leaves the line."""
    return KinematicMotionPlanner(solver, max_flip_deviation=0.0)


def test_unsafe_line_t_holds_tcp_on_the_line_through_the_wrist_flip(planner, solver):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    assert planner.max_flip_deviation == pytest.approx(3e-3)
    with pytest.warns(UserWarning, match="TCP held") as record:
        actions = planner.plan_line_t(
            q0, goal, velocity=10, acc_time=0.2, on_discontinuity="jump"
        )
    assert _unsafe_warnings(record) == []
    assert all(a.velocity <= 100.0 for a in actions)
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    steps = [np.max(np.abs(b - a)) for a, b in zip(qs, qs[1:])]
    assert max(steps) <= planner.max_joint_step + 1e-9  # no joint jump left
    p0, p1 = solver.fk(q0)[:3, 3], goal[:3, 3]
    assert _max_line_deviation(solver, qs, p0, p1) <= planner.max_flip_deviation
    lin, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert lin < 1e-5 and ang < 1e-5


def test_max_flip_deviation_must_not_be_negative(solver):
    with pytest.raises(ValueError, match="max_flip_deviation"):
        KinematicMotionPlanner(solver, max_flip_deviation=-1e-3)


def test_max_flip_angle_defaults_to_3_deg_and_must_not_be_negative(solver):
    assert KinematicMotionPlanner(solver).max_flip_angle == pytest.approx(math.radians(3.0))
    with pytest.raises(ValueError, match="max_flip_angle"):
        KinematicMotionPlanner(solver, max_flip_angle=-0.01)


def test_unsafe_line_t_flip_beyond_max_flip_deviation_falls_back_to_direct_jump(solver):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    tight = KinematicMotionPlanner(solver, max_flip_deviation=1e-4, max_flip_angle=1e-3)
    with pytest.warns(UserWarning, match="joint jump") as record:
        tight.plan_line_t(q0, goal, velocity=10, acc_time=0.2, on_discontinuity="jump")
    assert not any("TCP held" in str(w.message) for w in record)


def test_unsafe_line_t_held_flip_does_not_slow_the_line(planner, solver):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    pct, ta = 10, 0.2
    with pytest.warns(UserWarning, match="TCP held"):
        actions = planner.plan_line_t(
            q0, goal, velocity=pct, acc_time=ta, on_discontinuity="jump"
        )
    assert all(a.velocity <= 100.0 and a.acc_time <= ta + 1e-12 for a in actions)
    nominal = 0.2 / (pct / 100 * planner.max_linear_velocity) + ta
    assert _segment_durations(solver, q0, actions).sum() < 1.5 * nominal


def test_unsafe_line_t_falls_back_to_a_direct_jump(direct_planner, solver):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    with pytest.warns(UserWarning) as record:
        actions = direct_planner.plan_line_t(
            q0, goal, velocity=10, acc_time=0.2, on_discontinuity="jump"
        )
    msgs = _unsafe_warnings(record)
    assert len(msgs) == 1 and "joint jump" in msgs[0]
    assert all(a.velocity <= 100.0 for a in actions)
    # every waypoint is still an IK solution of a point on the line
    p0, p1 = solver.fk(q0)[:3, 3], goal[:3, 3]
    assert all(_on_line(solver, a.positions, p0, p1, 1e-6) for a in actions)
    lin, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert lin < 1e-5 and ang < 1e-5


def test_unsafe_line_t_jumps_to_the_closest_ik_solution_that_reaches_the_goal(direct_planner, solver):
    planner = direct_planner
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    with pytest.warns(UserWarning):
        actions = planner.plan_line_t(
            q0, goal, velocity=10, acc_time=0.2, on_discontinuity="jump"
        )
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    steps = [np.max(np.abs(b - a)) for a, b in zip(qs, qs[1:])]
    i = int(np.argmax(steps))
    q_prev = qs[i]
    seeds = [q_prev, *solver.ik_all(goal, seeds=[q0])]
    candidates = solver.ik_all(solver.fk(qs[i + 1]), seeds=seeds)
    assert len(candidates) > 1

    def continues(c):  # the rest of the line runs without another jump
        try:
            planner.plan_line_t(c, goal, velocity=10, acc_time=0.2)
            return True
        except PathDiscontinuityError:
            return False

    closer = [c for c in candidates if np.max(np.abs(c - q_prev)) < steps[i] - 1e-6]
    assert not any(continues(c) for c in closer)


def test_unsafe_line_t_jump_runs_at_line_speed_without_slowing_the_line(direct_planner, solver):
    planner = direct_planner
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    pct, ta = 10, 0.2
    with pytest.warns(UserWarning):
        actions = planner.plan_line_t(
            q0, goal, velocity=pct, acc_time=ta, on_discontinuity="jump"
        )
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    steps = [np.max(np.abs(b - a)) for a, b in zip(qs, qs[1:])]
    i = int(np.argmax(steps))
    assert np.degrees(steps[i]) > 90  # the wrist flip
    jump = actions[i]
    # the jump is a plain PTP_J at the requested speed %
    assert jump.velocity == pytest.approx(pct)
    assert jump.acc_time == pytest.approx(ta)
    # the rest of the line is not crawling because of the jump
    dts = _segment_durations(solver, q0, actions)
    line_time = dts.sum() - dts[i]
    nominal = 0.2 / (pct / 100 * planner.max_linear_velocity) + ta
    assert line_time < 5 * nominal


def _same_waypoints(a, b, rel=1e-9):
    """Same waypoints and flags; speed % / acc_time equal up to rounding (hold / jump
    time the line with the smooth profile, which equals the trapezoid of a clean
    line only up to floating-point rounding)."""
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.positions, y.positions)
        assert x.velocity == pytest.approx(y.velocity, rel=rel)
        assert x.acc_time == pytest.approx(y.acc_time, rel=rel)
        assert (x.blend_percentage, x.fine_goal, x.blocked) == (
            y.blend_percentage, y.fine_goal, y.blocked)


@pytest.mark.parametrize("mode", ["hold", "jump"])
def test_unsafe_line_t_without_problems_equals_safe_line_t(planner, solver, mode):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.1, -0.05))
    safe = planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unsafe = planner.plan_line_t(
            Q_HOME, goal, velocity=10, acc_time=0.2, on_discontinuity=mode
        )
    _same_waypoints(unsafe, safe)


@pytest.mark.parametrize("mode", ["hold", "jump"])
def test_unsafe_line_t_unreachable_goal_still_raises_ik_error(planner, solver, mode):
    goal = _shifted_goal(solver, Q_HOME, (3.0, 0.0, 0.0))
    with pytest.raises(IKError):
        planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2, on_discontinuity=mode)


def test_line_t_rejects_unknown_discontinuity_mode(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="on_discontinuity"):
        planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2, on_discontinuity="skip")


def _base_axis_line(solver):
    """Horizontal line through the joint-1 axis: no IK where the wrist is too close
    to it (s ~0.36 to 0.69)."""
    start, goal = solver.fk(Q_HOME), solver.fk(Q_HOME)
    start[:3, 3] = [0.4, 0.05, 0.4]
    goal[:3, 3] = [-0.4, 0.05, 0.4]
    q0 = min(solver.ik_analytical(start), key=lambda q: float(np.max(np.abs(q - Q_HOME))))
    return q0, goal


def test_unsafe_line_t_skips_points_without_ik(planner, solver):
    q0, goal = _base_axis_line(solver)
    with pytest.raises(IKError):
        planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2)
    with pytest.raises(IKError):
        planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2, on_discontinuity="hold")
    with pytest.warns(UserWarning) as record:
        actions = planner.plan_line_t(
            q0, goal, velocity=10, acc_time=0.2, on_discontinuity="jump"
        )
    msgs = _unsafe_warnings(record)
    assert len(msgs) == 1 and "without IK" in msgs[0]
    xs = [solver.fk(a.positions)[0, 3] for a in actions]
    assert not any(-0.1 < x < 0.1 for x in xs)
    lin, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert lin < 1e-5 and ang < 1e-5


# ---------------------------------------------------------------- held reconfiguration


def test_hold_line_t_reconfigures_with_the_tcp_on_the_line(planner, solver):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    with pytest.warns(UserWarning, match="TCP held") as record:
        actions = planner.plan_line_t(
            q0, goal, velocity=10, acc_time=0.2, on_discontinuity="hold"
        )
    assert _unsafe_warnings(record) == []
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    assert max(np.max(np.abs(b - a)) for a, b in zip(qs, qs[1:])) <= planner.max_joint_step + 1e-9
    p0, p1 = solver.fk(q0)[:3, 3], goal[:3, 3]
    assert _max_line_deviation(solver, qs, p0, p1) <= planner.max_flip_deviation
    lin, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert lin < 1e-5 and ang < 1e-5


def test_hold_line_t_leaves_the_line_for_at_most_max_held_length(solver):
    planner = KinematicMotionPlanner(solver, max_held_length=0.05)
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    with pytest.warns(UserWarning, match="TCP held") as record:
        actions = planner.plan_line_t(
            q0, goal, velocity=10, acc_time=0.2, on_discontinuity="hold"
        )
    held = [tuple(float(x) for x in re.findall(r"s in \[([\d.]+), ([\d.]+)\]", str(w.message))[0])
            for w in record if "TCP held" in str(w.message)]
    assert len(held) == 1
    s_from, s_to = held[0]
    assert (s_to - s_from) * 0.2 <= 0.05 + 1e-9
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    p0, p1 = solver.fk(q0)[:3, 3], goal[:3, 3]
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    for qa, qb in zip(qs, qs[1:]):
        for t in np.linspace(0.0, 1.0, 21):
            d = solver.fk((1 - t) * qa + t * qb)[:3, 3] - p0
            s = np.dot(d, u) / 0.2
            if not s_from - 1e-3 <= s <= s_to + 1e-3:
                assert np.linalg.norm(d - np.dot(d, u) * u) <= planner.linear_tolerance + 1e-9


def test_hold_line_t_cut_detours_stay_within_max_flip_deviation(solver):
    # Long joint steps allow chords far off the line; only close ones may cut a detour.
    planner = KinematicMotionPlanner(solver, max_joint_step=0.6)
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    with pytest.warns(UserWarning, match="TCP held"):
        actions = planner.plan_line_t(
            q0, goal, velocity=10, acc_time=0.2, on_discontinuity="hold"
        )
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    assert max(np.max(np.abs(b - a)) for a, b in zip(qs, qs[1:])) <= planner.max_joint_step + 1e-9
    p0, p1 = solver.fk(q0)[:3, 3], goal[:3, 3]
    assert _max_line_deviation(solver, qs, p0, p1) <= planner.max_flip_deviation


@pytest.mark.parametrize("limit", [1e-4, 0.0])
def test_hold_line_t_raises_when_the_tcp_cannot_be_held(solver, limit):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    tight = KinematicMotionPlanner(solver, max_flip_deviation=limit, max_flip_angle=10 * limit)
    with pytest.raises(PathDiscontinuityError, match="TCP"):
        tight.plan_line_t(q0, goal, velocity=10, acc_time=0.2, on_discontinuity="hold")


# Experiment cases (long25, random112): the start configuration is the most
# singular point of its branch, which never reaches the goal; the only held flip
# is right at the start.
_SINGULAR_START_LINES = {
    "long25": (
        [2.305104033994315, 1.659214225690647, -0.2563279826809608,
         0.2810343360586116, 1.0748915642772774, -0.9976130072193667],
        [[-0.7028930303875554, -0.65657108249125, -0.2735978827930997, 0.34886928282210355],
         [-0.7101868175337376, 0.6263352012621201, 0.32146368358065125, 0.2432734088248941],
         [-0.03969977372606681, 0.42026019237635026, -0.9065347752126732, -0.7166870031854617],
         [0.0, 0.0, 0.0, 1.0]],
    ),
    "random112": (
        [0.6255065749875941, -2.6532698063684976, 0.2609755697879468,
         -3.28876755585088, -1.4561516814810416, -0.3628395941137854],
        [[0.7803395687533382, 0.6114590321179717, -0.13110304908432496, -0.07175859282596737],
         [0.13281635418558665, -0.36691217673302456, -0.9207254045728739, -0.38767518892148206],
         [-0.6110891698424239, 0.7010658361426455, -0.36752784914195386, -0.5460133139561196],
         [0.0, 0.0, 0.0, 1.0]],
    ),
}


@pytest.mark.parametrize("name", sorted(_SINGULAR_START_LINES))
def test_hold_line_t_reconfigures_at_a_singular_start(robot_planner, name):
    q0, goal = (np.asarray(x, dtype=float) for x in _SINGULAR_START_LINES[name])
    _assert_held_line(robot_planner, q0, goal, (0.0, 1e-3))


# Experiment case wrist0: q5 stays within ~2 deg of 0 over s ~0.17-0.24; at the
# sigma_min minimum (s ~0.197) no self-motion run attaches to another branch on
# the line, but a few mm either side they do.
_FLAT_WRIST_LINE = (
    [-0.5909527510065298, -2.5522043479214163, -1.312221482728291,
     -1.8910751478471486, -0.25731491653680183, -0.42722212711701685],
    [[0.584087706335018, 0.4123552432580479, -0.6991456963079252, -0.14391604981842093],
     [-0.20842469007722048, -0.7562709850257978, -0.62017202917765, -0.4157899339904151],
     [-0.784474792376729, 0.5079540831273093, -0.35578357123366766, -0.4197416961649044],
     [0.0, 0.0, 0.0, 1.0]],
)


def test_hold_line_t_tries_next_to_a_singularity_where_no_flip_attaches_at_it(robot_planner):
    q0, goal = (np.asarray(x, dtype=float) for x in _FLAT_WRIST_LINE)
    _assert_held_line(robot_planner, q0, goal, (0.15, 0.25))


def _assert_held_line(planner, q0, goal, flip_range):
    solver = planner.solver
    with pytest.warns(UserWarning, match="TCP held") as record:
        actions = planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2, on_discontinuity="hold")
    assert _unsafe_warnings(record) == []
    held = _held_ranges(record)
    assert len(held) == 1 and flip_range[0] <= held[0][0] <= flip_range[1]
    qs = [q0] + [np.asarray(a.positions) for a in actions]
    steps = [np.max(np.abs(b - a)) for a, b in zip(qs, qs[1:])]
    assert 0.0 < min(steps) and max(steps) <= planner.max_joint_step + 1e-9
    assert _max_line_deviation(solver, qs, solver.fk(q0)[:3, 3], goal[:3, 3]) \
        <= planner.max_flip_deviation
    lin, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert lin < 1e-5 and ang < 1e-5


class _DecoySolver(PinocchioSolver):
    """``ik_all`` also returns a wrong "solution" right next to the first seed.

    It is the closest candidate to the current joints but no route holding the
    TCP reaches it, so the planner has to fall back to the next candidate.
    """

    OFFSET = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

    def ik_all(self, target, seeds=(), **kwargs):
        seeds = list(seeds)
        found = super().ik_all(target, seeds=seeds, **kwargs)
        return found + [np.asarray(seeds[0], dtype=float) + self.OFFSET] if seeds else found


@pytest.mark.parametrize("mode", ["hold", "jump"])
def test_unreachable_closest_solution_falls_back_to_the_next_one(solver, mode):
    decoy = _DecoySolver()
    planner = KinematicMotionPlanner(decoy)
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    with pytest.warns(UserWarning, match="TCP held") as record:
        actions = planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2, on_discontinuity=mode)
    assert _unsafe_warnings(record) == []
    qs = [np.asarray(q0)] + [np.asarray(a.positions) for a in actions]
    assert max(np.max(np.abs(b - a)) for a, b in zip(qs, qs[1:])) <= planner.max_joint_step + 1e-9
    p0, p1 = solver.fk(q0)[:3, 3], goal[:3, 3]
    assert _max_line_deviation(solver, qs, p0, p1) <= planner.max_flip_deviation
    lin, ang = pose_distance(solver.fk(actions[-1].positions), goal)
    assert lin < 1e-5 and ang < 1e-5


def test_max_held_length_must_not_be_negative(solver):
    with pytest.raises(ValueError, match="max_held_length"):
        KinematicMotionPlanner(solver, max_held_length=-0.01)


def _held_ranges(record):
    return [tuple(float(x) for x in m)
            for w in record
            for m in re.findall(r"s in \[([\d.]+), ([\d.]+)\]", str(w.message))]


@pytest.mark.parametrize("mode", ["hold", "jump"])
def test_line_t_flip_is_deterministic_and_stable_under_small_perturbations(planner, solver, mode):
    q0, goal = _wrist_line(solver, -0.1, 0.1)
    with pytest.warns(UserWarning, match="TCP held") as record:
        ref = planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2, on_discontinuity=mode)
    ref_q = np.array([a.positions for a in ref])
    ref_held = _held_ranges(record)
    with pytest.warns(UserWarning, match="TCP held"):
        again = planner.plan_line_t(q0, goal, velocity=10, acc_time=0.2, on_discontinuity=mode)
    assert np.array_equal(np.array([a.positions for a in again]), ref_q)

    rng = np.random.default_rng(0)
    for _ in range(4):
        start = solver.fk(q0)
        start[:3, 3] += rng.uniform(-2e-4, 2e-4, 3)
        q0_p = solver.ik(start, q0)
        goal_p = goal.copy()
        goal_p[:3, 3] += rng.uniform(-2e-4, 2e-4, 3)
        with pytest.warns(UserWarning, match="TCP held") as record:
            out = planner.plan_line_t(q0_p, goal_p, velocity=10, acc_time=0.2, on_discontinuity=mode)
        held = _held_ranges(record)
        assert len(held) == len(ref_held) == 1
        assert abs(held[0][0] - ref_held[0][0]) < 0.05 and abs(held[0][1] - ref_held[0][1]) < 0.05
        # same flip, same direction: the same goal configuration
        assert np.max(np.abs(np.asarray(out[-1].positions) - ref_q[-1])) < 0.01


def _goal_and_solutions(solver):
    goal = _shifted_goal(solver, Q_HOME, (-0.05, 0.05, -0.05))
    return goal, solver.ik_analytical(goal)


def test_line_t_goal_joints_on_the_start_branch_keep_the_continuous_line(planner, solver):
    goal, _ = _goal_and_solutions(solver)
    plain = planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2)
    end = np.asarray(plain[-1].positions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actions = planner.plan_line_t(
            Q_HOME, goal, velocity=10, acc_time=0.2, goal_joints=end + 0.2
        )
    np.testing.assert_allclose(actions[-1].positions, end, atol=1e-9)


def test_line_t_goal_joints_on_another_branch_select_that_goal(planner, solver):
    goal, sols = _goal_and_solutions(solver)
    other = next(q for q in sols if solver.branch(q) != solver.branch(Q_HOME))
    with pytest.raises(PathDiscontinuityError, match="another configuration than requested"):
        planner.plan_line_t(Q_HOME, goal, velocity=10, acc_time=0.2, goal_joints=other)
    with pytest.warns(UserWarning, match="left the straight line"):
        actions = planner.plan_line_t(
            Q_HOME, goal, velocity=10, acc_time=0.2, on_discontinuity="jump", goal_joints=other
        )
    np.testing.assert_allclose(actions[-1].positions, other, atol=1e-6)
    ptp = planner.plan_ptp_t(Q_HOME, goal, velocity=10, acc_time=0.2, goal_joints=other)
    np.testing.assert_allclose(ptp[-1].positions, other, atol=1e-6)


def test_goal_joints_without_a_matching_goal_configuration_raise(planner, solver):
    goal, sols = _goal_and_solutions(solver)
    same = next(q for q in sols if solver.branch(q) == solver.branch(Q_HOME))
    # the same configuration one q6 turn further, beyond the q6 limit
    turned = same + np.array([0, 0, 0, 0, 0, -2 * np.pi if same[5] < 0 else 2 * np.pi])
    assert not solver.lower_limits[5] <= turned[5] <= solver.upper_limits[5]
    for plan in (planner.plan_line_t, planner.plan_ptp_t):
        with pytest.raises(IKError, match="goal_joints"):
            plan(Q_HOME, goal, velocity=10, acc_time=0.2, goal_joints=turned)
        with pytest.raises(ValueError, match="goal_joints"):
            plan(Q_HOME, goal, velocity=10, acc_time=0.2, goal_joints=same[:5])


# ---------------------------------------------------------------- min PTP_J speed


def test_line_t_speed_defaults_to_at_least_one_percent(planner, solver):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.0, 0.0))
    actions = planner.plan_line_t(Q_HOME, goal, velocity=0.1, acc_time=0.2)
    assert planner.min_velocity == 1.0
    assert all(1.0 <= a.velocity <= 100.0 for a in actions)
    assert any(a.velocity == 1.0 for a in actions)


def test_min_velocity_is_configurable(solver):
    planner = KinematicMotionPlanner(solver, min_velocity=5.0)
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.0, 0.0))
    actions = planner.plan_line_t(Q_HOME, goal, velocity=0.1, acc_time=0.2)
    assert all(a.velocity >= 5.0 for a in actions)
    assert any(a.velocity == 5.0 for a in actions)


def test_min_velocity_does_not_slow_fast_segments(solver):
    goal = _shifted_goal(solver, Q_HOME, (0.1, 0.1, -0.05))
    fast = KinematicMotionPlanner(solver, min_velocity=1e-9).plan_line_t(
        Q_HOME, goal, velocity=10, acc_time=0.2
    )
    assert all(a.velocity > 1.0 for a in fast)
    assert KinematicMotionPlanner(solver).plan_line_t(
        Q_HOME, goal, velocity=10, acc_time=0.2
    ) == fast


@pytest.mark.parametrize("bad", [0.0, -1.0, 100.5])
def test_min_velocity_must_be_a_speed_percentage(solver, bad):
    with pytest.raises(ValueError, match="min_velocity"):
        KinematicMotionPlanner(solver, min_velocity=bad)
