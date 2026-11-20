"""Offline tests for reach_volume_planner (fake planner; no pinocchio needed)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "reach_volume"))

import reach_volume_planner as RP  # noqa: E402
from motion_planner import IKError, PathDiscontinuityError  # noqa: E402


def phase_poses():
    """7 fake goal matrices; T[0, 3] is the phase index (carry == withdraw)."""
    out = []
    for k in (0, 1, 2, 3, 4, 5, 5):
        T = np.eye(4)
        T[0, 3] = k
        out.append(T)
    return out


class FakePlanner:
    """ik: phase -> list of solutions; line(q_start, phase) -> end q, None = fails."""

    def __init__(self, ik, line, timeout=False):
        self.solver = SimpleNamespace(ik_analytical=lambda T: list(ik.get(int(T[0, 3]), [])),
                                      lower_limits=np.full(6, -100.0),
                                      upper_limits=np.full(6, 100.0))
        self._line = line
        self._timeout = timeout
        self.calls = []

    def plan_line_t(self, q0, goal, velocity, acc_time, on_discontinuity="raise", diagnostics=None):
        k = int(goal[0, 3])
        q0 = np.asarray(q0, dtype=float)
        self.calls.append((float(q0[0]), k, on_discontinuity))
        end = self._line(q0, k)
        if end is None:
            if diagnostics is not None:
                diagnostics["search"] = {"stats": {"timed_out": self._timeout}}
            raise PathDiscontinuityError("no line")
        if end == "ikerror":
            raise IKError("no goal IK")
        if end == "empty":
            return []
        return [SimpleNamespace(positions=[float(end)] * 6)]


def q(v):
    return np.full(6, float(v))


ALL_IK = {k: [q(10 * k)] for k in range(6)}


def test_all_reached_gives_full_levels():
    p = FakePlanner(ALL_IK, lambda q0, k: 10 * k)
    assert RP.point_levels(p, phase_poses()) == (7, 7, 0)


def test_lines_are_insert_lift_withdraw_only_and_use_mode():
    p = FakePlanner(ALL_IK, lambda q0, k: 10 * k)
    RP.point_levels(p, phase_poses(), mode="hold")
    assert [k for _, k, _ in p.calls] == [3, 4, 5]
    assert {m for _, _, m in p.calls} == {"hold"}


def test_chain_starts_at_pre_insert_and_follows_last_waypoint():
    p = FakePlanner(ALL_IK, lambda q0, k: q0[0] + 1)
    RP.point_levels(p, phase_poses())
    # pre_insert IK = 20 -> insert ends at 21 -> lift at 22
    assert [s for s, _, _ in p.calls] == [20.0, 21.0, 22.0]


def test_empty_plan_keeps_start():
    p = FakePlanner(ALL_IK, lambda q0, k: "empty" if k == 4 else q0[0] + 1)
    assert RP.point_levels(p, phase_poses()) == (7, 7, 0)
    assert [s for s, _, _ in p.calls] == [20.0, 21.0, 21.0]


def test_ik_failure_stops_both_levels():
    ik = {k: v for k, v in ALL_IK.items() if k != 1}
    p = FakePlanner(ik, lambda q0, k: 10 * k)
    assert RP.point_levels(p, phase_poses()) == (1, 1, 0)
    assert p.calls == []


def test_line_failure_caps_level_but_not_ik_level():
    p = FakePlanner(ALL_IK, lambda q0, k: None if k == 4 else 10 * k)
    assert RP.point_levels(p, phase_poses()) == (7, 4, 0)


def test_ik_error_in_line_counts_as_failure():
    p = FakePlanner(ALL_IK, lambda q0, k: "ikerror" if k == 3 else 10 * k)
    assert RP.point_levels(p, phase_poses()) == (7, 3, 0)


def test_best_chain_over_every_pre_insert_solution():
    ik = dict(ALL_IK)
    ik[2] = [q(1), q(2), q(3)]
    # start 1 fails at insert, start 2 reaches lift only, start 3 goes all the way
    ends = {(1, 3): None, (2, 3): 2, (2, 4): None, (3, 3): 3, (3, 4): 3, (3, 5): 3}
    p = FakePlanner(ik, lambda q0, k: ends[(int(q0[0]), k)])
    assert RP.point_levels(p, phase_poses()) == (7, 7, 0)


def test_stops_trying_starts_once_full():
    ik = dict(ALL_IK)
    ik[2] = [q(1), q(2)]
    p = FakePlanner(ik, lambda q0, k: q0[0])
    RP.point_levels(p, phase_poses())
    assert {s for s, _, _ in p.calls} == {1.0}


def test_starts_tried_largest_joint_limit_margin_first():
    ik = dict(ALL_IK)
    ik[2] = [q(90), q(-50), q(0), q(30)]
    p = FakePlanner(ik, lambda q0, k: None)
    RP.point_levels(p, phase_poses())
    assert [s for s, _, _ in p.calls] == [0.0, 30.0, -50.0, 90.0]


def test_best_partial_chain_is_kept():
    ik = dict(ALL_IK)
    ik[2] = [q(1), q(2)]
    ends = {(1, 3): 1, (1, 4): 1, (1, 5): None, (2, 3): None}
    p = FakePlanner(ik, lambda q0, k: ends[(int(q0[0]), k)])
    assert RP.point_levels(p, phase_poses()) == (7, 5, 0)


def test_timeouts_are_counted():
    p = FakePlanner(ALL_IK, lambda q0, k: None, timeout=True)
    assert RP.point_levels(p, phase_poses()) == (7, 3, 1)


def test_phase_matrices_are_flange_poses_in_metres_radians():
    point = (0.1, 0.7, 0.2)
    Ts = RP.phase_matrices(point)
    P = RP.RV.flange_poses(point)
    assert len(Ts) == 7
    for T, p in zip(Ts, P):
        np.testing.assert_allclose(T[:3, 3], p[:3] / 1e3, atol=1e-12)
        np.testing.assert_allclose(T[:3, 2], [0, 1, 0], atol=1e-12)    # flange z = base +Y
        np.testing.assert_allclose(T[:3, 1], [0, 0, 1], atol=1e-12)    # flange y = base +Z


def test_column_levels_shapes():
    p = FakePlanner(ALL_IK, lambda q0, k: 10 * k)
    ik, lv, to = RP.column_levels(p, 0.0, 0.5, [0.0, 0.1, 0.2], "hold")
    assert ik.dtype == np.int8 and lv.dtype == np.int8
    np.testing.assert_array_equal(ik, [7, 7, 7])
    np.testing.assert_array_equal(lv, [7, 7, 7])
    assert to == 0


@pytest.mark.parametrize("mode", ["raise", "hold", "jump"])
def test_modes_accepted(mode):
    p = FakePlanner(ALL_IK, lambda q0, k: 10 * k)
    assert RP.point_levels(p, phase_poses(), mode=mode)[1] == 7
