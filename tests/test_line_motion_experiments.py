"""Smoke tests of the LINE_T experiment harness (scripts/line_motion_experiments)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "line_motion_experiments"


@pytest.fixture(scope="module")
def harness():
    sys.path.insert(0, str(HARNESS))
    try:
        import cases
        import common
        import report
        import run

        yield common, cases, run, report
    finally:
        sys.path.remove(str(HARNESS))


def test_generated_cases_roundtrip(harness, tmp_path):
    common, cases, _, _ = harness
    g = cases.Generator(0)
    built = g.user() + g.random(2) + g.tiny(1) + g.goal_joints(1)
    assert {c.suite for c in built} >= {"user", "random", "tiny"}
    path = tmp_path / "cases.json"
    common.save_cases(built, path)
    loaded = common.load_cases(path)
    assert [c.name for c in loaded] == [c.name for c in built]
    np.testing.assert_allclose(loaded[0].goal_matrix, built[0].goal_matrix)


def test_run_one_plans_and_checks_the_user_cases(harness):
    _, cases, run, _ = harness
    run._init(timeout=60.0)
    for case in cases.Generator(0).user():
        rec = run.run_one((case, "hold"))
        assert rec["status"] == "ok", rec.get("error")
        check = rec["check"]
        assert check["within_limits"]
        assert check["goal_lin_mm"] < 0.1
        assert check["jump_segments"] == 0
        assert check["max_lin_mm"] < 5.0  # held reconfigurations stay within 3 mm (+ sampling)
        json.dumps(rec)  # the record is written as JSON


def test_checker_flags_a_jump(harness):
    common, cases, _, _ = harness
    planner = common.get_planner()
    case = cases.Generator(0).user()[0]
    # straight joint interpolation from the start to a far configuration leaves the line
    q0 = np.asarray(case.q0)
    q1 = q0 + np.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0])
    check = common.Checker(planner).run(q0, planner.fk(q1), np.array([q0, q1]))
    assert check.goal_lin_mm < 1e-6
    assert check.jump_segments == 1


def test_perturbed_cases_keep_names_and_goal_joints(harness):
    _, cases, run, _ = harness
    base = cases.Generator(0).user()[:2]
    out = run.perturbed(base, 3)
    assert len(out) == 6
    assert out[0].name == f"{base[0].name}~p0"
    assert all(o.goal_joints == base[0].goal_joints for o in out[:3])
