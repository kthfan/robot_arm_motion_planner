"""Command-line scripts and examples."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

ROOT = Path(__file__).resolve().parents[1]
Q0_DEG = ["0", "17", "69", "11", "86", "6"]


@pytest.fixture(scope="module")
def plan_motion():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import plan_motion

        yield plan_motion
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def _goal_deg_mm(plan_motion, capsys, q_deg):
    assert plan_motion.main(["fk", "--q0", *q_deg, "--deg", "--mm"]) == 0
    return capsys.readouterr().out.split()


def test_fk_and_ik_roundtrip(plan_motion, capsys):
    goal = _goal_deg_mm(plan_motion, capsys, Q0_DEG)
    assert len(goal) == 6
    assert plan_motion.main(["ik", "--goal", *goal, "--deg", "--mm", "--q0", *Q0_DEG]) == 0
    first = capsys.readouterr().out.splitlines()[1].split()[:6]
    np.testing.assert_allclose([float(x) for x in first], [float(x) for x in Q0_DEG], atol=1e-2)


def test_line_json_reaches_the_goal(plan_motion, capsys):
    goal = _goal_deg_mm(plan_motion, capsys, ["5", "20", "65", "11", "86", "6"])
    assert plan_motion.main(["line", "--q0", *Q0_DEG, "--goal", *goal, "--deg", "--mm",
                             "--speed", "30", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["waypoints"]
    np.testing.assert_allclose(np.degrees(out["waypoints"][-1]["positions"]),
                               [5, 20, 65, 11, 86, 6], atol=0.5)


def test_unreachable_goal_fails_with_exit_code_1(plan_motion, capsys):
    rc = plan_motion.main(["line", "--q0", *Q0_DEG, "--goal", "5000", "0", "0", "180", "0", "0",
                           "--deg", "--mm"])
    assert rc == 1
    assert "IKError" in capsys.readouterr().err


def test_quickstart_example_runs(capsys):
    runpy.run_path(str(ROOT / "examples" / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "LINE_T:" in out and "PTP_T goal joints" in out
