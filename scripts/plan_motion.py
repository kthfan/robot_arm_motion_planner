"""Plan one motion from the command line (FK, IK, PTP_T or LINE_T).

Joint angles and the pose rotation are in radians and positions in metres,
unless ``--deg`` / ``--mm`` are given (then they are degrees / millimetres, as on
the TM teach pendant). A goal pose is ``x y z rx ry rz`` (TM convention,
``R = Rz(rz) Ry(ry) Rx(rx)``) of the flange in the robot base frame.

Usage::

    python scripts/plan_motion.py fk   --q0 0 17 69 11 86 6 --deg
    python scripts/plan_motion.py ik   --goal 400 -100 300 180 0 90 --deg --mm
    python scripts/plan_motion.py line --q0 0 17 69 11 86 6 --goal 400 -100 300 180 0 90 \
        --deg --mm --speed 50 --mode hold
    python scripts/plan_motion.py ptp  --q0 ... --goal ... --deg --mm --json

``line`` / ``ptp`` print the planned PTP_J waypoints; ``--json`` prints them with
the planner diagnostics as JSON instead.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from motion_planner import (  # noqa: E402
    DISCONTINUITY_MODES,
    IKError,
    PathDiscontinuityError,
    default_planner,
    matrix_to_pose,
    pose_to_matrix,
)


def _joints(values, deg: bool) -> np.ndarray:
    q = np.asarray(values, dtype=float)
    if q.shape != (6,):
        raise SystemExit(f"expected 6 joint values, got {q.size}")
    return np.radians(q) if deg else q


def _pose(values, deg: bool, mm: bool) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.shape != (6,):
        raise SystemExit(f"expected a pose x y z rx ry rz, got {p.size} values")
    if mm:
        p[:3] = p[:3] / 1000.0
    if deg:
        p[3:] = np.radians(p[3:])
    return pose_to_matrix(p.tolist())


def _fmt_joints(q, deg: bool) -> str:
    q = np.degrees(q) if deg else np.asarray(q)
    return " ".join(f"{x:9.3f}" if deg else f"{x:9.5f}" for x in q)


def _fmt_pose(T, deg: bool, mm: bool) -> str:
    p = np.asarray(matrix_to_pose(T), dtype=float)
    if mm:
        p[:3] *= 1000.0
    if deg:
        p[3:] = np.degrees(p[3:])
    return " ".join(f"{x:10.4f}" for x in p)


def _solver_planner(args):
    planner = default_planner("pinocchio")
    if args.uncalibrated:
        from motion_planner.pinocchio_solver import PinocchioSolver
        from motion_planner.planner import KinematicMotionPlanner

        planner = KinematicMotionPlanner(PinocchioSolver())
    return planner.solver, planner


def cmd_fk(args) -> int:
    solver, _ = _solver_planner(args)
    T = solver.fk(_joints(args.q0, args.deg))
    print(_fmt_pose(T, args.deg, args.mm))
    return 0


def cmd_ik(args) -> int:
    solver, _ = _solver_planner(args)
    goal = _pose(args.goal, args.deg, args.mm)
    sols = solver.ik_analytical(goal)
    if not sols:
        print("no IK solution", file=sys.stderr)
        return 1
    if args.q0 is not None:
        q0 = _joints(args.q0, args.deg)
        sols = sorted(sols, key=lambda q: float(np.linalg.norm(q - q0)))
    unit = "deg" if args.deg else "rad"
    print(f"{len(sols)} solution(s) [{unit}]   branch (shoulder, elbow, wrist)")
    for q in sols:
        print(f"  {_fmt_joints(q, args.deg)}   {solver.branch(q)}")
    return 0


def cmd_plan(args) -> int:
    solver, planner = _solver_planner(args)
    if args.q0 is None or args.goal is None:
        raise SystemExit("--q0 and --goal are required")
    q0 = _joints(args.q0, args.deg)
    goal = _pose(args.goal, args.deg, args.mm)
    kw = {"diagnostics": {}}
    if args.goal_joints is not None:
        kw["goal_joints"] = _joints(args.goal_joints, args.deg)
    if args.cmd == "line":
        kw["on_discontinuity"] = args.mode
    plan = planner.plan_line_t if args.cmd == "line" else planner.plan_ptp_t
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            waypoints = plan(q0, goal, args.speed, args.acc_time, **kw)
        except (IKError, PathDiscontinuityError) as exc:
            if args.json:
                print(json.dumps({"status": type(exc).__name__, "error": str(exc),
                                  "diagnostics": kw["diagnostics"]}, default=_json_default))
            else:
                print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps({
            "status": "ok",
            "waypoints": [w.__dict__ for w in waypoints],
            "warnings": [str(w.message) for w in caught],
            "diagnostics": kw["diagnostics"],
        }, default=_json_default))
        return 0
    unit = "deg" if args.deg else "rad"
    print(f"{len(waypoints)} PTP_J waypoint(s); joints [{unit}], speed %, acc_time [s], blend %")
    for i, w in enumerate(waypoints):
        print(f"  {i:3d} {_fmt_joints(w.positions, args.deg)}  {w.velocity:6.2f} "
              f"{w.acc_time:6.3f} {w.blend_percentage:4d}")
    events = kw["diagnostics"].get("events") or []
    for e in events:
        print(f"  event: {e['kind']} at s={e['s_from']:.4f}..{e['s_to']:.4f}, "
              f"joint change {math.degrees(e['joint_change']):.1f} deg")
    for w in caught:
        print(f"  warning: {w.message}")
    if waypoints:
        print(f"end pose: {_fmt_pose(solver.fk(waypoints[-1].positions), args.deg, args.mm)}")
    return 0


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fk", "ik", "ptp", "line"])
    ap.add_argument("--q0", nargs=6, type=float, help="start joints (ik: sort solutions by distance)")
    ap.add_argument("--goal", nargs=6, type=float, help="goal pose x y z rx ry rz")
    ap.add_argument("--goal-joints", nargs=6, type=float, help="preferred goal configuration")
    ap.add_argument("--speed", type=float, default=50.0, help="speed %% (default: %(default)s)")
    ap.add_argument("--acc-time", type=float, default=0.2, help="ramp time [s] (default: %(default)s)")
    ap.add_argument("--mode", choices=DISCONTINUITY_MODES, default="hold",
                    help="LINE_T on_discontinuity (default: %(default)s)")
    ap.add_argument("--deg", action="store_true", help="joints and pose rotation in degrees")
    ap.add_argument("--mm", action="store_true", help="pose position in millimetres")
    ap.add_argument("--uncalibrated", action="store_true", help="ignore models/tm5s_calibration.json")
    ap.add_argument("--json", action="store_true", help="ptp / line: print JSON")
    args = ap.parse_args(argv)
    if args.cmd == "fk":
        if args.q0 is None:
            raise SystemExit("--q0 is required")
        return cmd_fk(args)
    if args.cmd == "ik":
        if args.goal is None:
            raise SystemExit("--goal is required")
        return cmd_ik(args)
    return cmd_plan(args)


if __name__ == "__main__":
    sys.exit(main())
