#!/usr/bin/env python3
"""Reach volume of reach_volume_tm.py, planned locally with ``motion_planner``.

Same grid, phases and flange poses as reach_volume_tm.py, but IK and motions come
from the calibrated pinocchio model (``default_planner("pinocchio")``), no controller:

* sight_far, sight_near, pre_insert: reached if the pose has an IK solution within
  the joint limits (PTP: any configuration).
* insert, lift, withdraw: LINE_T from the previous phase, planned with
  ``plan_line_t(on_discontinuity=--mode)``. The chain starts from every IK solution
  of pre_insert (largest joint-limit margin first), each LINE_T starts where the
  previous one ended, and the point keeps the best chain.
* carry == withdraw (tilt 0): reached with withdraw.

``levels_ik`` = IK-only levels; ``levels`` = with the LINE_T motions. The npz has the
reach_volume_tm.py layout, so export_tm_meshes.py works unchanged. Checkpointed every
few columns; rerun to resume.

    python scripts/reach_volume/reach_volume_planner.py --step 0.1 --out out/preview_planner.npz
    python scripts/reach_volume/reach_volume_planner.py --out out/reach_volume_planner.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reach_volume_tm as RV  # noqa: E402
from motion_planner import IKError, PathDiscontinuityError, pose_to_matrix  # noqa: E402

#: Index of pre_insert; the phases after it up to withdraw are LINE_T motions.
LINE_FROM = RV.PHASES.index("pre_insert")
LINE_TO = RV.PHASES.index("withdraw")
VELOCITY, ACC_TIME = 100.0, 0.2


def phase_matrices(point) -> list[np.ndarray]:
    """4x4 flange poses (m) of the 7 phases for a label-centre point (m)."""
    return [pose_to_matrix([*p[:3] / 1e3, *np.radians(p[3:])]) for p in RV.flange_poses(point)]


def _chain(planner, q, goals, mode) -> tuple[int, bool]:
    """Number of LINE_T *goals* reached one after the other from *q*; timed out?"""
    for n, T in enumerate(goals):
        diag: dict = {}
        try:
            plan = planner.plan_line_t(q, T, VELOCITY, ACC_TIME, on_discontinuity=mode,
                                       diagnostics=diag)
        except (IKError, PathDiscontinuityError):
            return n, bool(diag.get("search", {}).get("stats", {}).get("timed_out", False))
        if plan:
            q = np.asarray(plan[-1].positions, dtype=float)
    return len(goals), False


def point_levels(planner, poses, mode: str = "hold") -> tuple[int, int, int]:
    """(IK-only level, LINE_T level, number of timed-out plans) of one grid point."""
    level_ik, starts = 0, []
    for k, T in enumerate(poses[:RV.DISTINCT]):
        sols = planner.solver.ik_analytical(T)
        if not sols:
            break
        level_ik = k + 1
        if k == LINE_FROM:
            starts = sols
    if level_ik == RV.DISTINCT:
        level_ik = len(RV.PHASES)                 # carry == withdraw
    if level_ik <= LINE_FROM + 1:
        return level_ik, level_ik, 0

    goals = poses[LINE_FROM + 1:LINE_TO + 1]
    lo, hi = planner.solver.lower_limits, planner.solver.upper_limits
    # Starts far from the joint limits succeed far more often: try them first.
    starts = sorted((np.asarray(q, dtype=float) for q in starts),
                    key=lambda q: -float(np.min(np.minimum(q - lo, hi - q))))
    best, timeouts = 0, 0
    for q in starts:
        n, late = _chain(planner, q, goals, mode)
        best, timeouts = max(best, n), timeouts + late
        if best == len(goals):
            break
    level = LINE_FROM + 1 + best
    if level == RV.DISTINCT:
        level = len(RV.PHASES)
    return level_ik, level, timeouts


def column_levels(planner, x: float, y: float, zs, mode: str):
    """(levels_ik, levels, timeouts) of one (x, y) column."""
    n = len(zs)
    lv_ik, lv = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    timeouts = 0
    for i, z in enumerate(zs):
        a, b, t = point_levels(planner, phase_matrices((x, y, z)), mode)
        lv_ik[i], lv[i] = a, b
        timeouts += t
    return lv_ik, lv, timeouts


_planner = None


def _init_worker(max_plan_time: float) -> None:
    global _planner
    from motion_planner import default_planner
    warnings.simplefilter("ignore")
    _planner = default_planner("pinocchio")
    _planner.max_plan_time = max_plan_time


def _run_column(task):
    i, j, x, y, zs, mode = task
    return (i, j, *column_levels(_planner, x, y, zs, mode))


def save(path, xs, ys, zs, lv_ik, lv, done, timeouts, mode, max_plan_time):
    tmp = path[:-4] + ".partial.npz"
    np.savez(tmp, xs=xs, ys=ys, zs=zs, levels=lv, levels_ik=lv_ik, done=done, timeouts=timeouts,
             phases=np.array(RV.PHASES), preset=np.array("strict"),
             line_phases=np.array(RV.PHASES[LINE_FROM + 1:LINE_TO + 1]),
             mode=np.array(mode), max_plan_time=np.array(max_plan_time),
             planner=np.array("motion_planner default_planner('pinocchio')"),
             frame=np.array("label centre; robot base frame (x, y approach, z up)"))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--x", type=float, nargs=2, default=(-1.2, 1.2))
    ap.add_argument("--y", type=float, nargs=2, default=(0.0, 1.4))
    ap.add_argument("--z", type=float, nargs=2, default=(-1.2, 1.2))
    ap.add_argument("--mode", choices=("raise", "hold", "jump"), default="hold",
                    help="plan_line_t on_discontinuity (default hold)")
    ap.add_argument("--max-plan-time", type=float, default=10.0,
                    help="LINE_T planning budget per motion, s (timeouts count as failures)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xs, ys, zs = RV.axis(*args.x, args.step), RV.axis(*args.y, args.step), RV.axis(*args.z, args.step)
    shape = (len(xs), len(ys), len(zs))
    lv_ik = np.zeros(shape, dtype=np.int8)
    lv = np.zeros(shape, dtype=np.int8)
    timeouts = np.zeros(shape[:2], dtype=np.int32)
    done = np.zeros(shape[:2], dtype=bool)
    if os.path.exists(args.out):
        with np.load(args.out) as d:
            if not (np.array_equal(d["xs"], xs) and np.array_equal(d["ys"], ys)
                    and np.array_equal(d["zs"], zs) and str(d["mode"]) == args.mode):
                print("existing --out has a different grid/mode; refusing to overwrite")
                return 2
            lv_ik, lv, done = d["levels_ik"].copy(), d["levels"].copy(), d["done"].copy()
            timeouts = d["timeouts"].copy()
        print(f"resuming: {int(done.sum())}/{done.size} columns done")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"grid {shape[0]}x{shape[1]}x{shape[2]} = {int(np.prod(shape))} points, mode={args.mode}, "
          f"max_plan_time={args.max_plan_time} s, {args.workers} workers", flush=True)
    todo = [(i, j, float(xs[i]), float(ys[j]), zs, args.mode)
            for i in range(shape[0]) for j in range(shape[1]) if not done[i, j]]
    t0 = time.time()
    try:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                 initargs=(args.max_plan_time,)) as pool:
            # as completed: a slow column (all starts fail -> every start is planned,
            # up to ~1 min) must not hold back progress and checkpoints of the others
            futures = [pool.submit(_run_column, task) for task in todo]
            for count, fut in enumerate(as_completed(futures), 1):
                i, j, a, b, t = fut.result()
                lv_ik[i, j], lv[i, j], timeouts[i, j], done[i, j] = a, b, t, True
                if count % 20 == 0 or count == len(todo):
                    save(args.out, xs, ys, zs, lv_ik, lv, done, timeouts, args.mode, args.max_plan_time)
                el = time.time() - t0
                eta = el / count * (len(todo) - count)
                print(f"\r  columns {count}/{len(todo)}  {el:6.0f}s  eta {eta:6.0f}s", end="", flush=True)
    finally:
        save(args.out, xs, ys, zs, lv_ik, lv, done, timeouts, args.mode, args.max_plan_time)
    print()

    total = int(np.prod(shape))
    print("level  phase        ik_only  line_t")
    for k in range(len(RV.PHASES) + 1):
        nm = "none" if k == 0 else RV.PHASES[k - 1]
        print(f"  {k}    {nm:<11} {int((lv_ik == k).sum()):7d}  {int((lv == k).sum()):7d}")
    assert (lv <= lv_ik).all(), "LINE_T level must not exceed IK level"
    print(f"sum = {total}; timed-out plans {int(timeouts.sum())}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
