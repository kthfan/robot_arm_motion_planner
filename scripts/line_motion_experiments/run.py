"""Run LINE_T experiment cases through the planner in this working tree.

Usage:
  python scripts/line_motion_experiments/run.py TAG [--modes hold,jump,raise]
         [--suites user,random,...] [--workers 16] [--perturb K] [--limit N]
         [--timeout 60] [--cases cases.json]

Writes results/TAG.jsonl (one record per case and mode, incl. waypoints).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import threading
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from common import RESULTS, Case, Checker, get_planner, load_cases

_planner = _checker = None
_timeout = 0.0


class JobTimeout(Exception):
    pass


class _Deadline:
    """Raise JobTimeout in the calling thread after ``seconds`` (between bytecodes,
    so a planner stuck in a Python loop is interrupted)."""

    def __init__(self, seconds: float) -> None:
        self.tid = threading.get_ident()
        self.lock = threading.Lock()
        self.done = False
        self.timer = threading.Timer(seconds, self._fire)
        self.timer.daemon = True
        if seconds > 0:
            self.timer.start()

    def _fire(self) -> None:
        with self.lock:
            if not self.done:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self.tid), ctypes.py_object(JobTimeout))

    def stop(self) -> None:
        with self.lock:
            self.done = True
            self.timer.cancel()
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self.tid), None)


def _init(timeout: float = 0.0):
    global _planner, _checker, _timeout
    _planner = get_planner()
    _checker = Checker(_planner)
    _timeout = timeout


def _plan(planner, case: Case, mode: str):
    kwargs = {"on_discontinuity": mode}
    if case.goal_joints is not None:
        kwargs["goal_joints"] = case.goal_joints
    return planner.plan_line_t(case.q0, case.goal_matrix, 100.0, 0.2, **kwargs)


def run_one(args):
    case, mode = args
    rec = {"name": case.name, "suite": case.suite, "mode": mode}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.perf_counter()
        deadline = _Deadline(_timeout)
        try:
            try:
                actions = _plan(_planner, case, mode)
                rec["status"] = "ok"
            finally:
                deadline.stop()
        except JobTimeout:
            actions = None
            rec["status"] = "Timeout"
            rec["error"] = f"no result within {_timeout:g} s"
        except Exception as exc:  # noqa: BLE001 - recorded for the report
            actions = None
            rec["status"] = type(exc).__name__
            rec["error"] = str(exc)
            if not isinstance(exc, RuntimeError):
                rec["traceback"] = traceback.format_exc()
        rec["time"] = time.perf_counter() - t
    rec["warnings"] = [str(w.message) for w in caught]
    if actions is not None:
        qs = np.array([case.q0] + [a.positions for a in actions])
        rec["qs"] = qs.round(7).tolist()
        rec["check"] = asdict(_checker.run(np.asarray(case.q0), case.goal_matrix, qs))
    return rec


def perturbed(cases: list[Case], k: int, seed: int = 1) -> list[Case]:
    rng = np.random.default_rng(seed)
    out = []
    for c in cases:
        for i in range(k):
            q0 = np.asarray(c.q0) + rng.normal(0, 1e-3, 6)
            T = c.goal_matrix.copy()
            T[:3, 3] += rng.normal(0, 0.5e-3, 3)
            T[:3, :3] = (Rotation.from_rotvec(rng.normal(0, math.radians(0.05), 3))
                         * Rotation.from_matrix(T[:3, :3])).as_matrix()
            out.append(Case(f"{c.name}~p{i}", c.suite, q0.tolist(), T.tolist(), c.goal_joints))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--modes", default="hold,jump,raise")
    ap.add_argument("--cases", default="cases.json", help="case file in results/")
    ap.add_argument("--suites", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--perturb", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="max cases per suite")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds per plan before it is recorded as Timeout (0: none)")
    a = ap.parse_args()

    cases = load_cases(RESULTS / a.cases)
    if a.suites:
        cases = [c for c in cases if c.suite in a.suites.split(",")]
    if a.names:
        cases = [c for c in cases if c.name in a.names.split(",")]
    if a.limit:
        count: dict = {}
        kept = []
        for c in cases:
            count[c.suite] = count.get(c.suite, 0) + 1
            if count[c.suite] <= a.limit:
                kept.append(c)
        cases = kept
    if a.perturb:
        cases = cases + perturbed(cases, a.perturb)
    jobs = [(c, m) for c in cases for m in a.modes.split(",")]
    out = RESULTS / f"{a.tag}.jsonl"
    t0 = time.perf_counter()
    n = 0
    with out.open("w", encoding="utf-8") as f:
        if a.workers <= 1:
            _init(a.timeout)
            results = map(run_one, jobs)
        else:
            pool = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.timeout,))
            results = pool.map(run_one, jobs, chunksize=1)
        for rec in results:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"done {n} jobs in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
