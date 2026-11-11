"""Dense branch-graph oracle: is a LINE_T case feasible at all with held flips?

Independent of the planner's search: every analytical IK solution (2 pi variants
included) at N+1 samples of the line is a node; nodes are linked along the line
by seeded IK continuation and, at the same sample, by a self-motion walk that
holds the TCP within the planner's max_flip_deviation / max_flip_angle.
0-1 BFS gives the fewest held flips from q0 to s = 1.

Usage: python scripts/line_motion_experiments/oracle.py RUN_TAG [--all] [--n 200]
       [--cases cases.json]
  -> results/RUN_TAG_oracle.jsonl for the cases whose hold run failed (or --all).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from common import RESULTS, get_planner, load_cases

_p = None


def _init():
    global _p
    _p = get_planner()


class Line:
    def __init__(self, T0, T1):
        self.p0, self.p1 = T0[:3, 3], T1[:3, 3]
        self.slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))

    def pose(self, s):
        T = np.eye(4)
        T[:3, :3] = self.slerp([s]).as_matrix()[0]
        T[:3, 3] = (1 - s) * self.p0 + s * self.p1
        return T


def _err(sv, q, target, rot_scale):
    T = sv.fk(q)
    rot = Rotation.from_matrix(target[:3, :3] @ T[:3, :3].T).as_rotvec()
    return np.concatenate([target[:3, 3] - T[:3, 3], rot_scale * rot])


def _dev(sv, q, target):
    T = sv.fk(q)
    lin = float(np.linalg.norm(T[:3, 3] - target[:3, 3]))
    ang = float(Rotation.from_matrix(T[:3, :3].T @ target[:3, :3]).magnitude())
    return lin, ang


def held_walks(p, q, target):
    """Self-motion walks from q at the target pose (both directions)."""
    sv = p.solver
    rs = p.linear_tolerance / p.angular_tolerance
    W = np.diag([1, 1, 1, rs, rs, rs])
    step = 0.5 * p.max_joint_step
    walks = []
    for sign in (1.0, -1.0):
        walk, x, v_prev = [q], q, None
        for _ in range(int(4 * math.pi / step)):
            J = W @ sv.jacobian(x)
            v = np.linalg.svd(J)[2][-1]
            v = sign * v if v_prev is None else (v if v @ v_prev >= 0 else -v)
            y = x + step * v / np.max(np.abs(v))
            P = np.eye(6) - np.outer(v, v)
            for _ in range(20):  # hold the pose, not moving along v
                e = _err(sv, y, target, rs)
                dq = P @ np.linalg.lstsq(W @ sv.jacobian(y) @ P, e, rcond=1e-6)[0]
                n = np.max(np.abs(dq))
                if n > p.max_joint_step:
                    dq *= p.max_joint_step / n
                y = np.clip(y + dq, sv.lower_limits, sv.upper_limits)
                if n < 1e-9:
                    break
            moved = np.max(np.abs(y - x))
            if not 0.1 * step <= moved <= p.max_joint_step:
                break
            if any(
                a > p.max_flip_deviation or b > p.max_flip_angle
                for a, b in (_dev(sv, x + t * (y - x), target) for t in (0.25, 0.5, 0.75, 1.0))
            ):
                break
            walk.append(y)
            x, v_prev = y, v
            if len(walk) > 3 and np.max(np.abs(x - q)) <= step:
                break
        walks.append(walk)
    return walks


def reaches(p, walks, c, target):
    sv = p.solver
    for walk in walks:
        for w in walk:
            if np.max(np.abs(c - w)) <= p.max_joint_step and all(
                a <= p.max_flip_deviation and b <= p.max_flip_angle
                for a, b in (_dev(sv, w + t * (c - w), target) for t in (0.25, 0.5, 0.75, 1.0))
            ):
                return True
    return False


def track(p, line, s_a, q, s_b, sub=4):
    sv = p.solver
    for k in range(1, sub + 1):
        x = sv.ik(line.pose(s_a + (s_b - s_a) * k / sub), q)
        if x is None or np.max(np.abs(x - q)) > p.max_joint_step:
            return track(p, line, s_a, q, s_b, sub * 4) if sub < 64 else None
        q = x
    return q


def same_branch(sv, q, ref):
    return sv.branch(q) == sv.branch(ref) and bool(np.all(np.abs((q - ref)[[0, 3, 5]]) < math.pi))


def oracle(args):
    case, n = args
    p, sv = _p, _p.solver
    t0 = time.perf_counter()
    q0 = np.asarray(case.q0)
    line = Line(sv.fk(q0), case.goal_matrix)
    ss = np.linspace(0, 1, n + 1)
    nodes = []
    for k, s in enumerate(ss):
        Q = [np.asarray(q) for q in sv.analytical_ik.ik(line.pose(s)).Q]
        if k == 0:
            Q = [q0] + [q for q in Q if np.max(np.abs(q - q0)) > 1e-3]
        nodes.append(Q)
    no_ik = [float(s) for s, Q in zip(ss, nodes) if not Q]
    goal_ok = lambda q: case.goal_joints is None or same_branch(sv, q, np.asarray(case.goal_joints))  # noqa: E731

    def line_next(k, i):
        x = track(p, line, ss[k], nodes[k][i], ss[k + 1])
        if x is None:
            return None
        d = [np.max(np.abs(x - c)) for c in nodes[k + 1]]
        j = int(np.argmin(d)) if d else None
        return j if j is not None and d[j] < 1e-2 else None

    flips_cache = {}

    def flips(k, i):
        if (k, i) not in flips_cache:
            target = line.pose(ss[k])
            walks = held_walks(p, nodes[k][i], target)
            if max(len(w) for w in walks) < 2:
                flips_cache[(k, i)] = []
            else:
                flips_cache[(k, i)] = [
                    j for j, c in enumerate(nodes[k]) if j != i and reaches(p, walks, c, target)
                ]
        return flips_cache[(k, i)]

    # raise: the q0 branch alone
    k, i = 0, 0
    while k < n:
        j = line_next(k, i)
        if j is None:
            break
        k, i = k + 1, j
    raise_ok = k == n and goal_ok(nodes[n][i])
    raise_end = float(ss[k])

    # hold: 0-1 BFS (line edge 0, held flip 1)
    dist = {(0, 0): 0}
    parent = {}
    dq = deque([(0, 0)])
    best = None
    flip_at = []
    while dq:
        k, i = dq.popleft()
        d = dist[(k, i)]
        if k == n:
            if goal_ok(nodes[k][i]):
                best = d
                x = (k, i)
                while x in parent:
                    if parent[x][0] == x[0]:
                        flip_at.append(float(ss[x[0]]))
                    x = parent[x]
                break
        else:
            j = line_next(k, i)
            if j is not None and dist.get((k + 1, j), 1 << 30) > d:
                dist[(k + 1, j)] = d
                parent[(k + 1, j)] = (k, i)
                dq.appendleft((k + 1, j))
        for j in flips(k, i):
            if dist.get((k, j), 1 << 30) > d + 1:
                dist[(k, j)] = d + 1
                parent[(k, j)] = (k, i)
                dq.append((k, j))
    reached = max(ss[k] for k, _ in dist)
    return {
        "name": case.name, "suite": case.suite, "n": n,
        "raise_feasible": raise_ok, "raise_end_s": raise_end,
        "hold_feasible": best is not None, "hold_min_flips": best,
        "hold_flip_s": flip_at[::-1],
        "hold_reached_s": float(reached), "no_ik_s": no_ik[:5],
        "time": time.perf_counter() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--names", default="")
    ap.add_argument("--cases", default="cases.json")
    a = ap.parse_args()
    cases = {c.name: c for c in load_cases(RESULTS / a.cases)}
    names = a.names.split(",") if a.names else None
    if names is None:
        with (RESULTS / f"{a.tag}.jsonl").open(encoding="utf-8") as f:
            recs = [json.loads(x) for x in f]
        names = sorted({r["name"] for r in recs if "~p" not in r["name"]
                        and (a.all or (r["mode"] == "hold" and r["status"] != "ok"))})
    jobs = [(cases[nm], a.n) for nm in names if nm in cases]
    out = RESULTS / f"{a.tag}_oracle.jsonl"
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as f, ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        for k, rec in enumerate(pool.map(oracle, jobs, chunksize=1), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{k}/{len(jobs)} {rec['name']} hold={rec['hold_feasible']} "
                  f"flips={rec['hold_min_flips']} raise={rec['raise_feasible']} "
                  f"no_ik={rec['no_ik_s'][:1]} {rec['time']:.0f}s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
