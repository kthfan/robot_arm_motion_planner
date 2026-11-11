"""Seeded LINE_T experiment cases.

Usage:
  python scripts/line_motion_experiments/cases.py [SCALE] [--out cases.json]

SCALE multiplies the number of generated cases per suite (default 1.0, about 900
cases; 0.5 gives about 460). Writes results/<out>.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from common import RESULTS, Case, get_planner, save_cases

# Reported user cases (start joints, TM goal pose [m, rad]).
USER = {
    "user1": ([2.44735982, 1.13695154, 2.05527087, -3.22704178, -0.10781322, -0.0331056],
              [0.04011968656694022, 0.5660800280188786, 0.07081906302980091,
               1.5783607252426803, 0.0021714530626639216, 2.348814935634403]),
    "user2": ([2.44796196, 1.1332355, 2.05063617, -3.24102364, -0.10785403, -0.05515234],
              [0.0404524802425845, 0.5662882533166673, 0.06992704940857346,
               1.5729843248128845, 0.0021030978989773014, 2.349234564732319]),
    "user3": ([2.4480203, 1.13224786, 2.04869034, -3.24347961, -0.10855246, -0.06072712],
              [0.04032131, 0.56616406, 0.07035293,
               1.5755799894488138, 0.0021154390954929436, 2.3491710326928166]),
    "user4": ([2.50256289, 0.50228798, 0.57586271, 2.04564119, 0.19129453, 0.01268078],
              [-0.21021457, 0.5934515, 0.56501399,
               -1.547089253432336, -0.005586216800832711, -0.790899909061723]),
    "user5": ([2.37628732, 0.27010514, 1.57430745, -1.20395067, -0.03484558, -2.50246421],
              [-0.08079602, 0.71796395, 0.56756084,
               -1.5441180210521142, -0.004519044520707949, -0.7849294678487391]),
    "user6": ([2.37763765, 1.72041564, -1.58324008, 0.59165777, -0.03120506, -2.41394083],
              [-0.08159062, 0.71783619, 0.56720372,
               -1.5393583485293365, -0.005086421734882718, -0.7829327231804721]),
}
# Case 7: goal given as position + quaternion (x, y, z, w).
USER7_Q0 = [2.4507007939103107, 1.14282550315689, 2.0616146447890866,
            3.034382500243299, -0.10155445512723975, -0.04073203431605356]
USER7_GOAL = ([0.019133, 0.81707898, 0.47240737], [0.26745484, 0.64396943, 0.66119858, 0.27674486])


def _pose(p, R) -> list:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T.tolist()


def _tm_pose(pose) -> list:
    return _pose(pose[:3], Rotation.from_euler("xyz", pose[3:6]).as_matrix())


class Generator:
    def __init__(self, seed: int = 0) -> None:
        self.p = get_planner()
        self.sv = self.p.solver
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = self.sv.lower_limits, self.sv.upper_limits

    def q_random(self, shrink: float = 0.95) -> np.ndarray:
        mid, half = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo) * shrink
        return self.rng.uniform(mid - half, mid + half)

    def back_track(self, q_mid, T_mid, T_start, steps: int = 64):
        """q at T_start on q_mid's branch (seeded continuation along the line)."""
        p0, p1 = T_mid[:3, 3], T_start[:3, 3]
        r = Rotation.from_matrix(np.stack([T_mid[:3, :3], T_start[:3, :3]]))
        rv = (r[0].inv() * r[1]).as_rotvec()
        q = np.asarray(q_mid, dtype=float)
        for k in range(1, steps + 1):
            t = k / steps
            T = np.eye(4)
            T[:3, :3] = (r[0] * Rotation.from_rotvec(t * rv)).as_matrix()
            T[:3, 3] = p0 + t * (p1 - p0)
            q = self.sv.ik(T, q)
            if q is None:
                return None
        return q

    def line_through(self, q_mid, length, rot_deg, name, suite, frac=None):
        """Line of ``length`` [m] and rotation through FK(q_mid), q_mid at fraction ``frac``."""
        T_mid = self.sv.fk(q_mid)
        frac = self.rng.uniform(0.2, 0.8) if frac is None else frac
        d = self.rng.normal(size=3)
        d /= np.linalg.norm(d)
        axis = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rv = axis * math.radians(rot_deg)
        R_mid = Rotation.from_matrix(T_mid[:3, :3])
        T_a = np.eye(4)
        T_a[:3, :3] = (R_mid * Rotation.from_rotvec(-frac * rv)).as_matrix()
        T_a[:3, 3] = T_mid[:3, 3] - frac * length * d
        T_b = np.eye(4)
        T_b[:3, :3] = (R_mid * Rotation.from_rotvec((1 - frac) * rv)).as_matrix()
        T_b[:3, 3] = T_mid[:3, 3] + (1 - frac) * length * d
        q0 = self.back_track(q_mid, T_mid, T_a)
        if q0 is None or np.any(q0 < self.lo) or np.any(q0 > self.hi):
            return None
        return Case(name, suite, q0.tolist(), T_b.tolist(), note=f"through q_mid={np.round(q_mid, 4).tolist()}")

    # ----------------------------------------------------------- suites

    def random(self, n):
        out = []
        while len(out) < n:
            q0 = self.q_random()
            qg = self.q_random() if len(out) % 2 else q0 + self.rng.normal(0, 0.5, 6)
            qg = np.clip(qg, self.lo, self.hi)
            out.append(Case(f"random{len(out)}", "random", q0.tolist(), self.sv.fk(qg).tolist()))
        return out

    def singular(self, n, suite):
        out, tries = [], 0
        while len(out) < n and tries < 50 * n:
            tries += 1
            q = self.q_random(0.9)
            if suite == "wrist":
                q[4] = math.radians(self.rng.normal(0, 2.0))
            elif suite == "elbow":
                q[2] = math.radians(self.rng.normal(0, 2.0))
            elif suite == "shoulder":
                if not self._near_shoulder(q):
                    continue
            c = self.line_through(q, self.rng.uniform(0.05, 0.5), self.rng.uniform(0, 30),
                                  f"{suite}{len(out)}", suite)
            if c is not None:
                out.append(c)
        return out

    def _near_shoulder(self, q, tol=0.01):
        """Move q2 onto a shoulder-branch boundary (+ noise ``tol`` [rad]); False if none."""
        side = lambda x: self.sv.branch(x)[0]  # noqa: E731
        grid = np.linspace(-2.5, 2.5, 51)
        signs = []
        for g in grid:
            x = q.copy()
            x[1] = g
            signs.append(side(x))
        for k in range(len(grid) - 1):
            if signs[k] != signs[k + 1]:
                lo_, hi_ = grid[k], grid[k + 1]
                for _ in range(30):
                    m = 0.5 * (lo_ + hi_)
                    x = q.copy()
                    x[1] = m
                    if side(x) == signs[k]:
                        lo_ = m
                    else:
                        hi_ = m
                q[1] = 0.5 * (lo_ + hi_) + self.rng.normal(0, tol)
                return True
        return False

    def limits(self, n):
        out = []
        while len(out) < n:
            q = self.q_random(0.9)
            j = (0, 3, 5)[len(out) % 3]
            q[j] = self.rng.choice([-1, 1]) * (self.hi[j] - math.radians(self.rng.uniform(1, 15)))
            c = self.line_through(q, self.rng.uniform(0.05, 0.4), self.rng.uniform(0, 60),
                                  f"limits{len(out)}", "limits", frac=0.0)
            if c is not None:
                out.append(c)
        return out

    def rotation(self, n):
        out = []
        while len(out) < n:
            q0 = self.q_random(0.9)
            T0 = self.sv.fk(q0)
            axis = self.rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            R = Rotation.from_rotvec(axis * math.radians(self.rng.uniform(20, 179))) * Rotation.from_matrix(T0[:3, :3])
            out.append(Case(f"rotation{len(out)}", "rotation", q0.tolist(), _pose(T0[:3, 3], R.as_matrix())))
        return out

    def long(self, n):
        out = []
        while len(out) < n:
            q0, qg = self.q_random(), self.q_random()
            T0, Tg = self.sv.fk(q0), self.sv.fk(qg)
            if np.linalg.norm(Tg[:3, 3] - T0[:3, 3]) > 0.8:
                out.append(Case(f"long{len(out)}", "long", q0.tolist(), Tg.tolist()))
        return out

    def tiny(self, n):
        out = []
        while len(out) < n:
            q0 = self.q_random(0.9)
            T0 = self.sv.fk(q0)
            d = self.rng.normal(size=3)
            d *= self.rng.uniform(1e-4, 2e-3) / np.linalg.norm(d)
            R = Rotation.from_rotvec(self.rng.normal(size=3) * 1e-3) * Rotation.from_matrix(T0[:3, :3])
            out.append(Case(f"tiny{len(out)}", "tiny", q0.tolist(), _pose(T0[:3, 3] + d, R.as_matrix())))
        return out

    def hole(self, n):
        """Lines passing within 0.15 m of the joint-1 axis: no IK in the middle."""
        out = []
        while len(out) < n:
            q0 = self.q_random(0.9)
            T0 = self.sv.fk(q0)
            p0 = T0[:3, 3]
            r = np.linalg.norm(p0[:2])
            if r < 0.2:
                continue
            normal = np.array([-p0[1], p0[0], 0.0]) / r
            p1 = np.array([-p0[0], -p0[1], p0[2] + self.rng.uniform(-0.1, 0.1)])
            p1 += normal * self.rng.uniform(-0.3, 0.3)
            out.append(Case(f"hole{len(out)}", "hole", q0.tolist(), _pose(p1, T0[:3, :3])))
        return out

    def goal_joints(self, n):
        """Random lines whose goal configuration is a random IK solution of the goal."""
        out = []
        while len(out) < n:
            q0 = self.q_random()
            qg = np.clip(q0 + self.rng.normal(0, 0.5, 6), self.lo, self.hi)
            T = self.sv.fk(qg)
            sols = [q for q in self.sv.ik_analytical(T)
                    if np.all(q >= self.lo) and np.all(q <= self.hi)]
            if not sols:
                continue
            gj = sols[int(self.rng.integers(len(sols)))]
            out.append(Case(f"goal_joints{len(out)}", "goal_joints", q0.tolist(), T.tolist(),
                            goal_joints=gj.tolist()))
        return out

    def user(self):
        out = [Case(k, "user", q0, _tm_pose(pose)) for k, (q0, pose) in USER.items()]
        p, quat = USER7_GOAL
        out.append(Case("user7", "user", USER7_Q0, _pose(p, Rotation.from_quat(quat).as_matrix()),
                        note="hold-infeasible by dense oracle"))
        return out


def build(scale: float = 1.0) -> list:
    g = Generator(0)
    n = lambda k: max(1, int(k * scale))  # noqa: E731
    return (
        g.user() + g.random(n(300)) + g.singular(n(100), "wrist") + g.singular(n(100), "elbow")
        + g.singular(n(60), "shoulder") + g.limits(n(90)) + g.rotation(n(60)) + g.long(n(60))
        + g.tiny(n(30)) + g.hole(n(40)) + g.goal_joints(n(60))
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the seeded LINE_T cases.")
    ap.add_argument("scale", nargs="?", type=float, default=1.0,
                    help="multiplier of the number of cases per suite")
    ap.add_argument("--out", default="cases.json", help="case file in results/")
    a = ap.parse_args()
    cases = build(a.scale)
    out = RESULTS / a.out
    save_cases(cases, out)
    from collections import Counter

    print(len(cases), dict(Counter(c.suite for c in cases)), "->", out)
