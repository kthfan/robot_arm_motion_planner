"""LINE_T joint path search.

The IK branch of the start configuration is followed along the line
(:meth:`LineSearch.trace`). Where the line cannot be completed on it (the branch
ends at a singularity or joint limit, or reaches the goal in the wrong
configuration), the joints are reconfigured at a point of the line where the
arm is closest to a singularity: there the arm can change branch through a
self-motion with the TCP held (:class:`SelfMotion`), or, in jump mode, jump.
Reconfigurations are found by a best-first search over (s, q) states, ranked
lexicographically by the number of jumps, then the number of held
reconfigurations, then a weighted sum of the joint travel, the largest jump and
the joint-limit margin shortfall.

Reconfiguring where the distance to the singularity is smallest keeps the plan
stable: small changes of the start or goal move that point only slightly.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from motion_planner.path import CartesianPath
from motion_planner.self_motion import SelfMotion, deviation, rotvec
from motion_planner.solver import IKError, KinematicsSolver

LINE, HELD, JUMP = "line", "held", "jump"

# Joint-limit margins above this [rad] are equally good.
MARGIN_CAP = math.radians(10.0)
# Weight [rad of travel per rad of margin shortfall] of the joint-limit margin.
MARGIN_WEIGHT = 20.0
# Weight of the largest jump in the score (jump mode).
JUMP_WEIGHT = 5.0
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_TWO_PI = 2.0 * math.pi
_NEWTON_ITERS = 8
_NEWTON_TOL = 1e-9  # pose error [m, rad] of exact IK


class PathDiscontinuityError(RuntimeError):
    """Raised when a Cartesian path cannot be followed with continuous joint motion."""


@dataclass
class Event:
    """A joint reconfiguration: the TCP is held near the line (``HELD``, with the
    max ``deviation`` [m]) or leaves it (``JUMP``) between ``s_from`` and ``s_to``;
    ``no_ik``: the line has no IK solution in between."""

    kind: str
    s_from: float
    s_to: float
    joint_change: float
    deviation: Optional[float] = None
    sigma: float = math.nan
    no_ik: bool = False


@dataclass
class LinePlan:
    """``samples``: (s, q, kind of the motion reaching q) after the start."""

    samples: list
    events: list


@dataclass
class Trace:
    """Samples of one IK branch from ``s[0]``; ``stuck``: it ends before s = 1."""

    s: list
    q: list
    sigma: list
    stuck: bool


@dataclass
class _Window:
    s: float
    q: np.ndarray
    sigma: float
    head: int  # trace samples [1, head) precede it
    end: bool  # the end of the trace (no line motion after it)
    alts: tuple = ()  # nearby trace samples to try if no held flip attaches here
    held_only: bool = False  # a fallback of another window, which tried the jumps


@dataclass
class _Cost:
    jumps: int = 0
    max_jump: float = 0.0
    flips: int = 0
    margin: float = math.inf
    travel: float = 0.0

    def key(self, jump_mode: bool, rest: float = 0.0) -> tuple:
        """Jumps and held flips are counted first; the rest is one smooth score,
        so that near-ties between plans do not flip on tiny input changes.
        ``rest``: lower bound of the travel still to go."""
        score = (self.travel + rest
                 + MARGIN_WEIGHT * max(0.0, MARGIN_CAP - self.margin))
        if jump_mode:
            return (self.jumps, self.flips, score + JUMP_WEIGHT * self.max_jump)
        return (self.flips, score)


@dataclass
class _Node:
    s: float
    q: np.ndarray
    cost: _Cost
    samples: list = field(default_factory=list)
    events: list = field(default_factory=list)
    terminal: bool = False
    window: Optional[_Window] = None  # reconfigure at it (q is the window point)
    parent: Optional[tuple] = None  # (node, trace samples) the window lies on


def _rotvec_batch(R: np.ndarray) -> np.ndarray:
    """:func:`rotvec` of many rotation matrices (N, 3, 3)."""
    c = np.clip(0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0), -1.0, 1.0)
    angle = np.arccos(c)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    sn = np.sin(angle)
    scale = np.where(angle < 1e-6, 0.5, angle / (2.0 * np.where(sn > 0.0, sn, 1.0)))
    out = scale[:, None] * w
    for i in np.nonzero(angle >= math.pi - 1e-4)[0]:  # near pi: the scalar formula
        out[i] = rotvec(R[i])
    return out


class LineSearch:
    """Plans one LINE_T motion. ``mode``: "raise" (start branch only), "hold" (held
    reconfigurations) or "jump" (also jumps). ``goal_ok`` accepts the configuration
    at s = 1."""

    max_flips = 4
    max_expansions = 60
    max_link_tries = 30
    max_effort = 4000  # link checks + projections per plan (~0.3 ms each)
    max_ik_queries = 8  # analytical IK calls per plan for jump targets
    jump_ahead = 2.0 ** -8  # jumps land this far (in s) past the window
    # [rad] a held link changes no joint by more than a wrist flip (pi) plus drift;
    # in the experiments 1 of ~3000 tried links beyond 3 rad held (at 3.45 rad)
    max_link_gap = 1.2 * math.pi
    window_sigma = 0.05  # [m/rad] scaled-Jacobian distance below which to try flips
    # trace samples next to a sigma_min minimum to try when no held flip attaches
    # at the minimum itself (flat singular stretches, e.g. q5 ~ 0 over a few cm)
    window_alts = 4

    def __init__(
        self,
        solver: KinematicsSolver,
        self_motion: SelfMotion,
        path: CartesianPath,
        q0: np.ndarray,
        mode: str,
        goal_ok: Callable[[np.ndarray], bool],
        linear_tolerance: float,
        angular_tolerance: float,
        max_joint_step: float,
        max_depth: int,
        max_held_length: float = 0.15,
        goal_solutions: Optional[list[np.ndarray]] = None,
        deadline: Optional[float] = None,
    ) -> None:
        self.solver = solver
        self.sm = self_motion
        self.path = path
        self.q0 = np.asarray(q0, dtype=float)
        self.mode = mode
        self.goal_ok = goal_ok
        self.lin_tol = linear_tolerance
        self.ang_tol = angular_tolerance
        self.max_step = max_joint_step
        self.h_min = 2.0 ** -max_depth
        self.h_max = 1.0 / 32
        length = path.translation_length
        self.max_held_length = max_held_length
        self.held_span = 1.0 if length <= max_held_length else max_held_length / length
        self._lo, self._hi = solver.lower_limits, solver.upper_limits
        self._poses: dict = {}
        self._traces: dict = {}
        self._goal_traces: Optional[list[Trace]] = None
        self._goals: Optional[list[np.ndarray]] = None
        self._ik_cache: dict = {} if goal_solutions is None else {1.0: list(goal_solutions)}
        self._tried: list = []  # (window, families reached, families attached) for diagnosis
        self._effort = 0  # link checks + projections, bounded by max_effort for bridges
        self._ik_queries = 0  # analytical IK calls, bounded by max_ik_queries for jumps
        # time.perf_counter() after which the search only returns plans already
        # complete (then fails), so a hopeless case does not take minutes
        self.deadline = deadline
        # where the planning time goes (diagnostics, logged per motion)
        self._fk_memo: Optional[dict] = None  # FK cache while smoothing
        self.stats = {"expansions": 0, "trace_s": 0.0, "goal_traces_s": 0.0, "links_s": 0.0,
                      "reconfigure_s": 0.0, "windows_s": 0.0, "ik_calls": 0, "ik_s": 0.0,
                      "failure_s": 0.0, "timed_out": False}

    # ------------------------------------------------------------ helpers

    def _late(self) -> bool:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            self.stats["timed_out"] = True
        return self.stats["timed_out"]

    def pose(self, s: float) -> np.ndarray:
        T = self._poses.get(s)
        if T is None:
            T = self._poses[s] = self.path.pose_at(s)
        return T

    def _segment_ok(self, s_a, q_a, s_b, q_b) -> bool:
        if np.max(np.abs(q_b - q_a)) > self.max_step:
            return False
        lin, ang = deviation(self.solver.fk(0.5 * (q_a + q_b)), self.pose(0.5 * (s_a + s_b)))
        return lin <= self.lin_tol and ang <= self.ang_tol

    def _span_ok(self, s_a, q_a, s_b, q_b, lin_tol=None, ang_tol=None, points: int = 16) -> bool:
        """Joint interpolation q_a -> q_b within one joint step and the LINE_T
        tolerance (or the given one) of the line between s_a and s_b (checked at
        ``points`` points)."""
        if np.max(np.abs(q_b - q_a)) > self.max_step:
            return False
        lin_tol = self.lin_tol if lin_tol is None else lin_tol
        ang_tol = self.ang_tol if ang_tol is None else ang_tol
        for k in range(1, points):
            t = k / float(points)
            lin, ang = deviation(
                self.solver.fk(q_a + t * (q_b - q_a)), self.pose(s_a + t * (s_b - s_a))
            )
            if lin > lin_tol or ang > ang_tol:
                return False
        return True

    def simplify(self, q0: np.ndarray, samples: list, held: bool = False) -> list:
        """Drop waypoints of runs on the line (kind LINE) whose neighbours can be
        joined directly within the LINE_T tolerance and joint step; jumps are
        kept as they are. Held reconfigurations are kept too, unless ``held``:
        then their waypoints are merged the same way within the deviation the
        held run already has (at least the LINE_T tolerance); the run's last
        waypoint stays."""
        out: list = []
        prev_s, prev_q = 0.0, np.asarray(q0, dtype=float)
        i, n = 0, len(samples)
        while i < n:
            s_i, q_i, kind = samples[i]
            if kind == HELD and held:
                j_end = i
                while j_end + 1 < n and samples[j_end + 1][2] == HELD:
                    j_end += 1
                run = samples[i:j_end + 1]
                lin_b, ang_b = self._route_error(prev_s, prev_q, [(x[0], x[1]) for x in run])
                lin_b, ang_b = max(lin_b, self.lin_tol), max(ang_b, self.ang_tol)
                k = i
                while k <= j_end:
                    j = k
                    while j + 1 <= j_end and self._span_ok(prev_s, prev_q, samples[j + 1][0],
                                                           samples[j + 1][1], lin_b, ang_b, 48):
                        j += 1
                    out.append(samples[j])
                    prev_s, prev_q = samples[j][0], samples[j][1]
                    k = j + 1
                i = j_end + 1
                continue
            if kind != LINE:
                out.append(samples[i])
                prev_s, prev_q = s_i, q_i
                i += 1
                continue
            # furthest j of this LINE run reachable from the previous waypoint
            j = i
            while j + 1 < n and samples[j + 1][2] == LINE \
                    and self._span_ok(prev_s, prev_q, samples[j + 1][0], samples[j + 1][1]):
                j += 1
            out.append(samples[j])
            prev_s, prev_q = samples[j][0], samples[j][1]
            i = j + 1
        return out

    def smooth_held(self, q0: np.ndarray, samples: list, iterations: int = 10,
                    tolerance: float = 1.0, deviation_weight: float = 0.0,
                    batch: bool = True) -> tuple[list, dict]:
        """Even out the joint motion of held reconfigurations.

        The waypoints of a held route are projected one by one into the
        reconfiguration tolerance, so the joint change is uneven: large on one
        segment, nearly zero on the next (the robot's joint speeds alternate).
        Each inner waypoint of a held run is moved onto the straight joint
        interpolation of its neighbours (by its share of ``s``; Gauss-Seidel,
        ``iterations`` sweeps) where both links stay within ``tolerance`` times
        the reconfiguration tolerance (``max_flip_deviation`` /
        ``max_flip_angle``; a route already beyond that stays allowed its own
        deviation) and the joint limits; else half way, else not at all. The run's first and last
        waypoints stay fixed.

        ``deviation_weight`` > 0 also pulls each waypoint towards the line: the
        step minimises ``w^2 |pose error / flip tolerance|^2 + |q - target|^2``
        (linearised), so the TCP comes as close to the line as the joint path
        allows without bending it; the bound only shrinks from sweep to sweep.

        ``batch``: update every other waypoint at once (red-black Gauss-Seidel;
        all checks of a half sweep in one batch). Much faster; the result
        differs from the one-by-one sweep (``batch=False``) only by where the
        iterations stop.
        Returns (samples, info).
        """
        self._fk_memo = {}
        try:
            return self._smooth_held(q0, samples, iterations, tolerance, deviation_weight, batch)
        finally:
            self._fk_memo = None

    def _smooth_held(self, q0, samples, iterations, tolerance, deviation_weight, batch=False):
        """Body of :meth:`smooth_held` (runs with the FK cache on)."""
        qs = [np.asarray(q0, dtype=float)] + [np.asarray(q, dtype=float) for _, q, _ in samples]
        ss = [0.0] + [float(s) for s, _, _ in samples]
        kinds = [None] + [kind for _, _, kind in samples]
        n = len(qs)
        runs, i = [], 1
        while i < n:
            if kinds[i] == HELD:
                j = i
                while j + 1 < n and kinds[j + 1] == HELD:
                    j += 1
                if j - i >= 1:  # anchors i - 1 and j, inner waypoints i .. j - 1
                    runs.append((i - 1, j))
                i = j + 1
            else:
                i += 1
        before = self._roughness(qs, runs)
        errs = [self._route_error(ss[a], qs[a], list(zip(ss[a + 1:b + 1], qs[a + 1:b + 1])))
                for a, b in runs]
        deviation_before = max([e[0] for e in errs], default=0.0)
        angle_before = max([e[1] for e in errs], default=0.0)
        moves = sweeps = reverted = 0
        frac = min(max(float(tolerance), 0.0), 1.0)
        coarse = (0.25, 0.5, 0.75, 1.0)  # quick checks first; verified, else redone with all
        for a, b in runs:
            original = [q.copy() for q in qs[a:b + 1]]
            for checks in (coarse, self._LINK_T):
                # the run may use what it already uses, at most the full tolerance
                dev0, ang0 = self._route_error(ss[a], qs[a], list(zip(ss[a + 1:b + 1], qs[a + 1:b + 1])))
                lin_tol = max(frac * self.sm.max_deviation, dev0)
                ang_tol = max(frac * self.sm.max_angle, ang0)
                w2 = float(deviation_weight) ** 2
                wv = np.array([1.0 / self.sm.max_deviation] * 3 + [1.0 / self.sm.max_angle] * 3)
                if batch:
                    m, n_sw = self._sweeps_batch(ss, qs, a, b, iterations, lin_tol, ang_tol, w2, wv, checks)
                    moves, sweeps = moves + m, sweeps + n_sw
                    iters = 0  # the one-by-one sweeps below are skipped
                else:
                    iters = max(int(iterations), 0)
                # waypoints whose last try did not move them and whose neighbours have not moved
                # since: the same try again (the bounds only shrink) gives the same answer
                settled = set()
                for _ in range(iters):
                    sweeps += 1
                    moved, largest = 0, 0.0
                    for k in range(a + 1, b):
                        if k in settled:
                            continue
                        settled.add(k)
                        span = ss[k + 1] - ss[k - 1]
                        t = (ss[k] - ss[k - 1]) / span if span > 1e-12 else 0.5
                        target = qs[k - 1] + t * (qs[k + 1] - qs[k - 1])
                        step = target - qs[k]
                        if w2 > 0.0:
                            T, F = self.pose(ss[k]), self.solver.fk(qs[k])
                            e = wv * np.concatenate([T[:3, 3] - F[:3, 3], rotvec(T[:3, :3] @ F[:3, :3].T)])
                            J = wv[:, None] * self.solver.jacobian(qs[k])
                            step = np.linalg.solve(w2 * (J.T @ J) + np.eye(len(step)), w2 * (J.T @ e) + step)
                        for lam in (1.0, 0.5):
                            cand = qs[k] + lam * step
                            if float(np.max(np.abs(cand - qs[k]))) < 1e-7:
                                break
                            if self._margin(cand) >= 0.0 \
                                    and self._links_ok(ss[k - 1], qs[k - 1], ss[k], cand, ss[k + 1], qs[k + 1],
                                                       lin_tol, ang_tol, checks):
                                largest = max(largest, float(np.max(np.abs(cand - qs[k]))))
                                qs[k] = cand
                                moved += 1
                                settled.difference_update((k - 1, k, k + 1))
                                break
                    moves += moved
                    if w2 > 0.0:  # keep what the pull gained
                        lin_now, ang_now = self._route_error(
                            ss[a], qs[a], list(zip(ss[a + 1:b + 1], qs[a + 1:b + 1])))
                        lin_tol, ang_tol = min(lin_tol, lin_now), min(ang_tol, ang_now)
                    if largest < 1e-4:  # [rad] settled
                        break
                lin_new, ang_new = self._route_error(ss[a], qs[a], list(zip(ss[a + 1:b + 1], qs[a + 1:b + 1])))
                if lin_new <= max(frac * self.sm.max_deviation, dev0) + 1e-9 \
                        and ang_new <= max(frac * self.sm.max_angle, ang0) + 1e-9:
                    break
                qs[a:b + 1] = [q.copy() for q in original]  # a coarse check missed a peak: redo with all checks
                reverted += 1
        out = [(s, qs[i + 1], kind) for i, (s, _, kind) in enumerate(samples)]
        info = {"runs": len(runs), "moves": moves, "sweeps": sweeps, "reverted": reverted,
                "roughness_before": before, "roughness_after": self._roughness(qs, runs),
                "deviation_m": deviation_before, "angle_rad": angle_before}
        errs = [self._route_error(ss[a], qs[a], list(zip(ss[a + 1:b + 1], qs[a + 1:b + 1])))
                for a, b in runs]
        info["deviation_after_m"] = max([e[0] for e in errs], default=0.0)
        info["angle_after_rad"] = max([e[1] for e in errs], default=0.0)
        return out, info

    def _deviations_of(self, F: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Like :meth:`_deviations` for poses F (N, 4, 4) already computed."""
        poses_at = getattr(self.path, "poses_at", None)
        P = poses_at(S) if poses_at is not None else np.array([self.pose(float(x)) for x in S])
        d = F[:, :3, 3] - P[:, :3, 3]
        lin = np.sqrt(np.einsum("ij,ij->i", d, d))
        c = 0.5 * (np.einsum("nij,nij->n", F[:, :3, :3], P[:, :3, :3]) - 1.0)
        return lin, np.arccos(np.clip(c, -1.0, 1.0))

    def _run_error(self, S: np.ndarray, Q: np.ndarray, a: int, b: int, ts) -> tuple[float, float]:
        """Max TCP distance / angle from the line along waypoints a..b (arrays S, Q)."""
        t = np.asarray(ts, dtype=float)
        qa, qb = Q[a:b], Q[a + 1:b + 1]
        pts = (qa[:, None, :] + t[None, :, None] * (qb - qa)[:, None, :]).reshape(-1, Q.shape[1])
        sp = (S[a:b, None] + t[None, :] * (S[a + 1:b + 1] - S[a:b])[:, None]).reshape(-1)
        lin, ang = self._deviations_of(self.solver.fk_batch(pts), sp)
        return float(lin.max()), float(ang.max())

    def _sweeps_batch(self, ss, qs, a, b, iterations, lin_tol, ang_tol, w2, wv, checks):
        """Red-black version of the smoothing sweeps of run a..b (updates qs in place).
        Returns (moves, sweeps)."""
        S = np.asarray(ss, dtype=float)
        Q = np.array(qs, dtype=float)
        t = np.asarray(checks, dtype=float)
        nt = len(t)
        K = np.arange(a + 1, b)
        lo, hi = self._lo, self._hi
        moves = sweeps = 0
        for _ in range(max(int(iterations), 0)):
            sweeps += 1
            largest = 0.0
            for parity in (0, 1):
                k = K[(K - a - 1) % 2 == parity]
                if len(k) == 0:
                    continue
                span = S[k + 1] - S[k - 1]
                safe = np.where(span > 1e-12, span, 1.0)
                f = np.where(span > 1e-12, (S[k] - S[k - 1]) / safe, 0.5)
                step = Q[k - 1] + f[:, None] * (Q[k + 1] - Q[k - 1]) - Q[k]
                if w2 > 0.0:
                    F = self.solver.fk_batch(Q[k])
                    poses_at = getattr(self.path, "poses_at", None)
                    T = poses_at(S[k]) if poses_at is not None else np.array([self.pose(float(x)) for x in S[k]])
                    Rerr = T[:, :3, :3] @ np.transpose(F[:, :3, :3], (0, 2, 1))
                    e = wv[None] * np.concatenate([T[:, :3, 3] - F[:, :3, 3], _rotvec_batch(Rerr)], axis=1)
                    J = wv[None, :, None] * self.solver.jacobian_batch(Q[k])
                    Jt = np.transpose(J, (0, 2, 1))
                    A = w2 * (Jt @ J) + np.eye(J.shape[2])[None]
                    rhs = w2 * np.einsum("nij,nj->ni", Jt, e) + step
                    step = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
                todo = np.ones(len(k), dtype=bool)
                for lam in (1.0, 0.5):
                    idx = np.nonzero(todo)[0]
                    if len(idx) == 0:
                        break
                    kk = k[idx]
                    cand = Q[kk] + lam * step[idx]
                    move = np.max(np.abs(cand - Q[kk]), axis=1)
                    tiny = move < 1e-7
                    todo[idx[tiny]] = False  # nothing left to gain here
                    ok = ~tiny & (np.min(cand - lo, axis=1) >= 0.0) & (np.min(hi - cand, axis=1) >= 0.0) \
                        & (np.max(np.abs(cand - Q[kk - 1]), axis=1) <= self.max_step) \
                        & (np.max(np.abs(Q[kk + 1] - cand), axis=1) <= self.max_step)
                    sel = np.nonzero(ok)[0]
                    self._effort += 2 * len(sel)
                    if len(sel) == 0:
                        continue
                    c, k1 = cand[sel], kk[sel]
                    p1 = Q[k1 - 1][:, None, :] + t[None, :, None] * (c - Q[k1 - 1])[:, None, :]
                    p2 = c[:, None, :] + t[None, :, None] * (Q[k1 + 1] - c)[:, None, :]
                    s1 = S[k1 - 1][:, None] + t[None, :] * (S[k1] - S[k1 - 1])[:, None]
                    s2 = S[k1][:, None] + t[None, :] * (S[k1 + 1] - S[k1])[:, None]
                    pts = np.concatenate([p1, p2], axis=1).reshape(-1, Q.shape[1])
                    sp = np.concatenate([s1, s2], axis=1).reshape(-1)
                    lin, ang = self._deviations_of(self.solver.fk_batch(pts), sp)
                    good = ((lin <= lin_tol) & (ang <= ang_tol)).reshape(len(sel), 2 * nt).all(axis=1)
                    acc = sel[good]
                    if len(acc):
                        Q[kk[acc]] = cand[acc]
                        largest = max(largest, float(move[acc].max()))
                        moves += len(acc)
                        todo[idx[acc]] = False
            if w2 > 0.0:  # keep what the pull gained
                lin_now, ang_now = self._run_error(S, Q, a, b, self._LINK_T)
                lin_tol, ang_tol = min(lin_tol, lin_now), min(ang_tol, ang_now)
            if largest < 1e-4:  # [rad] settled
                break
        for i in K:
            qs[i] = Q[i].copy()
        return moves, sweeps

    @staticmethod
    def _roughness(qs: list, runs: list) -> float:
        """Joint direction change through the inner waypoints of held runs per
        joint travel (0: evenly spread; ~2: all motion on every other segment)."""
        turn = travel = 0.0
        for a, b in runs:
            d = np.diff(np.asarray(qs[a:b + 1]), axis=0)
            travel += float(np.sum(np.linalg.norm(d, axis=1)))
            turn += float(np.sum(np.linalg.norm(np.diff(d, axis=0), axis=1)))
        return turn / travel if travel > 1e-12 else 0.0

    _LINK_T = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)

    def _link_ok(self, s_a, q_a, s_b, q_b, lin_tol=None, ang_tol=None, ts=_LINK_T) -> bool:
        """Joint interpolation q_a -> q_b within one joint step and the reconfiguration
        tolerance of the line between s_a and s_b."""
        self._effort += 1
        if np.max(np.abs(q_b - q_a)) > self.max_step:
            return False
        lin_tol = self.sm.max_deviation if lin_tol is None else lin_tol
        ang_tol = self.sm.max_angle if ang_tol is None else ang_tol
        for t in ts:
            lin, ang = deviation(
                self.solver.fk(q_a + t * (q_b - q_a)), self.pose(s_a + t * (s_b - s_a))
            )
            if lin > lin_tol or ang > ang_tol:
                return False
        return True

    def _deviations(self, Q, S) -> tuple[np.ndarray, np.ndarray]:
        """:func:`deviation` of fk(Q[i]) from the line pose at S[i], for many points at
        once (arrays of TCP distance [m] and angle [rad])."""
        memo = self._fk_memo
        if memo is None:
            F = np.array([self.solver.fk(q) for q in Q])
        else:  # the smoothing evaluates the same joint vectors again and again
            F = np.empty((len(Q), 4, 4))
            for i, q in enumerate(Q):
                key = q.tobytes()
                f = memo.get(key)
                if f is None:
                    f = memo[key] = self.solver.fk(q)
                F[i] = f
        P = np.array([self.pose(float(x)) for x in S])
        d = F[:, :3, 3] - P[:, :3, 3]
        lin = np.sqrt(np.einsum("ij,ij->i", d, d))
        c = 0.5 * (np.einsum("nij,nij->n", F[:, :3, :3], P[:, :3, :3]) - 1.0)
        return lin, np.arccos(np.clip(c, -1.0, 1.0))

    def _links_ok(self, s_a, q_a, s_b, q_b, s_c, q_c, lin_tol, ang_tol, ts=_LINK_T) -> bool:
        """Both links a -> b -> c within the bounds, checked together (one batch)."""
        self._effort += 2
        if np.max(np.abs(q_b - q_a)) > self.max_step or np.max(np.abs(q_c - q_b)) > self.max_step:
            return False
        t = np.asarray(ts, dtype=float)
        Q = np.concatenate([q_a + t[:, None] * (q_b - q_a), q_b + t[:, None] * (q_c - q_b)])
        S = np.concatenate([s_a + t * (s_b - s_a), s_b + t * (s_c - s_b)])
        lin, ang = self._deviations(Q, S)
        return bool(np.all(lin <= lin_tol) and np.all(ang <= ang_tol))

    def _route_error(self, s: float, q: np.ndarray, route: list) -> tuple[float, float]:
        """Max TCP distance [m] and angle [rad] from the line along a route (at the
        points :meth:`_link_ok` checks)."""
        if not route:
            return 0.0, 0.0
        t = np.asarray(self._LINK_T)
        Q, S = [], []
        for s1, q1 in route:
            Q.append(q + t[:, None] * (q1 - q))
            S.append(s + t * (s1 - s))
            s, q = s1, q1
        lin, ang = self._deviations(np.concatenate(Q), np.concatenate(S))
        return float(lin.max()), float(ang.max())

    def _project(self, s: float, q: np.ndarray) -> Optional[np.ndarray]:
        """Damped Gauss-Newton from q into half the reconfiguration tolerance of the
        line at s (not onto the exact IK, so branches can be crossed near a
        singularity)."""
        self._effort += 1
        T = self.pose(s)
        w = np.array([1.0 / self.sm.max_deviation] * 3 + [1.0 / self.sm.max_angle] * 3)
        for _ in range(12):
            F = self.solver.fk(q)
            e = w * np.concatenate([T[:3, 3] - F[:3, 3], rotvec(T[:3, :3] @ F[:3, :3].T)])
            if max(np.linalg.norm(e[:3]), np.linalg.norm(e[3:])) <= 0.5:
                return q if self._margin(q) >= 0.0 else None
            J = w[:, None] * self.solver.jacobian(q)
            q = q + J.T @ np.linalg.solve(J @ J.T + np.eye(6), e)
        return None

    def bridge(self, s_a, q_a, s_b, q_b, depth: int = 0) -> Optional[list]:
        """Held joint path q_a (at s_a) -> q_b (at s_b): the joint interpolation
        projected into the reconfiguration tolerance of the line. Waypoints
        [(s, q)] after q_a, ending with (s_b, q_b); None if it leaves the tolerance."""
        if self._link_ok(s_a, q_a, s_b, q_b):
            return [(s_b, q_b)]
        if self.sm.max_deviation <= 0.0 or self.sm.max_angle <= 0.0 \
                or self._effort > self.max_effort or self._late():
            return None
        n = max(2, math.ceil(float(np.max(np.abs(q_b - q_a))) / (0.5 * self.max_step)),
                math.ceil((s_b - s_a) / self.h_max))
        if depth > 0:
            n = 2
        out, prev = [], (s_a, q_a)
        for k in range(1, n + 1):
            s = s_a + k / n * (s_b - s_a)
            # steer from the previous point towards q_b, then back into the tolerance
            q = q_b if k == n else self._project(s, prev[1] + (q_b - prev[1]) / (n - k + 1))
            if q is None:
                return None
            if self._link_ok(prev[0], prev[1], s, q):
                out.append((s, q))
            elif depth < 3:
                sub = self.bridge(prev[0], prev[1], s, q, depth + 1)
                if sub is None:
                    return None
                out += sub
            else:
                return None
            prev = (s, q)
        return out

    def _margin(self, q: np.ndarray) -> float:
        return float(min(np.min(q - self._lo), np.min(self._hi - q)))

    def _route_deviation(self, s: float, q: np.ndarray, route: list) -> float:
        """Max TCP distance [m] from the line along the joint interpolation of a route."""
        dev = 0.0
        for s1, q1 in route:
            for t in (0.25, 0.5, 0.75, 1.0):
                lin, _ = deviation(self.solver.fk(q + t * (q1 - q)), self.pose(s + t * (s1 - s)))
                dev = max(dev, lin)
            s, q = s1, q1
        return dev

    @staticmethod
    def _add(cost: _Cost, q_prev: np.ndarray, points: list) -> _Cost:
        c = _Cost(cost.jumps, cost.max_jump, cost.flips, cost.margin, cost.travel)
        for q in points:
            c.travel += float(np.sum(np.abs(q - q_prev)))
            q_prev = q
        return c

    def _with_margin(self, cost: _Cost, points: list) -> _Cost:
        if points:
            cost.margin = min(cost.margin, min(self._margin(q) for q in points))
        return cost

    # ------------------------------------------------------------ tracing

    def _newton(self, T: np.ndarray, q: np.ndarray) -> Optional[np.ndarray]:
        """Exact IK of T near q (a few Newton steps; None if they do not converge or
        leave the joint limits)."""
        prev = math.inf
        for _ in range(_NEWTON_ITERS):
            F = self.solver.fk(q)
            e = np.concatenate([T[:3, 3] - F[:3, 3], rotvec(T[:3, :3] @ F[:3, :3].T)])
            err = float(e @ e)
            if err < _NEWTON_TOL**2:
                return q if self._margin(q) >= 0.0 else None
            if err > 0.25 * prev:
                return None
            prev = err
            J = self.solver.jacobian(q)
            q = q + np.linalg.lstsq(J, e, rcond=1e-6)[0]
        return None

    def _step(self, s: float, q: np.ndarray, s1: float) -> Optional[np.ndarray]:
        T0, T1 = self.pose(s), self.pose(s1)
        dx = np.concatenate([T1[:3, 3] - T0[:3, 3], rotvec(T1[:3, :3] @ T0[:3, :3].T)])
        dq = np.linalg.lstsq(self.solver.jacobian(q), dx, rcond=1e-4)[0]
        size = float(np.max(np.abs(dq)))
        if size > self.max_step:
            return None
        q1 = self._newton(T1, q + dq)
        if q1 is None or not self._segment_ok(s, q, s1, q1):
            return None
        return q1

    def trace(self, s0: float, q0: np.ndarray, s_end: float = 1.0) -> Trace:
        """Follow the IK branch of ``q0`` from ``s0`` towards ``s_end`` with adaptive
        steps (samples in the order visited)."""
        key = (s0, s_end, q0.tobytes())
        if key in self._traces:
            return self._traces[key]
        d = 1.0 if s_end >= s0 else -1.0
        ss, qs, sig = [s0], [q0], [self.sm.sigma_min(q0)]
        s, q, h = s0, q0, self.h_max
        stuck = False
        while s != s_end:
            s1 = min(s_end, s + h) if d > 0 else max(s_end, s - h)
            q1 = self._step(s, q, s1)
            if q1 is None:
                h *= 0.5
                if h < self.h_min:
                    stuck = True
                    break
                continue
            s, q = s1, q1
            ss.append(s)
            qs.append(q)
            sig.append(self.sm.sigma_min(q))
            h = min(2.0 * h, self.h_max)
        tr = self._traces[key] = Trace(ss, qs, sig, stuck)
        return tr

    def goals(self) -> list[np.ndarray]:
        """Accepted goal configurations within the joint limits."""
        if self._goals is None:
            self._goals = [q for q in self._ik_at(1.0)
                           if self.goal_ok(q) and self._margin(q) >= 0.0]
        return self._goals

    def goal_traces(self) -> list[Trace]:
        """The branches of the goal configurations traced back along the line, as
        traces in increasing s ending at s = 1. Goals differing by full turns of
        joints share one trace (shifted, cut where it leaves the joint limits)."""
        if self._goal_traces is None:
            t0 = time.perf_counter()
            out, base = [], {}
            for g in self.goals():
                turns = np.round(g / _TWO_PI) * _TWO_PI
                key = (tuple(self.solver.branch(g)), tuple(np.round(g - turns, 3)))
                if key not in base:
                    b = self.trace(1.0, g - turns, 0.0)
                    base[key] = (np.asarray(b.s[::-1]), np.asarray(b.q[::-1]),
                                 np.asarray(b.sigma[::-1]))
                ss, qs, sig = base[key]
                qs = qs + turns
                inside = np.all((qs >= self._lo) & (qs <= self._hi), axis=1)
                k = len(qs)
                while k > 0 and inside[k - 1]:
                    k -= 1
                out.append(Trace(list(ss[k:]), list(qs[k:]), list(sig[k:]), False))
            self._goal_traces = out
            self.stats["goal_traces_s"] += time.perf_counter() - t0
        return self._goal_traces

    def _rest(self, q: np.ndarray) -> float:
        """Lower bound of the joint travel from q to an accepted goal."""
        goals = self.goals()
        return min(float(np.sum(np.abs(g - q))) for g in goals) if goals else 0.0

    def links(self, tr: Trace) -> list[tuple[int, Trace, int, list]]:
        """Held reconfigurations from ``tr`` onto the goal branches: (i, goal trace
        b, j, route) where ``route`` [(s, q)] leads from tr.q[i] (at tr.s[i]) to b.q[j]
        (at b.s[j] >= tr.s[i]) within the reconfiguration tolerance of the line,
        at most ``max_held_length`` along it. Pairs are tried in the order of the
        joint travel of the whole motion; the least for each goal branch."""
        return self._links(tr, self.goal_traces())

    def _links(self, tr: Trace, targets: list[Trace]) -> list[tuple[int, Trace, int, list]]:
        qa, sa = np.asarray(tr.q), np.asarray(tr.s)
        done = np.concatenate([[0.0], np.cumsum(np.sum(np.abs(np.diff(qa, axis=0)), axis=1))])
        cand = []
        for n, b in enumerate(targets):
            if not b.s:
                continue
            qb, sb = np.asarray(b.q), np.asarray(b.s)
            seg = np.sum(np.abs(np.diff(qb, axis=0)), axis=1)
            rest = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
            span = sb[None, :] - sa[:, None]
            ii, jj = np.nonzero((span >= 0.0) & (span <= self.held_span))
            if len(ii):
                gap = np.abs(qa[ii] - qb[jj])
                keep = gap.max(axis=1) <= self.max_link_gap
                ii, jj, gap = ii[keep], jj[keep], gap[keep]
            if len(ii):
                cost = done[ii] + gap.sum(axis=1) + rest[jj]
                cand.append(np.column_stack([cost, np.full(len(ii), n), ii, jj]))
        if not cand:
            return []
        cand = np.concatenate(cand)
        cand = cand[np.lexsort((cand[:, 3], cand[:, 2], cand[:, 1], cand[:, 0]))]
        out, linked, failed = [], set(), []
        tries, best = 0, math.inf
        for c, n, i, j in cand:
            n, i, j = int(n), int(i), int(j)
            # the joint interpolation is the shortest route: no later pair saves travel
            if tries >= self.max_link_tries or c >= best or self._effort > self.max_effort \
                    or self._late():
                break
            if n in linked or any(m == n and abs(i - fi) <= 1 and abs(j - fj) <= 1
                                  for m, fi, fj in failed):
                continue
            tries += 1
            b = targets[n]
            route = self.bridge(tr.s[i], tr.q[i], b.s[j], b.q[j])
            if route is not None:
                out.append((int(i), b, int(j), route))
                linked.add(n)
                qs = [tr.q[i]] + [q for _, q in route]
                travel = sum(float(np.sum(np.abs(y - x))) for x, y in zip(qs, qs[1:]))
                best = min(best, c - float(np.sum(np.abs(qa[i] - b.q[j]))) + travel)
            else:
                failed.append((n, i, j))
        return out

    def _shortcut(self, ss: list, qs: list) -> tuple[list, list]:
        """Held route (ss, qs) cut short: from one of its points straight onto the
        branch it ends on, further along the line, where that saves joint travel."""
        after = self.trace(ss[-1], qs[-1])
        found = self._links(Trace(ss, qs, [math.nan] * len(ss), False), [after])
        if not found:
            return ss, qs
        i, _, j, route = found[0]
        if i == len(ss) - 1 and j == 0:
            return ss, qs
        return ss[: i + 1] + [s for s, _ in route], qs[: i + 1] + [q for _, q in route]

    # ------------------------------------------------------------ windows

    def _q_on(self, tr: Trace, i0: int, i1: int, s: float) -> Optional[np.ndarray]:
        seed = np.array([np.interp(s, tr.s[i0 : i1 + 1], [q[j] for q in tr.q[i0 : i1 + 1]])
                         for j in range(len(tr.q[0]))])
        q = self._newton(self.pose(s), seed)
        if q is None or np.max(np.abs(q - seed)) > self.max_step:
            return None
        return q

    def _refine_min(self, tr: Trace, i: int) -> tuple[float, np.ndarray, float]:
        """Golden-section minimum of sigma_min on the branch around sample i."""
        i0, i1 = i - 1, min(i + 1, len(tr.s) - 1)
        best = (tr.sigma[i], tr.s[i], tr.q[i])

        def f(s):
            nonlocal best
            q = self._q_on(tr, i0, i1, s)
            if q is None:
                return math.inf
            v = self.sm.sigma_min(q)
            if v < best[0]:
                best = (v, s, q)
            return v

        a, b = tr.s[i0], tr.s[i1]
        x1, x2 = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
        f1, f2 = f(x1), f(x2)
        for _ in range(14):
            if f1 <= f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - _GOLDEN * (b - a)
                f1 = f(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + _GOLDEN * (b - a)
                f2 = f(x2)
        sigma, s, q = best
        return s, q, sigma

    def windows(self, tr: Trace, end: bool) -> list[_Window]:
        """Points of ``tr`` to reconfigure at: local minima of sigma_min below
        ``window_sigma`` (refined; the line's start counts if it is one), plus its
        end if ``end``."""
        out = []
        n = len(tr.s)
        if tr.s[0] == 0.0 and n > 1 and tr.sigma[0] <= tr.sigma[1] \
                and tr.sigma[0] < self.window_sigma:
            out.append(_Window(tr.s[0], tr.q[0], tr.sigma[0], 1, False))
        for i in range(1, n - 1):
            if tr.sigma[i] <= tr.sigma[i - 1] and tr.sigma[i] <= tr.sigma[i + 1] \
                    and tr.sigma[i] < self.window_sigma:
                s, q, sigma = self._refine_min(tr, i)
                head = i + 1 if s > tr.s[i] else i
                if not self._segment_ok(tr.s[head - 1], tr.q[head - 1], s, q):
                    s, q, sigma, head = tr.s[i], tr.q[i], tr.sigma[i], i
                if s > tr.s[0]:
                    near = ([i] if s != tr.s[i] else []) + [
                        j for k in range(1, n) for j in (i - k, i + k) if 0 < j < n - 1]
                    alts = tuple(_Window(tr.s[j], tr.q[j], tr.sigma[j], j, False, (), True)
                                 for j in near if tr.sigma[j] < self.window_sigma)
                    out.append(_Window(s, q, sigma, head, False, alts[: self.window_alts]))
        if end:
            out.append(_Window(tr.s[-1], tr.q[-1], tr.sigma[-1], n, True))
        return out

    # ------------------------------------------------------------ reconfigurations

    def _attach(self, s: float, w: np.ndarray, family: tuple, ref: np.ndarray):
        """Exact IK of ``family`` at (or just after) s, one joint step from walk point w."""
        for h in (0.0, 2.0 ** -14, 2.0 ** -12, 2.0 ** -10, 2.0 ** -8):
            s1 = min(1.0, s + h)
            r = self._newton(self.pose(s1), w)
            if r is not None and self.sm.family(r, ref) == family and self._link_ok(s, w, s1, r):
                return s1, r
            if s1 >= 1.0:
                break
        return None

    def held_flips(self, win: _Window) -> list[tuple[list, float, np.ndarray, float]]:
        """Held reconfigurations at ``win``: (route waypoints, s after, q after, deviation)."""
        target = self.pose(win.s)
        runs = self.sm.runs(win.q, target)
        out, attached = [], []
        for run in runs:
            hit = None
            for w in run.points:
                hit = self._attach(win.s, w, run.family, win.q)
                if hit is not None:
                    break
            if hit is None:
                continue
            s1, r = hit
            walk = run.prefix + run.points
            for k, w in enumerate(walk):
                if self._link_ok(win.s, w, s1, r):
                    route = walk[: k + 1] + [r]
                    break
            else:
                continue
            lin, _ = self.sm.interpolation_error([win.q] + walk[: k + 1], target)
            out.append((route, s1, r, lin))
            attached.append(run.family)
        self._tried.append((win, [r.family for r in runs], attached))
        return out

    def _ik_at(self, s: float) -> list[np.ndarray]:
        sols = self._ik_cache.get(s)
        if sols is None:
            self._ik_queries += 1
            t0 = time.perf_counter()
            sols = self._ik_cache[s] = self.solver.ik_analytical(self.pose(s))
            self.stats["ik_calls"] += 1
            self.stats["ik_s"] += time.perf_counter() - t0
        return sols

    def jumps(self, win: _Window) -> list[tuple]:
        """Jump targets just past ``win`` (its IK is slow and ambiguous right at the
        singularity): (s, q, points without IK skipped, goal trace or None, index)."""
        ref = win.q
        own = self.sm.family(ref, ref)
        s_n = min(1.0, win.s + self.jump_ahead)
        budget = self._ik_queries < self.max_ik_queries or s_n in self._ik_cache
        ahead = self._ik_at(s_n) if budget else []
        out = [(s_n, c, False, None, 0) for c in ahead
               if (win.end and win.s < 1.0) or self.sm.family(c, ref) != own]
        no_ik = budget and win.end and not ahead
        # onto the goal branches (already traced): one jump reaches the goal
        for b in self.goal_traces():
            j = next((k for k, s in enumerate(b.s) if s >= s_n), None)
            if j is not None and (win.end or self.sm.family(b.q[j], ref) != own):
                out.append((b.s[j], b.q[j], no_ik, b, j))
        if win.end and win.s < 1.0:
            # no IK ahead: jump to where the line is reachable again (a goal branch,
            # traced back to that point, already gives it)
            if no_ik and not any(t[3] is not None for t in out):
                lo, hi = s_n, None
                for k in itertools.count(1):
                    if self._ik_queries >= 2 * self.max_ik_queries:
                        break
                    s = min(1.0, s_n + k * 2.0 ** -6)
                    if self._ik_at(s):
                        hi = s
                        break
                    lo = s
                    if s >= 1.0:
                        break
                if hi is not None:
                    while hi - lo > 1e-4:
                        mid = 0.5 * (lo + hi)
                        if self._ik_at(mid):
                            hi = mid
                        else:
                            lo = mid
                    out += [(hi, c, True, None, 0) for c in self._ik_at(hi)]
        return out

    # ------------------------------------------------------------ search

    def run(self) -> LinePlan:
        jump_mode = self.mode == "jump"
        q0 = self.q0
        counter = itertools.count()
        heap: list = []

        def push(node: _Node) -> None:
            rest = 0.0 if node.terminal else self._rest(node.q)
            heapq.heappush(heap, (node.cost.key(jump_mode, rest), next(counter), node))

        push(_Node(0.0, q0, _Cost(margin=self._margin(q0))))
        seen: set = set()
        expansions = 0
        first: Optional[Trace] = None
        while heap:
            _, _, node = heapq.heappop(heap)
            if node.terminal:
                return LinePlan(node.samples, node.events)
            if expansions >= self.max_expansions or self._late():
                continue  # budget spent: only return plans already complete
            if node.window is not None:
                expansions += 1
                self.stats["expansions"] += 1
                t0 = time.perf_counter()
                self._reconfigure(node, push, jump_mode)
                self.stats["reconfigure_s"] += time.perf_counter() - t0
                continue
            fam = (round(node.s, 4), self.sm.family(node.q, q0))
            if fam in seen:
                continue
            seen.add(fam)
            expansions += 1
            self.stats["expansions"] += 1
            t0 = time.perf_counter()
            tr = self.trace(node.s, node.q)
            self.stats["trace_s"] += time.perf_counter() - t0
            first = first or tr
            line = [(s, q, LINE) for s, q in zip(tr.s[1:], tr.q[1:])]
            reached = not tr.stuck and self.goal_ok(tr.q[-1])
            if reached:
                cost = self._with_margin(self._add(node.cost, node.q, tr.q[1:]), tr.q[1:])
                push(_Node(1.0, tr.q[-1], cost, node.samples + line, node.events, True))
                continue  # any reconfiguration ranks behind
            if self.mode == "raise" or node.cost.flips + node.cost.jumps >= self.max_flips:
                continue
            t0 = time.perf_counter()
            links = self.links(tr)
            self.stats["links_s"] += time.perf_counter() - t0
            for i, b, j, route in links:
                pts = tr.q[1 : i + 1] + [q for _, q in route] + b.q[j + 1 :]
                cost = self._with_margin(self._add(node.cost, node.q, pts), pts)
                cost.flips += 1
                dev = self._route_deviation(tr.s[i], tr.q[i], route)
                change = float(np.max(np.abs(np.asarray([q for _, q in route]) - tr.q[i])))
                ev = Event(HELD, tr.s[i], b.s[j], change, dev, tr.sigma[i])
                samples = (line[:i] + [(s, q, HELD) for s, q in route]
                           + [(s, q, LINE) for s, q in zip(b.s[j + 1 :], b.q[j + 1 :])])
                push(_Node(1.0, b.q[-1], cost, node.samples + samples, node.events + [ev], True))
            t0 = time.perf_counter()
            wins = self.windows(tr, end=True)
            self.stats["windows_s"] += time.perf_counter() - t0
            for win in wins:
                self._push_window(node, line, win, push)
        t0 = time.perf_counter()
        exc = self._failure(first)
        self.stats["failure_s"] = time.perf_counter() - t0
        if self.stats["timed_out"]:
            exc = type(exc)(f"{exc} (the search stopped at its planning time budget)")
        raise exc

    def _push_window(self, node: _Node, line: list, win: _Window, push) -> None:
        # trace samples [1, head) precede the window, then the window point
        head = line[: win.head - 1]
        if not win.end and win.s > node.s:
            head = head + [(win.s, win.q, LINE)]
        pts = [q for _, q, _ in head]
        base = self._with_margin(self._add(node.cost, node.q, pts), pts)
        # lower bound of the reconfigurations at the window, expanded when popped
        bound = _Cost(base.jumps, base.max_jump, base.flips + 1, base.margin, base.travel)
        q_w = pts[-1] if pts else node.q
        push(_Node(win.s, q_w, bound, node.samples + head, node.events, window=win,
                   parent=(node, line)))

    def _reconfigure(self, node: _Node, push, jump_mode: bool) -> None:
        win, q_w = node.window, node.q
        base = _Cost(node.cost.jumps, node.cost.max_jump, node.cost.flips - 1,
                     node.cost.margin, node.cost.travel)
        flips = self.held_flips(win)
        if not flips:
            for alt in win.alts:
                self._push_window(*node.parent, alt, push)
        for route, s1, _, _ in flips:
            ss, route = self._shortcut([win.s] * (len(route) - 1) + [s1], route)
            s1, r = ss[-1], route[-1]
            cost = self._with_margin(self._add(base, q_w, route), route)
            cost.flips += 1
            change = float(np.max(np.abs(np.asarray(route) - q_w)))
            samples = [(s, w, HELD) for s, w in zip(ss, route)]
            dev = self._route_deviation(win.s, q_w, list(zip(ss, route)))
            ev = Event(HELD, win.s, s1, change, dev, win.sigma)
            push(_Node(s1, r, cost, node.samples + samples, node.events + [ev]))
        if jump_mode and not win.held_only:
            for s1, c, no_ik, b, j in self.jumps(win):
                # onto a goal branch: its traced rest reaches the goal (terminal)
                tail = [] if b is None else list(zip(b.s[j + 1 :], b.q[j + 1 :]))
                pts = [c] + [q for _, q in tail]
                cost = self._with_margin(self._add(base, q_w, pts), pts)
                cost.jumps += 1
                change = float(np.max(np.abs(c - q_w)))
                cost.max_jump = max(cost.max_jump, change)
                ev = Event(JUMP, win.s, s1, change, None, win.sigma, no_ik)
                samples = [(s1, c, JUMP)] + [(s, q, LINE) for s, q in tail]
                push(_Node(1.0 if b is not None else s1, pts[-1], cost,
                           node.samples + samples, node.events + [ev], b is not None))

    # ------------------------------------------------------------ diagnosis

    def _why_end(self, tr: Trace) -> str:
        q, s = tr.q[-1], tr.s[-1]
        if not tr.stuck:
            return "it reaches the goal in another configuration than requested"
        reasons = []
        at_limit = [j + 1 for j in range(len(q))
                    if min(q[j] - self._lo[j], self._hi[j] - q[j]) < 1e-2]
        if at_limit:
            reasons.append("joint " + ", ".join(map(str, at_limit)) + " at its limit")
        if tr.sigma[-1] < self.window_sigma:
            v = np.abs(self.sm.weakest_direction(q))
            joints = ", ".join(str(j + 1) for j in np.flatnonzero(v >= 0.3 * v.max()))
            reasons.append(f"singularity: a motion of joints {joints} barely moves the TCP")
        if not self._ik_at(min(1.0, s + 2.0 ** -8)):
            reasons.append("no IK solution beyond it")
        return ", ".join(reasons) or "IK does not converge"

    def _failure(self, tr: Optional[Trace]) -> Exception:
        s_end = tr.s[-1] if tr else 0.0
        why = self._why_end(tr) if tr else "search budget exhausted"
        if tr is not None and tr.stuck and not self._ik_at(min(1.0, s_end + 2.0 ** -8)) \
                and self.mode != "jump":
            return IKError(
                f"no IK solution on the LINE_T path near s={s_end:.4f} "
                f"(the line leaves the reachable workspace; {why})"
            )
        if self.mode == "raise":
            return PathDiscontinuityError(
                f"LINE_T path needs discontinuous joint motion near s={s_end:.4f}: the "
                f"start branch ends there ({why}); it crosses a singularity or a joint "
                "limit blocks it (on_discontinuity 'hold' / 'jump' reconfigure)"
            )
        tried = []
        for win, reached, attached in self._tried:
            if attached:
                note = "reconfigured, but no continuation reaches the goal"
            elif reached:
                note = "self-motion reaches other branches, but none continues on the line"
            else:
                note = f"no self-motion (singularity distance {win.sigma:.3g})"
            tried.append(f"s={win.s:.4f}: {note}")
        tried_msg = "; ".join(list(dict.fromkeys(tried))[:6]) or "no point close to a singularity"
        targets = [b for b in self.goal_traces() if b.s]
        if targets:
            goal_msg = (
                f"no held stretch of at most {self.max_held_length * 1000:g} mm along the line "
                f"joins it to the goal's {len(targets)} configuration(s), whose branches "
                f"reach back to s={min(b.s[0] for b in targets):.4f}"
            )
        else:
            goal_msg = "the goal pose has no allowed configuration within the joint limits"
        return PathDiscontinuityError(
            f"LINE_T path cannot be followed with the TCP held within "
            f"{self.sm.max_deviation * 1000:g} mm / {math.degrees(self.sm.max_angle):g} deg: "
            f"the start branch ends near s={s_end:.4f} ({why}); {goal_msg}; reconfigurations "
            f"tried: {tried_msg}. Use on_discontinuity 'jump' (motion code 20) to allow a "
            "joint jump."
        )
