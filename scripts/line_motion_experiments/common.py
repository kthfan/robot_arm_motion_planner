"""Shared helpers for the LINE_T experiments: planner, cases, independent checker."""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# LINE_EXP_ROOT: plan with the motion_planner package of another checkout (e.g. a baseline).
ROOT = Path(os.environ.get("LINE_EXP_ROOT") or Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def get_planner():
    from motion_planner import default_planner

    return default_planner("pinocchio")


@dataclass
class Case:
    name: str
    suite: str
    q0: list
    goal: list  # 4x4 nested list
    goal_joints: list | None = None
    note: str = ""

    @property
    def goal_matrix(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=float)


def save_cases(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in cases]), encoding="utf-8")


def load_cases(path: Path) -> list[Case]:
    return [Case(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


# ------------------------------------------------------------------ checker


@dataclass
class Check:
    goal_lin_mm: float = math.nan
    goal_ang_deg: float = math.nan
    max_lin_mm: float = 0.0
    max_ang_deg: float = 0.0
    jump_segments: int = 0
    max_jump_deg: float = 0.0
    within_limits: bool = True
    max_step_deg: float = 0.0
    travel_deg: float = 0.0
    excess_travel_deg: float = 0.0
    backtrack_mm: float = 0.0
    branches: list = field(default_factory=list)
    flip_s: list = field(default_factory=list)


class Checker:
    """Checks a joint waypoint list against the line, independently of the planner:
    the robot interpolates joints linearly between waypoints, so every segment is
    sampled densely through FK."""

    def __init__(self, planner, samples: int = 8) -> None:
        self.p = planner
        self.solver = planner.solver
        self.samples = samples

    def run(self, q0, goal: np.ndarray, qs: np.ndarray) -> Check:
        sv, p = self.solver, self.p
        T0 = sv.fk(q0)
        p0, p1 = T0[:3, 3], goal[:3, 3]
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / L if L > 1e-9 else np.zeros(3)
        rots = Rotation.from_matrix(np.stack([T0[:3, :3], goal[:3, :3]]))
        slerp = Slerp([0.0, 1.0], rots)
        grid = np.linspace(0.0, 1.0, 2001)
        grid_pos = p0[None, :] + grid[:, None] * (p1 - p0)[None, :]
        grid_rots_inv = slerp(grid).inv()

        c = Check()
        qs = np.asarray(qs, dtype=float)
        lin_g, ang_g = _pose_distance(sv.fk(qs[-1]), goal)
        c.goal_lin_mm, c.goal_ang_deg = lin_g * 1e3, math.degrees(ang_g)
        c.within_limits = bool(
            np.all(qs >= sv.lower_limits - 1e-9) and np.all(qs <= sv.upper_limits + 1e-9)
        )
        d = np.diff(qs, axis=0)
        if len(d):
            c.max_step_deg = math.degrees(float(np.max(np.abs(d))))
        c.travel_deg = math.degrees(float(np.sum(np.abs(d))))
        c.excess_travel_deg = c.travel_deg - math.degrees(float(np.sum(np.abs(qs[-1] - qs[0]))))

        flip_lin = p.max_flip_deviation + 1e-4
        flip_ang = p.max_flip_angle + 1e-4
        s_max = 0.0
        for qa, qb in zip(qs, qs[1:]):
            seg_lin = seg_ang = 0.0
            for t in np.arange(1, self.samples + 1) / self.samples:
                T = sv.fk(qa + t * (qb - qa))
                pos, R = T[:3, 3], Rotation.from_matrix(T[:3, :3])
                d_pos = np.linalg.norm(grid_pos - pos, axis=1)
                d_ang = (grid_rots_inv * R).magnitude()
                k = int(np.argmin(np.maximum(d_pos / flip_lin, d_ang / flip_ang)))
                ang = float(d_ang[k])
                s = float(grid[k])
                if L > 1e-6:
                    lo_s, hi_s = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
                    s = float(np.clip((pos - p0) @ u / L, lo_s, hi_s))
                lin = float(np.linalg.norm(pos - (p0 + s * (p1 - p0))))
                seg_lin, seg_ang = max(seg_lin, lin), max(seg_ang, ang)
                c.backtrack_mm = max(c.backtrack_mm, (s_max - s) * max(L, 1e-9) * 1e3)
                s_max = max(s_max, s)
            c.max_lin_mm = max(c.max_lin_mm, seg_lin * 1e3)
            c.max_ang_deg = max(c.max_ang_deg, math.degrees(seg_ang))
            if seg_lin > flip_lin or seg_ang > flip_ang:
                c.jump_segments += 1
                c.max_jump_deg = max(c.max_jump_deg, math.degrees(float(np.max(np.abs(qb - qa)))))

        # branch sequence and where it changes (path parameter of the waypoint)
        prev = None
        for q in qs:
            b = sv.branch(q)
            if b != prev:
                if prev is not None:
                    pos = sv.fk(q)[:3, 3]
                    c.flip_s.append(round(float((pos - p0) @ u / L), 4) if L > 1e-6 else math.nan)
                c.branches.append(list(b))
                prev = b
        return c


def _pose_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    lin = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    ang = float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
    return lin, ang


def joint_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance [rad, max-norm] between two waypoint lists."""
    a, b = np.asarray(a), np.asarray(b)
    d = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))
