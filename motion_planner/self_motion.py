"""Self-motion: moving the joints while the TCP pose is (approximately) held.

Near a singularity the Jacobian loses rank and the arm can move without moving
the TCP. LINE_T uses such motions to reconfigure the joints (wrist, elbow or
shoulder flip) with the TCP held at a point of the line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from motion_planner.solver import KinematicsSolver


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    c = max(-1.0, min(1.0, 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1.0)))
    angle = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * w
    if angle < math.pi - 1e-4:
        return angle / (2.0 * math.sin(angle)) * w
    # near pi: axis from the symmetric part
    B = 0.5 * (R + np.eye(3))
    k = int(np.argmax(np.diag(B)))
    axis = B[:, k] / math.sqrt(max(B[k, k], 1e-300))
    if axis @ w < 0:
        axis = -axis
    return angle * axis


def deviation(T: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(translation [m], rotation [rad]) between two poses."""
    lin = float(np.linalg.norm(T[:3, 3] - target[:3, 3]))
    c = 0.5 * (float(np.sum(T[:3, :3] * target[:3, :3])) - 1.0)
    return lin, math.acos(max(-1.0, min(1.0, c)))


class SelfMotion:
    """Self-motion walks and held reconfigurations for one solver and tolerances.

    Pose errors weigh rotation by ``linear_tolerance / angular_tolerance`` [m/rad],
    so a pose that can only be held approximately (next to, not on, a
    singularity) spreads the error over position and orientation in
    proportion to their tolerances.
    """

    def __init__(
        self,
        solver: KinematicsSolver,
        linear_tolerance: float,
        angular_tolerance: float,
        max_joint_step: float,
        max_deviation: float,
        max_angle: float,
    ) -> None:
        self.solver = solver
        self.rot_scale = linear_tolerance / angular_tolerance
        self.max_joint_step = max_joint_step
        self.max_deviation = max_deviation
        self.max_angle = max_angle
        self._w = np.array([1.0, 1.0, 1.0] + [self.rot_scale] * 3)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Tolerance-scaled TCP Jacobian."""
        return self._w[:, None] * self.solver.jacobian(q)

    def sigma_min(self, q: np.ndarray) -> float:
        """Smallest singular value of :meth:`jacobian` [m/rad]: distance to a singularity."""
        return float(np.linalg.svd(self.jacobian(q), compute_uv=False)[-1])

    def error(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Scaled pose error to ``target``; error(q + dq) ~ error(q) - jacobian(q) dq."""
        T = self.solver.fk(q)
        rot = rotvec(target[:3, :3] @ T[:3, :3].T)
        return np.concatenate([target[:3, 3] - T[:3, 3], self.rot_scale * rot])

    def interpolation_error(
        self, points: Sequence[np.ndarray], target: np.ndarray
    ) -> tuple[float, float]:
        """Max (translation [m], rotation [rad]) deviation from ``target`` along the
        joint interpolation through ``points`` (start excluded)."""
        lin = ang = 0.0
        for a, b in zip(points, points[1:]):
            for t in (0.25, 0.5, 0.75, 1.0):
                e_lin, e_ang = deviation(self.solver.fk(a + t * (b - a)), target)
                lin, ang = max(lin, e_lin), max(ang, e_ang)
        return lin, ang

    def hold_pose(
        self, q: np.ndarray, target: np.ndarray, frozen: np.ndarray, iterations: int = 30
    ) -> np.ndarray:
        """Gauss-Newton towards ``target`` without moving along unit direction ``frozen``
        (least-squares pose if ``target`` is not exactly reachable)."""
        P = np.eye(len(q)) - np.outer(frozen, frozen)
        lo, hi = self.solver.lower_limits, self.solver.upper_limits
        q = np.array(q, dtype=float)
        for _ in range(iterations):
            dq = P @ np.linalg.lstsq(self.jacobian(q) @ P, self.error(q, target), rcond=1e-6)[0]
            size = float(np.max(np.abs(dq)))
            if size > self.max_joint_step:
                dq *= self.max_joint_step / size
            q = np.clip(q + dq, lo, hi)
            if size < 1e-10:
                break
        return q

    def walk(self, q0: np.ndarray, target: np.ndarray, sign: float) -> list[tuple[np.ndarray, float]]:
        """Waypoints along the self-motion at ``target`` from ``q0`` (included), each
        with the max TCP deviation [m] up to it. Stops where the motion stalls (joint
        limit), the pose can no longer be held (no singularity) or the loop closes."""
        step = 0.5 * self.max_joint_step
        q, v_prev, worst = np.asarray(q0, dtype=float), None, 0.0
        walk = [(q, 0.0)]
        for _ in range(int(math.ceil(4 * math.pi / step))):
            v = np.linalg.svd(self.jacobian(q))[2][-1]
            if v_prev is None:
                v = sign * v
            elif v @ v_prev < 0:
                v = -v
            q_next = self.hold_pose(q + step * v / np.max(np.abs(v)), target, v)
            moved = float(np.max(np.abs(q_next - q)))
            if not 0.1 * step <= moved <= self.max_joint_step:
                break
            lin, ang = self.interpolation_error([q, q_next], target)
            if lin > self.max_deviation or ang > self.max_angle:
                break
            worst = max(worst, lin)
            walk.append((q_next, worst))
            q, v_prev = q_next, v
            if len(walk) > 3 and np.max(np.abs(q - q0)) <= step:
                break
        return walk

    def family(self, q: np.ndarray, q_ref: np.ndarray) -> tuple:
        """Configuration family of ``q``: IK branch signs and the 2 pi turn of the
        unbounded joints relative to ``q_ref``."""
        turns = np.round((np.asarray(q) - q_ref)[list(_TURN_JOINTS)] / (2.0 * math.pi))
        return tuple(self.solver.branch(q)) + tuple(int(t) for t in turns)

    def runs(self, q0: np.ndarray, target: np.ndarray) -> list[Run]:
        """Other configuration families the self-motion at ``target`` reaches from
        ``q0``, walking both ways; each run lists the walk up to it and its points.
        A family reached both ways keeps the shorter walk."""
        if self.max_deviation <= 0.0:
            return []
        q0 = np.asarray(q0, dtype=float)
        start = self.family(q0, q0)
        best: dict = {}
        for sign in (1.0, -1.0):
            walk = self.walk(q0, target, sign)
            k = 1
            while k < len(walk):
                key = self.family(walk[k][0], q0)
                j = k
                while j + 1 < len(walk) and self.family(walk[j + 1][0], q0) == key:
                    j += 1
                if key != start and (key not in best or k < len(best[key].prefix)):
                    best[key] = Run(
                        key,
                        [q for q, _ in walk[1:k]],
                        [q for q, _ in walk[k : j + 1]],
                        [d for _, d in walk[k : j + 1]],
                    )
                k = j + 1
        return list(best.values())


_TURN_JOINTS = (0, 3, 5)


@dataclass
class Run:
    """Walk points of one configuration family reached by a self-motion walk.

    ``prefix``: walk waypoints after the start before the run; ``points``: the
    run's waypoints; ``deviation[i]``: max TCP deviation [m] up to ``points[i]``.
    """

    family: tuple
    prefix: list
    points: list
    deviation: list
