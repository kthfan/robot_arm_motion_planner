"""Cartesian paths parameterised by s in [0, 1]."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation


class CartesianPath(ABC):
    """A TCP path from ``pose_at(0)`` to ``pose_at(1)``.

    Subclasses define the geometric constraint (line, arc, ...).
    """

    @abstractmethod
    def pose_at(self, s: float) -> np.ndarray:
        """Return the 4x4 pose at path parameter ``s`` in [0, 1]."""

    @property
    @abstractmethod
    def translation_length(self) -> float:
        """Length of the TCP translation [m]."""

    @property
    @abstractmethod
    def rotation_angle(self) -> float:
        """Total TCP rotation angle [rad]."""

    @staticmethod
    def from_poses(start: np.ndarray, goal: np.ndarray) -> "CartesianPath":
        """Build the path whose constraint is implied by the start and goal poses.

        Translation is a straight line and rotation is about a single fixed axis
        (slerp). A pure translation therefore keeps the orientation, and a pure
        rotation keeps the TCP position.
        """
        return LinearPath(start, goal)


class LinearPath(CartesianPath):
    """Straight-line translation with simultaneous slerp of orientation."""

    def __init__(self, start: np.ndarray, goal: np.ndarray) -> None:
        self._p0 = np.asarray(start, dtype=float)[:3, 3].copy()
        self._p1 = np.asarray(goal, dtype=float)[:3, 3].copy()
        rots = Rotation.from_matrix(
            np.stack([np.asarray(start)[:3, :3], np.asarray(goal)[:3, :3]])
        )
        # slerp: R(s) = R0 exp(s [w]x) with w the rotation vector of R0^T R1
        self._R0 = rots[0].as_matrix()
        w = (rots[0].inv() * rots[1]).as_rotvec()
        self._angle = float(np.linalg.norm(w))
        axis = w / self._angle if self._angle > 0.0 else np.zeros(3)
        self._K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                            [-axis[1], axis[0], 0.0]])
        self._K2 = self._K @ self._K

    def poses_at(self, S: np.ndarray) -> np.ndarray:
        """:meth:`pose_at` for many path parameters at once (N, 4, 4)."""
        S = np.asarray(S, dtype=float)
        if S.size and (S.min() < 0.0 or S.max() > 1.0):
            raise ValueError("path parameters must be in [0, 1]")
        a = S * self._angle
        M = (np.eye(3)[None] + np.sin(a)[:, None, None] * self._K[None]
             + (1.0 - np.cos(a))[:, None, None] * self._K2[None])
        T = np.zeros((len(S), 4, 4))
        T[:, :3, :3] = self._R0[None] @ M
        T[:, :3, 3] = (1.0 - S)[:, None] * self._p0[None] + S[:, None] * self._p1[None]
        T[:, 3, 3] = 1.0
        return T

    def pose_at(self, s: float) -> np.ndarray:
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"path parameter must be in [0, 1], got {s}")
        a = s * self._angle
        T = np.eye(4)
        T[:3, :3] = self._R0 @ (np.eye(3) + np.sin(a) * self._K + (1.0 - np.cos(a)) * self._K2)
        T[:3, 3] = (1.0 - s) * self._p0 + s * self._p1
        return T

    @property
    def translation_length(self) -> float:
        return float(np.linalg.norm(self._p1 - self._p0))

    @property
    def rotation_angle(self) -> float:
        return self._angle
