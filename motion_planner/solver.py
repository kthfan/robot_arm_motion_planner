"""Kinematics solver interface (FK / IK), independent of the backing library."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

IKSelector = Callable[[Sequence[np.ndarray], np.ndarray], np.ndarray]


class IKError(RuntimeError):
    """Raised when no inverse-kinematics solution can be found."""


class KinematicsSolver(ABC):
    """Forward and inverse kinematics for a serial arm.

    Poses are 4x4 homogeneous transforms of the TCP in the robot base frame.
    Joint vectors are numpy arrays in radians.
    """

    @property
    @abstractmethod
    def lower_limits(self) -> np.ndarray:
        """Lower joint position limits [rad]."""

    @property
    @abstractmethod
    def upper_limits(self) -> np.ndarray:
        """Upper joint position limits [rad]."""

    @property
    @abstractmethod
    def velocity_limits(self) -> np.ndarray:
        """Maximum joint speeds [rad/s] (100% PTP speed)."""

    @property
    def nq(self) -> int:
        return len(self.lower_limits)

    @abstractmethod
    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Return the TCP pose for joint vector ``q``."""

    def fk_batch(self, Q: np.ndarray) -> np.ndarray:
        """TCP poses (N, 4, 4) of N joint vectors (default: :meth:`fk` per row)."""
        return np.array([self.fk(q) for q in np.atleast_2d(Q)])

    def jacobian_batch(self, Q: np.ndarray) -> np.ndarray:
        """Jacobians (N, 6, nq) of N joint vectors (default: :meth:`jacobian` per row)."""
        return np.array([self.jacobian(q) for q in np.atleast_2d(Q)])

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """6 x nq Jacobian of the TCP twist (linear velocity, angular velocity)
        in base-frame axes. Defaults to finite differences of :meth:`fk`."""
        from scipy.spatial.transform import Rotation

        q = np.asarray(q, dtype=float)
        T = self.fk(q)
        h = 1e-7
        J = np.empty((6, len(q)))
        for j in range(len(q)):
            x = q.copy()
            x[j] += h
            Tx = self.fk(x)
            J[:3, j] = (Tx[:3, 3] - T[:3, 3]) / h
            J[3:, j] = Rotation.from_matrix(Tx[:3, :3] @ T[:3, :3].T).as_rotvec() / h
        return J

    def branch(self, q: Sequence[float]) -> tuple[int, int, int]:
        """(shoulder, elbow, wrist) configuration signs (+1 / -1) of ``q``; the
        analytical IK solutions of a pose differ in it, up to 2 pi joint turns."""
        raise NotImplementedError(f"{type(self).__name__} does not define IK branches")

    @abstractmethod
    def ik(self, target: np.ndarray, seed: Sequence[float]) -> Optional[np.ndarray]:
        """Solve IK locally from ``seed``; return ``None`` if it does not converge."""

    def ik_all(
        self,
        target: np.ndarray,
        seeds: Iterable[Sequence[float]] = (),
        n_random: int = 32,
        rng_seed: int = 0,
        tol: float = 1e-3,
    ) -> list[np.ndarray]:
        """Collect distinct IK solutions via multi-start from ``seeds`` plus random seeds."""
        rng = np.random.default_rng(rng_seed)
        all_seeds = [np.asarray(s, dtype=float) for s in seeds]
        all_seeds += [
            rng.uniform(self.lower_limits, self.upper_limits) for _ in range(n_random)
        ]
        solutions: list[np.ndarray] = []
        for seed in all_seeds:
            q = self.ik(target, seed)
            if q is not None and all(np.max(np.abs(q - s)) > tol for s in solutions):
                solutions.append(q)
        return solutions

    def ik_candidates(
        self, target: np.ndarray, seeds: Iterable[Sequence[float]] = ()
    ) -> list[np.ndarray]:
        """As many distinct IK solutions as possible, e.g. for choosing a joint
        reconfiguration. Defaults to :meth:`ik_all`; solvers with an analytical
        IK add its solutions."""
        return self.ik_all(target, seeds=seeds)

    def ik_analytical(self, target: np.ndarray) -> list[np.ndarray]:
        """All IK solutions within the joint limits (2 pi variants included), fast and
        deterministic where the solver has an analytical IK. Defaults to :meth:`ik_all`."""
        return self.ik_all(target)


def closest_to_reference(solutions: Sequence[np.ndarray], q_ref: np.ndarray) -> np.ndarray:
    """Default IK selector: the solution with the smallest joint distance to ``q_ref``."""
    q_ref = np.asarray(q_ref, dtype=float)
    return min(solutions, key=lambda q: float(np.linalg.norm(q - q_ref)))
