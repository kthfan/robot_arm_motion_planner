"""Pinocchio-backed kinematics solver."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pinocchio as pin
from eaik.IK_HP import HPRobot

from motion_planner.robust_eaik import RobustIK
from motion_planner.solver import KinematicsSolver

DEFAULT_URDF = Path(__file__).parent / "models" / "tm5s.urdf"
DEFAULT_CALIBRATION = Path(__file__).parent / "models" / "tm5s_calibration.json"


class PinocchioSolver(KinematicsSolver):
    """FK and damped least-squares IK using pinocchio.

    Args:
        urdf_path: Robot URDF. Defaults to the bundled TM5S model.
        ee_frame: URDF frame the TCP is attached to.
        tcp_offset: Optional 4x4 TCP transform relative to ``ee_frame``.
        tol: IK convergence threshold on the SE3 log error.
        max_iter: IK iteration cap per solve.
        calibration: Optional :class:`~motion_planner.calibration.KinematicCalibration`
            or path to a calibration JSON, applied on top of the URDF.
        eaik_parallel_tol: Axes closer than this to parallel [rad] are treated as
            parallel by the analytical IK (:attr:`analytical_ik`).
        eaik_intersect_tol: Axes closer than this to intersecting [m] are treated
            as intersecting by the analytical IK.
    """

    def __init__(
        self,
        urdf_path: Union[str, Path] = DEFAULT_URDF,
        ee_frame: str = "flange",
        tcp_offset: Optional[np.ndarray] = None,
        tol: float = 1e-10,
        max_iter: int = 200,
        calibration=None,
        eaik_parallel_tol: float = 5e-3,
        eaik_intersect_tol: float = 2e-3,
    ) -> None:
        self._model = pin.buildModelFromUrdf(str(urdf_path))
        if calibration is not None:
            from motion_planner.calibration import KinematicCalibration

            if not isinstance(calibration, KinematicCalibration):
                calibration = KinematicCalibration.load(calibration)
            calibration.apply(self._model)
        if not self._model.existFrame(ee_frame):
            raise ValueError(f"frame '{ee_frame}' not found in {urdf_path}")
        parent_id = self._model.getFrameId(ee_frame)
        parent = self._model.frames[parent_id]
        offset = pin.SE3.Identity() if tcp_offset is None else pin.SE3(np.asarray(tcp_offset))
        self._fid = self._model.addFrame(
            pin.Frame(
                "_tcp",
                parent.parentJoint,
                parent_id,
                parent.placement * offset,
                pin.FrameType.OP_FRAME,
            )
        )
        self._data = self._model.createData()
        self._lower = self._model.lowerPositionLimit.copy()
        self._upper = self._model.upperPositionLimit.copy()
        self._vel = self._model.velocityLimit.copy()
        self._tol = tol
        self._max_iter = max_iter
        self._eaik_tols = (eaik_parallel_tol, eaik_intersect_tol)
        self._analytical: Optional[RobustIK] = None

    @property
    def analytical_ik(self) -> RobustIK:
        """EAIK analytical IK of this model (calibration and TCP included),
        built on first use.

        Raises:
            ValueError: EAIK has no decomposition for this kinematic chain.
        """
        if self._analytical is None:
            self._analytical = self._build_analytical_ik()
        return self._analytical

    def _build_analytical_ik(self) -> RobustIK:
        model, data = self._model, self._model.createData()
        if model.nq != model.nv or model.nv != model.njoints - 1:
            raise ValueError("EAIK needs a chain of single-DoF revolute joints")
        q0 = pin.neutral(model)
        pin.computeJointJacobians(model, data, q0)
        pin.framesForwardKinematics(model, data, q0)
        # Joint axes (angular part of the world Jacobian) and origins at q = 0.
        H = pin.getJointJacobian(model, data, model.njoints - 1, pin.WORLD)[3:].T.copy()
        origins = np.array([data.oMi[j].translation for j in range(1, model.njoints)])
        tip = data.oMf[self._fid]
        P = np.vstack([origins[0], np.diff(origins, axis=0), tip.translation - origins[-1]])
        parallel_tol, intersect_tol = self._eaik_tols
        robot = HPRobot(
            H, P, R6T=tip.rotation.copy(),
            parallel_tol=parallel_tol, intersect_tol=intersect_tol,
        )
        if not robot.hasKnownDecomposition():
            raise ValueError(
                f"EAIK has no decomposition for this model ({robot.getKinematicFamily()}); "
                f"try larger eaik_parallel_tol / eaik_intersect_tol (now {parallel_tol:g} rad / "
                f"{intersect_tol:g} m)"
            )
        return RobustIK(robot, self._lower, self._upper)

    @property
    def lower_limits(self) -> np.ndarray:
        return self._lower

    @property
    def upper_limits(self) -> np.ndarray:
        return self._upper

    @property
    def velocity_limits(self) -> np.ndarray:
        return self._vel

    def _placement(self, q: np.ndarray) -> pin.SE3:
        pin.framesForwardKinematics(self._model, self._data, q)
        return self._data.oMf[self._fid]

    def fk(self, q: Sequence[float]) -> np.ndarray:
        return self._placement(np.asarray(q, dtype=float)).homogeneous.copy()

    def _batch_model(self) -> Optional[RobustIK]:
        """The analytical IK's numpy chain (same model as pinocchio, built from it)
        for batched FK / Jacobians; None if unavailable or not matching."""
        if getattr(self, "_batch_ok", None) is None:
            self._batch_ok = False
            try:
                ik = self.analytical_ik
                if len(ik.free) == self.nq:
                    Q = np.random.default_rng(0).uniform(-3.0, 3.0, (4, self.nq))
                    R, p, J = ik._fk_jac_batch(Q)
                    F = np.array([self.fk(q) for q in Q])
                    Jp = np.array([self.jacobian(q) for q in Q])
                    self._batch_ok = bool(np.max(np.abs(R - F[:, :3, :3])) < 1e-9
                                          and np.max(np.abs(p - F[:, :3, 3])) < 1e-9
                                          and np.max(np.abs(J - Jp)) < 1e-9)
            except Exception:  # no EAIK decomposition: loop over pinocchio
                self._batch_ok = False
        return self._analytical if self._batch_ok else None

    def fk_batch(self, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        ik = self._batch_model()
        if ik is None:
            return np.array([self.fk(q) for q in Q])
        R, p, _ = ik._fk_jac_batch(Q, jac=False)
        T = np.zeros((len(Q), 4, 4))
        T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = R, p, 1.0
        return T

    def jacobian_batch(self, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        ik = self._batch_model()
        if ik is None:
            return np.array([self.jacobian(q) for q in Q])
        return ik._fk_jac_batch(Q)[2]

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        return pin.computeFrameJacobian(
            self._model, self._data, np.asarray(q, dtype=float), self._fid,
            pin.LOCAL_WORLD_ALIGNED,
        ).copy()

    def branch(self, q: Sequence[float]) -> tuple[int, int, int]:
        """Elbow = sign(q3), wrist = sign(q5); shoulder = side of joint 1's axis
        the wrist point (axis 6 where it passes axis 5) is on, along the arm's
        reach (perpendicular to joint 2's axis)."""
        q = np.asarray(q, dtype=float)
        pin.computeJointJacobians(self._model, self._data, q)
        axes, origins = [], []
        for j in (1, 2, 5, 6):
            axes.append(pin.getJointJacobian(self._model, self._data, j, pin.WORLD)[3:, j - 1])
            origins.append(self._data.oMi[j].translation)
        (a1, a2, a5, a6), (o1, _, o5, o6) = axes, origins
        w = o6 - o5
        b = a6 @ a5
        t = (b * (a5 @ w) - a6 @ w) / (1.0 - b * b)
        wrist_point = o6 + t * a6
        reach = np.cross(a2, a1)
        shoulder = 1 if (wrist_point - o1) @ reach >= 0.0 else -1
        return shoulder, 1 if q[2] >= 0.0 else -1, 1 if q[4] >= 0.0 else -1

    def ik_analytical(self, target: np.ndarray) -> list[np.ndarray]:
        return [np.asarray(q, dtype=float) for q in self.analytical_ik.ik(target).Q]

    def ik_candidates(
        self, target: np.ndarray, seeds: Iterable[Sequence[float]] = ()
    ) -> list[np.ndarray]:
        """Analytical (EAIK) solutions plus the distinct ones of :meth:`ik_all`."""
        solutions = [np.asarray(q, dtype=float) for q in self.analytical_ik.ik(target).Q]
        for q in self.ik_all(target, seeds=seeds):
            if all(np.max(np.abs(q - s)) > 1e-3 for s in solutions):
                solutions.append(q)
        return solutions

    def ik(self, target: np.ndarray, seed: Sequence[float]) -> Optional[np.ndarray]:
        oMdes = pin.SE3(np.asarray(target, dtype=float))
        q = np.clip(np.asarray(seed, dtype=float), self._lower, self._upper)
        for _ in range(self._max_iter):
            iMd = self._placement(q).actInv(oMdes)
            err = pin.log6(iMd).vector
            err_sq = float(err @ err)
            if err_sq < self._tol**2:
                return q
            J = pin.computeFrameJacobian(self._model, self._data, q, self._fid)
            J = -pin.Jlog6(iMd.inverse()) @ J
            # Levenberg-Marquardt damping: robust far away / near singularities,
            # vanishes near the solution for fast final convergence.
            damping = 1e-8 + 1e-2 * err_sq
            dq = -J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            step = np.max(np.abs(dq))
            if step > 0.5:
                dq *= 0.5 / step
            q = np.clip(pin.integrate(self._model, q, dq), self._lower, self._upper)
        return None
