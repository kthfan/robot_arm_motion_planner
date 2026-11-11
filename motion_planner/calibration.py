"""Kinematic calibration of the pinocchio robot model from joint/pose pairs.

The model is corrected by replacing the placement of every joint (relative to
its parent joint) and of the flange frame. Each placement is a full rigid
transform, so joint zero offsets, link lengths/twists, the base and the flange
are all covered. For a 6R arm only 30 of the 42 placement parameters are
identifiable from end-effector poses; the fit uses a minimum-norm
(truncated-SVD) Gauss-Newton step so the unidentifiable directions stay at
their URDF values.

Calibration files are JSON::

    {"joints": {"joint_1": {"xyz": [...], "rpy": [...]}, ...},
     "frames": {"flange": {"xyz": [...], "rpy": [...]}},
     "metadata": {...}}

``xyz`` in metres and ``rpy`` in radians, with the URDF ``<origin>`` convention.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from motion_planner.pinocchio_solver import DEFAULT_URDF
from motion_planner.pose import pose_to_matrix
from motion_planner.solver import KinematicsSolver

Placement = tuple[Sequence[float], Sequence[float]]  # (xyz [m], rpy [rad])
PathLike = Union[str, Path]


def _to_se3(placement: Placement) -> pin.SE3:
    xyz, rpy = placement
    return pin.SE3(pin.rpy.rpyToMatrix(np.asarray(rpy, dtype=float)), np.asarray(xyz, dtype=float))


def _from_se3(M: pin.SE3) -> Placement:
    return M.translation.copy(), pin.rpy.matrixToRpy(M.rotation).copy()


@dataclass
class KinematicCalibration:
    """Calibrated joint and frame placements, applicable to a pinocchio model."""

    joints: dict[str, Placement]
    frames: dict[str, Placement]
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: pin.Model, frames: Sequence[str] = ("flange",)) -> "KinematicCalibration":
        joints = {model.names[j]: _from_se3(model.jointPlacements[j]) for j in range(1, model.njoints)}
        return cls(
            joints=joints,
            frames={name: _from_se3(model.frames[model.getFrameId(name)].placement) for name in frames},
        )

    @classmethod
    def from_urdf(cls, urdf_path: PathLike = DEFAULT_URDF, frames: Sequence[str] = ("flange",)) -> "KinematicCalibration":
        """The uncalibrated placements of the URDF."""
        return cls.from_model(pin.buildModelFromUrdf(str(urdf_path)), frames)

    @classmethod
    def load(cls, path: PathLike) -> "KinematicCalibration":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def parse(entries):
            return {name: (list(v["xyz"]), list(v["rpy"])) for name, v in entries.items()}

        return cls(parse(raw["joints"]), parse(raw.get("frames", {})), raw.get("metadata", {}))

    def save(self, path: PathLike) -> None:
        def dump(entries):
            return {
                name: {"xyz": [float(x) for x in xyz], "rpy": [float(x) for x in rpy]}
                for name, (xyz, rpy) in entries.items()
            }

        out = {"joints": dump(self.joints), "frames": dump(self.frames), "metadata": self.metadata}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")

    def apply(self, model: pin.Model) -> None:
        """Overwrite the placements of ``model`` in place."""
        for name, placement in self.joints.items():
            if not model.existJointName(name):
                raise ValueError(f"calibration joint '{name}' not found in the model")
            model.jointPlacements[model.getJointId(name)] = _to_se3(placement)
        for name, placement in self.frames.items():
            if not model.existFrame(name):
                raise ValueError(f"calibration frame '{name}' not found in the model")
            model.frames[model.getFrameId(name)].placement = _to_se3(placement)


def load_pairs(path: PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a ``record_joint_pose.py`` JSONL file.

    Returns (joints [N,6] rad, flange poses [N,4,4], record indices [N]).
    Blank and malformed lines are skipped.
    """
    qs, poses, idx = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                q, pose = rec["joints"], rec["pose_flange"]
            except (ValueError, KeyError, TypeError):
                continue
            qs.append(q)
            poses.append(pose_to_matrix(pose))
            idx.append(int(rec.get("index", len(idx))))
    return np.array(qs, dtype=float), np.array(poses), np.array(idx)


def fk_errors(solver: KinematicsSolver, Q: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (position error [m], rotation error [rad]) of ``solver.fk`` vs ``T``."""
    pos, rot = [], []
    for q, t in zip(Q, T):
        F = solver.fk(q)
        pos.append(np.linalg.norm(F[:3, 3] - t[:3, 3]))
        rot.append(np.linalg.norm(pin.log3(t[:3, :3].T @ F[:3, :3])))
    return np.array(pos), np.array(rot)


@dataclass
class CalibrationResult:
    calibration: KinematicCalibration
    inliers: np.ndarray  # bool [N]
    errors_before: tuple[np.ndarray, np.ndarray]  # (pos [m], rot [rad]) per sample, uncalibrated
    errors_after: tuple[np.ndarray, np.ndarray]  # per sample, calibrated (all samples)
    rank: int  # identifiable parameters
    singular_values: np.ndarray  # of the weighted identification Jacobian (inliers)


class _Problem:
    """Placement perturbations ``M_k <- M_k0 * exp6(delta_k)`` and their residuals."""

    def __init__(self, urdf_path, nominal: KinematicCalibration, Q, T, rotation_weight):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        nominal.apply(self.model)
        self.data = self.model.createData()
        self.nominal = nominal
        self.joint_ids = [self.model.getJointId(n) for n in nominal.joints]
        self.frame_ids = [self.model.getFrameId(n) for n in nominal.frames]
        self.M0 = [self.model.jointPlacements[j].copy() for j in self.joint_ids] + [
            self.model.frames[f].placement.copy() for f in self.frame_ids
        ]
        self.ee = self.frame_ids[0]
        self.Q, self.T, self.w = Q, T, rotation_weight
        self.n = 6 * len(self.M0)

    def set(self, p: np.ndarray) -> None:
        for k, M0 in enumerate(self.M0):
            M = M0 * pin.exp6(p[6 * k:6 * k + 6])
            if k < len(self.joint_ids):
                self.model.jointPlacements[self.joint_ids[k]] = M
            else:
                self.model.frames[self.frame_ids[k - len(self.joint_ids)]].placement = M

    def residuals(self, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Stacked [dp (m), w * dtheta (rad)] per sample."""
        self.set(p)
        out = np.empty((len(idx), 6))
        for r, i in enumerate(idx):
            pin.framesForwardKinematics(self.model, self.data, self.Q[i])
            F = self.data.oMf[self.ee]
            out[r, :3] = F.translation - self.T[i][:3, 3]
            out[r, 3:] = self.w * pin.log3(self.T[i][:3, :3].T @ F.rotation)
        return out.ravel()

    def jacobian(self, p: np.ndarray, idx: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        r0 = self.residuals(p, idx)
        J = np.empty((r0.size, self.n))
        for k in range(self.n):
            dp = p.copy()
            dp[k] += eps
            J[:, k] = (self.residuals(dp, idx) - r0) / eps
        return J

    def sample_errors(self, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.residuals(p, idx).reshape(-1, 6), axis=1)

    def gauss_newton(self, idx, p0=None, rcond=1e-6, max_iter=30, tol=1e-13):
        """Minimum-norm Gauss-Newton: unidentifiable directions are never moved."""
        p = np.zeros(self.n) if p0 is None else p0.copy()
        for _ in range(max_iter):
            J = self.jacobian(p, idx)
            step = -np.linalg.pinv(J, rcond=rcond) @ self.residuals(p, idx)
            p += step
            if np.linalg.norm(step) < tol:
                break
        return p

    def calibration(self, p: np.ndarray) -> KinematicCalibration:
        self.set(p)
        return KinematicCalibration.from_model(self.model, list(self.nominal.frames))


def calibrate(
    Q: np.ndarray,
    T: np.ndarray,
    nominal: Optional[KinematicCalibration] = None,
    urdf_path: PathLike = DEFAULT_URDF,
    rotation_weight: float = 0.5,
    reject_outliers: bool = True,
    outlier_floor: float = 1e-4,
    outlier_mad_k: float = 5.0,
    rcond: float = 1e-6,
) -> CalibrationResult:
    """Fit joint/flange placements so FK(Q[i]) matches the measured poses T[i].

    Args:
        Q: Joint vectors [N, 6] (rad).
        T: Measured flange poses [N, 4, 4] in the robot base frame.
        nominal: Starting placements (default: the URDF's). Only the joints and
            frames it lists are calibrated; the first frame is the end effector.
        rotation_weight: Metres per radian when mixing position and rotation errors.
        reject_outliers: Robustly detect and exclude inconsistent samples.
        outlier_floor: A sample is never an outlier below this error [m, weighted].
        outlier_mad_k: Outlier threshold = median + k * 1.4826 * MAD of the errors.
        rcond: Relative singular-value cutoff deciding identifiable directions.
    """
    Q, T = np.asarray(Q, dtype=float), np.asarray(T, dtype=float)
    nominal = nominal or KinematicCalibration.from_urdf(urdf_path)
    prob = _Problem(urdf_path, nominal, Q, T, rotation_weight)
    all_idx = np.arange(len(Q))
    zero = np.zeros(prob.n)
    before = fk_errors(_ModelFK(prob, zero), Q, T)

    inliers = np.ones(len(Q), dtype=bool)
    if reject_outliers:
        # Robust start so outliers cannot drag the model toward themselves.
        robust = least_squares(
            lambda p: prob.residuals(p, all_idx), zero, loss="soft_l1",
            f_scale=outlier_floor, xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        p = robust.x
        for _ in range(10):
            err = prob.sample_errors(p, all_idx)
            ok = err[inliers] if inliers.any() else err
            med = np.median(ok)
            thresh = max(outlier_floor, med + outlier_mad_k * 1.4826 * np.median(np.abs(ok - med)))
            new = err <= thresh
            p = prob.gauss_newton(np.flatnonzero(new), p0=zero, rcond=rcond)
            if np.array_equal(new, inliers):
                break
            inliers = new
        # Final classification with the inlier-only model.
        err = prob.sample_errors(p, all_idx)
        med = np.median(err[inliers])
        thresh = max(outlier_floor, med + outlier_mad_k * 1.4826 * np.median(np.abs(err[inliers] - med)))
        inliers = err <= thresh
    p = prob.gauss_newton(np.flatnonzero(inliers), p0=zero, rcond=rcond)

    sv = np.linalg.svd(prob.jacobian(p, np.flatnonzero(inliers)), compute_uv=False)
    rank = int(np.sum(sv > rcond * sv[0]))
    cal = prob.calibration(p)
    after = fk_errors(_ModelFK(prob, p), Q, T)
    pos_in, rot_in = after[0][inliers], after[1][inliers]
    cal.metadata = {
        "n_samples": int(len(Q)),
        "n_inliers": int(inliers.sum()),
        "outliers": [int(i) for i in np.flatnonzero(~inliers)],
        "identifiable_parameters": rank,
        "rotation_weight_m_per_rad": rotation_weight,
        "pos_rms_before_m": float(np.sqrt(np.mean(before[0][inliers] ** 2))),
        "pos_rms_after_m": float(np.sqrt(np.mean(pos_in ** 2))),
        "pos_max_after_m": float(pos_in.max()),
        "rot_rms_after_rad": float(np.sqrt(np.mean(rot_in ** 2))),
        "rot_max_after_rad": float(rot_in.max()),
    }
    return CalibrationResult(cal, inliers, before, after, rank, sv)


class _ModelFK:
    """Minimal FK adapter over a perturbed problem model (for :func:`fk_errors`)."""

    def __init__(self, prob: _Problem, p: np.ndarray) -> None:
        self._prob, self._p = prob, p

    def fk(self, q):
        self._prob.set(self._p)
        pin.framesForwardKinematics(self._prob.model, self._prob.data, np.asarray(q, dtype=float))
        return self._prob.data.oMf[self._prob.ee].homogeneous
