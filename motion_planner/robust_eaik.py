"""
Numerical layer on top of the analytical EAIK fork (eaik >= 1.2.2+robust.1).

Division of work
----------------
EAIK (analytical, C++):
  * tip_link            the IK target / FK output is the real URDF tip frame
  * parallel_tol,       the analytical solver works on an idealized model in which near-parallel /
    intersect_tol       near-intersecting consecutive axes are exactly parallel / intersecting
  * pos_error,          every solution is evaluated on the ORIGINAL (calibrated) model; is_LS is
    rot_error, is_LS    True iff it misses the target by more than ls_pos_tol / ls_rot_tol
This module (numerical, Python):
  * Newton refinement of every candidate on the original model (removes the idealization error)
  * acceptance only of candidates that reach the target within tol_pos / tol_rot
    (drops least-squares fill-ins of unreachable branches)
  * twin seeds near singularities, where EAIK merges two nearby solutions into one candidate
  * joint limits, +-2*pi variants, de-duplication

Works for any serial revolute chain that EAIK can decompose (optionally with fixed_axes).
Requires numpy and the EAIK fork (plus urchin, an EAIK dependency, for URDF input).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- math helpers
def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R, accurate for small angles and near pi."""
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(w)
    c = 0.5 * (np.trace(R) - 1.0)
    th = math.atan2(s, c)
    if th < 1e-6:
        return w
    if math.pi - th < 1e-4:
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        ax = A[:, k] / math.sqrt(max(A[k, k], 1e-300))
        if np.dot(ax, w) < 0:
            ax = -ax
        return th * ax
    return th / s * w


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def _wrap(a):
    return (np.asarray(a) + math.pi) % _TWO_PI - math.pi


def pose_error(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(position error [m], rotation error [rad]) between two 4x4 poses."""
    return (float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])),
            float(np.linalg.norm(_log_so3(Ta[:3, :3] @ Tb[:3, :3].T))))


# --------------------------------------------------------------------------- result type
@dataclass
class IKResult:
    Q: list = field(default_factory=list)        # verified joint vectors (full length, incl. fixed axes)
    errors: list = field(default_factory=list)   # (dp [m], dr [rad]) on the original model, per solution
    n_raw: int = 0                                # candidates returned by EAIK
    n_rejected: int = 0                           # EAIK candidates from which no seed reached the target

    def __len__(self):
        return len(self.Q)


# --------------------------------------------------------------------------- solver
class RobustIK:
    """
    Build with RobustIK.from_urdf(...) or RobustIK(robot, ...) for any eaik IKRobot.

    Parameters
    ----------
    robot            : eaik IKRobot (UrdfRobot, HPRobot, ...) from the EAIK fork.
    lower, upper     : joint limits (full length); None = no limits (solutions wrapped to [-pi, pi)).
    fixed_axes       : the same list of (index, value) the robot was built with.
    tol_pos, tol_rot : acceptance tolerance on the original model [m], [rad].
    near_singular    : sigma_min / sigma_max of the Jacobian below which twin seeds are added.
    near_zero        : |sin q_b| below which a wrist-flip twin seed is added.
    kick             : size of the smallest twin-seed perturbation [rad]; near-singular candidates are
                       also sampled along the Jacobian's weakest direction at 0.2, 0.8 and 2.0 rad.
    """

    def __init__(self, robot, lower=None, upper=None, fixed_axes=None,
                 tol_pos: float = 1e-9, tol_rot: float = 1e-9,
                 near_singular: float = 1e-2, near_zero: float = 0.05, kick: float = 0.02):
        self.robot = robot
        self.tol_pos, self.tol_rot = tol_pos, tol_rot
        self.near_singular, self.near_zero, self.kick = near_singular, near_zero, kick
        self.valley_amplitudes = (kick, 0.2, 0.8, 2.0)
        self.valley_amplitudes_dense = (kick, 0.1, 0.2, 0.4, 0.6, 0.8, 1.1, 1.4, 1.8, 2.2, 2.6, 3.0)
        self.valley_trigger = 1e-3      # [rad] expected joint error above which the valley is sampled
        self.max_seed_error = 0.02      # [m] / [rad] candidates missing the target by more than this are
                                        # least-squares fill-ins: no twin seeds, no continuation

        self.H = np.asarray(robot.getOriginal_H(), float).T.copy()     # (n, 3)
        self.P = np.asarray(robot.getOriginal_P(), float).T.copy()     # (n+1, 3), last row = tip offset
        self.H /= np.linalg.norm(self.H, axis=1, keepdims=True)
        self.R6T = np.asarray(robot.getR6T(), float)
        self.n = len(self.H)
        self._rodrigues = []
        for h in self.H:
            hh = np.outer(h, h)
            K = np.array([[0.0, -h[2], h[1]], [h[2], 0.0, -h[0]], [-h[1], h[0], 0.0]])
            self._rodrigues.append((hh[None], (np.eye(3) - hh)[None], K[None]))

        fixed = dict(fixed_axes or [])
        self.fixed = fixed
        self.free = np.array([i for i in range(self.n) if i not in fixed], dtype=int)

        self.lower = None if lower is None else np.asarray(lower, float)
        self.upper = None if upper is None else np.asarray(upper, float)

        # Idealized model the analytical solver used (for continuation). Only available as a full chain
        # when no axes are fixed (with fixed axes EAIK idealizes the reduced chain).
        self.H_ideal = self.P_ideal = None
        if not fixed and hasattr(robot, "getIdealized_H"):
            Hi = np.asarray(robot.getIdealized_H(), float).T.copy()
            Pi = np.asarray(robot.getIdealized_P(), float).T.copy()
            if Hi.shape == self.H.shape and Pi.shape == self.P.shape:
                Hi /= np.linalg.norm(Hi, axis=1, keepdims=True)
                if max(np.abs(Hi - self.H).max(), np.abs(Pi - self.P).max()) > 0.0:
                    self.H_ideal, self.P_ideal = Hi, Pi

        # Wrist-flip triples (a, b, c) of consecutive free joints: h_a || h_c and h_b _|_ h_a, h_c.
        # Then R(h_a, qa) R(h_b, qb) R(h_c, qc) = R(h_a, qa + pi) R(h_b, -qb) R(h_c, qc + pi):
        # near qb = 0 (or pi) EAIK can merge the two wrist configurations into one candidate.
        self.flip_triples = []
        Hf = self.H[self.free]
        for k in range(len(self.free) - 2):
            ha, hb, hc = Hf[k], Hf[k + 1], Hf[k + 2]
            if (np.linalg.norm(np.cross(ha, hc)) < 1e-2 and abs(ha @ hb) < 1e-2 and abs(hc @ hb) < 1e-2):
                self.flip_triples.append(tuple(self.free[k:k + 3]))

    @classmethod
    def from_urdf(cls, urdf_path: str, tip_link: str | None = "flange", fixed_axes=None,
                  parallel_tol: float = 2e-3, intersect_tol: float = 1e-3, **kwargs) -> "RobustIK":
        """UrdfRobot with idealization switched on, joint limits taken from the URDF."""
        from eaik.IK_URDF import UrdfRobot
        from urchin import URDF

        robot = UrdfRobot(urdf_path, fixed_axes or [], tip_link=tip_link,
                          parallel_tol=parallel_tol, intersect_tol=intersect_tol)
        u = URDF.load(urdf_path, lazy_load_meshes=True)
        joints = u._sort_joints(u.actuated_joints)
        lower = [j.limit.lower if (j.limit is not None and j.joint_type == "revolute") else -np.inf for j in joints]
        upper = [j.limit.upper if (j.limit is not None and j.joint_type == "revolute") else np.inf for j in joints]
        obj = cls(robot, lower, upper, fixed_axes, **kwargs)
        obj.joint_names = [j.name for j in joints]
        return obj

    # ----------------------------------------------------------------- original-model kinematics
    def fk(self, q) -> np.ndarray:
        """FK of the original (calibrated) model; equals robot.fwdKin(q)."""
        return self._fk_jac(np.asarray(q, float), jac=False)

    def _fk_jac(self, q, jac=True, model=None):
        H, P = (self.H, self.P) if model is None else model
        R = np.eye(3)
        p = P[0].copy()
        axes, origins = [], []
        for i in range(self.n):
            if jac:
                axes.append(R @ H[i])
                origins.append(p.copy())
            R = R @ _rot(H[i], q[i])
            p = p + R @ P[i + 1]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R @ self.R6T, p
        if not jac:
            return T
        J = np.empty((6, len(self.free)))
        for c, i in enumerate(self.free):
            J[:3, c] = _cross(axes[i], p - origins[i])
            J[3:, c] = axes[i]
        return T, J

    def _residual(self, T_target, T):
        return np.concatenate([T_target[:3, 3] - T[:3, 3], _log_so3(T_target[:3, :3] @ T[:3, :3].T)])

    # ----------------------------------------------------------------- batched kinematics
    def _fk_jac_batch(self, Q, jac=True):
        """FK (rotation (N,3,3), position (N,3)) and Jacobian (N,6,n_free) of the original
        model for N joint vectors at once (same arithmetic as :meth:`_fk_jac`)."""
        N = len(Q)
        R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        p = np.broadcast_to(self.P[0], (N, 3)).copy()
        axes, origins = [], []
        C_ = np.cos(Q)
        S_ = np.sin(Q)
        for i in range(self.n):
            if jac:
                axes.append(R @ self.H[i])
                origins.append(p)
            # Rodrigues: R(h, q) = h h^T + cos q (I - h h^T) + sin q [h]x
            hh, I_hh, K = self._rodrigues[i]
            Ri = hh + C_[:, i, None, None] * I_hh + S_[:, i, None, None] * K
            R = R @ Ri
            p = p + R @ self.P[i + 1]
        if not jac:
            return R @ self.R6T, p, None
        J = np.empty((N, 6, len(self.free)))
        for col, i in enumerate(self.free):
            u, v = axes[i], p - origins[i]
            J[:, 0, col] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
            J[:, 1, col] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
            J[:, 2, col] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            J[:, 3:, col] = u
        return R @ self.R6T, p, J

    def _residual_batch(self, T_target, Rb, pb):
        Rerr = T_target[:3, :3][None] @ np.transpose(Rb, (0, 2, 1))
        w = 0.5 * np.stack([Rerr[:, 2, 1] - Rerr[:, 1, 2], Rerr[:, 0, 2] - Rerr[:, 2, 0],
                            Rerr[:, 1, 0] - Rerr[:, 0, 1]], axis=1)
        s = np.linalg.norm(w, axis=1)
        c = 0.5 * (np.trace(Rerr, axis1=1, axis2=2) - 1.0)
        th = np.arctan2(s, c)
        rv = w * np.where(th < 1e-6, 1.0, th / np.maximum(s, 1e-300))[:, None]
        for k in np.nonzero(math.pi - th < 1e-4)[0]:  # near pi: the scalar formula
            rv[k] = _log_so3(Rerr[k])
        return np.concatenate([T_target[:3, 3][None] - pb, rv], axis=1)

    def refine_batch(self, T_target, Q0, max_iter: int = 60):
        """:meth:`refine` (original model) for many start vectors at once; each row runs
        the same damped Gauss-Newton iterations as the scalar version.
        Returns (Q, dp, dr) as arrays."""
        Q = np.array(Q0, float, copy=True)
        N = len(Q)
        if N == 0:
            return Q, np.zeros(0), np.zeros(0)
        Rb, pb, J = self._fk_jac_batch(Q)
        E = self._residual_batch(T_target, Rb, pb)
        cost = np.einsum("ij,ij->i", E, E)
        lam = np.full(N, 1e-9)
        active = np.ones(N, bool)
        free = self.free
        for it in range(max_iter):
            done = (np.linalg.norm(E[:, :3], axis=1) < 0.1 * self.tol_pos) \
                & (np.linalg.norm(E[:, 3:], axis=1) < 0.1 * self.tol_rot)
            if it >= 12:
                done |= cost > 1e-8
            active &= ~done
            idx = np.nonzero(active)[0]
            if len(idx) == 0:
                break
            Ja, Ea = J[idx], E[idx]
            A = np.transpose(Ja, (0, 2, 1)) @ Ja
            dA = np.diagonal(A, axis1=1, axis2=2)
            M = A + (lam[idx, None] * (dA + 1e-12))[:, :, None] * np.eye(A.shape[1])[None]
            g = np.einsum("nji,nj->ni", Ja, Ea)
            dq = np.linalg.solve(M, g[:, :, None])[:, :, 0]
            step = np.linalg.norm(dq, axis=1)
            big = step > 0.5
            dq[big] *= (0.5 / step[big])[:, None]
            Qn = Q[idx].copy()
            Qn[:, free] += dq
            Rn, pn, Jn = self._fk_jac_batch(Qn)
            En = self._residual_batch(T_target, Rn, pn)
            cn = np.einsum("ij,ij->i", En, En)
            better = cn < cost[idx]
            ib, iw = idx[better], idx[~better]
            Q[ib], J[ib], E[ib], cost[ib] = Qn[better], Jn[better], En[better], cn[better]
            lam[ib] = np.maximum(lam[ib] * 0.1, 1e-12)
            lam[iw] *= 10.0
            active[iw[lam[iw] > 1e3]] = False
        return Q, np.linalg.norm(E[:, :3], axis=1), np.linalg.norm(E[:, 3:], axis=1)

    # ----------------------------------------------------------------- refinement
    def refine(self, T_target, q0, max_iter: int = 60, model=None):
        """Damped Gauss-Newton over the free joints (original model unless `model` = (H, P)).
        Returns (q, dp, dr)."""
        q = np.asarray(q0, float).copy()
        T, J = self._fk_jac(q, model=model)
        e = self._residual(T_target, T)
        cost = e @ e
        lam = 1e-9
        for it in range(max_iter):
            if np.linalg.norm(e[:3]) < 0.1 * self.tol_pos and np.linalg.norm(e[3:]) < 0.1 * self.tol_rot:
                break
            if it >= 12 and cost > 1e-8:                     # not converging to this target (> ~0.1 mm / mrad)
                break
            A = J.T @ J
            dq = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), J.T @ e)
            step = np.linalg.norm(dq)
            if step > 0.5:                                    # trust region: stay on the local branch
                dq *= 0.5 / step
            qn = q.copy()
            qn[self.free] += dq
            Tn, Jn = self._fk_jac(qn, model=model)
            en = self._residual(T_target, Tn)
            cn = en @ en
            if cn < cost:
                q, T, J, e, cost = qn, Tn, Jn, en, cn
                lam = max(lam * 0.1, 1e-12)
            else:
                lam *= 10.0
                if lam > 1e3:
                    break
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    def _converged(self, dp, dr):
        return dp <= self.tol_pos and dr <= self.tol_rot

    def continuation(self, T_target, q0, min_dt: float = 1.0 / 32):
        """
        Track a solution from the idealized model (where EAIK's candidate is exact) to the original
        model while the model parameters are blended from one to the other. Stays on the same branch
        where a direct Newton step from the idealized candidate would be ill-conditioned (near
        singularities the idealization error delta turns into a joint error of order delta/sigma_min).
        Returns (q, dp, dr) on the original model, or None if the branch cannot be tracked.
        """
        if self.H_ideal is None:
            return None
        q, dp, dr = self.refine(T_target, q0, max_iter=30, model=(self.H_ideal, self.P_ideal))
        if not self._converged(dp, dr):
            return None                                      # no exact solution of the idealized model here
        t, dt = 0.0, 0.25
        while t < 1.0:
            t1 = min(1.0, t + dt)
            Ht = (1.0 - t1) * self.H_ideal + t1 * self.H
            Ht /= np.linalg.norm(Ht, axis=1, keepdims=True)
            Pt = (1.0 - t1) * self.P_ideal + t1 * self.P
            qn, dp, dr = self.refine(T_target, q, max_iter=15, model=(Ht, Pt))
            if self._converged(dp, dr) and np.max(np.abs(qn - q)) < 0.5:
                q, t, dt = qn, t1, min(0.5, 2.0 * dt)
            else:
                dt *= 0.5
                if dt < min_dt:
                    return None
        return self.refine(T_target, q)

    # ----------------------------------------------------------------- twin seeds (4)
    def _seeds(self, q_raw, model_error: float = 0.0):
        """
        model_error: size of the candidate's error on the original model (EAIK pos_error, rot_error),
        i.e. roughly the idealization error at this pose.
        """
        q = np.asarray(q_raw, float)
        k = self.kick

        # Wrist-flip twins merged by EAIK near q_b = 0 (or pi)
        bases = [q]
        for a, b, c in self.flip_triples:
            if abs(math.sin(q[b])) < self.near_zero:
                base = round(q[b] / math.pi) * math.pi
                for s in (+k, -k):
                    t = q.copy(); t[b] = base + s
                    f = t.copy(); f[a] += math.pi; f[c] += math.pi; f[b] = 2 * base - t[b]
                    bases += [t, f]

        seeds = list(bases)
        self._last_amplified = 0.0
        _, J = self._fk_jac(q)
        _, S, Vt = np.linalg.svd(J)
        if S[-1] < self.near_singular * S[0]:
            v = np.zeros(self.n)
            v[self.free] = Vt[-1]
            # Local twins (elbow/shoulder-type) merged by EAIK: perturb along the weakest direction
            amps, around = (k,), [q]
            # Near a singularity the model error turns into a joint error ~ model_error / sigma_min
            # along that direction, so several solutions may share one "valley": sample along it.
            amplified = model_error / S[-1]
            self._last_amplified = amplified
            if amplified > self.valley_trigger:
                amps = self.valley_amplitudes
            if amplified > 0.1:                      # valley position essentially unknown: sample densely,
                amps = self.valley_amplitudes_dense  # also around the wrist-flipped seeds
                around = bases
            for b0 in around:
                for amp in amps:
                    seeds += [b0 + amp * v, b0 - amp * v]
        return seeds

    # ----------------------------------------------------------------- post-processing
    def _postprocess(self, T, sol, within_limits, expand_2pi, dedup_tol, limit_tol):
        Q = np.atleast_2d(np.asarray(sol.Q, float)) if len(sol.Q) else np.zeros((0, self.n))
        pe = np.asarray(sol.pos_error, float)
        re = np.asarray(sol.rot_error, float)
        res = IKResult(n_raw=len(Q))

        accepted = []
        for i, q_raw in enumerate(Q):
            ok = False
            model_error = max(float(pe[i]), float(re[i]))
            if model_error > self.max_seed_error:
                # Least-squares fill-in for a branch that does not reach the target: one direct try only
                q, dp, dr = self.refine(T, q_raw)
                if self._converged(dp, dr):
                    accepted.append((q, dp, dr))
                else:
                    res.n_rejected += 1
                continue
            seeds = self._seeds(q_raw, model_error=model_error)
            # Fast path: EAIK candidate already exact on the original model and not near a singularity
            if len(seeds) == 1 and pe[i] <= self.tol_pos and re[i] <= self.tol_rot:
                accepted.append((q_raw.copy(), float(pe[i]), float(re[i])))
                continue
            ambiguous = self._last_amplified > self.valley_trigger
            q, dp, dr = self.refine(T, seeds[0], max_iter=60)
            if self._converged(dp, dr):
                accepted.append((q, dp, dr))
                ok = True
            if ambiguous or not self._converged(dp, dr):
                # Track the EAIK candidate's own branch from the idealized model
                c = self.continuation(T, seeds[0])
                if c is not None and self._converged(c[1], c[2]):
                    accepted.append(c)
                    ok = True
            if len(seeds) > 1:  # twin / valley seeds: all at once
                Qs, dps, drs = self.refine_batch(T, np.asarray(seeds[1:]), max_iter=25)
                for q, dp, dr in zip(Qs, dps, drs):
                    if self._converged(float(dp), float(dr)):
                        accepted.append((q, float(dp), float(dr)))
                        ok = True
            if not ok:
                res.n_rejected += 1                           # unreachable branch / LS fill-in

        has_limits = within_limits and self.lower is not None
        kept = np.zeros((max(len(accepted), 1) * 8, self.n))  # res.Q as an array, for the duplicate check
        n_kept = 0
        seen = np.zeros((max(len(accepted), 1), self.n))  # wrapped accepted solutions expanded so far
        n_seen = 0
        for q, dp, dr in accepted:
            q = q.copy()
            q[self.free] = _wrap(q[self.free])
            if n_seen and float(np.min(np.max(np.abs(seen[:n_seen] - q), axis=1))) < dedup_tol:
                continue  # a duplicate expands to the same candidates
            seen[n_seen] = q
            n_seen += 1
            if has_limits:
                lo, hi = self.lower, self.upper
                if expand_2pi:
                    choices = []
                    for i in range(self.n):
                        if i in self.fixed:
                            choices.append([q[i]])
                            continue
                        if not (np.isfinite(lo[i]) and np.isfinite(hi[i])):
                            choices.append([q[i]])
                            continue
                        ks = range(math.floor((lo[i] - q[i]) / _TWO_PI) - 1, math.ceil((hi[i] - q[i]) / _TWO_PI) + 2)
                        choices.append([q[i] + m * _TWO_PI for m in ks
                                        if lo[i] - limit_tol <= q[i] + m * _TWO_PI <= hi[i] + limit_tol])
                    cands = [np.array(c) for c in itertools.product(*choices)]
                else:
                    cands = [q]
                cands = [c for c in cands if np.all(c >= lo - limit_tol) and np.all(c <= hi + limit_tol)]
            else:
                cands = [q]
            for c in cands:
                e = (dp, dr)
                if has_limits and (np.any(c < lo) or np.any(c > hi)):
                    # Slightly outside a limit: clip onto it only if the clipped q still reaches the target
                    cc = np.clip(c, lo, hi)
                    ec = pose_error(self.fk(cc), T)
                    if self._converged(*ec):
                        c, e = cc, ec
                if n_kept and float(np.min(np.max(np.abs(kept[:n_kept] - c), axis=1))) < dedup_tol:
                    continue
                if n_kept == len(kept):
                    kept = np.vstack([kept, np.zeros_like(kept)])
                kept[n_kept] = c
                n_kept += 1
                res.Q.append(c)
                res.errors.append(e)
        return res

    def _prepare(self, T):
        T = np.array(T, float, copy=True)
        T[:3, :3] = _orthonormalize(T[:3, :3])
        T[3] = [0.0, 0.0, 0.0, 1.0]
        return T

    def ik(self, T_tip, within_limits: bool = True, expand_2pi: bool = True,
           dedup_tol: float = 1e-6, limit_tol: float = 1e-6) -> IKResult:
        """
        All verified IK solutions for a tip pose (4x4, base frame).

        within_limits : drop solutions outside the joint limits (if limits are known).
        expand_2pi    : with limits, also return q_i +- 2*pi variants that fit the limits.
        limit_tol     : solutions up to this far outside a limit [rad] are kept; they are clipped onto
                        the limit only if the clipped q still reaches the target within tolerance.
        """
        T = self._prepare(T_tip)
        return self._postprocess(T, self.robot.IK(T), within_limits, expand_2pi, dedup_tol, limit_tol)

    def ik_batch(self, T_tips, num_threads: int = 4, **kwargs) -> list[IKResult]:
        Ts = [self._prepare(T) for T in T_tips]
        sols = self.robot.IK_batched(Ts, num_threads)
        opts = dict(within_limits=True, expand_2pi=True, dedup_tol=1e-6, limit_tol=1e-6)
        opts.update(kwargs)
        return [self._postprocess(T, s, **opts) for T, s in zip(Ts, sols)]
