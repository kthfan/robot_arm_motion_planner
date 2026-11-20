#!/usr/bin/env python3
"""Binarize a reach-volume npz: sphere fit -> inner area -> valid / invalid.

The reachable set (level > 0) looks like a half sphere with holes. Three steps:

1. **Envelope surface.** Voxels with level > 0 form the envelope. Background
   connected to the grid border is *exterior*; background enclosed by the
   envelope is a hole and does not count as outside. Every face between an
   envelope voxel and an exterior voxel yields one surface sample at the face
   midpoint (unbiased by half a step, unlike the voxel centres). Faces on the
   grid border yield nothing, so the flat cut where the grid ends adds no samples.
2. **Sphere.** A robust sphere fit (algebraic start, then trimmed geometric
   least squares) to those samples. The concave surface around any unreachable
   core, and notches in the shell, are rejected as outliers.
3. **Split.** ``inner`` = grid points within the sphere plus *margin* (default half
   a step, so a voxel cell that touches the sphere counts as inside; everything
   else is outer). ``valid`` = inner points whose level equals the number of phases,
   i.e. every action passed. ``invalid`` = inner and not valid.

Output ``label``: 0 outer, 1 inner-invalid, 2 inner-valid; ``valid`` and ``inner``
are the boolean masks. ``levels`` is kept as is.

    python scripts/reach_volume/binarize_reach_volume.py --src <npz>
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage, optimize

OUTER, INVALID, VALID = 0, 1, 2


def exterior(envelope: np.ndarray) -> np.ndarray:
    """Background voxels connected to a face of the grid (holes are not)."""
    lab, n = ndimage.label(~envelope)
    border = np.zeros(envelope.shape, dtype=bool)
    for ax in range(3):
        border[(slice(None),) * ax + (0,)] = True
        border[(slice(None),) * ax + (-1,)] = True
    keep = np.unique(lab[border & ~envelope])
    return np.isin(lab, keep[keep > 0])


def surface_samples(envelope: np.ndarray, axes) -> np.ndarray:
    """Midpoints of every face between an envelope voxel and an exterior voxel."""
    ext = exterior(envelope)
    pts = []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        # each pair is (lower voxel, upper voxel) along this axis
        for faces in (envelope[a] & ext[b], ext[a] & envelope[b]):
            lower = np.argwhere(faces)
            upper = lower.copy()
            upper[:, ax] += 1
            pts.append(np.stack([(axes[k][lower[:, k]] + axes[k][upper[:, k]]) / 2
                                 for k in range(3)], 1))
    return np.concatenate(pts) if pts else np.empty((0, 3))


def _algebraic(P):
    A = np.c_[2 * P, np.ones(len(P))]
    s = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    c = s[:3]
    return c, float(np.sqrt(max(s[3] + c @ c, 0.0)))


def fit_sphere(P: np.ndarray, tol: float, iters: int = 20):
    """Robust sphere fit. Returns (centre, radius, inlier mask)."""
    if len(P) < 4:
        raise ValueError("need at least 4 surface samples")
    c, r = _algebraic(P)
    keep = np.ones(len(P), dtype=bool)
    for _ in range(iters):
        res = np.linalg.norm(P - c, axis=1) - r
        # inner-void and notch samples sit far from the shell: trim, never grow
        new = np.abs(res) < max(tol, 2.5 * 1.4826 * np.median(np.abs(res[keep])))
        if new.sum() < 4:
            break
        sol = optimize.least_squares(
            lambda q: np.linalg.norm(P[new] - q[:3], axis=1) - q[3],
            np.r_[c, r], loss="soft_l1", f_scale=tol)
        c, r = sol.x[:3], float(sol.x[3])
        if (new == keep).all():
            break
        keep = new
    res = np.linalg.norm(P - c, axis=1) - r
    return c, r, np.abs(res) < tol


def binarize(d, tol: float | None = None, margin: float | None = None) -> dict:
    """*d*: mapping with xs, ys, zs, levels, phases, preset (and optional extras)."""
    xs, ys, zs = (np.asarray(d[k], dtype=float) for k in ("xs", "ys", "zs"))
    levels = np.asarray(d["levels"])
    full = len(d["phases"])
    step = float(np.min([np.diff(xs).min(), np.diff(ys).min(), np.diff(zs).min()]))
    tol = step if tol is None else tol
    margin = step / 2 if margin is None else margin

    pts = surface_samples(levels > 0, (xs, ys, zs))
    centre, radius, inl = fit_sphere(pts, tol)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    dist = np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2)
    inner = dist <= radius + margin
    valid = inner & (levels == full)
    label = np.full(levels.shape, OUTER, dtype=np.int8)
    label[inner] = INVALID
    label[valid] = VALID

    out = {k: d[k] for k in ("xs", "ys", "zs", "levels", "phases", "preset", "frame") if k in d}
    out.update(inner=inner, valid=valid, label=label,
               sphere_centre=centre, sphere_radius=np.array(radius), sphere_margin=np.array(margin),
               surface_samples=np.array(len(pts)), surface_inliers=np.array(int(inl.sum())))
    return out


def report(out) -> str:
    lv, inner, valid = out["levels"], out["inner"], out["valid"]
    reached_outside = int(((lv > 0) & ~inner).sum())
    return (f"sphere centre {np.round(out['sphere_centre'], 4).tolist()} "
            f"radius {float(out['sphere_radius']):.4f} + margin {float(out['sphere_margin']):.4f}  "
            f"({int(out['surface_inliers'])}/{int(out['surface_samples'])} surface samples fit)\n"
            f"  inner {int(inner.sum())}  valid {int(valid.sum())}  "
            f"invalid {int((inner & ~valid).sum())}  outer {int((~inner).sum())}\n"
            f"  reached voxels outside the sphere: {reached_outside}  "
            f"(valid fraction of inner {valid.sum() / inner.sum():.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="reach-volume npz files")
    ap.add_argument("--tol", type=float, default=None, help="inlier distance, m (default: grid step)")
    ap.add_argument("--margin", type=float, default=None, help="added to the radius, m (default: half the grid step)")
    args = ap.parse_args()
    for src in args.src:
        dst = src[:-4] + "_binary.npz"
        with np.load(src) as d:
            out = binarize({k: d[k] for k in d.files}, args.tol, args.margin)
        np.savez(dst, **out)
        print("wrote", dst)
        print(report(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
