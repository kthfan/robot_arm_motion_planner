"""Calibrate the pinocchio kinematic model from recorded joint/pose pairs.

Reads a JSONL file written by ``record_joint_pose.py`` (joints + controller
reported flange pose), fits the joint and flange placements of the URDF,
reports data coverage / outliers / held-out accuracy, and writes a calibration
JSON that :class:`PinocchioSolver` (and the default pinocchio planner) loads.

Usage::

    python scripts/calibrate_kinematics.py joint_pose_XXXX.jsonl
    python scripts/calibrate_kinematics.py data.jsonl -o my_cal.json --folds 5
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))

from motion_planner import pinocchio_solver  # noqa: E402
from motion_planner.calibration import calibrate, fk_errors, load_pairs  # noqa: E402
from motion_planner.pinocchio_solver import PinocchioSolver  # noqa: E402

LEARNING_SIZES = (10, 15, 20, 30, 50, 80)


def _mm(x):
    return f"{1e3 * x:.4f}"


def _deg(x):
    return f"{np.degrees(x):.5f}"


def _stats(pos, rot):
    return (f"pos rms {_mm(np.sqrt(np.mean(pos ** 2)))} / max {_mm(pos.max())} mm, "
            f"rot rms {_deg(np.sqrt(np.mean(rot ** 2)))} / max {_deg(rot.max())} deg")


def _held_out(Q, T, train, test):
    fit = calibrate(Q[train], T[train], reject_outliers=False)
    return fk_errors(PinocchioSolver(calibration=fit.calibration), Q[test], T[test])[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="JSONL joint/pose pairs")
    ap.add_argument("-o", "--out", default=str(pinocchio_solver.DEFAULT_CALIBRATION),
                    help="output calibration JSON (default: %(default)s)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    Q, T, rec = load_pairs(args.data)
    print(f"Loaded {len(Q)} pairs from {args.data}")

    # ---- coverage
    plain = PinocchioSolver()
    lo, hi = np.degrees(plain.lower_limits), np.degrees(plain.upper_limits)
    qmin, qmax = np.degrees(Q.min(axis=0)), np.degrees(Q.max(axis=0))
    print("\nJoint coverage [deg]       data min..max      URDF limits")
    for j in range(Q.shape[1]):
        flag = "  <-- outside URDF limits" if qmin[j] < lo[j] - 1e-6 or qmax[j] > hi[j] + 1e-6 else ""
        print(f"  joint_{j + 1}: {qmin[j]:9.1f} .. {qmax[j]:7.1f}   {lo[j]:7.1f} .. {hi[j]:6.1f}{flag}")
    p = T[:, :3, 3]
    print("  flange xyz range [mm]: " + ", ".join(
        f"{a} {1e3 * p[:, i].min():.0f}..{1e3 * p[:, i].max():.0f}" for i, a in enumerate("xyz")))

    # ---- fit
    result = calibrate(Q, T)
    inl = result.inliers
    sv = result.singular_values
    print(f"\nIdentifiable parameters: {result.rank} of {sv.size} "
          f"(condition {sv[0] / sv[result.rank - 1]:.1f}, next singular value ratio "
          f"{sv[result.rank] / sv[0] if result.rank < sv.size else 0:.1e})")
    print(f"Samples per identifiable parameter: {inl.sum() * 6 / result.rank:.1f} residuals")
    print(f"Uncalibrated URDF (inliers): {_stats(result.errors_before[0][inl], result.errors_before[1][inl])}")
    print(f"Calibrated       (inliers): {_stats(result.errors_after[0][inl], result.errors_after[1][inl])}")

    out_idx = np.flatnonzero(~inl)
    if out_idx.size == 0:
        print("\nOutliers: none")
    else:
        print(f"\nOutliers: {out_idx.size} (excluded from the fit)")
        print("  record   calibrated err [mm, deg]   URDF err [mm, deg]")
        for i in out_idx:
            print(f"  {rec[i]:6d}   {_mm(result.errors_after[0][i]):>8} {_deg(result.errors_after[1][i]):>8}"
                  f"        {_mm(result.errors_before[0][i]):>8} {_deg(result.errors_before[1][i]):>8}")

    # ---- held-out accuracy
    rng = np.random.default_rng(args.seed)
    good = rng.permutation(np.flatnonzero(inl))
    if args.folds >= 2 and good.size >= 2 * args.folds:
        errs = np.concatenate([
            _held_out(Q, T, np.setdiff1d(good, fold), fold) for fold in np.array_split(good, args.folds)
        ])
        print(f"\n{args.folds}-fold cross-validation (held-out inliers): "
              f"pos rms {_mm(np.sqrt(np.mean(errs ** 2)))} / max {_mm(errs.max())} mm")

    sizes = [n for n in LEARNING_SIZES if n <= good.size - 5]
    if sizes:
        print("Learning curve (train size -> held-out pos rms / max mm, median of 3 draws):")
        for n in sizes:
            rms, mx = [], []
            for _ in range(3):
                perm = rng.permutation(good)
                e = _held_out(Q, T, perm[:n], perm[n:])
                rms.append(np.sqrt(np.mean(e ** 2)))
                mx.append(e.max())
            print(f"  {n:4d} -> {_mm(np.median(rms))} / {_mm(np.median(mx))}")

    # ---- save
    cal = result.calibration
    cal.metadata.update({
        "source": os.path.basename(args.data),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "urdf": os.path.basename(str(pinocchio_solver.DEFAULT_URDF)),
        "outlier_records": [int(rec[i]) for i in out_idx],
        "joint_range_min_rad": Q.min(axis=0).tolist(),
        "joint_range_max_rad": Q.max(axis=0).tolist(),
    })
    del cal.metadata["outliers"]  # array positions; outlier_records holds the record indices
    cal.save(args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
