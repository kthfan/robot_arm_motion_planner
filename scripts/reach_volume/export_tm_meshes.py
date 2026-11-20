#!/usr/bin/env python3
"""Valid / invalid (cleaned) meshes from a reach_volume_tm.py result.

Grid points are re-keyed from the label centre (spec Section 4) to the tray's
front-centre-bottom (spec Section 5 offsets), but stay in the ROBOT BASE frame
(x, y = approach, z = up); no axis flip or permutation. Then: binarize (sphere
fit, Section 6) -> export valid + cleaned invalid as OBJ (+ .mtl).

    python scripts/reach_volume/export_tm_meshes.py out/reach_volume_tm.npz
    python scripts/reach_volume/export_tm_meshes.py <npz> --use-ik-levels   # IK-only verdict
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import binarize_reach_volume as B      # noqa: E402
import export_reach_volume_obj as E    # noqa: E402

#: Label centre -> tray front-centre-bottom, in base axes (m).
APPROACH_OFFSET = 0.022 + 0.0115       # base +Y
UP_OFFSET = 0.035 + 0.015              # base +Z
AXES = "robot base frame (x, y approach, z up); point = tray front-centre-bottom"


def to_tray_point(d: dict, level_key: str = "levels") -> dict:
    return {
        "xs": np.asarray(d["xs"], dtype=float),
        "ys": np.round(np.asarray(d["ys"], dtype=float) + APPROACH_OFFSET, 6),
        "zs": np.round(np.asarray(d["zs"], dtype=float) + UP_OFFSET, 6),
        "levels": np.asarray(d[level_key]),
        "phases": d["phases"],
        "preset": d["preset"],
        "frame": np.array(AXES),
    }


def signed_volume(v, t) -> float:
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--use-ik-levels", action="store_true", help="mesh the IK-only verdict")
    args = ap.parse_args()

    with np.load(args.src) as f:
        d = {k: f[k] for k in f.files}
    if "done" in d and not d["done"].all():
        print(f"incomplete run: {int(d['done'].sum())}/{d['done'].size} columns")
        return 2
    key = "levels_ik" if args.use_ik_levels else "levels"
    stem = args.src[:-4] + ("_ik" if args.use_ik_levels and key != "levels" else "") + "_tray_base"

    out = B.binarize(to_tray_point(d, key))
    np.savez(stem + "_binary.npz", **out)
    print(B.report(out))

    step = [float(np.diff(out[k]).mean()) for k in ("xs", "ys", "zs")]
    cell = float(np.prod(step))
    masks = {"valid": out["valid"],
             "invalid": E.invalid_area(out["inner"], out["valid"], out["levels"],
                                       out["xs"], out["ys"], out["zs"])}
    for name, m in masks.items():
        v, t = E.voxel_mesh(m, out["xs"], out["ys"], out["zs"])
        vol, want = signed_volume(v, t), int(m.sum()) * cell
        ok = np.isclose(vol, want, rtol=1e-9, atol=1e-12)
        print(f"  {name:<8} {int(m.sum()):6d} voxels  mesh volume {vol:.6f} m^3  "
              f"expected {want:.6f}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            return 3
    E.export(out, stem, axes=AXES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
