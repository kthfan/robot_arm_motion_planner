#!/usr/bin/env python3
"""Reach-volume experiment (tmp/REACH_VOLUME_EXPERIMENT.md) solved on a Techman controller.

IK and collision come from TMscript through the Listen node (port 5890):

* IK: ``inversekin(pose, {c1,c2,c3})`` for all 8 configurations; a phase is reached
  if any configuration returns a solution (inversekin enforces the joint limits).
* Collision (``--collision``): a phase also needs one IK solution for which
  ``CollisionCheck(safety, PTP("JPP", q, ...))`` returns "". NOTE: CollisionCheck
  simulates the PTP *path* from the controller's current pose to q, not just q.

The controller walks a whole (x, y) column of z values per script and returns
``int[2n]``: n IK-only levels, then n collision levels (= IK levels without
``--collision``). Results are checkpointed after every column; rerun to resume.

    python scripts/reach_volume/reach_volume_tm.py --step 0.1 --out out/preview.npz
    python scripts/reach_volume/reach_volume_tm.py --out out/reach_volume_tm.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PHASES = ("sight_far", "sight_near", "pre_insert", "insert", "lift", "withdraw", "carry")

# Section 3, flange position relative to the label centre (m): (approach = base +Y, height = base Z).
CAM_Z, CAM_Y = 0.044217, 0.075155            # EIH camera optical centre, flange z / y
PLATE_TIP = 0.321
LIFT = 0.02125
PHASE_OFFSETS = {
    "sight_far":  (-0.350 - CAM_Z, -CAM_Y),
    "sight_near": (-0.2768 - CAM_Z, -CAM_Y),
    "pre_insert": (-0.010 - PLATE_TIP, 0.07725),
    "insert":     (0.022 + 0.218 - PLATE_TIP, 0.07725),
    "lift":       (0.022 + 0.218 - PLATE_TIP, 0.07725 + LIFT),
    "withdraw":   (-0.010 - PLATE_TIP, 0.07725 + LIFT),
    "carry":      (-0.010 - PLATE_TIP, 0.07725 + LIFT),   # carry tilt 0 deg -> same pose as withdraw
}
# Fixed flange orientation, TM Euler (deg): flange z = base +Y, flange y = base +Z, flange x = -base X.
# Verified on the controller: forwardkin(inversekin({0,619,277,90,0,180})) round-trips.
RPY = (90.0, 0.0, 180.0)
CONFIGS = [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]
#: carry == withdraw (tilt 0), so the controller only walks the first 6 phases.
DISTINCT = 6


def flange_poses(point) -> np.ndarray:
    """(7, 6) TM poses {X,Y,Z mm, RX,RY,RZ deg} of the 7 phases for a label-centre point (m)."""
    x, y, z = point
    return np.array([[x * 1e3, (y + da) * 1e3, (z + dh) * 1e3, *RPY]
                     for da, dh in (PHASE_OFFSETS[p] for p in PHASES)])


def _arr(vals, fmt="{:.4f}") -> str:
    return "{" + ",".join(fmt.format(v) for v in vals) + "}"


def column_script(x: float, y: float, zs, collision: bool, safety_mm: float = 0.0) -> str:
    """TMscript that walks one (x, y) column and ListenSends int[2n] levels."""
    n = len(zs)
    da = [PHASE_OFFSETS[p][0] * 1e3 for p in PHASES[:DISTINCT]]
    dh = [PHASE_OFFSETS[p][1] * 1e3 for p in PHASES[:DISTINCT]]
    zero = _arr([0] * (2 * n), "{}")
    col = "true" if collision else "false"
    rx, ry, rz = RPY
    lines = [
        "if (true) {",
        f"float rvX = {x * 1e3:.4f}",
        f"float rvY = {y * 1e3:.4f}",
        f"float[] rvZ = {_arr([z * 1e3 for z in zs])}",
        f"float[] rvDA = {_arr(da)}",
        f"float[] rvDH = {_arr(dh)}",
        f"int[] rvC1 = {_arr([c[0] for c in CONFIGS], '{}')}",
        f"int[] rvC2 = {_arr([c[1] for c in CONFIGS], '{}')}",
        f"int[] rvC3 = {_arr([c[2] for c in CONFIGS], '{}')}",
        f"int[] rvLV = {zero}",
        f"for (int i = 0; i < {n}; i++) {{",
        " bool aik = true",
        f" bool acol = {col}",
        " int k = 0",
        f" while (aik && k < {DISTINCT}) {{",
        f"  float[] pose = {{rvX, rvY + rvDA[k], rvZ[i] + rvDH[k], {rx}, {ry}, {rz}}}",
        "  bool okik = false",
        "  bool okcol = false",
        "  int c = 0",
        "  while (c < 8 && !(okik && (okcol || !acol))) {",
        "   float[] q = inversekin(pose, {rvC1[c], rvC2[c], rvC3[c]})",
        "   if (Length(q) == 6) {",
        "    okik = true",
        "    if (acol) {",
        f"     string r = CollisionCheck({safety_mm}, PTP(\"JPP\", q, 100, 200, 0, false))",
        "     if (r == \"\") {",
        "      okcol = true",
        "     }",
        "    }",
        "   }",
        "   c++",
        "  }",
        "  aik = okik",
        "  acol = acol && okcol",
        "  if (aik) {",
        "   rvLV[i] = k + 1",
        "  }",
        "  if (acol) {",
        f"   rvLV[i + {n}] = k + 1",
        "  }",
        "  k++",
        " }",
        "}",
        "ListenSend(90, rvLV)",
        "}",
    ]
    return "\r\n".join(lines)


def decode_column(payload: bytes, n: int, collision: bool) -> tuple[np.ndarray, np.ndarray]:
    """(levels_ik, levels) for one column; level 6 -> 7 because carry == withdraw."""
    v = np.frombuffer(payload, dtype="<i4")        # tm_eval.as_i32, without the robot package
    if len(v) != 2 * n:
        raise ValueError(f"expected {2 * n} ints, got {len(v)}")
    ik, lc = v[:n].astype(np.int8), v[n:].astype(np.int8)
    ik[ik == DISTINCT] = len(PHASES)
    lc[lc == DISTINCT] = len(PHASES)
    return ik, (lc if collision else ik.copy())


def axis(lo, hi, step):
    return np.round(np.arange(lo, hi + step / 2, step), 6)


def save(path, xs, ys, zs, lv_ik, lv, done, collision, safety):
    tmp = path[:-4] + ".partial.npz"
    np.savez(tmp, xs=xs, ys=ys, zs=zs, levels=lv, levels_ik=lv_ik, done=done,
             phases=np.array(PHASES), preset=np.array("strict"),
             collision=np.array(collision), safety_mm=np.array(safety),
             frame=np.array("label centre; robot base frame (x, y approach, z up)"))
    os.replace(tmp, path)


def main() -> int:
    from tm_eval import TMEval

    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="10.105.20.164")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--x", type=float, nargs=2, default=(-1.2, 1.2))
    ap.add_argument("--y", type=float, nargs=2, default=(0.0, 1.4))
    ap.add_argument("--z", type=float, nargs=2, default=(-1.2, 1.2))
    ap.add_argument("--collision", action="store_true", help="also run CollisionCheck (slow)")
    ap.add_argument("--safety", type=float, default=0.0, help="CollisionCheck safety distance, mm")
    ap.add_argument("--only-ik-reached", default=None,
                    help="IK-only npz: skip columns whose IK levels are all 0 (collision runs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xs, ys, zs = axis(*args.x, args.step), axis(*args.y, args.step), axis(*args.z, args.step)
    shape = (len(xs), len(ys), len(zs))
    lv_ik = np.zeros(shape, dtype=np.int8)
    lv = np.zeros(shape, dtype=np.int8)
    done = np.zeros(shape[:2], dtype=bool)
    if os.path.exists(args.out):
        with np.load(args.out) as d:
            if not (np.array_equal(d["xs"], xs) and np.array_equal(d["ys"], ys)
                    and np.array_equal(d["zs"], zs) and bool(d["collision"]) == args.collision):
                print("existing --out has a different grid/mode; refusing to overwrite")
                return 2
            lv_ik, lv, done = d["levels_ik"].copy(), d["levels"].copy(), d["done"].copy()
        print(f"resuming: {int(done.sum())}/{done.size} columns done")

    skip = np.zeros(shape[:2], dtype=bool)
    if args.only_ik_reached:
        with np.load(args.only_ik_reached) as d:
            if not (np.array_equal(d["xs"], xs) and np.array_equal(d["ys"], ys)
                    and np.array_equal(d["zs"], zs)):
                print("--only-ik-reached grid differs")
                return 2
            skip = ~(d["levels_ik"] > 0).any(axis=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"grid {shape[0]}x{shape[1]}x{shape[2]} = {int(np.prod(shape))} points, "
          f"collision={args.collision} safety={args.safety} mm, controller {args.ip}", flush=True)
    tm = TMEval(args.ip)
    t0, n_run = time.time(), 0
    todo = [(i, j) for i in range(shape[0]) for j in range(shape[1]) if not done[i, j]]
    try:
        for count, (i, j) in enumerate(todo, 1):
            if skip[i, j]:
                done[i, j] = True             # all-zero column: levels stay 0
                continue
            script = column_script(xs[i], ys[j], zs, args.collision, args.safety)
            timeout = 60 + (len(zs) * DISTINCT * 8 * 1.0 if args.collision else 0)
            payload = tm.run(script, timeout=timeout)
            lv_ik[i, j], lv[i, j] = decode_column(payload, len(zs), args.collision)
            done[i, j] = True
            n_run += 1
            if n_run % 10 == 0 or count == len(todo):
                save(args.out, xs, ys, zs, lv_ik, lv, done, args.collision, args.safety)
            el = time.time() - t0
            eta = el / count * (len(todo) - count)
            print(f"\r  columns {count}/{len(todo)}  {el:6.0f}s  eta {eta:6.0f}s", end="", flush=True)
    finally:
        save(args.out, xs, ys, zs, lv_ik, lv, done, args.collision, args.safety)
        tm.close()
    print()

    total = int(np.prod(shape))
    print("level  phase        ik_only  verdict")
    for k in range(len(PHASES) + 1):
        nm = "none" if k == 0 else PHASES[k - 1]
        print(f"  {k}    {nm:<11} {int((lv_ik == k).sum()):7d}  {int((lv == k).sum()):7d}")
    assert int(np.bincount(lv.ravel(), minlength=8).sum()) == total
    assert (lv <= lv_ik).all(), "collision level must not exceed IK level"
    print(f"sum = {total}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
