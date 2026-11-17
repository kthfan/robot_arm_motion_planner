"""Analyse repeated measurements of a chart's pose relative to the robot base.

Typical use: the chart is removed and re-installed several times and its pose in
the base frame is measured after each installation. The report shows

* the repeatability of all samples and of each cluster of installations
  (installations that ended in the same seated position form one cluster);
* every sample relative to a reference (default: mean of the largest cluster),
  in the base frame and in the chart frame;
* for samples that are rotated relative to the reference, the screw axis of
  that motion: axis direction, the point on the axis nearest to the chart
  origin, and the slide along the axis. A chart that swings about one locating
  feature shows a consistent axis point (the "hinge");
* optionally, the deviation from a nominal (design) orientation / position;
* optionally, how each difference moves points at given offsets in the chart
  frame (e.g. a target 140 mm in front of the chart: ``--offset-mm 0,0,140``);
* optionally, an installation acceptance check against the reference.

Poses are TM poses ``[x, y, z, rx, ry, rz]`` in metres and radians, with
``R = Rz(rz) @ Ry(ry) @ Rx(rx)`` (same convention as :mod:`motion_planner.pose`).
Only numpy and scipy are needed.

Input (file path, or ``-`` for stdin), any of:

* a JSON array of poses: ``[[x, y, z, rx, ry, rz], ...]``
* JSON lines, one pose per line: ``[x, y, z, rx, ry, rz]``
* JSON lines of objects holding the pose under ``--key`` (default ``pose``);
  an optional ``label`` / ``time`` / ``index`` field names the sample.

Usage::

    python scripts/analyze_chart_pose.py chart_poses.json
    python scripts/analyze_chart_pose.py chart_poses.json --nominal-rot-deg 90,0,135 \\
        --offset-mm 0,0,140 --accept-deg 0.05 --accept-mm 0.1 --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- input
def load_poses(path: str, key: str = "pose") -> tuple[np.ndarray, list[str]]:
    """Read poses [N, 6] and sample labels from a JSON / JSON-lines file."""
    text = sys.stdin.read() if path == "-" else open(path, "r", encoding="utf-8").read()
    text = text.strip()
    try:
        records = json.loads(text)
        if isinstance(records, dict) or (records and not isinstance(records[0], (list, dict))):
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    poses, labels = [], []
    for i, rec in enumerate(records):
        if isinstance(rec, dict):
            label = rec.get("label", rec.get("time", rec.get("index", i + 1)))
            rec = rec[key]
        else:
            label = i + 1
        poses.append([float(v) for v in rec])
        labels.append(str(label))
    P = np.asarray(poses, dtype=float)
    if P.ndim != 2 or P.shape[1] != 6:
        raise ValueError(f"expected poses [x, y, z, rx, ry, rz], got array of shape {P.shape}")
    return P, labels


def parse_vec(s: Optional[str], n: int = 3) -> Optional[np.ndarray]:
    if s is None:
        return None
    v = np.array([float(x) for x in s.replace(" ", "").split(",")], dtype=float)
    if v.size != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got '{s}'")
    return v


# ------------------------------------------------------------------------- geometry
def rotations(P: np.ndarray) -> Rotation:
    return Rotation.from_euler("xyz", P[:, 3:6])


def mean_pose(P: np.ndarray, idx) -> tuple[Rotation, np.ndarray]:
    return rotations(P[idx]).mean(), P[idx, :3].mean(axis=0)


def angle_deg(R: Rotation) -> np.ndarray:
    return np.degrees(R.magnitude())


def clusters(P: np.ndarray, tol_mm: float, tol_deg: float) -> list[list[int]]:
    """Single-linkage clusters: two samples are linked when both their position
    difference <= tol_mm and their rotation difference <= tol_deg."""
    n = len(P)
    R = rotations(P)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dp = np.linalg.norm(P[i, :3] - P[j, :3]) * 1e3
            da = np.degrees((R[i] * R[j].inv()).magnitude())
            if dp <= tol_mm and da <= tol_deg:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def screw_axis(R_ref: Rotation, t_ref: np.ndarray, R_i: Rotation, t_i: np.ndarray) -> dict:
    """Screw axis of the rigid motion taking the reference chart pose to sample i.

    The motion in the base frame is p' = Rd p + td with Rd = R_i R_ref^-1.
    Returns the rotation angle, the axis direction (base and reference chart
    frame), the point on the axis nearest to the reference chart origin
    (base and chart frame, metres) and the slide along the axis (metres)."""
    Rd = R_i * R_ref.inv()
    td = t_i - Rd.apply(t_ref)
    rv = Rd.as_rotvec()
    ang = float(np.linalg.norm(rv))
    w = rv / ang
    slide = float(w @ td)
    # points c on the axis satisfy (I - Rd) c = td - slide * w; pinv gives the
    # one nearest to the base origin
    c = np.linalg.pinv(np.eye(3) - Rd.as_matrix(), rcond=1e-9) @ (td - slide * w)
    c_chart = R_ref.inv().apply(c - t_ref)
    w_chart = R_ref.inv().apply(w)
    c_chart = c_chart - (c_chart @ w_chart) * w_chart  # nearest to the chart origin
    return {
        "angle_deg": float(np.degrees(ang)),
        "axis_base": w,
        "axis_chart": w_chart,
        "point_chart_m": c_chart,
        "point_base_m": t_ref + R_ref.apply(c_chart),
        "slide_m": slide,
    }


# ---------------------------------------------------------------------------- report
def fmt(v, prec=3, width=8) -> str:
    return " ".join(f"{x:{width}.{prec}f}" for x in np.atleast_1d(v))


def repeatability(P: np.ndarray, idx: list[int]) -> dict:
    R_m, t_m = mean_pose(P, idx)
    R = rotations(P[idx])
    dev = R * R_m.inv()
    ang = angle_deg(dev)
    rv_chart = np.degrees(R_m.inv().apply(dev.as_rotvec()))
    return {
        "n": len(idx),
        "pos_std_mm": P[idx, :3].std(axis=0, ddof=1) * 1e3 if len(idx) > 1 else np.zeros(3),
        "pos_dev_max_mm": float(np.max(np.linalg.norm(P[idx, :3] - t_m, axis=1)) * 1e3),
        "rot_dev_rms_deg": float(np.sqrt(np.mean(ang ** 2))),
        "rot_dev_max_deg": float(ang.max()),
        "rot_std_chart_deg": rv_chart.std(axis=0, ddof=1) if len(idx) > 1 else np.zeros(3),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="poses file (JSON array or JSON lines), '-' for stdin")
    ap.add_argument("--key", default="pose", help="pose field of JSON-lines objects (default: %(default)s)")
    ap.add_argument("--cluster-mm", type=float, default=0.3,
                    help="position tolerance linking two samples into one cluster [mm] (default: %(default)s)")
    ap.add_argument("--cluster-deg", type=float, default=0.1,
                    help="rotation tolerance linking two samples into one cluster [deg] (default: %(default)s)")
    ap.add_argument("--reference", default="largest",
                    help="'largest' (mean of the largest cluster), 'all' (mean of all samples) "
                         "or a 1-based sample number (default: %(default)s)")
    ap.add_argument("--pivot-min-deg", type=float, default=0.1,
                    help="report the screw axis of samples rotated more than this [deg] (default: %(default)s)")
    ap.add_argument("--nominal-rot-deg", help="nominal chart orientation rx,ry,rz [deg], TM convention")
    ap.add_argument("--nominal-pos-mm", help="nominal chart position x,y,z [mm]")
    ap.add_argument("--offset-mm", action="append", default=[],
                    help="point in the chart frame x,y,z [mm] whose displacement is reported; repeatable")
    ap.add_argument("--accept-deg", type=float, help="acceptance: max rotation from the reference [deg]")
    ap.add_argument("--accept-mm", type=float, help="acceptance: max position change from the reference [mm]")
    ap.add_argument("--json", help="also write the results to this JSON file")
    args = ap.parse_args(argv)

    P, labels = load_poses(args.data, args.key)
    n = len(P)
    R_all = rotations(P)
    nom_R = Rotation.from_euler("xyz", np.radians(parse_vec(args.nominal_rot_deg))) if args.nominal_rot_deg else None
    nom_t = parse_vec(args.nominal_pos_mm) / 1e3 if args.nominal_pos_mm else None
    offsets = [parse_vec(o) / 1e3 for o in args.offset_mm]
    out: dict = {"n_samples": n, "labels": labels}

    print(f"Loaded {n} chart poses from {args.data}\n")
    print("Samples (position mm, TM euler deg):")
    print(f"  {'#':>3} {'label':>12} {'x':>10} {'y':>10} {'z':>10} {'rx':>9} {'ry':>9} {'rz':>9}")
    for i in range(n):
        print(f"  {i + 1:3d} {labels[i][-12:]:>12} {fmt(P[i, :3] * 1e3, 3, 10)} {fmt(np.degrees(P[i, 3:]), 4, 9)}")

    # ---- clusters and repeatability
    groups = clusters(P, args.cluster_mm, args.cluster_deg)
    out["clusters"] = [[i + 1 for i in g] for g in groups]
    print(f"\nClusters (linked when <= {args.cluster_mm} mm and <= {args.cluster_deg} deg apart):")
    reps = []
    for k, g in enumerate(groups):
        rep = repeatability(P, g)
        reps.append(rep)
        line = f"  C{k + 1}: samples {[i + 1 for i in g]}"
        if len(g) > 1:
            line += (f"  pos std {fmt(rep['pos_std_mm'], 3, 6)} mm (max dev {rep['pos_dev_max_mm']:.3f}),"
                     f" rot dev rms {rep['rot_dev_rms_deg']:.3f} / max {rep['rot_dev_max_deg']:.3f} deg")
        print(line)
    allrep = repeatability(P, list(range(n)))
    print(f"  all samples: pos std {fmt(allrep['pos_std_mm'], 3, 6)} mm (max dev {allrep['pos_dev_max_mm']:.3f}),"
          f" rot dev rms {allrep['rot_dev_rms_deg']:.3f} / max {allrep['rot_dev_max_deg']:.3f} deg")
    out["repeatability"] = {"clusters": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                                         for r in reps],
                            "all": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in allrep.items()}}

    # ---- reference
    if args.reference == "largest":
        ref_idx, ref_name = groups[0], f"mean of cluster C1 (samples {[i + 1 for i in groups[0]]})"
    elif args.reference == "all":
        ref_idx, ref_name = list(range(n)), "mean of all samples"
    else:
        j = int(args.reference) - 1
        ref_idx, ref_name = [j], f"sample {j + 1}"
    R_ref, t_ref = mean_pose(P, ref_idx)
    ref_pose = np.r_[t_ref, R_ref.as_euler("xyz")]
    out["reference"] = {"name": ref_name, "pose": ref_pose.tolist()}
    print(f"\nReference: {ref_name}")
    print(f"  pose [m, rad]: {ref_pose.tolist()}")
    print("  chart axes in base:  x " + fmt(R_ref.apply([1, 0, 0]), 3, 6)
          + " | y " + fmt(R_ref.apply([0, 1, 0]), 3, 6) + " | z " + fmt(R_ref.apply([0, 0, 1]), 3, 6))

    # ---- per-sample difference to the reference
    print("\nDifference to the reference (rotation as rotation vector, deg):")
    print(f"  {'#':>3} | {'dpos base [mm]':^26} | {'dpos chart [mm]':^26} | {'drot base [deg]':^23} |"
          f" {'drot chart [deg]':^23} | {'angle':>6}")
    rows = []
    for i in range(n):
        d = R_all[i] * R_ref.inv()
        dp = (P[i, :3] - t_ref)
        dp_c = R_ref.inv().apply(dp)
        rv = d.as_rotvec()
        rv_c = R_ref.inv().apply(rv)
        ang = float(np.degrees(np.linalg.norm(rv)))
        rows.append({"sample": i + 1, "dpos_base_mm": (dp * 1e3).tolist(), "dpos_chart_mm": (dp_c * 1e3).tolist(),
                     "drot_base_deg": np.degrees(rv).tolist(), "drot_chart_deg": np.degrees(rv_c).tolist(),
                     "angle_deg": ang})
        print(f"  {i + 1:3d} | {fmt(dp * 1e3)} | {fmt(dp_c * 1e3)} | {fmt(np.degrees(rv), 3, 7)} |"
              f" {fmt(np.degrees(rv_c), 3, 7)} | {ang:6.3f}")
    out["vs_reference"] = rows

    # ---- screw axes ("hinge") of rotated samples
    piv = [i for i in range(n) if rows[i]["angle_deg"] > args.pivot_min_deg]
    out["screw_axes"] = []
    if piv:
        print(f"\nScrew axis of the motion reference -> sample (samples rotated > {args.pivot_min_deg} deg):")
        print("  point = point on the axis nearest to the reference chart origin, in the chart frame;")
        print("  a consistent point across samples means the chart swings about one locating feature.")
        for i in piv:
            s = screw_axis(R_ref, t_ref, R_all[i], P[i, :3])
            out["screw_axes"].append({"sample": i + 1, "angle_deg": s["angle_deg"],
                                      "axis_base": s["axis_base"].tolist(), "axis_chart": s["axis_chart"].tolist(),
                                      "point_chart_mm": (s["point_chart_m"] * 1e3).tolist(),
                                      "point_base_mm": (s["point_base_m"] * 1e3).tolist(),
                                      "slide_mm": s["slide_m"] * 1e3})
            print(f"  {i + 1:3d}: angle {s['angle_deg']:6.3f} deg  axis chart {fmt(s['axis_chart'], 3, 6)}"
                  f"  (base {fmt(s['axis_base'], 3, 6)})  point chart {fmt(s['point_chart_m'] * 1e3, 1, 7)} mm"
                  f"  slide {s['slide_m'] * 1e3:6.3f} mm")
    else:
        print(f"\nNo sample is rotated more than {args.pivot_min_deg} deg from the reference (no screw axes).")

    # ---- nominal
    if nom_R is not None or nom_t is not None:
        print("\nDeviation from nominal:")
        out["vs_nominal"] = []
        for i in range(n):
            rec, line = {"sample": i + 1}, f"  {i + 1:3d}:"
            if nom_R is not None:
                d = R_all[i] * nom_R.inv()
                rv, rv_c = d.as_rotvec(), nom_R.inv().apply(d.as_rotvec())
                rec.update(drot_base_deg=np.degrees(rv).tolist(), drot_chart_deg=np.degrees(rv_c).tolist(),
                           angle_deg=float(np.degrees(np.linalg.norm(rv))))
                line += (f" drot base {fmt(np.degrees(rv), 3, 7)} | chart {fmt(np.degrees(rv_c), 3, 7)} deg"
                         f" | angle {rec['angle_deg']:6.3f}")
            if nom_t is not None:
                dp = P[i, :3] - nom_t
                ref_R = nom_R if nom_R is not None else R_ref
                rec.update(dpos_base_mm=(dp * 1e3).tolist(), dpos_chart_mm=(ref_R.inv().apply(dp) * 1e3).tolist())
                line += f" | dpos base {fmt(dp * 1e3)} chart {fmt(ref_R.inv().apply(dp) * 1e3)} mm"
            out["vs_nominal"].append(rec)
            print(line)

    # ---- offset points (e.g. a target in front of the chart)
    out["offsets"] = []
    for o in offsets:
        print(f"\nPoint at chart-frame offset {fmt(o * 1e3, 1, 6)} mm:")
        print("  shift vs ref : full displacement of this point (chart moved, point attached to the chart),")
        print("                 in the reference chart frame [mm]")
        if nom_R is not None:
            print("  rot vs nominal: displacement caused by the orientation error alone, i.e. the error of")
            print("                 'chart pose + offset' for a point fixed at the nominal orientation [mm, nominal chart frame]")
        p_ref = t_ref + R_ref.apply(o)
        recs = []
        for i in range(n):
            p = P[i, :3] + R_all[i].apply(o)
            sh = R_ref.inv().apply(p - p_ref) * 1e3
            rec = {"sample": i + 1, "shift_vs_ref_mm": sh.tolist()}
            line = f"  {i + 1:3d}: shift vs ref {fmt(sh)} (|{np.linalg.norm(sh):.3f}|)"
            if nom_R is not None:
                e = nom_R.inv().apply(R_all[i].apply(o) - nom_R.apply(o)) * 1e3
                rec["rot_vs_nominal_mm"] = e.tolist()
                line += f" | rot vs nominal {fmt(e)} (|{np.linalg.norm(e):.3f}|)"
            recs.append(rec)
            print(line)
        out["offsets"].append({"offset_mm": (o * 1e3).tolist(), "samples": recs})

    # ---- acceptance check
    if args.accept_deg is not None or args.accept_mm is not None:
        a_deg = np.inf if args.accept_deg is None else args.accept_deg
        a_mm = np.inf if args.accept_mm is None else args.accept_mm
        print(f"\nAcceptance vs reference (rotation <= {a_deg} deg, position <= {a_mm} mm):")
        out["acceptance"] = []
        for i, r in enumerate(rows):
            dpos = float(np.linalg.norm(r["dpos_base_mm"]))
            ok = r["angle_deg"] <= a_deg and dpos <= a_mm
            out["acceptance"].append({"sample": i + 1, "ok": bool(ok), "angle_deg": r["angle_deg"], "dpos_mm": dpos})
            print(f"  {i + 1:3d}: {'OK  ' if ok else 'FAIL'}  rot {r['angle_deg']:.3f} deg, pos {dpos:.3f} mm")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"\nWrote {args.json}")
    return out


if __name__ == "__main__":
    main()
