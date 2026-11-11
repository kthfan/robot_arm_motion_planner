"""Summarise experiment runs.

Usage: python scripts/line_motion_experiments/report.py TAG [TAG2 ...] [--details]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import RESULTS, joint_hausdorff


def load(tag):
    with (RESULTS / f"{tag}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def valid(rec) -> str:
    """'' if an ok plan passes the independent check, else the reason."""
    c = rec["check"]
    if c["goal_lin_mm"] > 0.1 or c["goal_ang_deg"] > 0.01:
        return "goal"
    if not c["within_limits"]:
        return "limits"
    if rec["mode"] != "jump" and c["jump_segments"]:
        return "jump-in-" + rec["mode"]
    if rec["mode"] == "raise" and c["max_lin_mm"] > 1.5:
        return "line-dev"
    return ""


def outcome(rec) -> str:
    if rec["status"] != "ok":
        return rec["status"]
    reason = valid(rec)
    return "INVALID:" + reason if reason else "ok"


def summary(recs, title):
    print(f"\n=== {title} ({len(recs)} runs)")
    base = [r for r in recs if "~p" not in r["name"]]
    table = defaultdict(lambda: defaultdict(int))
    times = defaultdict(list)
    for r in base:
        key = (r["mode"], r["suite"])
        table[key][outcome(r)] += 1
        times[key].append(r["time"])
    for mode in sorted({k[0] for k in table}):
        print(f"-- mode {mode}")
        tot = defaultdict(int)
        tt = []
        for (m, suite), counts in sorted(table.items()):
            if m != mode:
                continue
            t = np.array(times[(m, suite)])
            n = sum(counts.values())
            for k, v in counts.items():
                tot[k] += v
            tt += list(t)
            print(f"  {suite:9s} n={n:4d} ok={counts.get('ok', 0) / n:6.1%}  "
                  f"t p50={np.median(t):5.2f} p95={np.percentile(t, 95):5.2f} max={t.max():6.2f}  "
                  + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "ok"))
        n = sum(tot.values())
        tt = np.array(tt)
        print(f"  {'ALL':9s} n={n:4d} ok={tot.get('ok', 0) / n:6.1%}  "
              f"t p50={np.median(tt):5.2f} p95={np.percentile(tt, 95):5.2f} max={tt.max():6.2f}  "
              + ", ".join(f"{k}={v}" for k, v in sorted(tot.items()) if k != "ok"))
        ok = [r for r in base if r["mode"] == mode and outcome(r) == "ok"]
        if ok:
            lin = np.array([r["check"]["max_lin_mm"] for r in ok])
            flips = np.array([len(r["check"]["branches"]) - 1 for r in ok])
            exc = np.array([r["check"]["excess_travel_deg"] for r in ok])
            back = np.array([r["check"]["backtrack_mm"] for r in ok])
            print(f"  ok plans: max dev p95={np.percentile(lin, 95):.2f} max={lin.max():.2f} mm; "
                  f"branch changes hist={np.bincount(flips).tolist()}; excess travel p95="
                  f"{np.percentile(exc, 95):.0f} max={exc.max():.0f} deg; backtrack max={back.max():.1f} mm")
    stability(recs)


def stability(recs):
    by = {(r["name"], r["mode"]): r for r in recs}
    groups = defaultdict(list)
    for r in recs:
        if "~p" in r["name"]:
            groups[(r["name"].split("~p")[0], r["mode"])].append(r)
    if not groups:
        return
    stats = defaultdict(lambda: defaultdict(int))
    worst = []
    for (name, mode), perts in groups.items():
        b = by.get((name, mode))
        if b is None:
            continue
        for p in perts:
            s = stats[mode]
            s["n"] += 1
            if b["status"] != p["status"]:
                s["status-changed"] += 1
                worst.append((mode, name, "status", b["status"], p["status"]))
                continue
            if b["status"] != "ok":
                continue
            if b["check"]["branches"] != p["check"]["branches"]:
                s["branches-changed"] += 1
                worst.append((mode, name, "branches", b["check"]["branches"], p["check"]["branches"]))
            qb, qp = np.array(b["qs"]), np.array(p["qs"])
            if np.max(np.abs(qb[-1] - qp[-1])) > 0.1:
                s["goal-config-changed"] += 1
            h = joint_hausdorff(qb, qp)
            if h > 0.2:
                s["hausdorff>0.2rad"] += 1
                worst.append((mode, name, "hausdorff", round(h, 3), ""))
    print("-- stability (perturbed q0 1e-3 rad, goal 0.5 mm / 0.05 deg)")
    for mode, s in sorted(stats.items()):
        print(f"  {mode}: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    for w in worst[:15]:
        print("   ", w)


def details(recs):
    for r in recs:
        o = outcome(r)
        if o != "ok" and "~p" not in r["name"]:
            msg = r.get("error", "") or str(r.get("check", {}))
            print(f"  {r['mode']:5s} {r['name']:12s} {o:28s} {r['time']:6.2f}s {msg[:150]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--details", action="store_true")
    a = ap.parse_args()
    for tag in a.tags:
        recs = load(tag)
        summary(recs, tag)
        if a.details:
            details(recs)
