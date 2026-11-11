"""Field logs of the planner, as written by the robot application (RobotServer's
robot/low_level/motion_log.py, one JSON record per line).

Usage:
  python motion_log_tool.py summary LOG.jsonl [LOG2.jsonl ...]
  python motion_log_tool.py track   LOG.jsonl [...] [--limit 30]
  python motion_log_tool.py export  LOG.jsonl [...] [--out field_cases.json] [--failed-only]
  python motion_log_tool.py motions LOG.jsonl [...]
  python motion_log_tool.py profile LOG.jsonl [...] --seq N [--session PREFIX]

summary: sessions, planner settings and outcomes; track: planned vs executed
deviation from the line; export: LINE_T motions as cases for run.py
(results/<out>); motions: one line per motion (planning time, wrist sweep, speed
dips); profile: per-segment speed %, acc_time and duration of one motion.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np

from common import RESULTS, Case, Checker, get_planner, save_cases
from motion_planner.pose import pose_to_matrix

LINE_CODES = (18, 19, 20)


def load(paths):
    sessions, planners, motions, notes = {}, [], [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                kind = r.get("type")
                if kind == "session":
                    sessions[r["session"]] = r
                elif kind == "planner":
                    planners.append(r)
                elif kind == "motion":
                    motions.append(r)
                elif kind == "note":
                    notes.append(r)
    return sessions, planners, motions, notes


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else math.nan


def wkind(w: str) -> str:
    for k in ("reconfigured", "left the straight line", "slowed down", "capped"):
        if k in w:
            return k
    return w[:30]


def cmd_summary(a):
    sessions, planners, motions, notes = load(a.logs)
    sv = get_planner().solver
    print(f"== {len(sessions)} sessions, {len(motions)} motions, {len(notes)} notes")
    for s in sessions.values():
        code = s.get("code_sha256", {})
        ls_hash = next((v for k, v in code.items() if k.endswith("line_search.py")), None)
        print(f"  session {s['session']} {s['ts']} host={s.get('host')} "
              f"git={str(s.get('git_commit'))[:10]} dirty={s.get('git_dirty')} "
              f"line_search={str(ls_hash)[:10]}")
    seen = set()
    for p in planners:
        key = (p["session"], p.get("name"))
        if key in seen:
            continue
        seen.add(key)
        cal = p.get("default_calibration", {})
        diffs = [float(np.linalg.norm(sv.fk(np.asarray(x["q"]))[:3, 3]
                                      - np.asarray(x["T"])[:3, 3])) * 1e3
                 for x in p.get("fk_probe", [])]
        print(f"  planner {p.get('name')} (session {p['session']}): calibration "
              f"exists={cal.get('exists')} sha={str(cal.get('sha256'))[:10]} "
              f"created={cal.get('metadata', {}).get('created')}; "
              f"FK probe vs current model max {max(diffs, default=math.nan):.4f} mm; "
              f"upper limits {np.round(p.get('upper_limits', []), 3).tolist()}")

    local = [m for m in motions if m.get("local")]
    remote = [m for m in motions if not m.get("local")]
    print(f"\n== local motions: {len(local)}, remote: {len(remote)}")
    tab = Counter((m["motion_name"], m.get("plan_status", "no-plan"),
                   str(m.get("robot_result") or m.get("error") or "-")[:40]) for m in local)
    for (name, st, res), n in sorted(tab.items()):
        print(f"  {name:30s} plan={st:24s} robot={res:40s} n={n}")
    times = [m["plan_time_s"] for m in local if "plan_time_s" in m]
    if times:
        print(f"  plan time p50={pct(times, 50):.2f}s p95={pct(times, 95):.2f}s "
              f"max={max(times):.2f}s")
    print(f"  q0 source: {dict(Counter(m.get('q0_source') for m in local))}")
    print("  reconfigurations: " + str(dict(Counter(
        e["kind"] for m in local for e in m.get("diagnostics", {}).get("events", [])))))
    print("  warnings: " + str(dict(Counter(
        wkind(w) for m in local for w in m.get("diagnostics", {}).get("warnings", [])))))
    errs = [m["q_end_error_rad"] for m in local if m.get("q_end_error_rad") is not None]
    if errs:
        print(f"  |q_end - planned end| p95={math.degrees(pct(errs, 95)):.3f} "
              f"max={math.degrees(max(errs)):.3f} deg")
    lo, hi = sv.lower_limits, sv.upper_limits
    out = [m for m in local if m.get("q0") is not None
           and (np.any(np.asarray(m["q0"]) < lo) or np.any(np.asarray(m["q0"]) > hi))]
    print(f"  q0 outside the model's joint limits: {len(out)}")
    for m in out[:10]:
        print(f"    {m['session']}#{m['seq']} q0={np.round(m['q0'], 3).tolist()}")

    fails = [m for m in local if m.get("plan_status") not in (None, "ok") or m.get("error")]
    print(f"\n== failures ({len(fails)})")
    for m in fails[:50]:
        print(f"  {m['session']}#{m['seq']} {m['ts']} {m['motion_name']} {m.get('plan_status')} "
              f"{(m.get('plan_error') or m.get('error') or '')[:200]}")
    if notes:
        print(f"\n== notes ({len(notes)})")
        for n in notes:
            print(f"  {n['session']}#{n['seq']} {n['ts']} {n['text'][:200]}")


def cmd_track(a):
    _, _, motions, _ = load(a.logs)
    chk = Checker(get_planner())
    rows = []
    for m in motions:
        if m.get("motion_type") not in LINE_CODES or m.get("plan_status") != "ok":
            continue
        tr = m.get("trajectory") or {}
        if len(tr.get("q", [])) < 2 or not m.get("actions"):
            continue
        q0 = np.asarray(m["q0"], dtype=float)
        goal = pose_to_matrix(m["positions"])
        planned = np.array([q0] + [x[0] for x in m["actions"]])
        executed = np.vstack([q0, np.asarray(tr["q"], dtype=float)])
        rows.append((m, chk.run(q0, goal, planned), chk.run(q0, goal, executed),
                     len(tr["q"]), tr["t"][-1]))
    print(f"== {len(rows)} LINE_T motions with a sampled trajectory")
    print("   deviation from the straight line: planned waypoints vs executed samples")
    for m, cp, ce, n, dur in rows[: a.limit]:
        print(f"  {m['session']}#{m['seq']} {m['motion_name']:30s} samples={n:5d} {dur:6.2f}s "
              f"planned {cp.max_lin_mm:6.2f} mm {cp.max_ang_deg:5.2f} deg | executed "
              f"{ce.max_lin_mm:6.2f} mm {ce.max_ang_deg:5.2f} deg | end error "
              f"{ce.goal_lin_mm:6.3f} mm")
    if rows:
        d = np.array([ce.max_lin_mm - cp.max_lin_mm for _, cp, ce, _, _ in rows])
        print(f"  executed - planned max deviation: p50={np.median(d):.2f} "
              f"p95={np.percentile(d, 95):.2f} max={d.max():.2f} mm")


def cmd_motions(a):
    """One line per local motion: planning time, wrist sweep, segments, wrist speed
    alternation (alt: up/down swings of the J4/J6 segment speeds), planned vs
    executed time. Failed motions add a line with where the planning time went."""
    _, planners, motions, _ = load(a.logs)
    vmax = {p["session"]: np.asarray(p.get("velocity_limits") or [1.0] * 6) for p in planners}
    print("session#seq  mode  status   L[mm] goal_sel     events  segs calc[s]  min|q5| "
          "q4/q6 sweep[deg] alt  plan[s] moving[s]  speed dips")
    for m in motions:
        if not m.get("local"):
            print(f"  {m['session'][:6]}#{m['seq']:<3d} remote {m.get('motion_name')} "
                  f"{m.get('robot_result') or m.get('error')}")
            continue
        d = m.get("diagnostics") or {}
        acts = m.get("actions") or []
        ev = ",".join(e["kind"] for e in d.get("events", [])) or "-"
        line = (f"  {m['session'][:6]}#{m['seq']:<3d} {str(m.get('mode')):5s} "
                f"{str(m.get('plan_status'))[:8]:8s} "
                f"{1e3 * (d.get('path') or {}).get('translation_m', math.nan):6.1f} "
                f"{str(d.get('goal_selection')):11s} {ev:7s} {len(acts):4d} "
                f"{m.get('plan_time_s') or math.nan:7.2f}")
        if acts and m.get("q0") is not None:
            Q = np.array([m["q0"]] + [x[0] for x in acts])
            vm = vmax.get(m["session"], np.ones(6))
            pct_ = np.array([x[1] for x in acts]) / 100.0
            dq = np.diff(Q, axis=0)
            dur = np.max(np.abs(dq) / (pct_[:, None] * vm), axis=1)
            plan_t = float(np.sum(dur))
            alt = 0
            for j in (3, 5):
                sp_j = np.abs(dq[:, j]) / np.maximum(dur, 1e-9)
                dd = np.diff(sp_j)
                dd = dd[np.abs(dd) > 0.05 * max(float(sp_j.max(initial=0.0)), 1e-9)]
                alt += int(np.sum(np.sign(dd[1:]) != np.sign(dd[:-1])))
            line += (f"  {math.degrees(np.min(np.abs(Q[:, 4]))):6.1f} "
                     f"{math.degrees(np.ptp(Q[:, 3])):7.1f}/{math.degrees(np.ptp(Q[:, 5])):6.1f} "
                     f"{alt:4d} {plan_t:8.2f}")
            tr = m.get("trajectory") or {}
            if len(tr.get("t", [])) > 5:
                t, q = np.asarray(tr["t"]), np.asarray(tr["q"])
                sp = np.max(np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None] / vm, axis=1)
                mv = np.where(sp > 0.01)[0]
                if len(mv):
                    w = sp[mv[0]: mv[-1] + 1]
                    dips = sum(1 for i in range(2, len(w) - 2) if w[i] < w[i - 1] and w[i] <= w[i + 1]
                               and w[i] < 0.5 * min(w[:i].max(), w[i + 1:].max()))
                    line += f" {t[mv[-1] + 1] - t[mv[0]]:9.2f} {dips:6d}"
        print(line)
        if m.get("plan_status") not in (None, "ok"):
            tm = d.get("timing_s") or {}
            st = [(k, (d.get(k) or {}).get("stats") or {}) for k in ("search_any", "search_closest", "search")]
            print("      timing_s " + json.dumps({k: round(v, 3) for k, v in tm.items()})
                  + "  " + "  ".join(f"{k}: " + json.dumps(v) for k, v in st if v))


def cmd_profile(a):
    """Per PTP_J segment of one motion: speed %, acc_time, planned duration and
    TCP speed; then the TCP speed measured from the joint feedback (every
    sample). Shows speed steps between segments and where the robot crawls."""
    _, _, motions, _ = load(a.logs)
    sel = [m for m in motions if m.get("seq") == a.seq and m.get("actions")
           and (a.session is None or str(m.get("session", "")).startswith(a.session))]
    if not sel:
        print(f"no local motion with seq {a.seq}")
        return
    sv = get_planner().solver
    vm = np.asarray(sv.velocity_limits, dtype=float)
    for m in sel[:1]:
        acts = m["actions"]
        Q = np.array([m["q0"]] + [x[0] for x in acts])
        dq = np.diff(Q, axis=0)
        pct_ = np.array([x[1] for x in acts])
        acc = np.array([x[2] for x in acts])
        dur = np.max(np.abs(dq) / (pct_[:, None] / 100.0 * vm), axis=1)
        P = np.array([sv.fk(q)[:3, 3] for q in Q])
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        print(f"{str(m.get('session'))[:6]}#{m['seq']} {m['motion_name']} mode={m.get('mode')} "
              f"segments={len(acts)} planned={dur.sum():.2f}s executed={m.get('exec_time_s') or math.nan:.2f}s")
        print("  seg  speed%  acc[s]  dur[s]  step[mm]  TCP[mm/s]  change")
        prev = None
        for i in range(len(acts)):
            v = step[i] / max(dur[i], 1e-9) * 1e3
            ch = "" if prev is None or prev <= 0 else f"{100.0 * (v / prev - 1.0):+6.0f}%"
            print(f"  {i:3d} {pct_[i]:7.1f} {acc[i]:7.3f} {dur[i]:7.3f} {step[i] * 1e3:8.2f} {v:9.1f}  {ch}")
            prev = v
        tr = m.get("trajectory") or {}
        if len(tr.get("t") or []) > 2:
            t = np.array(tr["t"])
            p = np.array([sv.fk(np.asarray(q))[:3, 3] for q in tr["q"]])
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9) * 1e3
            print("  measured TCP speed [mm/s] (t from the first sample):")
            line = []
            for k in range(len(v)):
                line.append(f"{0.5 * (t[k] + t[k + 1]) - t[0]:5.2f}:{v[k]:5.1f}")
                if len(line) == 8:
                    print("   " + "  ".join(line))
                    line = []
            if line:
                print("   " + "  ".join(line))


def cmd_export(a):
    _, _, motions, _ = load(a.logs)
    cases = []
    for m in motions:
        if m.get("motion_type") not in LINE_CODES or m.get("q0") is None:
            continue
        if a.failed_only and m.get("plan_status") == "ok":
            continue
        goal = pose_to_matrix(m["positions"])
        name = f"field_{m['session'][:6]}_{m['seq']}"
        note = f"{m['motion_name']} mode={m.get('mode')} status={m.get('plan_status')} ts={m['ts']}"
        cases.append(Case(name, "field", list(m["q0"]), goal.tolist(), m.get("goal_joints"), note))
    out = RESULTS / a.out
    save_cases(cases, out)
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "track", "export", "motions", "profile"])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default="field_cases.json")
    ap.add_argument("--failed-only", action="store_true")
    ap.add_argument("--seq", type=int, help="profile: motion seq")
    ap.add_argument("--session", help="profile: session id prefix")
    a = ap.parse_args()
    {"summary": cmd_summary, "track": cmd_track, "export": cmd_export,
     "motions": cmd_motions, "profile": cmd_profile}[a.cmd](a)
