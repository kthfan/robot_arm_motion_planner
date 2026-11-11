"""Recompute the independent check of recorded plans with the current Checker
(no re-planning).

Usage: python recheck.py TAG [--perturb 3] [--workers 16] [--cases cases.json]  -> results/TAG_rc.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from common import RESULTS, Checker, get_planner, load_cases
from run import perturbed

_checker = None
_cases = None


def _init(n_perturb: int, cases_file: str = "cases.json"):
    global _checker, _cases
    _checker = Checker(get_planner())
    cases = load_cases(RESULTS / cases_file)
    _cases = {c.name: c for c in cases + perturbed(cases, n_perturb)}


def recheck(rec):
    if "qs" in rec:
        c = _cases[rec["name"]]
        rec["check"] = asdict(_checker.run(np.asarray(c.q0, dtype=float), c.goal_matrix,
                                           np.asarray(rec["qs"], dtype=float)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--perturb", type=int, default=3)
    ap.add_argument("--cases", default="cases.json")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    a = ap.parse_args()
    recs = [json.loads(x) for x in (RESULTS / f"{a.tag}.jsonl").open(encoding="utf-8")]
    out = RESULTS / f"{a.tag}_rc.jsonl"
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as f, ProcessPoolExecutor(
            a.workers, initializer=_init, initargs=(a.perturb, a.cases)) as pool:
        for n, rec in enumerate(pool.map(recheck, recs, chunksize=8), 1):
            f.write(json.dumps(rec) + "\n")
            if n % 500 == 0:
                print(f"{n}/{len(recs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
