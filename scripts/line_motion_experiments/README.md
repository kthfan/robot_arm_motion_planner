# LINE_T experiments

A regression and diagnosis harness for `KinematicMotionPlanner.plan_line_t`.
It generates seeded cases, plans each one in the requested `on_discontinuity`
modes, and checks every plan with an independent checker. The checker samples
each PTP_J segment densely through FK, because the robot interpolates joints
linearly between waypoints. It does not trust the planner's own diagnostics.

Results go to `results/` (git-ignored). All commands run from the repository root
and use the default planner: the TM5S model plus `motion_planner/models/tm5s_calibration.json`.

| Script | Purpose |
| --- | --- |
| `cases.py` | Generates `results/cases.json` (seeded suites: `user`, `random`, `wrist` / `elbow` / `shoulder` singularities, `limits`, `rotation`, `long`, `tiny`, `hole`, `goal_joints`). |
| `run.py` | Plans every case and mode, then writes `results/TAG.jsonl`: one record per case and mode with status, time, warnings, waypoints and check. |
| `report.py` | Prints per-suite success rate, planning time, error counts, deviation from the line, branch changes and joint travel for one or more tags. |
| `recheck.py` | Recomputes the checks of a recorded run with the current `Checker`, without re-planning. |
| `oracle.py` | Dense branch-graph oracle: decides whether a failed case is feasible at all with held reconfigurations. It runs independently of the planner's search. |
| `motion_log_tool.py` | Reads field logs written by the robot application: summary, executed vs planned deviation, per-segment speed profile, and export of field motions as cases. |
| `common.py` | Shared code: `Case`, `Checker`, `get_planner()`. |

## Typical loop

```bash
python scripts/line_motion_experiments/cases.py 0.5             # ~460 cases
python scripts/line_motion_experiments/run.py base --workers 8   # modes hold,jump,raise
# ... change the planner ...
python scripts/line_motion_experiments/run.py new --workers 8
python scripts/line_motion_experiments/report.py base new
```

Useful `run.py` options:

| Option | Effect |
| --- | --- |
| `--modes hold` | Plans only the given modes (comma-separated). |
| `--suites user,wrist` / `--names user1,random7` | Restricts the run to some cases. |
| `--perturb 3` | Adds 3 perturbed copies per case (1 mrad joints, 0.5 mm / 0.05 deg goal) to test stability under small input changes. |
| `--timeout 60` | Records a plan as `Timeout` after 60 s. |
| `--cases field_cases.json` | Uses another case file in `results/`. |

To compare against another checkout (for example a baseline commit), set
`LINE_EXP_ROOT=/path/to/other/checkout`. The scripts then import
`motion_planner` from there.

## Field logs

The robot application can log every planned motion: start joints, goal, waypoints,
and the sampled executed trajectory. To turn those logs into cases and plan them
again with the current code:

```bash
python scripts/line_motion_experiments/motion_log_tool.py summary motions_20261106.jsonl
python scripts/line_motion_experiments/motion_log_tool.py export  motions_20261106.jsonl --out field_cases.json
python scripts/line_motion_experiments/run.py field --cases field_cases.json --modes hold
python scripts/line_motion_experiments/report.py field
```

`track` compares the planned deviation from the line with the executed one.
`profile --seq N` lists speed %, acc_time and duration per PTP_J segment of one
motion, which is useful for jitter and slow-segment analysis.
