# motion_planner

Kinematics (FK / IK) and Cartesian motion planning (`PTP_T`, `LINE_T`) for the Techman TM5S arm.

The package only computes. It has no robot, gRPC or Techman dependency. A planner takes the current joint
configuration and a goal pose, and returns a list of `JointWaypoint`s. The caller (for example RobotServer)
sends them to the robot as blended `PTP_J` motions.

- **LINE_T** follows the straight line within 1 mm / 0.5 deg. Near singularities it can reconfigure the joints
  with the TCP held near the line (`on_discontinuity="hold"`), or jump (`"jump"`).
- **PTP_T** goes to the IK solution closest to the start, or closest to `goal_joints`.
- **Timing** turns the TCP speed % into per-segment PTP_J speed % and `acc_time`, within the joint velocity and
  acceleration limits.
- **Calibration** fits the joint and flange placements of the model from recorded joint/pose pairs.

## Install

Requires Python >= 3.10. Analytical IK uses a fork of [EAIK](https://github.com/OstermD/EAIK)
(`eaik==1.2.2+robust.1`). The fork is **not on PyPI**, so install the provided wheel first:

```bash
pip install /path/to/eaik-1.2.2+robust.1-*.whl
pip install -r requirements.txt
pip install -e .                  # or: pip install -e ".[test,docs]"
```

## Quick start

```python
import numpy as np
from motion_planner import default_planner, matrix_to_pose, pose_to_matrix

planner = default_planner("pinocchio")           # TM5S URDF + models/tm5s_calibration.json
q0 = np.radians([0, 17, 69, 11, 86, 6])          # current joints [rad]

x, y, z, rx, ry, rz = matrix_to_pose(planner.fk(q0))       # TM pose [m, rad]
goal = pose_to_matrix([x - 0.10, y, z, rx, ry, rz])        # 10 cm along -x

for wp in planner.plan_line_t(q0, goal, velocity=50, acc_time=0.2, on_discontinuity="hold"):
    print(np.degrees(wp.positions), wp.velocity, wp.acc_time, wp.blend_percentage)
```

See also `examples/quickstart.py`. From the shell (`--deg` / `--mm` for teach-pendant units):

```bash
python scripts/plan_motion.py fk   --q0 0 17 69 11 86 6 --deg --mm
python scripts/plan_motion.py ik   --goal 400 -100 300 180 0 90 --deg --mm
python scripts/plan_motion.py line --q0 0 17 69 11 86 6 --goal 300 -100 300 180 0 90 --deg --mm --mode hold
```

## Layout

| Path | Content |
| --- | --- |
| `motion_planner/planner.py` | `MotionPlanner` ABC, `KinematicMotionPlanner`, `JointWaypoint` |
| `motion_planner/line_search.py` | LINE_T joint path search (traces, held reconfigurations, jumps, smoothing) |
| `motion_planner/self_motion.py` | Self-motion walks with the TCP held (singularity reconfigurations) |
| `motion_planner/solver.py` | `KinematicsSolver` ABC (fk / ik / ik_all / branch), `IKError` |
| `motion_planner/pinocchio_solver.py` | `PinocchioSolver` (pinocchio + analytical IK) |
| `motion_planner/robust_eaik.py` | `RobustIK`: numerical layer on the EAIK fork (refinement, twin seeds, limits) |
| `motion_planner/path.py`, `pose.py` | Cartesian line, TM pose conversions |
| `motion_planner/calibration.py` | Fits the model placements from joint/pose pairs |
| `motion_planner/defaults.py` | `default_planner("pinocchio")` |
| `motion_planner/models/` | `tm5s.urdf`, `tm5s_calibration.json` |
| `scripts/plan_motion.py` | FK / IK / PTP_T / LINE_T from the command line |
| `scripts/calibrate_kinematics.py` | Calibration CLI |
| `scripts/line_motion_experiments/` | LINE_T regression harness and field-log tools ([README](scripts/line_motion_experiments/README.md)) |
| `examples/` | Usage examples |
| `docs/` | Sphinx documentation (guide + API) |
| `tests/` | pytest suite |

## Test

```bash
pytest
```

The tests need pinocchio and the EAIK fork. Without pinocchio, the solver and planner tests are skipped.

## Calibrate

```bash
python scripts/calibrate_kinematics.py joint_pose_XXXX.jsonl     # writes motion_planner/models/tm5s_calibration.json
```

The input has one JSON object per line: `{"index": i, "joints": [6 x rad], "pose_flange": [x, y, z, rx, ry, rz]}`
(m, rad). See `docs/calibration.rst`.

## LINE_T experiments

```bash
python scripts/line_motion_experiments/cases.py 0.5
python scripts/line_motion_experiments/run.py base --workers 8
python scripts/line_motion_experiments/report.py base
```

## Docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The guide covers concepts (units, waypoints, modes), the LINE_T algorithm and its tuning parameters,
calibration, the experiment harness, and integration with a robot application.
