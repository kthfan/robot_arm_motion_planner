# motion_planner

Kinematics (FK/IK) and Cartesian motion planning (`PTP_T`, `LINE_T`) for the Techman TM5S arm.
Pure computation: no robot, gRPC or Techman dependency. Planners return `JointWaypoint` lists;
the caller (e.g. RobotServer) turns them into robot commands.

## Layout

| Path | Content |
| --- | --- |
| `motion_planner/solver.py` | `KinematicsSolver` ABC (fk / ik / ik_all), `IKError` |
| `motion_planner/pinocchio_solver.py` | `PinocchioSolver` (pinocchio + EAIK analytical IK) |
| `motion_planner/planner.py` | `MotionPlanner` ABC, `KinematicMotionPlanner`, `JointWaypoint` |
| `motion_planner/line_search.py`, `self_motion.py`, `path.py`, `pose.py` | LINE_T search, wrist self-motion, Cartesian paths, pose helpers |
| `motion_planner/calibration.py` | Fit the model placements from joint/pose pairs |
| `motion_planner/defaults.py` | `default_planner("pinocchio")` (applies `models/tm5s_calibration.json` if present) |
| `motion_planner/models/` | `tm5s.urdf`, `tm5s_calibration.json` |
| `scripts/calibrate_kinematics.py` | Calibration CLI |

## Install

```bash
pip install -r requirements.txt     # EAIK fork (eaik==1.2.2+robust.1) is not on PyPI
pip install -e .
```

## Test

```bash
pytest
```

## Calibrate

```bash
python scripts/calibrate_kinematics.py joint_pose_XXXX.jsonl
```

## Docs

```bash
pip install sphinx
sphinx-build -b html docs docs/_build/html
```
