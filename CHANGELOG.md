# Changelog

## Unreleased

- `KinematicsSolver.turn_joints`: the joints whose 2 pi turn is part of the configuration family are derived
  from the joint limits (range above 2 pi) instead of the hard-coded TM5S joints 1, 4, 5, 6.
- `SelfMotion` raises `NotImplementedError` for more than 6 joints (redundant arms are not supported) instead
  of walking along the null space; `SelfMotion.weakest_direction`.
- Held smoothing no longer assumes 6 joints (`np.eye(6)` in the deviation pull).
- LINE_T failure messages name the joints of the weakest Jacobian direction at a singularity instead of
  TM5S-specific wrist / elbow / shoulder guesses from `q5` and `q3`.

## 0.1.0

First release as a standalone package, split out of RobotServer (`robot/kinematics`).

- `motion_planner` package: `KinematicMotionPlanner` (PTP_T, LINE_T with raise / hold / jump), `PinocchioSolver`
  with analytical IK (`RobustIK` on the EAIK fork), kinematic calibration.
- Planners return `JointWaypoint` instead of RobotServer's `Action`. The caller converts them to PTP_J commands.
- Public API exported from `motion_planner` (`default_planner`, planner types, pose helpers, errors);
  `motion_planner.__version__`.
- Scripts: `plan_motion.py` (FK / IK / PTP_T / LINE_T CLI), `calibrate_kinematics.py`, LINE_T experiment
  harness and field-log tools in `scripts/line_motion_experiments/`.
- Sphinx docs: getting started, concepts, LINE_T algorithm and tuning, calibration, experiments, integration,
  API reference.
- EAIK fork installed from https://github.com/kthfan/EAIK (pinned commit of v1.2.2+robust.1) by
  `requirements.txt`; GitHub Actions CI runs the tests and builds the docs.
- Wrist branch label is `sign(sin q5)` and the 2 pi turn of J5 is part of the configuration family
  (J5 ranges beyond +-180 deg). Fixes jump-mode plans with `goal_joints` that ended in the other wrist
  posture.
- Tests for the package API, `RobustIK`, the scripts and the experiment harness. Two planner tests were
  updated for the current joint limits and smooth timing.
