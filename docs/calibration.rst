Calibration
===========

The URDF describes the nominal TM5S. A real arm differs by fractions of a
millimetre and degree in its joint zero offsets, link lengths and twists, base
and flange. Those differences become the TCP error of every plan.
:mod:`motion_planner.calibration` fits these placements from recorded pairs of
joint angles and controller-reported flange poses.

Model
-----

The fit replaces the placement of every joint relative to its parent, and of the
``flange`` frame, with a full rigid transform. For a 6R arm only 30 of these 42
parameters are identifiable from end-effector poses. A truncated-SVD
Gauss-Newton step keeps the unidentifiable directions at their URDF values.
Outliers are detected, then excluded from the fit.

Data
----

The input is one JSON object per line:

.. code-block:: text

   {"index": 0, "joints": [q1, q2, q3, q4, q5, q6], "pose_flange": [x, y, z, rx, ry, rz]}

``joints`` are in radians. ``pose_flange`` is the TM pose of the flange, as
reported by the controller (m, rad). The robot application records this file;
in RobotServer that is ``record_joint_pose.py``. Spread the samples over the
whole workspace and every joint range. The calibration script prints the
coverage, plus a learning curve that shows whether more samples would still
help.

Fit
---

.. code-block:: bash

   python scripts/calibrate_kinematics.py joint_pose_XXXX.jsonl              # -> models/tm5s_calibration.json
   python scripts/calibrate_kinematics.py data.jsonl -o my_cal.json --folds 5

The report shows:

* joint coverage against the URDF limits;
* the number of identifiable parameters and the condition of the problem;
* errors of the uncalibrated and calibrated model;
* outliers (record index and error);
* k-fold cross-validation, i.e. the error on samples not used in the fit.

Use
---

``default_planner("pinocchio")`` applies ``models/tm5s_calibration.json``
automatically when the file exists. You can also pass a calibration explicitly:

.. code-block:: python

   from motion_planner.pinocchio_solver import PinocchioSolver
   from motion_planner.planner import KinematicMotionPlanner

   solver = PinocchioSolver(calibration="my_cal.json")
   planner = KinematicMotionPlanner(solver)

The calibration file format is::

    {"joints": {"joint_1": {"xyz": [...], "rpy": [...]}, ...},
     "frames": {"flange": {"xyz": [...], "rpy": [...]}},
     "metadata": {...}}

``xyz`` is in metres and ``rpy`` in radians, using the URDF ``<origin>``
convention.
