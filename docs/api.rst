API reference
=============

Package
-------

.. automodule:: motion_planner
   :no-members:

Planner
-------

.. automodule:: motion_planner.planner
   :members: MotionPlanner, KinematicMotionPlanner, JointWaypoint, PathDiscontinuityError,
             DISCONTINUITY_MODES, GOAL_SELECTIONS
   :show-inheritance:

Defaults
--------

.. automodule:: motion_planner.defaults
   :members:

Kinematics solvers
------------------

.. automodule:: motion_planner.solver
   :members:
   :show-inheritance:

.. automodule:: motion_planner.pinocchio_solver
   :members:
   :show-inheritance:

Analytical IK
-------------

.. automodule:: motion_planner.robust_eaik
   :members: RobustIK, IKResult

LINE_T search
-------------

.. automodule:: motion_planner.line_search
   :members: LineSearch, LinePlan, Event, Trace, PathDiscontinuityError

Self-motion
-----------

.. automodule:: motion_planner.self_motion
   :members:

Paths and poses
---------------

.. automodule:: motion_planner.path
   :members:

.. automodule:: motion_planner.pose
   :members:

Calibration
-----------

.. automodule:: motion_planner.calibration
   :members: KinematicCalibration, CalibrationResult, calibrate, load_pairs, fk_errors
