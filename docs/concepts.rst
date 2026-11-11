Concepts
========

Units and frames
----------------

* Joint vectors are numpy arrays of 6 angles in **radians**.
* Poses are 4x4 homogeneous transforms of the TCP in the robot base frame, in
  **metres**. By default the TCP is the ``flange`` frame of the URDF;
  ``PinocchioSolver(tcp_offset=...)`` adds a tool.
* A *TM pose* is ``[x, y, z, rx, ry, rz]`` (m, rad) with fixed-axis XYZ Euler
  angles, ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``. Convert with
  :func:`~motion_planner.pose.pose_to_matrix` and
  :func:`~motion_planner.pose.matrix_to_pose`. The TM teach pendant shows
  millimetres and degrees.

Components
----------

:class:`~motion_planner.solver.KinematicsSolver`
    Abstract FK / IK interface: ``fk``, ``jacobian``, ``ik`` (seeded),
    ``ik_all`` (all solutions), ``ik_analytical``, ``branch`` and the joint and
    velocity limits.

:class:`~motion_planner.pinocchio_solver.PinocchioSolver`
    The implementation. Pinocchio provides FK, Jacobians and damped
    least-squares IK. :class:`~motion_planner.robust_eaik.RobustIK` provides
    analytical IK: EAIK on an idealized model, then a Newton refinement on the
    original (calibrated) model. A calibration JSON can be applied on top of the
    URDF (see :doc:`calibration`).

:class:`~motion_planner.planner.KinematicMotionPlanner`
    Turns Cartesian goals into joint waypoints. Its constructor arguments are
    the tuning parameters (tolerances, speeds, reconfiguration limits, timing
    options).

:func:`~motion_planner.defaults.default_planner`
    ``default_planner("pinocchio")`` returns a shared planner. It uses the
    bundled TM5S URDF and ``models/tm5s_calibration.json`` when that file
    exists.

IK branches
-----------

The TM5S has up to 8 analytical IK solutions per pose, plus ``2 pi`` variants
of the joints whose range exceeds a full turn. ``solver.branch(q)`` labels a
configuration with the signs ``(shoulder, elbow, wrist)``. Near the wrist
singularity (``q5 ~ 0``) the label does not tell the two wrist postures apart,
so LINE_T selects goals by joint-space distance by default
(``goal_selection="closest"``).

Waypoints
---------

Both planners return a list of :class:`~motion_planner.planner.JointWaypoint`:

=====================  ==========================================================
``positions``          6 joint angles [rad] at the end of the segment
``velocity``           PTP_J speed %, in (0, 100] of the joint speed limits
``acc_time``           ramp time [s]
``blend_percentage``   100 on every LINE_T waypoint except the last one, which
                       gets the requested blend
``fine_goal``          only the last waypoint gets the requested value
``blocked``            only the last waypoint gets the requested value
=====================  ==========================================================

On a TM controller, a PTP_J segment lasts ``max_j |dq_j| / (velocity/100 *
vmax_j)``. The planner chooses ``velocity`` per segment so that the segments
together follow the requested TCP speed profile.

PTP_T
-----

:meth:`~motion_planner.planner.KinematicMotionPlanner.plan_ptp_t` returns one
waypoint: the IK solution of the goal closest to ``q0`` (or, with
``goal_joints``, the closest one in that configuration). The TCP path between
start and goal is not constrained.

LINE_T
------

:meth:`~motion_planner.planner.KinematicMotionPlanner.plan_line_t` follows the
straight line from ``FK(q0)`` to the goal. Position is interpolated linearly
and orientation by slerp. Every joint interpolation between consecutive
waypoints stays within ``linear_tolerance`` / ``angular_tolerance`` of the
line. ``velocity`` is the speed % like TM LINE_T: 100 % is
``max_linear_velocity`` (translation) and ``angular_velocity`` (rotation).

``on_discontinuity`` decides what happens where the start IK branch cannot
reach the goal (singularity, joint limit, goal in another configuration):

``"raise"``
    :class:`~motion_planner.planner.PathDiscontinuityError` (or
    :class:`~motion_planner.solver.IKError` if the goal has no IK).
``"hold"``
    The joints reconfigure to another branch near a singularity while the TCP
    stays within ``max_flip_deviation`` / ``max_flip_angle`` of the line
    (a *held* reconfiguration). Fewest reconfigurations first.
``"jump"``
    Like ``"hold"``. Where no held route exists, the joints jump (the TCP
    leaves the line) and points without IK are skipped.

``goal_joints`` fixes the goal configuration. In ``"raise"`` / ``"hold"`` mode
the goal is the solution closest to ``goal_joints``. If the line cannot reach
it, the planner falls back to the IK branch of ``goal_joints`` and warns.
Without ``goal_joints`` in ``"hold"`` mode, the plan to the solution closest to
the start and the plan to any solution are both computed, and the one with less
joint travel wins.

Diagnostics and errors
----------------------

Pass ``diagnostics={}`` to either planner to have it filled with what the
planner found: path length, goal selection, reconfiguration events, search
counters, timing and warnings. It is filled even when the planner raises. The
same warnings are issued through :mod:`warnings`.

* :class:`~motion_planner.solver.IKError`: the goal (or a required
  configuration) has no IK solution.
* :class:`~motion_planner.planner.PathDiscontinuityError`: no continuous joint
  path along the line in the requested mode, or the search ran past
  ``max_plan_time``.
* :class:`ValueError`: invalid arguments (speed % outside (0, 100], unknown
  mode, ``goal_joints`` of the wrong length, ...).
