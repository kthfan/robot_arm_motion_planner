Integration
===========

The planner does not talk to a robot. A robot application:

1. reads the current joint configuration ``q0`` (radians) from the controller;
2. calls :meth:`~motion_planner.planner.KinematicMotionPlanner.plan_line_t` or
   :meth:`~motion_planner.planner.KinematicMotionPlanner.plan_ptp_t` with the
   goal pose, speed % and ramp time;
3. sends each :class:`~motion_planner.planner.JointWaypoint` as a ``PTP_J``
   motion with its ``velocity``, ``acc_time``, ``blend_percentage``,
   ``fine_goal`` and ``blocked``.

A minimal adapter:

.. code-block:: python

   from motion_planner import PathDiscontinuityError, IKError, default_planner, pose_to_matrix

   PTP_J = 1
   planner = default_planner("pinocchio")

   def line_t(robot, pose, velocity, acc_time, blend=0, fine_goal=False, blocked=True):
       q0 = robot.read_joints()                        # [rad]
       goal = pose_to_matrix(pose)                     # TM pose [m, rad]
       try:
           waypoints = planner.plan_line_t(q0, goal, velocity, acc_time, blend, fine_goal,
                                           blocked, on_discontinuity="hold")
       except (IKError, PathDiscontinuityError) as exc:
           raise RuntimeError(f"LINE_T not executable: {exc}") from exc
       for wp in waypoints:
           robot.send(PTP_J, wp.positions, wp.velocity, wp.acc_time,
                      wp.blend_percentage, wp.fine_goal, wp.blocked)

Notes:

* Plan against the same model the controller uses. The TCP of the planner is
  the ``flange`` unless ``tcp_offset`` is given. If the controller pose
  includes a tool, convert the goal to the flange first or configure the same
  tool.
* The URDF joint limits must not exceed the controller's limits. Plans stay
  within the URDF limits.
* ``default_planner`` builds the analytical IK when it is first called, which
  takes some time. Call it once at start-up.
* On the field robot, a TMflow project speed below 100 % was observed to
  stretch the segments but not their blending, which changes the executed
  speed profile. Plan with the speed you intend to run.
* Log ``diagnostics`` with the motion (the robot application's motion log)
  so that field failures can be planned again with
  ``scripts/line_motion_experiments/motion_log_tool.py``.
