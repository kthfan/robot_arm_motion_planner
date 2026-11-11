LINE_T planning
===============

This page describes how
:meth:`~motion_planner.planner.KinematicMotionPlanner.plan_line_t` turns a
straight Cartesian line into PTP_J waypoints, and which parameters control each
stage. The line is parameterised by ``s`` in [0, 1]
(:class:`~motion_planner.path.CartesianPath`).

Pipeline
--------

1. **Goal selection.** Compute every IK solution of the goal and filter them
   with ``goal_joints`` / ``goal_selection`` (see :doc:`concepts`). The goal
   pose must be reachable, or :class:`~motion_planner.solver.IKError` is raised.
2. **Search.** :class:`~motion_planner.line_search.LineSearch` finds a joint
   path along the line, with reconfigurations where needed.
3. **Held smoothing** (``"hold"`` / ``"jump"``). Even out the joint motion of
   held reconfigurations.
4. **Simplification.** Drop the waypoints that are not needed.
5. **Timing.** Assign a speed % and ``acc_time`` to each segment.

Search
------

The IK branch of ``q0`` is *traced* along the line with adaptive steps. Each
step is seeded by the previous one and refined by Newton iterations. Every
joint interpolation between samples stays within ``linear_tolerance`` /
``angular_tolerance`` and ``max_joint_step``, with steps down to
``2**-max_depth`` of the line. A trace stops at a singularity, at a joint
limit, or at the goal.

If the trace reaches an accepted goal configuration, that is the plan.
Otherwise the joints must reconfigure. Reconfigurations are tried at *windows*:
local minima of the distance to a singularity (``sigma_min`` of the
tolerance-scaled Jacobian) below ``LineSearch.window_sigma``, and at the end of
the trace. Reconfiguring where the arm is closest to a singularity keeps plans
stable, because small changes of the start or goal move that point only
slightly.

At a window:

* **Held reconfiguration.** :class:`~motion_planner.self_motion.SelfMotion`
  walks the self-motion of the arm with the TCP held. It finds the other
  configuration families reachable within ``max_flip_deviation`` /
  ``max_flip_angle``. The goal branches are traced back along the line, and
  *links* connect the current trace to them. A link is a held route along at
  most ``max_held_length`` of the line, projected into the reconfiguration
  tolerance.
* **Jump** (``"jump"`` mode only). The joints move directly to an IK solution
  just past the window, and the TCP leaves the line. Where the line has no IK,
  the planner skips ahead to where it is reachable again.

The candidates are ranked by a best-first search over ``(s, q)`` states:

1. fewest jumps,
2. fewest held reconfigurations,
3. then one smooth score: joint travel, plus the largest jump, plus a penalty
   for joint-limit margins below 10 deg.

Discrete counts come first and the rest is a single scalar, so near-ties do not
flip the plan on tiny input changes. The search is bounded by
``LineSearch.max_flips``, ``max_expansions``, an effort budget, and the planner's
``max_plan_time``. When no plan exists,
:class:`~motion_planner.planner.PathDiscontinuityError` explains where the
start branch ends and why.

Held smoothing
--------------

A held route is projected into the tolerance waypoint by waypoint. The
resulting joint change is uneven, so the robot's joint speeds would alternate
from one segment to the next (visible jitter).
:meth:`~motion_planner.line_search.LineSearch.smooth_held` moves each inner
waypoint toward the straight joint interpolation of its neighbours, while both
links stay within ``held_smoothing_tolerance`` times the reconfiguration
tolerance and within the joint limits.

* ``held_smoothing``: number of sweeps; 0 disables smoothing.
* ``held_deviation_weight``: also pulls the TCP toward the line.
* ``held_smoothing_batch``: smooths every other waypoint at once (red-black
  ordering, much faster) instead of one by one.

Simplification
--------------

With ``simplify_line``, waypoints on the line are dropped when their neighbours
can be joined directly within the tolerances and ``max_joint_step``. The robot
then gets fewer, longer PTP_J segments. Jumps are always kept. Held routes are
also merged only with ``simplify_held``. Use it when the controller's minimum
time per segment caps the speed of routes made of many short segments.

Timing
------

The nominal duration is the slower of translation at
``velocity/100 * max_linear_velocity`` and rotation at
``velocity/100 * angular_velocity``. It uses a trapezoidal profile with
``acc_time`` ramps.

* ``"raise"`` mode: if a segment would exceed the joint velocity or
  acceleration limits (``max_joint_acceleration``), the whole line is slowed
  down uniformly, with a warning.
* ``"hold"`` / ``"jump"`` with ``smooth_timing``: one speed profile along the
  line keeps every joint within its limits. Slow sections, such as those near a
  singularity, are entered and left gradually. With ``timing_on_dense``, the
  profile is computed on the waypoints before simplification, and a merged
  segment takes the time of the waypoints it replaces.
* Reconfigurations at one point of the line, and jumps, are plain PTP_J moves
  at the requested speed %.

Each segment's speed % is then
``100 * max_j(|dq_j| / vmax_j) / duration``, clipped to
[``min_velocity``, 100]. Segments clipped at 100 % are reported as *capped*.

Tuning summary
--------------

==============================  =====================  =======================================
Parameter                       Default                Effect
==============================  =====================  =======================================
``linear_tolerance``            1 mm                   max TCP deviation on the line
``angular_tolerance``           0.5 deg                max orientation deviation on the line
``max_joint_step``              0.2 rad                max joint change per waypoint
``max_flip_deviation``          3 mm                   TCP tolerance of held reconfigurations
``max_flip_angle``              3 deg                  orientation tolerance of held ones
``max_held_length``             0.15 m                 longest held route along the line
``goal_selection``              ``"closest"``          goal by joint distance or IK branch
``smooth_timing``               True                   one speed profile (hold / jump)
``simplify_line``               True                   fewer PTP_J segments
``simplify_held``               False                  also merge held routes
``held_smoothing``              10                     smoothing sweeps (0: off)
``held_smoothing_tolerance``    0.5                    share of the flip tolerance used
``held_deviation_weight``       0.1                    pull of held routes toward the line
``held_smoothing_batch``        True                   red-black smoothing
``timing_on_dense``             True                   time before simplification
``max_plan_time``               30 s                   planning budget (None: none)
``min_velocity``                1 %                    lowest PTP_J speed %
==============================  =====================  =======================================
