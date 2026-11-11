motion_planner
==============

Kinematics (FK / IK) and Cartesian motion planning (``PTP_T``, ``LINE_T``) for
the Techman TM5S arm.

The package is pure computation. It has no robot, gRPC or Techman SDK
dependency. A planner takes the current joint configuration and a Cartesian
goal, and returns a list of joint-space waypoints
(:class:`~motion_planner.planner.JointWaypoint`). The caller sends these to the
robot as ``PTP_J`` motions with blending.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   getting_started
   concepts
   line_t
   calibration
   experiments
   integration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
