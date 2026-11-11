"""Shared default planners."""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=None)
def default_planner(name: str):
    """Shared default planner per library name.

    The pinocchio planner applies ``DEFAULT_CALIBRATION`` when that file exists
    and builds its analytical IK (EAIK) up front, so an undecomposable model
    fails here rather than during a motion.
    """
    if name == "pinocchio":
        from motion_planner import pinocchio_solver
        from motion_planner.planner import KinematicMotionPlanner

        cal = pinocchio_solver.DEFAULT_CALIBRATION
        solver = pinocchio_solver.PinocchioSolver(calibration=cal if cal.exists() else None)
        solver.analytical_ik
        return KinematicMotionPlanner(solver)
    raise ValueError(f"no default planner named '{name}'")
