"""Library-agnostic kinematics and motion planning (FK/IK, PTP_T, LINE_T).

The planner types depend on numpy only; :class:`~motion_planner.pinocchio_solver.PinocchioSolver`
(pinocchio + EAIK) is imported on first use by :func:`default_planner`.
"""
from motion_planner.defaults import default_planner
from motion_planner.planner import (
    DISCONTINUITY_MODES,
    GOAL_SELECTIONS,
    JointWaypoint,
    KinematicMotionPlanner,
    MotionPlanner,
    PathDiscontinuityError,
)
from motion_planner.pose import matrix_to_pose, pose_distance, pose_to_matrix
from motion_planner.solver import IKError, KinematicsSolver

__version__ = "0.1.0"

__all__ = [
    "DISCONTINUITY_MODES",
    "GOAL_SELECTIONS",
    "IKError",
    "JointWaypoint",
    "KinematicMotionPlanner",
    "KinematicsSolver",
    "MotionPlanner",
    "PathDiscontinuityError",
    "default_planner",
    "matrix_to_pose",
    "pose_distance",
    "pose_to_matrix",
    "__version__",
]
