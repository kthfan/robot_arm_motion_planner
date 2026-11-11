"""Plan a LINE_T and a PTP_T motion and print the joint waypoints.

Run from the repository root: ``python examples/quickstart.py``
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from motion_planner import (  # noqa: E402
    PathDiscontinuityError,
    default_planner,
    matrix_to_pose,
    pose_to_matrix,
)

planner = default_planner("pinocchio")  # TM5S model + models/tm5s_calibration.json

# Start configuration [rad] and the flange pose there (x, y, z [m], rx, ry, rz [rad]).
q0 = np.radians([0.0, 17.0, 69.0, 11.0, 86.0, 6.0])
start = matrix_to_pose(planner.fk(q0))
print("start pose:", np.round(start, 4))

# LINE_T: 10 cm along -x of the base frame, same orientation, 50% speed.
goal = pose_to_matrix([start[0] - 0.10, start[1], start[2], *start[3:]])
diagnostics = {}
try:
    line = planner.plan_line_t(q0, goal, velocity=50, acc_time=0.2,
                               on_discontinuity="hold", diagnostics=diagnostics)
except PathDiscontinuityError as exc:  # no continuous joint path, not even with held flips
    sys.exit(f"LINE_T failed: {exc}")
print(f"LINE_T: {len(line)} PTP_J waypoints, {len(diagnostics['events'])} reconfiguration(s)")
for wp in line:
    print("  ", np.round(np.degrees(wp.positions), 2), f"{wp.velocity:5.1f}%", f"{wp.acc_time:.3f}s")

# PTP_T: joint-space move to a pose; the TCP path is not constrained.
ptp = planner.plan_ptp_t(q0, goal, velocity=30, acc_time=0.2)
print("PTP_T goal joints [deg]:", np.round(np.degrees(ptp[-1].positions), 2))

lin_err = np.linalg.norm(planner.fk(line[-1].positions)[:3, 3] - goal[:3, 3])
print(f"LINE_T end position error: {lin_err * 1e3:.4f} mm, "
      f"joint travel {math.degrees(float(np.abs(np.diff([q0] + [w.positions for w in line], axis=0)).sum())):.1f} deg")
