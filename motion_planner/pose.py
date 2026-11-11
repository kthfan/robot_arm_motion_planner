"""Conversions between TM poses and 4x4 homogeneous transforms.

A TM pose is ``[x, y, z, rx, ry, rz]`` in metres and radians, with fixed-axis
XYZ Euler angles, i.e. ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Convert a TM pose to a 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose(T: np.ndarray) -> list[float]:
    """Convert a 4x4 homogeneous transform to a TM pose."""
    T = np.asarray(T)
    euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return [float(v) for v in np.concatenate([T[:3, 3], euler])]


def pose_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (translation distance [m], rotation angle [rad]) between two transforms."""
    a, b = np.asarray(a), np.asarray(b)
    lin = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    ang = float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
    return lin, ang
