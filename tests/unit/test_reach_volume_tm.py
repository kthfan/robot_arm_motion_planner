"""Offline tests for reach_volume_tm (no controller needed)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "reach_volume"))

import reach_volume_tm as RV  # noqa: E402


def test_flange_poses_match_spec_table():
    L, zc, x = 0.7, 0.2, -0.3
    P = RV.flange_poses((x, L, zc))
    # spec Section 3: (approach distance, height) per action, metres
    want = [(L - 0.394217, zc - 0.075155), (L - 0.321017, zc - 0.075155),
            (L - 0.331, zc + 0.07725), (L - 0.081, zc + 0.07725),
            (L - 0.081, zc + 0.09850), (L - 0.331, zc + 0.09850), (L - 0.331, zc + 0.09850)]
    np.testing.assert_allclose(P[:, 0], x * 1e3)                      # lateral offset 0
    np.testing.assert_allclose(P[:, 1:3], np.array(want) * 1e3, atol=1e-9)
    np.testing.assert_allclose(P[:, 3:], [[90, 0, 180]] * 7)


def test_orientation_axes():
    """TM Euler RX,RY,RZ = Rz @ Ry @ Rx; flange z = +Y, y = +Z, x = -X."""
    rx, ry, rz = np.radians(RV.RPY)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    np.testing.assert_allclose(R[:, 0], [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(R[:, 1], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(R[:, 2], [0, 1, 0], atol=1e-12)


def test_carry_equals_withdraw_and_distinct_phases():
    P = RV.flange_poses((0.0, 0.5, 0.0))
    np.testing.assert_array_equal(P[6], P[5])
    assert len({tuple(p) for p in P[:RV.DISTINCT]}) == RV.DISTINCT


def test_decode_column_maps_6_to_7_and_splits():
    payload = np.array([0, 3, 6, 6, 0, 3, 2, 6], dtype="<i4").tobytes()
    ik, lc = RV.decode_column(payload, 4, collision=True)
    np.testing.assert_array_equal(ik, [0, 3, 7, 7])
    np.testing.assert_array_equal(lc, [0, 3, 2, 7])
    ik2, lc2 = RV.decode_column(payload, 4, collision=False)
    np.testing.assert_array_equal(lc2, ik2)


def test_decode_column_rejects_wrong_length():
    with pytest.raises(ValueError):
        RV.decode_column(np.zeros(3, dtype="<i4").tobytes(), 4, collision=False)


def test_column_script_uses_given_column():
    s = RV.column_script(0.1, 0.65, [-0.05, 0.0], collision=False)
    assert "float rvX = 100.0000" in s and "float rvY = 650.0000" in s
    assert "{-50.0000,0.0000}" in s and "CollisionCheck" in s     # guarded by acol=false
    assert "bool acol = false" in s
    assert s.rstrip().endswith("}") and "ListenSend(90, rvLV)" in s


def test_import_does_not_need_the_controller_client():
    """The phase setting is reused offline (reach_volume_planner): no tm_eval / robot import."""
    import subprocess
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "reach_volume")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import reach_volume_tm; "
            "bad = [m for m in ('tm_eval', 'robot') if m in sys.modules]; sys.exit(str(bad) if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code, here], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_axis_counts_match_spec():
    assert (len(RV.axis(-1.2, 1.2, 0.05)), len(RV.axis(0.0, 1.4, 0.05))) == (49, 29)
