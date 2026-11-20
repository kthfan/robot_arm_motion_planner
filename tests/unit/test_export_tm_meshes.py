"""Offline tests for export_tm_meshes."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "reach_volume"))

import export_tm_meshes as X  # noqa: E402


def test_tray_point_offsets_in_base_frame():
    d = {"xs": np.array([-0.05, 0.0, 0.05]), "ys": np.array([0.0, 0.05]), "zs": np.array([-0.1, 0.1]),
         "levels": np.arange(12, dtype=np.int8).reshape(3, 2, 2), "levels_ik": np.zeros((3, 2, 2)),
         "phases": np.array(["a"]), "preset": np.array("strict")}
    out = X.to_tray_point(d)
    np.testing.assert_allclose(out["xs"], d["xs"])                      # no flip
    np.testing.assert_allclose(out["ys"], d["ys"] + 0.0335)
    np.testing.assert_allclose(out["zs"], d["zs"] + 0.050)
    np.testing.assert_array_equal(out["levels"], d["levels"])           # no permutation
    np.testing.assert_array_equal(X.to_tray_point(d, "levels_ik")["levels"], d["levels_ik"])
