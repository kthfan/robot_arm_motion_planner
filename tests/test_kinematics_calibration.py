import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from motion_planner import calibration as calib_mod
from motion_planner.calibration import (
    KinematicCalibration,
    calibrate,
    fk_errors,
    load_pairs,
)
from motion_planner.pinocchio_solver import PinocchioSolver
from motion_planner.pose import matrix_to_pose, pose_distance

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
Q_HOME = np.array([0.0, 0.3, 1.2, 0.2, 1.5, 0.1])
PLACEMENTS = [f"joint_{i}" for i in range(1, 7)]


def _perturbed(seed=0, lin=2e-3, ang=np.radians(0.2)):
    """A calibration whose placements are the nominal ones plus random offsets."""
    rng = np.random.default_rng(seed)
    nominal = KinematicCalibration.from_urdf()
    joints = {}
    for name, (xyz, rpy) in nominal.joints.items():
        joints[name] = (
            np.asarray(xyz) + rng.uniform(-lin, lin, 3),
            np.asarray(rpy) + rng.uniform(-ang, ang, 3),
        )
    frames = {
        name: (np.asarray(xyz) + rng.uniform(-lin, lin, 3), np.asarray(rpy) + rng.uniform(-ang, ang, 3))
        for name, (xyz, rpy) in nominal.frames.items()
    }
    return KinematicCalibration(joints=joints, frames=frames)


def _samples(solver, n, seed=1):
    rng = np.random.default_rng(seed)
    qs = rng.uniform(0.8 * solver.lower_limits, 0.8 * solver.upper_limits, size=(n, 6))
    return qs, np.array([solver.fk(q) for q in qs])


# ------------------------------------------------------------ config


def test_from_urdf_lists_all_joint_placements_and_flange():
    nominal = KinematicCalibration.from_urdf()
    assert sorted(nominal.joints) == PLACEMENTS
    assert list(nominal.frames) == ["flange"]
    np.testing.assert_allclose(nominal.joints["joint_3"][0], [0.429, 0.0, 0.0])


def test_json_roundtrip(tmp_path):
    cal = _perturbed()
    cal.metadata = {"source": "x.jsonl", "n_inliers": 3}
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = KinematicCalibration.load(path)
    assert loaded.metadata == cal.metadata
    for name in PLACEMENTS:
        np.testing.assert_allclose(loaded.joints[name][0], cal.joints[name][0])
        np.testing.assert_allclose(loaded.joints[name][1], cal.joints[name][1])
    np.testing.assert_allclose(loaded.frames["flange"][0], cal.frames["flange"][0])


def test_solver_with_nominal_calibration_matches_uncalibrated():
    plain = PinocchioSolver()
    same = PinocchioSolver(calibration=KinematicCalibration.from_urdf())
    np.testing.assert_allclose(same.fk(Q_HOME), plain.fk(Q_HOME), atol=1e-12)


def test_solver_applies_calibration_from_object_and_file(tmp_path):
    nominal = KinematicCalibration.from_urdf()
    xyz, rpy = nominal.frames["flange"]
    shifted = KinematicCalibration(
        joints=nominal.joints,
        frames={"flange": (np.asarray(xyz) + [0.0, 0.0, 0.01], rpy)},
    )
    path = tmp_path / "cal.json"
    shifted.save(path)
    plain = PinocchioSolver()
    for cal in (shifted, path, str(path)):
        s = PinocchioSolver(calibration=cal)
        lin, _ = pose_distance(s.fk(Q_HOME), plain.fk(Q_HOME))
        assert lin == pytest.approx(0.01, abs=1e-4)


def test_solver_rejects_calibration_for_unknown_joint():
    cal = KinematicCalibration(joints={"joint_9": ([0, 0, 0], [0, 0, 0])}, frames={})
    with pytest.raises(ValueError, match="joint_9"):
        PinocchioSolver(calibration=cal)


# ------------------------------------------------------------ data


def test_load_pairs_reads_joints_and_flange_pose(tmp_path):
    s = PinocchioSolver()
    q = Q_HOME
    rec = {"index": 0, "joints": list(q), "pose_flange": matrix_to_pose(s.fk(q)), "pose_tool": [0] * 6}
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(rec) + "\n\n" + json.dumps({**rec, "index": 1}) + "\n{\"trunc", encoding="utf-8")
    Q, T, idx = load_pairs(path)
    assert Q.shape == (2, 6) and T.shape == (2, 4, 4)
    assert idx.tolist() == [0, 1]
    np.testing.assert_allclose(T[0], s.fk(q), atol=1e-12)


# ------------------------------------------------------------ fitting


def test_calibrate_recovers_a_perturbed_robot():
    truth = PinocchioSolver(calibration=_perturbed())
    Q, T = _samples(truth, 40)
    result = calibrate(Q, T)
    assert result.inliers.all()
    assert result.rank == 30
    fitted = PinocchioSolver(calibration=result.calibration)
    Qt, Tt = _samples(truth, 20, seed=7)
    pos, rot = fk_errors(fitted, Qt, Tt)
    assert pos.max() < 1e-6 and rot.max() < 1e-6
    # uncalibrated model is millimetres off, so the fit did real work
    assert fk_errors(PinocchioSolver(), Qt, Tt)[0].max() > 1e-3


def test_calibrate_flags_and_ignores_outliers():
    truth = PinocchioSolver(calibration=_perturbed(seed=3))
    Q, T = _samples(truth, 40, seed=4)
    bad = [3, 17, 29]
    for i in bad:
        T[i][:3, 3] += [0.002, -0.001, 0.0015]
    result = calibrate(Q, T)
    assert np.flatnonzero(~result.inliers).tolist() == bad
    fitted = PinocchioSolver(calibration=result.calibration)
    good = np.setdiff1d(np.arange(40), bad)
    assert fk_errors(fitted, Q[good], T[good])[0].max() < 1e-6
    assert result.errors_after[0][bad].min() > 1e-3


def test_calibration_metadata_records_fit_summary():
    truth = PinocchioSolver(calibration=_perturbed())
    Q, T = _samples(truth, 30)
    meta = calibrate(Q, T).calibration.metadata
    assert meta["n_samples"] == 30 and meta["n_inliers"] == 30
    assert meta["identifiable_parameters"] == 30
    assert meta["pos_rms_after_m"] < 1e-6


# ------------------------------------------------------------ wiring


def test_default_planner_uses_calibration_file(tmp_path, monkeypatch):
    from motion_planner import default_planner, pinocchio_solver

    path = tmp_path / "cal.json"
    cal = _perturbed()
    cal.save(path)
    monkeypatch.setattr(pinocchio_solver, "DEFAULT_CALIBRATION", path)
    default_planner.cache_clear()
    try:
        planner = default_planner("pinocchio")
        expected = PinocchioSolver(calibration=cal).fk(Q_HOME)
        np.testing.assert_allclose(planner.fk(Q_HOME), expected, atol=1e-12)
    finally:
        default_planner.cache_clear()


def test_default_planner_without_calibration_file_uses_urdf(tmp_path, monkeypatch):
    from motion_planner import default_planner, pinocchio_solver

    monkeypatch.setattr(pinocchio_solver, "DEFAULT_CALIBRATION", tmp_path / "missing.json")
    default_planner.cache_clear()
    try:
        planner = default_planner("pinocchio")
        np.testing.assert_allclose(planner.fk(Q_HOME), PinocchioSolver().fk(Q_HOME), atol=1e-12)
    finally:
        default_planner.cache_clear()


def test_calibrate_script_writes_config(tmp_path, monkeypatch, capsys):
    truth = PinocchioSolver(calibration=_perturbed())
    Q, T = _samples(truth, 30)
    data = tmp_path / "pairs.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for i, (q, t) in enumerate(zip(Q, T)):
            f.write(json.dumps({"index": i, "joints": q.tolist(), "pose_flange": matrix_to_pose(t)}) + "\n")
    out = tmp_path / "cal.json"
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    import calibrate_kinematics

    calibrate_kinematics.main([str(data), "-o", str(out), "--folds", "3"])
    report = capsys.readouterr().out
    assert "outliers: none" in report.lower()
    fitted = PinocchioSolver(calibration=out)
    assert fk_errors(fitted, Q, T)[0].max() < 1e-6
    assert KinematicCalibration.load(out).metadata["source"] == "pairs.jsonl"
