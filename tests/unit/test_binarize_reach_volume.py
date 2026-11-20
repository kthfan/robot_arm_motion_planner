"""Synthetic check of the sphere fit and the outer / invalid / valid split: a
half sphere (flat cut where the grid ends) with enclosed holes, a surface notch
and an unreachable core touching the cut."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "reach_volume"))
import binarize_reach_volume as B   # noqa: E402

STEP = 0.025
C0 = np.array([0.02, 0.10, 0.25])
R0 = 0.5
FULL = 7


def _volume():
    xs = np.round(np.arange(-0.7, 0.7001, STEP), 6)
    ys = np.round(np.arange(-0.6, 0.8001, STEP), 6)
    zs = np.round(np.arange(0.0, 1.2001, STEP), 6)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X, Y, Z], -1)
    dist = np.linalg.norm(P - C0, axis=-1)
    inside = dist <= R0
    levels = np.where(inside, FULL, 0).astype(np.int8)
    rng = np.random.default_rng(0)
    partial = inside & (rng.random(inside.shape) < 0.08)
    levels[partial] = 3                                      # reached, not every action
    hole = np.linalg.norm(P - (C0 + [0.1, 0.1, 0.25]), axis=-1) < 0.07
    levels[hole] = 0                                         # enclosed void
    core = np.linalg.norm(P - [C0[0], C0[1], 0.0], axis=-1) < 0.15
    levels[core] = 0                                         # unreachable core on the cut face
    notch = (dist > R0 - 0.1) & (X > 0.2) & (Y > 0.35) & (Y < 0.45)
    levels[notch] = 0                                        # notch open to the outside
    d = dict(xs=xs, ys=ys, zs=zs, levels=levels, phases=np.arange(FULL).astype(str),
             preset=np.array("synthetic"))
    return d, inside, hole


def test_sphere_recovered_despite_holes_notch_and_core():
    d, _, _ = _volume()
    out = B.binarize(d)
    assert np.linalg.norm(out["sphere_centre"] - C0) < 0.02
    assert abs(float(out["sphere_radius"]) - R0) < 0.02


def test_split_follows_inside_and_level():
    d, inside, hole = _volume()
    out = B.binarize(d)
    lv, lab = d["levels"], out["label"]
    assert (lab[hole] == B.INVALID).all()                    # enclosed hole is inner, invalid
    assert (lab[(lv == FULL)] == B.VALID).all()              # all actions passed -> valid
    assert (lab[(lv > 0) & (lv < FULL)] == B.INVALID).all()  # partial -> invalid
    assert (lab == B.VALID).sum() == (lv == FULL).sum()      # nothing else is valid
    far = np.linalg.norm(np.stack(np.meshgrid(d["xs"], d["ys"], d["zs"], indexing="ij"), -1) - C0,
                         axis=-1) > R0 + 2 * STEP
    assert (lab[far] == B.OUTER).all()
    assert out["valid"].dtype == bool and out["inner"].dtype == bool
    assert ((lab == B.OUTER) == ~out["inner"]).all()
    # the core touching the cut face is part of the inner area too
    assert (lab[(inside) & (lv == 0) & ~hole] != B.OUTER).sum() > 0
