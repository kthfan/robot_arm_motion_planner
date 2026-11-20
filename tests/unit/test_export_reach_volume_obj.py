"""The exported mesh must enclose exactly the labelled voxels, facing outward."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "reach_volume"))
import export_reach_volume_obj as E   # noqa: E402


def _signed_volume(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _grid(shape=(6, 5, 7)):
    return (np.round(np.arange(shape[0]) * 0.05 - 0.1, 6),
            np.round(np.arange(shape[1]) * 0.04 + 0.2, 6),
            np.round(np.arange(shape[2]) * 0.06 - 0.3, 6))


def test_single_voxel_is_a_unit_box():
    xs, ys, zs = _grid()
    m = np.zeros((6, 5, 7), bool)
    m[2, 3, 4] = True
    v, t = E.voxel_mesh(m, xs, ys, zs)
    assert len(v) == 8 and len(t) == 12
    lo, hi = v.min(0), v.max(0)
    assert np.allclose(lo, [xs[2] - 0.025, ys[3] - 0.02, zs[4] - 0.03])
    assert np.allclose(hi - lo, [0.05, 0.04, 0.06])
    assert np.isclose(_signed_volume(v, t), 0.05 * 0.04 * 0.06)       # positive: outward


def test_volume_and_watertight_with_hole_and_edge_voxels():
    xs, ys, zs = _grid()
    m = np.zeros((6, 5, 7), bool)
    m[0:5, 0:4, 0:6] = True                                           # touches the grid border
    m[2, 2, 2] = False                                                # enclosed cavity
    v, t = E.voxel_mesh(m, xs, ys, zs)
    assert np.isclose(_signed_volume(v, t), m.sum() * 0.05 * 0.04 * 0.06)
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts % 2 == 0).all()                                    # every edge is shared


def test_obj_file_roundtrip(tmp_path):
    xs, ys, zs = _grid()
    m = np.zeros((6, 5, 7), bool)
    m[1:3, 1:3, 1:3] = True
    v, t = E.voxel_mesh(m, xs, ys, zs)
    p = tmp_path / "a.obj"
    E.write_obj(str(p), v, t, "a", "test")
    lines = p.read_text().splitlines()
    v2 = np.array([[float(x) for x in l.split()[1:]] for l in lines if l.startswith("v ")])
    t2 = np.array([[int(x) for x in l.split()[1:]] for l in lines if l.startswith("f ")]) - 1
    assert np.allclose(v2, v, atol=1e-6) and (t2 == t).all()
    assert np.isclose(_signed_volume(v2, t2), 8 * 0.05 * 0.04 * 0.06)


def test_materials_green_red_semi_transparent(tmp_path):
    xs, ys, zs = _grid((8, 8, 8))
    d = dict(xs=xs, ys=ys, zs=zs, valid=np.zeros((8, 8, 8), bool), inner=np.zeros((8, 8, 8), bool),
             levels=np.full((8, 8, 8), 3, np.int8))
    d["inner"][1:7, 1:7, 1:7] = True
    d["valid"][1:7, 1:7, 1:7] = True
    d["valid"][2:5, 2:5, 2:5] = False                    # 27-voxel hole: the invalid area
    stem = str(tmp_path / "vol")
    E.export(d, stem)
    mtl = {}
    cur = None
    for line in (tmp_path / "vol.mtl").read_text().splitlines():
        k, _, rest = line.partition(" ")
        if k == "newmtl":
            cur = mtl.setdefault(rest, {})
        elif cur is not None and rest:
            cur[k] = [float(x) for x in rest.split()]
    g, r = mtl["valid"]["Kd"], mtl["invalid"]["Kd"]
    assert g[1] > g[0] and g[1] > g[2]                 # green dominant
    assert r[0] > r[1] and r[0] > r[2]                 # red dominant
    for m in mtl.values():
        assert 0.0 < m["d"][0] < 1.0                   # slightly transparent
    for name in ("valid", "invalid"):
        lines = (tmp_path / f"vol_{name}.obj").read_text().splitlines()
        assert "mtllib vol.mtl" in lines and f"usemtl {name}" in lines


def _ball(n=21):
    g = np.indices((n, n, n)) - n // 2
    axis = (np.arange(n) - n // 2) * 0.05
    return g, np.sqrt((g ** 2).sum(0)), (axis, axis, axis)


def test_invalid_area_is_clipped_to_the_hull_of_the_valid_area():
    g, rho, (xs, ys, zs) = _ball()
    inner = rho <= 9
    levels = np.zeros(rho.shape, np.int8)
    levels[rho <= 6] = 7                                     # valid body
    band = (rho > 3.5) & (rho <= 5.5)
    levels[band] = 3                                         # failed ring inside the valid body
    pocket = (np.abs(g) <= 1).all(0)
    levels[pocket] = 0                                       # unreached pocket inside it
    speck = (g[0] == 6) & (g[1] == 0) & (g[2] == 0)
    levels[speck] = 3                                        # lone failed voxel
    shell = (rho > 7.5) & (rho <= 9)
    levels[shell] = 3                                        # failed, but outside the valid body
    valid = levels == 7

    inv = E.invalid_area(inner, valid, levels, xs, ys, zs)
    hull = E.hull_mask(valid, xs, ys, zs)

    assert inv[band].all()                                   # inside the hull: kept
    assert inv[pocket].all()
    assert not inv[speck].any()                              # speck: dropped
    assert not inv[shell].any()                              # outside the hull: dropped
    assert not (inv & ~hull).any()                           # nothing outside the hull
    assert not (inv & valid).any() and not (inv & ~inner).any()
    assert hull[valid].all()


def test_invalid_area_drops_unreached_skin_inside_the_hull():
    g, rho, (xs, ys, zs) = _ball()
    inner = rho <= 9
    levels = np.where(inner, 7, 0).astype(np.int8)
    dent = inner & (g[0] >= 6) & (np.abs(g[1]) <= 2) & (np.abs(g[2]) <= 2)
    levels[dent] = 0                                         # unreached notch in the valid cap
    valid = levels == 7
    core = dent & (g[0] == 6) & (np.abs(g[1]) <= 1) & (np.abs(g[2]) <= 1)    # deep: >= 4 cells
    skin = dent & (g[0] == 8) & (np.abs(g[1]) <= 1) & (np.abs(g[2]) <= 1)   # within 2 cells

    inv = E.invalid_area(inner, valid, levels, xs, ys, zs)
    loose = E.invalid_area(inner, valid, levels, xs, ys, zs, skin_cells=0)

    assert loose[skin].all() and loose[core].all()           # both are inside the hull
    assert inv[core].all()                                   # deep unreached: kept
    assert not inv[skin].any()                               # unreached skin: dropped