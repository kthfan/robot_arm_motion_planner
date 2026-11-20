#!/usr/bin/env python3
"""Export the valid / invalid areas of a binarized reach volume as Wavefront OBJ.

OBJ: plain text, no dependency to write or read, opened by Blender, MeshLab,
trimesh, CAD importers and ROS/Gazebo alike.

Each area becomes one closed mesh made of the boundary faces of its voxels (a
voxel is the grid cell centred on a point, one step wide), so the mesh is exactly
the labelled region, with no smoothing. Triangles, outward-facing (CCW), vertices
shared. Units are metres, axes are those of the npz (tray frame: x lateral, y up,
z approach), origin at the robot base.

Colours live in a sidecar ``<stem>.mtl`` next to the OBJs (valid green, invalid red,
both ``d 0.5`` = 50 % opaque). Keep the .mtl beside the .obj files. Some viewers
ignore ``d``; in Blender set the material's blend mode if it shows opaque.

    python scripts/reach_volume/export_reach_volume_obj.py --src <stem>_binary.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

#: Never-reached voxels this close (in cells) to the outer area are the fit's skin.
SKIN_CELLS = 2
#: Invalid pieces smaller than this (voxels) are specks.
MIN_PIECE = 10


def hull_mask(valid, xs, ys, zs) -> np.ndarray:
    """Grid points whose whole voxel cell lies inside the convex hull of the valid area.

    The hull is built from the corners of the valid voxels, i.e. from the same
    cells the valid mesh is made of. A cell counts only if all 8 of its corners are
    inside, so no exported face pokes out of the hull.
    """
    axes = [np.asarray(a, dtype=float) for a in (xs, ys, zs)]
    half = np.array([float(np.diff(a).mean()) / 2.0 for a in axes])
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    centres = np.stack([X[valid], Y[valid], Z[valid]], 1)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (centres[:, None, :] + signs * half).reshape(-1, 3)
    eq = ConvexHull(corners).equations                  # n . p + offset <= 0 inside
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    inside = np.ones(len(pts), dtype=bool)
    for s in signs:
        inside &= ((pts + s * half) @ eq[:, :3].T + eq[:, 3] <= 1e-9).all(1)
    return inside.reshape(X.shape)


def invalid_area(inner, valid, levels, xs, ys, zs, skin_cells=SKIN_CELLS, min_piece=MIN_PIECE):
    """``inner & ~valid``, kept only inside the convex hull of the valid area.

    Anything outside that hull is not part of the region the valid area spans,
    so it is excluded. Also dropped: the unreached skin and specks.

    The sphere is only an approximation of the reach envelope, so a one-to-two
    cell layer just inside it is level 0 simply because the real surface is a
    little smaller. Those voxels are not a finding, so they are dropped, together
    with any separate piece under *min_piece* voxels. Reached-but-failed voxels
    (level 1..N-1) are kept, as long as they are inside the hull.
    """
    inner = np.asarray(inner, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    inv = inner & ~valid & hull_mask(valid, xs, ys, zs)
    # distance to the outer area; the grid border is not outer, so pad by edge
    dist = ndimage.distance_transform_edt(np.pad(inner, 1, mode="edge"))[1:-1, 1:-1, 1:-1]
    inv &= ~((np.asarray(levels) == 0) & (dist <= skin_cells))
    lab, n = ndimage.label(inv)
    if n:
        sizes = np.bincount(lab.ravel())
        inv &= (sizes >= min_piece)[lab] & (lab > 0)
    return inv


def voxel_mesh(mask: np.ndarray, xs, ys, zs):
    """(vertices (n,3) float, triangles (m,3) int) of the surface of *mask*."""
    mask = np.asarray(mask, dtype=bool)
    axes = [np.asarray(a, dtype=float) for a in (xs, ys, zs)]
    step = [float(np.diff(a).mean()) for a in axes]
    padded = np.pad(mask, 1)
    quads = []                                           # (k, 4, 3) in half-step integer units
    for ax in range(3):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        for s in (-1, 1):
            neighbour = np.roll(padded, -s, axis=ax)     # neighbour[i] = padded[i + s]
            idx = np.argwhere(padded & ~neighbour) - 1   # back to unpadded voxel index
            if not len(idx):
                continue
            c = 2 * idx
            c[:, ax] += s
            corners = []
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # u x v = +ax, CCW seen from +ax
                p = c.copy()
                p[:, u] += du
                p[:, v] += dv
                corners.append(p)
            q = np.stack(corners, 1)
            quads.append(q if s > 0 else q[:, ::-1])               # flip winding for the -ax faces
    if not quads:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    q = np.concatenate(quads)
    uniq, inv = np.unique(q.reshape(-1, 3), axis=0, return_inverse=True)
    inv = inv.reshape(-1, 4)
    tris = np.concatenate([inv[:, [0, 1, 2]], inv[:, [0, 2, 3]]])
    verts = np.stack([axes[k][0] + uniq[:, k] / 2.0 * step[k] for k in range(3)], 1)
    return verts, tris


#: material name -> (diffuse RGB, opacity)
MATERIALS = {"valid": ((0.0, 0.8, 0.0), 0.5), "invalid": ((0.9, 0.0, 0.0), 0.5)}


def write_mtl(path: str) -> None:
    with open(path, "w", newline="\n") as f:
        for name, (rgb, alpha) in MATERIALS.items():
            r, g, b = rgb
            f.write(f"newmtl {name}\nKa {r} {g} {b}\nKd {r} {g} {b}\nKs 0 0 0\n"
                    f"d {alpha}\nTr {1 - alpha}\nillum 1\n\n")


TRAY_AXES = "x lateral, y up, z approach"


def write_obj(path: str, verts, tris, name: str, comment: str, mtl: str | None = None,
              axes: str = TRAY_AXES) -> None:
    with open(path, "w", newline="\n") as f:
        f.write(f"# {comment}\n# units: metres; {axes}; origin at the robot base\n")
        if mtl:
            f.write(f"mtllib {mtl}\n")
        f.write(f"o {name}\n")
        if mtl:
            f.write(f"usemtl {name}\n")
        np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, tris + 1, fmt="f %d %d %d")


def export(d, stem: str, axes: str = TRAY_AXES) -> list[str]:
    paths = []
    mtl = os.path.basename(stem) + ".mtl"
    write_mtl(os.path.join(os.path.dirname(stem), mtl))
    areas = {"valid": d["valid"], "invalid": invalid_area(d["inner"], d["valid"], d["levels"],
                                              d["xs"], d["ys"], d["zs"])}
    for name, mask in areas.items():
        verts, tris = voxel_mesh(mask, d["xs"], d["ys"], d["zs"])
        path = f"{stem}_{name}.obj"
        write_obj(path, verts, tris, name,
                  f"{name} area of {os.path.basename(stem)}: {int(mask.sum())} voxels", mtl, axes)
        print(f"wrote {path}  {len(verts)} vertices, {len(tris)} triangles")
        paths.append(path)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="*_binary.npz files")
    args = ap.parse_args()
    for src in args.src:
        with np.load(src) as d:
            export({k: d[k] for k in d.files}, src[:-len("_binary.npz")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
