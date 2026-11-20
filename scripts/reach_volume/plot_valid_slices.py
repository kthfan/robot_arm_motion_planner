"""Scatter-plot the valid (level 7 = carry) points of a reach_volume npz, one panel per z slice.

    python scripts/reach_volume/plot_valid_slices.py out/preview_planner.npz [--out out/valid_slices.png]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_LEVEL = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()
    d = np.load(args.src)
    xs, ys, zs, lv = d["xs"], d["ys"], d["zs"], d["levels"]
    out = args.out or args.src[:-4] + "_valid_slices.png"
    valid = lv == VALID_LEVEL  # (x, y, z)
    step = xs[1] - xs[0]
    xlim = (xs[0] - step, xs[-1] + step)
    ylim = (ys[0] - step, ys[-1] + step)
    xt = np.round(np.arange(np.ceil(xs[0] * 2) / 2, xs[-1] + 1e-9, 0.5), 1)
    yt = np.round(np.arange(0, ys[-1] + 1e-9, 0.5), 1)
    n = len(zs)
    rows = int(np.ceil(n / args.cols))
    fig, axes = plt.subplots(rows, args.cols, figsize=(3.2 * args.cols, 2.9 * rows),
                             sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        ix, iy = np.nonzero(valid[:, :, k])
        ax.scatter(xs[ix], ys[iy], s=14, c="tab:blue")
        ax.set_title(f"z = {zs[k]:+.1f} m  ({len(ix)} pts)", fontsize=9)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=True, labelleft=True, labelsize=7)
    fig.supxlabel("x (m)"); fig.supylabel("y (m)")
    fig.suptitle(f"Valid points per z slice ({int(valid.sum())} total)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
