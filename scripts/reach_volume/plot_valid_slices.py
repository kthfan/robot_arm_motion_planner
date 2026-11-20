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
    ap.add_argument("--tick", type=float, default=0.1, help="tick spacing (m)")
    ap.add_argument("--size", type=float, default=26, help="scatter marker size")
    ap.add_argument("--only-nonempty", action="store_true", help="skip slices without valid points")
    args = ap.parse_args()
    d = np.load(args.src)
    xs, ys, zs, lv = d["xs"], d["ys"], d["zs"], d["levels"]
    out = args.out or args.src[:-4] + "_valid_slices.png"
    valid = lv == VALID_LEVEL  # (x, y, z)
    step = xs[1] - xs[0]
    xlim = (xs[0] - step, xs[-1] + step)
    ylim = (ys[0] - step, ys[-1] + step)
    xt = np.round(np.arange(xs[0], xs[-1] + 1e-9, args.tick), 6)
    yt = np.round(np.arange(ys[0], ys[-1] + 1e-9, args.tick), 6)
    kz = [k for k in range(len(zs)) if valid[:, :, k].any()] if args.only_nonempty else list(range(len(zs)))
    n = len(kz)
    rows = int(np.ceil(n / args.cols))
    fig, axes = plt.subplots(rows, args.cols, figsize=(6.6 * args.cols, 4.6 * rows),
                             sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        kk = kz[k]
        ix, iy = np.nonzero(valid[:, :, kk])
        ax.scatter(xs[ix], ys[iy], s=args.size, c="tab:blue")
        ax.set_title(f"z = {zs[kk]:+.2f} m  ({len(ix)} pts)", fontsize=11)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=True, labelleft=True, labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=90)
    fig.supxlabel("x (m)"); fig.supylabel("y (m)")
    fig.suptitle(f"Valid points per z slice ({int(valid.sum())} total)", fontsize=18)
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.975), h_pad=2.5, w_pad=1.5)
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()
