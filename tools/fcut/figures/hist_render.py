"""
Render a historical fcut diagnostic as vector, from the code that
implemented it.

Runs the package as it stood at a given commit, checked out in a
worktree. The old code imports flat, so the package directory itself
goes on sys.path. `r2_plots` hardcodes a PNG path, so `plt.savefig` is
wrapped to emit SVG alongside.

Usage:
    hist_render.py WORKTREE SIGNAL.txt OUTDIR
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 200
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def main():
    worktree, signal, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    pkg = os.path.join(worktree, 'weaselytics')
    sys.path.insert(0, pkg)

    os.makedirs(outdir, exist_ok=True)
    for sub in ('images', 'r2_plots'):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    os.chdir(outdir)

    # The old plot code writes PNG to a fixed relative path. Wrap the
    # call so the same figure is also written as vector.
    _orig = plt.savefig

    # The old code writes `fill_between(..., color="none", ec="white",
    # fc="purple", hatch="//")`. `color="none"` suppresses the hatch: the
    # bands render as solid purple, which is what the figures in issue #4
    # show. It also leaves the hatch colour unresolvable, so matplotlib's
    # SVG writer raises while emitting the pattern; Agg does not, which is
    # why the PNG always worked and the vector export never did.
    #
    # Dropping `hatch` therefore removes something that was already not
    # drawn, and keeps the figure faithful to what the author saw. The
    # alternative, dropping `color`, makes the hatch appear and changes
    # the image. Verified by comparing PNGs both ways.
    _orig_fb = Axes.fill_between

    def fill_between(self, *a, **k):
        if k.get('hatch') and k.get('color') == 'none':
            k = {kk: vv for kk, vv in k.items()
                 if kk not in ('hatch', 'hatch_linewidth')}
        return _orig_fb(self, *a, **k)

    Axes.fill_between = fill_between

    def savefig(fname, *a, **k):
        out = _orig(fname, *a, **k)
        root, _ = os.path.splitext(str(fname))
        _orig(root + '.svg', format='svg', bbox_inches='tight')
        print(f"    wrote {root}.svg")
        return out

    plt.savefig = savefig

    from parsers import ParsedData      # noqa: E402
    from baseline import auto_beads     # noqa: E402

    d = ParsedData(signal).data
    x, y = np.asarray(d[0], float), np.asarray(d[1], float)
    print(f"signal {os.path.basename(signal)}  n={len(y)}")

    res = auto_beads(y, x, print_plot=True, path=signal)
    print(f"    auto_beads returned {type(res)}")


if __name__ == '__main__':
    main()
