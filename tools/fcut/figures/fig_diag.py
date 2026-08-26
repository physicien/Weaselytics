"""
Figure 3 for segmentation.md: the real diagnostic, on the same signal.

The last section of the document currently describes this figure in
prose. `plot.r2_plots` writes PNG, so `plt.savefig` is wrapped to emit a
vector copy of the same figure.
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 200
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/esteban/Simulation/DFT/separation_part2/'
                   'Weaselytics')
from weaselytics.baseline import auto_beads  # noqa: E402
from weaselytics.parsers import ParsedData  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
STEM = 'Chlorobenzene__LPYE__60-70__2'
CACHE = ('/home/esteban/Simulation/DFT/separation_part2/runs/'
         'PROD_2026-08-24/r2_cache')


def main():
    src = glob.glob('/home/esteban/Simulation/DFT/separation_part2/data/'
                    f'*/{STEM}.txt')[0]
    d = ParsedData(src).data
    x, y = np.asarray(d[0], float), np.asarray(d[1], float)

    _orig = plt.savefig

    def savefig(fname, *a, **k):
        out = _orig(fname, *a, **k)
        root, _ = os.path.splitext(str(fname))
        _orig(root + '.svg', format='svg', bbox_inches='tight')
        print(f"  wrote {root}.svg")
        return out

    plt.savefig = savefig
    try:
        # The cache is COPIED, never pointed at in place: a miss deletes
        # the entries of the same stem before writing.
        import shutil
        work = os.path.join(OUT, 'diag_cache')
        if not os.path.isdir(work):
            shutil.copytree(CACHE, work)
        auto_beads(y, x, print_plot=True, path=src, output_dir=OUT,
                   cache_dir=work)
    finally:
        plt.savefig = _orig


if __name__ == '__main__':
    main()
