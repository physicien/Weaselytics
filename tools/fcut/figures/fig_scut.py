"""
Replacement for the `scut` figure, as vector and from current code.

The original is a raster drawn by the derivative-era diagnostic, with
tolerance bands and a "Case" label that no longer exist. The message it
carried is still right: truncating the record past the last peak changes
the autocorrelation curve, because the flat tail dilutes the statistic.

One run per row: (a) a single sharp peak early in a long record, (b) a
multi-peak run. Left, the raw trace with the cut marked, no baseline
fitted, the shaded part being what is discarded. Right, the curve
computed on the truncated record against the curve computed on the whole
of it.
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 200
import matplotlib.pyplot as plt
from pybaselines import Baseline

sys.path.insert(0, '/home/esteban/Simulation/DFT/separation_part2/'
                   'Weaselytics')
from weaselytics.baseline import (  # noqa: E402
    _custom_beads,
    _log_transform,
    _relevant_regions,
)
from weaselytics.parsers import ParsedData  # noqa: E402
from weaselytics.utils import r2_dw  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
STEMS = ['2-Xylene__LPYE__CS2__15', '2-Xylene__LPYE__90-96-100__2']
TAGS = ['(a)', '(b)']
N_GRID = 220
KEEP, DROP = '#1f4e79', '#b3402f'


def sweep(z, x_used, regions, sampling):
    fitter = Baseline(x_data=x_used)
    fcuts = np.geomspace(1e-5, 0.45, N_GRID)
    out = np.full(N_GRID, np.nan)
    for i, fc in enumerate(fcuts):
        try:
            bl, _ = _custom_beads(fitter, z, regions=regions,
                                  sampling=sampling, freq_cutoff=float(fc),
                                  asymmetry=1.0, fit_parabola=True,
                                  alpha=1.0, parabola_len=3)
            out[i] = r2_dw(z - bl)
        except Exception:
            pass
    return fcuts, out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.6),
                             gridspec_kw={'width_ratios': [1, 1.25]})
    for row, (stem, tag) in enumerate(zip(STEMS, TAGS)):
        src = glob.glob('/home/esteban/Simulation/DFT/separation_part2/'
                        f'data/*/{stem}.txt')[0]
        d = ParsedData(src).data
        x, y = np.asarray(d[0], float), np.asarray(d[1], float)
        regions, sampling, scut = _relevant_regions(y, x)
        pct = 100.0 * (1 - scut / len(y))
        print(f"  {stem:32s} n={len(y):5d} scut={scut:5d} "
              f"discards {pct:.0f}%")

        f_c, r_c = sweep(_log_transform(y[:scut]), x[:scut], regions, sampling)
        f_a, r_a = sweep(_log_transform(y), x, regions, sampling)

        # Raw trace only. No baseline is fitted here: the panel exists to
        # show where the cut lands and how much it discards.
        ax = axes[row, 0]
        ax.plot(x, y, lw=.6, color='0.45')
        ax.axvspan(x[scut], x[-1], color=DROP, alpha=.10, lw=0)
        ax.axvline(x[scut], color=KEEP, lw=1.5)
        ax.axvline(x[-1], color=DROP, lw=1.5, ls='--')
        top, span = float(np.nanmax(y)), x[-1] - x[0]
        ax.annotate('scut', xy=(x[scut], top * .52),
                    xytext=(x[scut] + .07 * span, top * .80), fontsize=8.5,
                    color=KEEP, ha='left',
                    arrowprops=dict(arrowstyle='->', color=KEEP, lw=1.0,
                                    shrinkA=0, shrinkB=2))
        ax.annotate('whole record', xy=(x[-1], top * .22),
                    xytext=(x[-1] - .06 * span, top * .48), fontsize=8.5,
                    color=DROP, ha='right',
                    arrowprops=dict(arrowstyle='->', color=DROP, lw=1.0,
                                    shrinkA=0, shrinkB=2))
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel('time (min)', fontsize=8.5)
        ax.set_ylabel('signal (mV)', fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
        ax.set_title(f'{stem}\n'
                     f'scut = {scut} of {len(y)} points, '
                     f'discarding {pct:.0f}% of the record',
                     fontsize=8.5, loc='left', pad=5)
        ax.text(-0.24, 1.16, tag, transform=ax.transAxes, fontsize=13,
                fontweight='bold', va='top', ha='left')

        ax = axes[row, 1]
        ax.plot(f_c, r_c, lw=1.7, color=KEEP,
                label=f'truncated at scut, {scut} points')
        ax.plot(f_a, r_a, lw=1.5, ls='--', color=DROP,
                label=f'whole record, {len(y)} points')
        ax.set_xscale('log')
        ax.set_xlim(f_c[0], f_c[-1])
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel('cutoff frequency  $f_{cut}$', fontsize=9)
        ax.set_ylabel('$r^2$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='lower left')

    fig.tight_layout()
    for ext in ('svg', 'png'):
        p = os.path.join(OUT, f'fig_scut.{ext}')
        fig.savefig(p, format=ext, bbox_inches='tight', dpi=200)
        print(f"  {p}")


if __name__ == '__main__':
    main()
