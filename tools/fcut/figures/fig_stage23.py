"""
Remaining figures for segmentation.md: the fallback channel, the stage 2
exclusions, and the stage 3 pick. One worked example throughout.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/esteban/Simulation/DFT/separation_part2/Weaselytics')
from weaselytics.segmentation import (  # noqa: E402
    _rolling_std, classify_segments, detect_dips, dip_curve, pelt_linear,
    segment_features, select_center, trim_plateaus)

CACHE = ('/home/esteban/Simulation/DFT/separation_part2/runs/'
         'PROD_2026-08-24/r2_cache')
OUT = os.path.dirname(os.path.abspath(__file__))
STEM = 'Chlorobenzene__LPYE__60-70__2'
N_USED, DPI = 400, 200
CURVE, KEEP, CUT, ORANGE, RED = (
    '#1f4e79', '#4c9f70', '#b3402f', '#d9a441', '#b3402f')


def shade(ax, fr, mask, color, label, alpha=.30):
    if not mask.any():
        return
    idx = np.flatnonzero(mask)
    for k, g in enumerate(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)):
        ax.axvspan(fr[g[0]], fr[g[-1]], color=color, alpha=alpha, lw=0,
                   label=label if k == 0 else None, zorder=1)


def main():
    f = glob.glob(os.path.join(CACHE, f'{STEM}__r2__*.npz'))[0]
    d = np.load(f)
    fr, r2, sens = d['fcut_range'], d['r2_val'], d['sensitivity']
    segs = classify_segments(segment_features(fr, r2, pelt_linear(r2)))
    dips = detect_dips(fr, r2)
    tr = trim_plateaus(fr, segs, dips, N_USED, exclude_past_drop=True,
                       sensitivity=sens)
    flat = np.zeros(len(r2), bool)
    for s in segs:
        if s['flat']:
            flat[s['start']:s['end']] = True

    # ---- the fallback channel -------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(10.6, 7.4), sharex=True)
    ax[0].plot(fr, r2, lw=1.5, color=CURVE)
    ax[0].set_ylabel('$r^2$')
    ax[0].set_title('the curve the fallback reads', fontsize=9.5,
                    loc='left', pad=4)
    ax[1].plot(fr, _rolling_std(r2, window=3), lw=1.0, color='#7a3fa0')
    ax[1].set_ylabel('rolling std\n(window 3)')
    ax[1].set_title('step 1: rolling standard deviation, raw',
                    fontsize=9.5, loc='left', pad=4)
    curve = dip_curve(r2)
    ax[2].plot(fr, curve, lw=1.4, color=ORANGE)
    for k, dp in enumerate(dips):
        ax[2].plot([dp['fcut']], [curve[dp['floor']]], 'v', ms=8,
                   color=CUT, zorder=5,
                   label='accepted proto-plateau' if k == 0 else None)
        ax[2].axvspan(fr[dp['start']], fr[dp['end']], color=ORANGE,
                      alpha=.25, lw=0, zorder=1)
    ax[2].set_ylabel('dip curve\n(smoothed, /max)')
    ax[2].set_ylim(bottom=-0.06, top=1.05)
    ax[2].set_title(r'step 2: Gaussian smoothed by $\sigma$ = 8 and '
                    'normalised; its local minima are the candidates',
                    fontsize=9.5, loc='left', pad=4)
    if dips:
        ax[2].legend(fontsize=8.5, loc='upper left')
    for a in ax:
        a.set_xscale('log')
        a.set_xlim(fr[0], fr[-1])
        a.grid(alpha=.2)
    ax[2].set_xlabel('cutoff frequency  $f_{cut}$')
    fig.tight_layout()
    p = os.path.join(OUT, 'stage1_fallback.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight')
    print(f"  {p}  ({len(dips)} dips)")

    # ---- the stage 2 exclusions -----------------------------------
    from weaselytics.segmentation import dips_to_mask
    detected = flat | dips_to_mask(fr, dips)
    rows = [('what stage 2 receives: flat segments and proto-plateaus',
             detected, KEEP),
            ('removed by the sub-fundamental clip and the region trim',
             tr['removed'], CUT),
            ('removed by the past-drop exclusion (SNR gate on)',
             tr['snr_removed'], '#7a3fa0'),
            ('removed by the stiff-side instability boundary',
             tr['instab_removed'], ORANGE),
            ('surviving', tr['surviving'], KEEP)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10.6, 8.6), sharex=True)
    for a, (title, mask, col) in zip(axes, rows):
        a.plot(fr, r2, lw=1.3, color=CURVE, zorder=3)
        shade(a, fr, mask, col, None, alpha=.32)
        a.set_xscale('log')
        a.set_xlim(fr[0], fr[-1])
        a.set_ylim(-0.06, 1.1)
        a.set_ylabel('$r^2$')
        a.grid(alpha=.2)
        a.set_title(f'{title}   ({int(mask.sum())} grid points)',
                    fontsize=9.5, loc='left', pad=4)
    axes[-1].set_xlabel('cutoff frequency  $f_{cut}$')
    fig.tight_layout()
    p = os.path.join(OUT, 'stage2_exclusions.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight')
    print(f"  {p}")

    # ---- stage 3 ---------------------------------------------------
    fc = select_center(fr, tr['surviving'])
    idx = np.flatnonzero(tr['surviving'])
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    last = groups[-1]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.plot(fr, r2, lw=1.6, color=CURVE, zorder=3)
    for k, g in enumerate(groups):
        ax.axvspan(fr[g[0]], fr[g[-1]], color=KEEP,
                   alpha=.34 if g is last else .13, lw=0, zorder=1,
                   label=('the last surviving region' if g is last else
                          ('earlier surviving regions' if k == 0 else None)))
    i = int(np.argmin(np.abs(fr - fc)))
    ax.axvline(fc, color=RED, lw=1.8, zorder=5)
    ax.plot([fc], [r2[i]], 'o', ms=7, color=RED, zorder=6)
    ax.annotate(f'geometric midpoint of that region,\nsnapped to a swept '
                f'grid point\n$f_{{cut}}$ = {fc:.4g}',
                xy=(fc, r2[i]), xytext=(fc * 0.06, 0.30), fontsize=9,
                color=RED, ha='left', zorder=6,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2,
                                shrinkA=2, shrinkB=6))
    ax.set_xscale('log')
    ax.set_xlim(fr[0], fr[-1])
    ax.set_ylim(-0.06, 1.1)
    ax.set_xlabel('cutoff frequency  $f_{cut}$')
    ax.set_ylabel('$r^2$')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=.92)
    ax.set_title('Stage 3 takes the last region; the midpoint within it '
                 'is a placeholder', fontsize=9.5, loc='left', pad=6)
    fig.tight_layout()
    p = os.path.join(OUT, 'stage3_select.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight')
    print(f"  {p}  (fcut={fc:.4g}, {len(groups)} surviving regions)")


if __name__ == '__main__':
    main()
