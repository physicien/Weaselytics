"""
Stage 1 figures for segmentation.md: how the segments are identified.

Two panels of one worked example, on the curve of Figures 1 and 3.

  stage1_segments  the partition pelt_linear returns, with the straight
                   line it fits on each segment, so the reader sees what
                   a "segment" is and where the boundaries land.
  stage1_features  every segment placed in (rel_slope, rel_noise) with
                   the classify_segments thresholds drawn on top, so the
                   rule is visible as a region of the plane rather than
                   as three numbers in a table.
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
    classify_segments, pelt_linear, segment_features)

CACHE = ('/home/esteban/Simulation/DFT/separation_part2/runs/'
         'PROD_2026-08-24/r2_cache')
OUT = os.path.dirname(os.path.abspath(__file__))
STEM = 'Chlorobenzene__LPYE__60-70__2'
DPI = 200

REL_SLOPE_MAX, REL_SLOPE_LOOSE = 0.2, 0.6
REL_NOISE_MAX, CLIFF_MIN = 0.006, 1.0

CURVE, FLAT, NOTFLAT, CLIFF = '#1f4e79', '#1a7a44', '#c2521f', '#7a3fa0'


def load():
    f = glob.glob(os.path.join(CACHE, f'{STEM}__r2__*.npz'))[0]
    d = np.load(f)
    return d['fcut_range'], d['r2_val']


def fig_segments(fr, r2, segs):
    """Two panels: the partition, and the residual it leaves behind.

    The residual panel is the point of the figure. The boundaries inside
    the initial plateau are not there because the mean moves, it does
    not; they are there because the noise level changes, and a segment
    cost that fits its own variance sees that.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.8), sharex=True,
                             gridspec_kw={'height_ratios': [2.0, 1.0]})
    t = np.arange(len(r2), dtype=float)

    for ax in axes:
        for k, s in enumerate(segs):
            if k % 2:
                ax.axvspan(fr[s['start']], fr[s['end'] - 1],
                           color='0.45', alpha=.15, lw=0, zorder=0)

    ax = axes[0]
    ax.plot(fr, r2, lw=4.0, color='0.86', zorder=2,
            label='the swept $r^2$ curve')
    for k, s in enumerate(segs):
        i, j = s['start'], s['end']
        tt = t[i:j]
        line = s['mean'] + s['slope'] * (tt - tt.mean())
        ax.plot(fr[i:j], line, lw=2.1,
                color=FLAT if s['flat'] else NOTFLAT, zorder=4,
                label=('fitted line, marked flat' if k == 0 else None))
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel('$r^2$')
    ax.grid(alpha=.2)
    ax.set_title(f'{len(segs)} segments, alternately shaded, with the '
                 'straight line fitted on each',
                 fontsize=9.5, loc='left', pad=6)
    hs, ls = ax.get_legend_handles_labels()
    hs.append(plt.Line2D([], [], color=NOTFLAT, lw=2.1))
    ls.append('fitted line, not flat')
    ax.legend(hs, ls, fontsize=8.5, loc='lower left', framealpha=.92)

    ax = axes[1]
    for k, sg in enumerate(segs):
        i, j = sg['start'], sg['end']
        ax.plot([fr[i], fr[j - 1]], [sg['rel_noise']] * 2, lw=3.2,
                color=FLAT if sg['flat'] else NOTFLAT, zorder=3,
                solid_capstyle='butt')
    ax.axhline(REL_NOISE_MAX, color='#b3402f', lw=1.3, ls='--', zorder=4)
    ax.text(fr[0] * 1.3, REL_NOISE_MAX * 1.4,
            r'$rel\_noise\_max$ = 0.006', fontsize=8.5, color='#b3402f',
            va='bottom')
    ax.axvline(1.0 / 400, color='#b3402f', lw=1.2, ls=':', zorder=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(fr[0], fr[-1])
    ax.set_xlabel('cutoff frequency  $f_{cut}$')
    ax.set_ylabel(r'$rel\_noise$')
    ax.grid(alpha=.2, which='both')
    ax.set_title('residual noise of each segment: across the initial '
                 'plateau the slope stays at zero while this climbs more than three '
                 'orders of magnitude', fontsize=9.5, loc='left', pad=6)

    fig.tight_layout()
    p = os.path.join(OUT, 'stage1_segments.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight')
    print(p)


def fig_features(segs):
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    rs = np.array([s['rel_slope'] for s in segs])
    # A perfectly level segment has rel_slope 0, which a log axis cannot
    # place. Park those on the left edge rather than dropping them.
    FLOOR = 1e-5
    clamped = rs < FLOOR
    rs = np.maximum(rs, FLOOR * 1.25)
    rn = np.array([s['rel_noise'] for s in segs])
    flat = np.array([s['flat'] for s in segs])

    ax.axhspan(0, REL_NOISE_MAX, xmax=1, color=FLAT, alpha=.09, lw=0)
    ax.axhline(REL_NOISE_MAX, color='#b3402f', lw=1.3, ls='--')
    ax.axvline(REL_SLOPE_MAX, color='#2e6b4a', lw=1.3)
    ax.axvline(REL_SLOPE_LOOSE, color='#8a6a1f', lw=1.3, ls='-.')
    ax.axvline(CLIFF_MIN, color=CLIFF, lw=1.3, ls=':')

    ax.scatter(rs[~flat], rn[~flat], s=42, facecolor='none',
               edgecolor=NOTFLAT, lw=1.3, label='not flat', zorder=4)
    ax.scatter(rs[flat], rn[flat], s=52, color=FLAT, lw=0,
               label='marked flat', zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$rel\_slope$   (segment slope / whole drop per decade)')
    ax.set_ylabel(r'$rel\_noise$   (residual std / whole drop)')
    ax.grid(alpha=.2, which='both')
    ax.set_xlim(FLOOR, max(rs.max() * 3, 4))
    ax.set_ylim(min(rn.min() * 0.5, 1e-5), max(rn.max() * 4, 0.05))
    # Offset in points rather than a leading space, so the gap to the
    # line is set here and does not depend on the font's space width.
    for xpos, txt, col in ((REL_SLOPE_MAX, 'tight 0.2', '#2e6b4a'),
                           (REL_SLOPE_LOOSE, 'loose 0.6', '#8a6a1f'),
                           (CLIFF_MIN, 'cliff 1.0', CLIFF)):
        ax.annotate(txt, xy=(xpos, ax.get_ylim()[1]),
                    xytext=(2.5, 0), textcoords='offset points',
                    fontsize=8.5, color=col, va='top', rotation=90,
                    ha='left')
    ax.text(ax.get_xlim()[0], REL_NOISE_MAX, ' quiet enough, 0.006',
            fontsize=8.5, color='#b3402f', va='bottom', ha='left')
    if clamped.any():
        ax.text(FLOOR * 1.35, ax.get_ylim()[0] * 3,
                f'{int(clamped.sum())} segments with slope 0,\nparked on '
                'the edge', fontsize=8, color='#2e6b4a', va='bottom')
    ax.legend(fontsize=8.5, loc='lower right', framealpha=.92)
    ax.set_title('Every segment of one curve, against the flatness rule',
                 fontsize=9.5, loc='left', pad=6)
    fig.tight_layout()
    p = os.path.join(OUT, 'stage1_features.png')
    fig.savefig(p, dpi=DPI, bbox_inches='tight')
    print(p)


def main():
    fr, r2 = load()
    segs = classify_segments(segment_features(fr, r2, pelt_linear(r2)))
    print(f"  {len(segs)} segments, {sum(s['flat'] for s in segs)} flat")
    fig_segments(fr, r2, segs)
    fig_features(segs)


if __name__ == '__main__':
    main()
