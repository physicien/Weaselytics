"""
Figure 3 for segmentation.md: the three stages on one curve.

Same signal as the vocabulary figure, so the reader follows a curve they
already know. Top: what stage 1 detects. Middle: what stage 2 leaves.
Bottom: where stage 3 commits.
"""

import argparse
import glob
import inspect
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 200
import matplotlib.pyplot as plt  # noqa: E402

from weaselytics.baseline import (  # noqa: E402
    _log_transform,
    _relevant_regions,
    _snr,
    auto_beads,
)
from weaselytics.parsers import ParsedData  # noqa: E402
from weaselytics.segmentation import (  # noqa: E402
    classify_segments,
    detect_dips,
    pelt_linear,
    segment_features,
    select_center,
    trim_plateaus,
)

OUT = os.path.dirname(os.path.abspath(__file__))
STEM = 'Chlorobenzene__LPYE__60-70__2'

CURVE = '#1f4e79'
KEEP = '#4c9f70'
CUT = '#b3402f'


def production_inputs(data):
    """`n_used` and the past-drop gate, as `_fcutoff` derives them."""
    src = glob.glob(os.path.join(data, '*', f'{STEM}.txt'))[0]
    x, y = ParsedData(src).data
    _, _, scut = _relevant_regions(y, x)
    gate = inspect.signature(auto_beads).parameters[
        'snr_threshold'].default
    return len(_log_transform(y[:scut])), bool(_snr(y) >= gate)


def mask_from_flat(segments, n):
    m = np.zeros(n, dtype=bool)
    for s in segments:
        if s['flat']:
            m[s['start']:s['end']] = True
    return m


def shade(ax, fr, mask, color, label, alpha=.30):
    """Shade the grid points a boolean mask selects."""
    if not mask.any():
        return
    idx = np.flatnonzero(mask)
    for k, grp in enumerate(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)):
        ax.axvspan(fr[grp[0]], fr[grp[-1]], color=color, alpha=alpha, lw=0,
                   label=label if k == 0 else None, zorder=1)


def main(data, cache):
    f = glob.glob(os.path.join(cache, f'{STEM}__r2__*.npz'))[0]
    d = np.load(f)
    fr, r2, sens = d['fcut_range'], d['r2_val'], d['sensitivity']

    segments = classify_segments(
        segment_features(fr, r2, pelt_linear(r2)))
    flat = mask_from_flat(segments, len(r2))
    dips = detect_dips(fr, r2)
    n_used, past_drop = production_inputs(data)
    trim = trim_plateaus(fr, segments, dips, n_used,
                         exclude_past_drop=past_drop, sensitivity=sens)
    surviving = trim['surviving']
    fcut = select_center(fr, surviving)
    print(f"  flat points      {flat.sum():4d}/{len(r2)}")
    print(f"  surviving points {surviving.sum():4d}/{len(r2)}")
    print(f"  selected fcut    {fcut:.4e}")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True)
    titles = [
        'Stage 1  ' + r'$\bf{detect}$' + ':  every stretch flat enough to '
        'be a plateau',
        'Stage 2  ' + r'$\bf{trim}$' + ':  what is left once the impossible '
        'regions are removed',
        'Stage 3  ' + r'$\bf{select}$' + ':  the centre of the last '
        'surviving region',
    ]
    for ax, title in zip(axes, titles):
        ax.plot(fr, r2, lw=1.4, color=CURVE, zorder=3)
        ax.set_xscale('log')
        ax.set_xlim(fr[0], fr[-1])
        ax.set_ylim(-0.06, 1.1)
        ax.set_ylabel('$r^2$')
        ax.grid(alpha=.2, zorder=0)
        ax.set_title(title, fontsize=9.5, loc='left', pad=4)

    shade(axes[0], fr, flat, KEEP, 'detected as flat')
    axes[0].legend(loc='lower left', fontsize=8, framealpha=.9)

    shade(axes[1], fr, surviving, KEEP, 'surviving')
    shade(axes[1], fr, flat & ~surviving, CUT, 'removed', alpha=.26)
    axes[1].legend(loc='lower left', fontsize=8, framealpha=.9)

    shade(axes[2], fr, surviving, KEEP, 'surviving', alpha=.18)
    axes[2].axvline(fcut, color=CUT, lw=1.8, zorder=5)
    i = int(np.argmin(np.abs(fr - fcut)))
    axes[2].plot([fcut], [r2[i]], 'o', ms=7, color=CUT, zorder=6)
    axes[2].annotate(f'selected  $f_{{cut}}$ = {fcut:.3g}',
                     xy=(fcut, r2[i]), xytext=(fcut * 0.35, 0.30),
                     fontsize=9, color=CUT, ha='right', zorder=6,
                     arrowprops=dict(arrowstyle='->', color=CUT, lw=1.2,
                                     shrinkA=0, shrinkB=5))
    axes[2].set_xlabel('cutoff frequency  $f_{cut}$')

    fig.tight_layout()
    for ext in ('svg', 'png'):
        p = os.path.join(OUT, f'fig_stages.{ext}')
        fig.savefig(p, format=ext, bbox_inches='tight', dpi=200)
        print(f"  {p}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True,
                        help='raw data directory, holding '
                             '<Molecule>/<stem>.txt')
    parser.add_argument('--cache', required=True,
                        help='r2_cache directory of a production run, '
                             'e.g. runs/PROD_<date>_<job>/r2_cache')
    args = parser.parse_args()
    main(args.data, args.cache)
