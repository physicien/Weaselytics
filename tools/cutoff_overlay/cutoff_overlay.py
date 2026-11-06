#!/usr/bin/python3
"""
Draw one synthetic signal's baseline at several cutoffs on one figure.

The left half is the production autocorrelation figure of
``plot.r2_plots``, with production's selected cutoff in red, one
vertical line per cutoff and the true optimum as a green dashed line.
The right half has three panels: the baselines against the true one,
the corrected traces against the true correction, and the baseline
error, fit minus true, with a grey band at +-sigma of the generator's
noise. Every cutoff is overlaid in its own colour, from dark (stiff) to
light (flexible).

The cutoffs are spaced by how much the fitted baseline moves. The
baseline is fitted on every tenth cutoff of the production grid, the
rms change between neighbouring fits is accumulated, and the cutoffs
are taken at equal shares of the total. Stretches where the fit barely
moves get one curve and the stretches where it changes fast get
several. A flat stretch around the optimum
can therefore hold no curve at all.

The signal's r2 cache entry is copied into a temporary directory before
it is read. A cache miss in ``_r2_array_cached`` deletes every entry of
the same stem, and the run's own cache must not be touched.

Usage
-----
python tools/cutoff_overlay/cutoff_overlay.py DATASET_DIR RUN_DIR STEM
    -o OUT.jpg [-n 8]
"""

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob

import numpy as np

from weaselytics.parsers import ParsedData

# The figure reuses the pieces of synth_figs, which lives in
# tools/synthetic and is not a package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'synthetic'))
import synth_figs as sf  # noqa: E402


def _stiffness_spaced(fcut_range, fit, n):
    """
    Pick ``n`` cutoffs at equal shares of how far the fit moves.

    Parameters
    ----------
    fcut_range : numpy.ndarray
        The production cutoff grid.
    fit : callable
        Returns the fitted baseline at a cutoff, or None when it fails.
    n : int
        Number of cutoffs.

    Returns
    -------
    grid : numpy.ndarray, shape (m,)
        The chosen cutoffs, taken from ``fcut_range``, sorted and without
        repeats. Where the fit jumps, two shares can fall on one cutoff,
        so ``m`` can be smaller than ``n``.

    Raises
    ------
    ValueError
        If no cutoff of the grid could be fitted.

    """
    fine = fcut_range[::10]
    fits = [fit(float(f)) for f in fine]
    ok = [i for i, b in enumerate(fits) if b is not None]
    if not ok:
        raise ValueError("no cutoff could be fitted.")
    step = [0.] + [float(np.sqrt(np.mean((fits[b] - fits[a]) ** 2)))
                   for a, b in zip(ok[:-1], ok[1:])]
    cum = np.cumsum(step)
    return np.unique([fine[ok[int(np.argmin(np.abs(cum - c)))]]
                      for c in np.linspace(0., cum[-1], n)])


def overlay(dataset, run, stem, out, n_curves, work):
    """Render the figure for one signal into ``out``."""
    path = os.path.join(dataset, 'signals', f'{stem}.txt')
    x, s = ParsedData(path).data
    truth = np.load(os.path.join(dataset, 'truth', f'{stem}__truth.npz'))
    b_true, sigma = truth['baseline'], float(np.std(truth['noise']))

    cache = os.path.join(work, 'r2_cache')
    os.makedirs(cache)
    for f in glob(os.path.join(run, 'r2_cache', f'{stem}__r2__*.npz')):
        shutil.copy2(f, cache)

    regions, sampling, scut = sf._relevant_regions(s, x)
    fcut_range, r2_val, sens, n_used, fc_best = sf._curve(
        stem, path, os.path.join(run, 'diag'), cache, x, s, b_true,
        regions, sampling, scut)
    segments = sf.classify_segments(
        sf.segment_features(fcut_range, r2_val, sf.pelt_linear(r2_val)))
    flat = np.zeros(len(fcut_range), dtype=bool)
    for seg in segments:
        if seg['flat']:
            flat[seg['start']:seg['end']] = True
    dips = sf.detect_dips(fcut_range, r2_val)
    trim = sf.trim_plateaus(
        fcut_range, segments, dips, n_used,
        exclude_past_drop=sf._snr(s) >= sf.SNR_THRESHOLD, sensitivity=sens)

    def fit(fc):
        return sf._beads_at(x, s, fc, regions, sampling, 3)

    grid = _stiffness_spaced(fcut_range, fit, n_curves)
    # Plasma has no green, so the green optimum line stays unique.
    cmap = sf.plt.get_cmap('plasma')
    cols = [cmap(v) for v in np.linspace(0., 0.9, len(grid))]
    fits = [(float(fc), fit(float(fc))) for fc in grid]

    # Left: the production r2 figure, one vertical line per cutoff.
    real_savefig = sf.plt.savefig

    def savefig(*a, **kw):
        for axis in sf.plt.gcf().axes:
            for (fc, _), c in zip(fits, cols):
                axis.axvline(fc, color=c, lw=1.6, zorder=6)
            if np.isfinite(fc_best):
                axis.axvline(fc_best, color=sf._TRUE, ls=(0, (6, 3)),
                             lw=1.6, zorder=7)
        return real_savefig(*a, **kw)

    # Production's own selection, NaN when no plateau survives.
    v = sf.select_center(fcut_range, trim['surviving'])
    selected = np.nan if v is None else float(v)
    r2_at = (np.nan if v is None
             else float(r2_val[int(np.argmin(np.abs(fcut_range - selected)))]))
    sf.plt.savefig = savefig
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sf.r2_plots(fcut_range, r2_val, sf.dip_curve(r2_val),
                        selected, r2_at, cp_flat=flat,
                        cp_dips=sf.dips_to_mask(fcut_range, dips),
                        cp_removed=trim['removed'],
                        cp_snr_removed=trim['snr_removed'],
                        cp_instab_removed=trim['instab_removed'],
                        sensitivity=sens, n_used=n_used, print_plot=True,
                        path=path, output_dir=work)
    finally:
        sf.plt.savefig = real_savefig

    # Right: three panels, every cutoff overlaid.
    fig, ax = sf.plt.subplots(3, 1, figsize=(11, sf._R2_HEIGHT_PX / sf._DPI),
                              sharex=True)
    ax[0].plot(x, s, lw=.7, color='0.6', label='signal')
    ax[1].axhline(0, color='k', lw=.6)
    ax[2].axhspan(-sigma, sigma, color='0.85', zorder=0,
                  label=f'noise, +-{sigma:.3f} mV')
    for (fc, bl), c in zip(fits, cols):
        if bl is None:
            continue
        rmse = float(np.sqrt(np.mean((bl - b_true) ** 2)))
        ax[0].plot(x, bl, color=c, lw=1.4, label=f'cutoff {fc:.2e}')
        ax[1].plot(x, s - bl, color=c, lw=1.0)
        ax[2].plot(x, bl - b_true, color=c, lw=1.4,
                   label=f'{fc:.2e}, RMSE {rmse:.3f}')
    ax[0].plot(x, b_true, color='k', lw=1.6, ls='--', zorder=5,
               label='true baseline')
    ax[1].plot(x, s - b_true, color='k', lw=1.0, ls='--', zorder=5,
               label='corrected with the true baseline')
    ax[2].axhline(0, color='k', lw=1.2, ls='--', zorder=5)
    ax[0].set_ylabel('mV')
    ax[1].set_ylabel('corrected (mV)')
    ax[2].set_ylabel('error (mV)')
    ax[2].set_xlabel('time (min)')
    opt = (f'true optimum {fc_best:.3e}, green dashes on the left'
           if np.isfinite(fc_best) else 'no true optimum')
    ax[0].set_title(f'{stem}: the baseline at {len(grid)} cutoffs, spaced '
                    f'by how much it moves ({opt})', fontsize=9)
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].legend(fontsize=7)
    ax[2].legend(fontsize=7, ncol=2)
    corr = s - b_true
    ax[1].set_ylim(-0.15 * np.ptp(corr), 1.1 * np.max(corr))
    fig.tight_layout(pad=0.4, h_pad=0.3)
    right = os.path.join(work, 'right.png')
    fig.savefig(right, dpi=sf._DPI)
    sf.plt.close(fig)
    subprocess.run(['convert',
                    os.path.join(work, 'r2_plots', f'{stem}_r2.png'),
                    right, '+append', out], check=True)
    return grid


def main() -> None:
    """CLI entry point of the overlay figure."""
    p = argparse.ArgumentParser(
        prog='cutoff_overlay',
        description="one synthetic signal's baseline at several cutoffs")
    p.add_argument('dataset', help='dataset holding signals/ and truth/')
    p.add_argument('run', help='run holding r2_cache/ and diag/')
    p.add_argument('stem', help='name of the signal')
    p.add_argument('-o', '--out', required=True,
                   help='output image; the extension sets the format, '
                        'e.g. .jpg')
    p.add_argument('-n', '--n-curves', type=int, default=8,
                   help='number of cutoffs (default: 8)')
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as work:
        grid = overlay(a.dataset, a.run, a.stem, a.out, a.n_curves, work)
    print('cutoffs:', ' '.join(f'{g:.2e}' for g in grid))
    print(a.out)


if __name__ == '__main__':
    main()
