#!/usr/bin/python3
"""
Measure how the optimal ``fcut`` scales with the peak width.

A controlled counterpart to ``tools/synth_diag.py``: everything is held
fixed (baseline shape, noise level, retention times, peak heights,
record length) and *only* the peak width varies, over a decade. For
each signal the true error curve against the known baseline gives an
objective optimum, and ``log fcut*`` is regressed on

* the **true** width — the exponent of the scaling law;
* the **measured** width (the production recipe of
  ``weaselytics.baseline._relevant_regions``) — which tells whether
  measurement error in the predictor biases the exponent, the
  regression-dilution effect that would flatten it toward zero.

Reference results (seed defaults, 60 signals): exponent -0.735 on the
true width (R2 0.857), -0.810 on the measured width, with the two
widths correlated at 0.998, so no dilution on clean synthetic peaks.
The exponent differs from the -0.88 of the full benchmark and the
-0.50 of the real dataset, i.e. it is a property of the signal
population and not a transferable constant.

The measured-width figures predate the removal of the coarse detrend
and of the absolute negative-depth gate from ``_relevant_regions`` and
have not been re-run; the true-width exponent is unaffected.

Usage
-----
python tools/synth_width_sweep.py -o width_sweep.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from synth_dataset import _FWHM_PER_SIGMA, emg_peak, make_baseline
from synth_diag import error_curve

from weaselytics.utils import peaks_params

DT = 1. / 60.
CENTERS = np.array([6., 10., 14., 18., 22.])
HEIGHTS = np.array([12., 20., 8., 15., 10.])
NOISES = {'low': 0.01, 'high': 0.06}


def measured_width(x: np.ndarray, s: np.ndarray) -> float:
    """
    Median relevant peak width with the production recipe.

    Mirrors ``weaselytics.baseline._relevant_regions``: weak smoothing,
    ``peaks_params`` and the relevance filter.

    Parameters
    ----------
    x : array-like, shape (N,)
        Time axis.
    s : array-like, shape (N,)
        The measured signal.

    Returns
    -------
    width : float
        Median FWHM of the relevant peaks, in points; NaN when none.

    """
    z = gaussian_filter1d(s, 3)
    peaks, widths = peaks_params(z, width=3, rel_prom_p=0.01, adapt=True,
                                 drop_enclosing=True)
    if len(peaks) == 0:
        return np.nan
    width_per_x = widths / np.where(x[peaks] > 0, x[peaks], np.inf)
    keep = (width_per_x < 6) | ((s[peaks] > 20) & (width_per_x < 11))
    return float(np.median(widths[keep])) if keep.any() else np.nan


def fit_exponent(width: np.ndarray, fcut: np.ndarray
                 ) -> tuple[float, float, int]:
    """
    Fit ``log fcut = a log width + b`` and report its quality.

    Parameters
    ----------
    width : array-like, shape (M,)
        Peak widths.
    fcut : array-like, shape (M,)
        Optimal cutoff frequencies.

    Returns
    -------
    exponent : float
        The fitted slope.
    r2 : float
        Fraction of the variance of ``log fcut`` explained.
    n : int
        Number of usable points.

    """
    ok = np.isfinite(width) & (width > 0)
    design = np.column_stack([np.ones(ok.sum()), np.log(width[ok])])
    coef, *_ = np.linalg.lstsq(design, np.log(fcut[ok]), rcond=None)
    resid = np.log(fcut[ok]) - design @ coef
    return coef[1], 1 - resid.var() / np.log(fcut[ok]).var(), int(ok.sum())


def main() -> None:
    """
    CLI entry point of the width sweep.

    """
    parser = argparse.ArgumentParser(
        prog='synth_width_sweep',
        description='scaling of the optimal fcut with the peak width')
    parser.add_argument('-o', '--output', default='width_sweep.csv',
                        help='output CSV path')
    parser.add_argument('-n', '--n-points', type=int, default=1500,
                        help='record length (default: 1500)')
    parser.add_argument('--replicates', type=int, default=3,
                        help='replicates per width (default: 3)')
    parser.add_argument('--widths', type=int, default=10,
                        help='number of widths from 8 to 120 points')
    args = parser.parse_args()

    widths = np.geomspace(8., 120., args.widths)
    fcuts = np.geomspace(1e-4, 0.45, 220)
    t = np.arange(args.n_points) * DT
    rows = []
    for rep in range(args.replicates):
        for noise_name, sigma in NOISES.items():
            for w_pts in widths:
                rng = np.random.default_rng(
                    1000 * rep + int(w_pts) + (0 if noise_name == 'low'
                                               else 7))
                signal = np.zeros(args.n_points)
                for tc, height in zip(CENTERS, HEIGHTS):
                    sg = w_pts * DT / _FWHM_PER_SIGMA
                    signal += emg_peak(t, tc, sg, 0.5 * sg, height)
                baseline = make_baseline(t, 'exp_hump_drift', rng)
                y = signal + baseline + rng.normal(0., sigma,
                                                   args.n_points)
                err = error_curve(t, y, baseline, fcuts)
                if not np.isfinite(err).any():
                    continue
                rows.append({'rep': rep, 'noise': noise_name,
                             'w_true': w_pts,
                             'w_meas': measured_width(t, y),
                             'fcut': float(fcuts[np.nanargmin(err)])})
                print(f"rep{rep} {noise_name:4s} w={w_pts:6.1f} -> "
                      f"fcut*={rows[-1]['fcut']:.4e}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)

    print('\n=== exponent of fcut* versus peak width ===')
    groups = [('all', table)] + [(k, g) for k, g in table.groupby('noise')]
    for label, sub in groups:
        for col in ['w_true', 'w_meas']:
            a, r2, n = fit_exponent(sub[col].to_numpy(),
                                    sub['fcut'].to_numpy())
            print(f'{label:5s} {col:7s}: exponent {a:+.3f}  R2 {r2:.3f} '
                  f' n={n}')

    ok = np.isfinite(table.w_meas)
    corr = np.corrcoef(np.log(table.w_true[ok]),
                       np.log(table.w_meas[ok]))[0, 1]
    product = table.fcut * table.w_true
    verdict = ('constant' if product.std() / product.mean() < 0.1
               else 'NOT constant')
    print('\n=== regression dilution check ===')
    print(f'corr(log w_true, log w_meas) = {corr:.3f}')
    print(f'fcut*·w_true: mean {product.mean():.4f} '
          f'median {product.median():.4f} std {product.std():.4f} '
          f'-> {verdict}')
    print(f'\n{len(table)} signals -> {args.output}')


if __name__ == '__main__':
    main()
