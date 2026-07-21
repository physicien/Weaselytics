#!/usr/bin/python3
"""
Diagnose the fcut machinery against synthetic ground truth.

For every signal of a dataset produced by ``tools/synth_dataset.py``:

* run the production autocorrelation sweep (cached r2 curve);
* run the changepoint chain (``trim_candidates`` +
  ``refine_candidates``);
* compute the TRUE error curve ``E(fcut)``: the RMSE between the
  baseline fitted with the final-correction configuration of
  ``auto_beads`` and the known true baseline, on a subsampled fcut
  grid.

The minimum of ``E`` is the objective optimal cutoff frequency — the
quantity the hand labels could only approximate visually — so the
containment and position of the true optimum inside the candidate and
bracket regions can be measured without label censoring.

Writes one ``diag/<stem>__diag.npz`` per signal and a summary CSV.

Usage
-----
python tools/synth_diag.py DATASET_DIR [--workers 4] [--stride 4]
"""

import argparse
import contextlib
import io
import os
from glob import glob

import numpy as np
from pybaselines import Baseline

from weaselytics.baseline import (
    _custom_beads,
    _log_transform,
    _relevant_regions,
    auto_beads,
)
from weaselytics.parsers import ParsedData
from weaselytics.segmentation import (
    classify_segments,
    pelt_linear,
    refine_candidates,
    segment_features,
    trim_candidates,
)
from weaselytics.utils import end_window


def error_curve(x: np.ndarray, s: np.ndarray, b_true: np.ndarray,
                fcuts: np.ndarray) -> np.ndarray:
    """
    Compute the true baseline-recovery error along the fcut grid.

    Mirrors the final-correction configuration of ``auto_beads``
    (custom_beads, per-region sampling, ``alpha=1``,
    ``parabola_len=end_window(s)``, asymmetry 1).

    Parameters
    ----------
    x : array-like, shape (N,)
        Time axis.
    s : array-like, shape (N,)
        The measured signal.
    b_true : array-like, shape (N,)
        The known true baseline.
    fcuts : array-like, shape (M,)
        Cutoff frequencies to evaluate.

    Returns
    -------
    err : numpy.ndarray, shape (M,)
        RMSE of the fitted baseline against `b_true`; NaN where the
        fit failed.

    """
    peak_regions, sampling, _ = _relevant_regions(s, x)
    fitter = Baseline(x_data=x)
    kwargs = {'regions': peak_regions, 'sampling': sampling,
              'asymmetry': 1.0, 'fit_parabola': True, 'alpha': 1.0,
              'parabola_len': end_window(s)}
    err = np.full(len(fcuts), np.nan)
    for k, fc in enumerate(fcuts):
        try:
            bl, _ = _custom_beads(fitter, s, freq_cutoff=fc, **kwargs)
            err[k] = float(np.sqrt(np.mean((bl - b_true) ** 2)))
        except Exception:
            pass
    return err


def diag_one(stem: str, sig_dir: str, truth_dir: str, cache_dir: str,
             diag_dir: str, workers: int, stride: int) -> dict:
    """
    Run the diagnostics of a single signal.

    Parameters
    ----------
    stem : str
        Signal name (file stem).
    sig_dir, truth_dir, cache_dir, diag_dir : str
        Dataset sub-directories.
    workers : int
        Workers for the r2 sweep.
    stride : int
        fcut grid stride of the error curve.

    Returns
    -------
    row : dict
        Summary of the signal's diagnostics.

    """
    path = os.path.join(sig_dir, f'{stem}.txt')
    x, s = ParsedData(path).data
    truth = np.load(os.path.join(truth_dir, f'{stem}__truth.npz'))
    b_true = truth['baseline']

    # production sweep (quiet); the r2 curve lands in the cache
    with contextlib.redirect_stdout(io.StringIO()):
        auto_beads(s, x, method='custom_beads', cache_dir=cache_dir,
                   path=path, workers=workers)
    npz = sorted(glob(os.path.join(cache_dir, f'{stem}__r2__*.npz')))[-1]
    d = np.load(npz)
    fcut_range, r2 = d['fcut_range'], d['r2_val']

    _, _, scut = _relevant_regions(s, x)
    n_used = len(_log_transform(s[:scut]))
    segments = classify_segments(
        segment_features(fcut_range, r2, pelt_linear(r2)))
    candidates = trim_candidates(fcut_range, segments, n_used)
    refined = refine_candidates(fcut_range, candidates)

    fcuts = fcut_range[::stride]
    err = error_curve(x, s, b_true, fcuts)
    k_best = int(np.nanargmin(err))
    fc_best = float(fcuts[k_best])

    np.savez(os.path.join(diag_dir, f'{stem}__diag.npz'),
             fcut_range=fcut_range, r2=r2, candidates=candidates,
             refined=refined, fcuts=fcuts, err=err, n_used=n_used)
    idx = min(np.searchsorted(fcut_range, fc_best),
              len(fcut_range) - 1)
    return {
        'stem': stem, 'n_used': n_used,
        'fc_best': fc_best, 'e_min': float(np.nanmin(err)),
        'e_med': float(np.nanmedian(err)),
        'n_fail': int((~np.isfinite(err)).sum()),
        'in_candidates': bool(candidates[idx]),
        'in_bracket': bool(refined[idx]),
    }


def main() -> None:
    """
    CLI entry point of the synthetic diagnostics runner.

    """
    parser = argparse.ArgumentParser(
        prog='synth_diag',
        description='score the fcut machinery on synthetic truth')
    parser.add_argument('dataset', help='directory from synth_dataset.py')
    parser.add_argument('--workers', type=int, default=4,
                        help='workers for the r2 sweep (default: 4)')
    parser.add_argument('--stride', type=int, default=4,
                        help='fcut grid stride of the error curve '
                             '(default: 4)')
    args = parser.parse_args()

    sig_dir = os.path.join(args.dataset, 'signals')
    truth_dir = os.path.join(args.dataset, 'truth')
    cache_dir = os.path.join(args.dataset, 'r2_cache')
    diag_dir = os.path.join(args.dataset, 'diag')
    os.makedirs(diag_dir, exist_ok=True)

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob(os.path.join(sig_dir, '*.txt')))
    rows = []
    failures = []
    for stem in stems:
        try:
            row = diag_one(stem, sig_dir, truth_dir, cache_dir,
                           diag_dir, args.workers, args.stride)
        except Exception as exc:
            failures.append(stem)
            print(f'{stem}: FAILED ({exc!r})')
            continue
        rows.append(row)
        print(f"{stem}: fc*={row['fc_best']:0.3e} "
              f"cand={row['in_candidates']} "
              f"bracket={row['in_bracket']}")

    out = os.path.join(args.dataset, 'diag_summary.csv')
    with open(out, 'w') as f:
        keys = list(rows[0].keys())
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r[k]) for k in keys) + '\n')
    n_cand = sum(r['in_candidates'] for r in rows)
    n_brk = sum(r['in_bracket'] for r in rows)
    print(f'\ntrue optimum inside candidates: {n_cand}/{len(rows)}, '
          f'inside bracket: {n_brk}/{len(rows)} -> {out}')
    if failures:
        print(f'{len(failures)} failures: {failures}')


if __name__ == '__main__':
    main()
