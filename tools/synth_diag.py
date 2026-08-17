#!/usr/bin/python3
"""
Diagnose the fcut machinery against synthetic ground truth.

For every signal of a dataset produced by ``tools/synth_dataset.py``:

* run the **production** selection through ``baseline._fcutoff``, which
  returns the selected cutoff together with every intermediate mask, so
  the harness cannot drift from what ships;
* recompute the stage-2 trimming twice, with the stiff-side instability
  exclusion on and off, giving every signal a matched pair;
* compute the TRUE error curve ``E(fcut)``: the RMSE between the
  baseline fitted with the final-correction configuration of
  ``auto_beads`` and the known true baseline, on a subsampled fcut grid.

The minimum of ``E`` is the objective optimal cutoff frequency. It is
defined by baseline fit alone -- peak-area error is deliberately not
computed and enters no decision here.

Writes one ``diag/<stem>__diag.npz`` per signal and a summary CSV.

Usage
-----
python tools/synth_diag.py DATASET_DIR [--workers 8] [--stride 4]
                           [--limit N] [--pattern GLOB]
"""

import argparse
import contextlib
import csv
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from pybaselines import Baseline

from weaselytics.baseline import (
    _custom_beads,
    _fcutoff,
    _relevant_regions,
    _snr,
    )
from weaselytics.parsers import ParsedData
from weaselytics.segmentation import (
    classify_segments,
    detect_dips,
    pelt_linear,
    segment_features,
    select_center,
    trim_plateaus,
    )
from weaselytics.utils import end_window

SUMMARY_FIELDS = [
    'stem', 'peak_case', 'baseline_case', 'noise_case', 'n_points',
    'replicate', 'n_used', 'snr',
    'fc_best', 'e_min', 'e_med', 'e_max',
    'fcut_selected', 'e_at_selected', 'd_decades', 'penalty',
    'fcut_selected_off', 'e_at_selected_off', 'd_decades_off',
    'penalty_off',
    'in_surviving', 'region_lo', 'region_hi', 'region_width_dec',
    'rel_pos', 'trim_fired', 'excluded_by',
    'r2_at_best', 'r2_at_selected', 'n_fail', 'consistent',
    ]


def error_curve(x: np.ndarray, s: np.ndarray, b_true: np.ndarray,
                fcuts: np.ndarray) -> np.ndarray:
    """
    Compute the true baseline-recovery error along the fcut grid.

    Mirrors the final-correction configuration of ``auto_beads``
    (custom_beads, per-region sampling, ``alpha=1``,
    ``parabola_len=end_window(s)``, asymmetry 1).

    The error is the RMSE against the known baseline on the **original**
    scale, which is Niezen et al. (2022) Eq. (15). Note the production
    sweep instead fits on ``_log_transform(s[:scut])``: the error curve
    and the r2 curve are therefore computed on different signals over
    the same grid, which is intended -- the baseline is a physical
    quantity in mV, while the log scale is an artefact of the method.

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

    References
    ----------
    Niezen, Schoenmakers & Pirok (2022), Anal. Chim. Acta 1201, 339605,
    Eq. (15).

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


def _region_of(fcut_range: np.ndarray, mask: np.ndarray
               ) -> tuple[float, float] | tuple[None, None]:
    """
    Bounds of the last contiguous run of a boolean mask.

    The last run rather than the first, matching `select_center`, which
    takes the last surviving region per Navarro-Huerta §3.4.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The cutoff frequencies.
    mask : array-like, shape (N,), dtype bool
        A surviving mask.

    Returns
    -------
    lo, hi : float or None
        The region bounds, or ``(None, None)`` when nothing survives.

    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None, None
    splits = np.where(np.diff(idx) > 1)[0] + 1
    region = np.split(idx, splits)[-1]
    return float(fcut_range[region[0]]), float(fcut_range[region[-1]])


def _err_at(x: np.ndarray, s: np.ndarray, b_true: np.ndarray,
            fcut: float | None) -> float:
    """
    Error at one cutoff, fitted exactly rather than interpolated.

    Parameters
    ----------
    x, s, b_true : array-like, shape (N,)
        Time axis, measured signal, and known baseline.
    fcut : float or None
        The cutoff to evaluate. None returns NaN.

    Returns
    -------
    err : float
        RMSE against `b_true`, or NaN.

    """
    if fcut is None:
        return float('nan')
    return float(error_curve(x, s, b_true, np.array([fcut]))[0])


def diag_one(stem: str, sig_dir: str, truth_dir: str, cache_dir: str,
             diag_dir: str, stride: int, snr_threshold: float = 25.0
             ) -> dict:
    """
    Run the diagnostics of a single signal.

    Parameters
    ----------
    stem : str
        Signal name (file stem).
    sig_dir, truth_dir, cache_dir, diag_dir : str
        Dataset sub-directories.
    stride : int
        fcut grid stride of the error curve.
    snr_threshold : float, optional
        Gate of the collapse exclusion, as in production. Default 25.

    Returns
    -------
    row : dict
        Summary of the signal's diagnostics, keyed by `SUMMARY_FIELDS`.

    """
    path = os.path.join(sig_dir, f'{stem}.txt')
    x, s = ParsedData(path).data
    truth = np.load(os.path.join(truth_dir, f'{stem}__truth.npz'))
    b_true = truth['baseline']

    # THE PRODUCTION PATH. `_fcutoff` holds the whole selection, so
    # reading its plot_data is the only way to score what ships --
    # but it must be called the way `auto_beads` calls it.
    #
    # MIRRORS baseline.auto_beads (the `method_kwargs` block and the
    # `_fcutoff` call). Getting this wrong is not a small error: the
    # defaults run plain `beads` with no peak regions instead of
    # `custom_beads` with them, which is a different algorithm on a
    # different signal, and it silently produced a 15% crash rate and
    # 54% containment on the first pilot. `test_synth_diag.py` pins
    # these against auto_beads' own signature so they cannot drift.
    peak_regions, sampling, scut = _relevant_regions(s, x)
    method_kwargs = {
        'asymmetry': 1.0,
        'fit_parabola': True,
        'alpha': 1.0,
        'parabola_len': 3,
        'regions': peak_regions,
        'sampling': sampling,
        }
    with contextlib.redirect_stdout(io.StringIO()):
        fcut_sel, pd = _fcutoff(s, x, scut, method='custom_beads',
                                cache_dir=cache_dir, path=path,
                                workers=1, snr_threshold=snr_threshold,
                                **method_kwargs)
    fcut_range = pd['fcut_range']
    r2 = pd['r2_val']
    sens = pd['sensitivity_val']
    n_used = pd['n_used']
    surviving = pd['cp_surviving']

    # Matched pair: the identical chain, with the stiff-side exclusion
    # on and off. `trim_plateaus` is the single source, so this differs
    # from production only in the one argument being tested.
    segments = classify_segments(
        segment_features(fcut_range, r2, pelt_linear(r2)))
    dips = detect_dips(fcut_range, r2)
    snr_val = float(_snr(s))
    common = dict(exclude_collapse=snr_val >= snr_threshold)
    trim_on = trim_plateaus(fcut_range, segments, dips, n_used,
                            sensitivity=sens, **common)
    trim_off = trim_plateaus(fcut_range, segments, dips, n_used,
                             sensitivity=None, **common)
    fcut_on = select_center(fcut_range, trim_on['surviving'])
    fcut_off = select_center(fcut_range, trim_off['surviving'])
    # If the re-derivation disagrees with production the harness is
    # wrong, not the pipeline; recorded rather than silently trusted.
    consistent = bool(fcut_on is not None
                      and np.isclose(fcut_on, fcut_sel, rtol=0, atol=0))

    fcuts = fcut_range[::stride]
    err = error_curve(x, s, b_true, fcuts)
    k_best = int(np.nanargmin(err))
    fc_best = float(fcuts[k_best])
    e_min = float(np.nanmin(err))

    e_sel = _err_at(x, s, b_true, fcut_sel)
    e_off = _err_at(x, s, b_true, fcut_off)

    lo, hi = _region_of(fcut_range, surviving)
    i_best = int(np.argmin(np.abs(fcut_range - fc_best)))
    i_sel = int(np.argmin(np.abs(fcut_range - fcut_sel)))
    rel_pos = float('nan')
    if lo is not None and hi is not None and hi > lo:
        rel_pos = float((np.log10(fc_best) - np.log10(lo))
                        / (np.log10(hi) - np.log10(lo)))

    excluded_by = ''
    if not surviving[i_best]:
        for name, key in (('instability', 'cp_instab_removed'),
                          ('collapse', 'cp_snr_removed'),
                          ('clip_or_tail', 'cp_removed')):
            if pd[key][i_best]:
                excluded_by = name
                break
        else:
            excluded_by = 'not_detected'

    np.savez(os.path.join(diag_dir, f'{stem}__diag.npz'),
             fcut_range=fcut_range, r2=r2, sensitivity=sens,
             fcuts=fcuts, err=err, n_used=n_used,
             fcut_selected=fcut_sel,
             fcut_selected_off=(np.nan if fcut_off is None else fcut_off),
             surviving=surviving, surviving_off=trim_off['surviving'],
             removed=pd['cp_removed'], snr_removed=pd['cp_snr_removed'],
             instab_removed=pd['cp_instab_removed'],
             dip_mask=pd['cp_dips'], flat_mask=pd['cp_flat'])

    parts = stem.split('__')
    return {
        'stem': stem,
        'peak_case': parts[2] if len(parts) > 2 else '',
        'baseline_case': parts[3] if len(parts) > 3 else '',
        'noise_case': parts[4] if len(parts) > 4 else '',
        'n_points': len(s), 'replicate': parts[6] if len(parts) > 6 else '',
        'n_used': n_used, 'snr': snr_val,
        'fc_best': fc_best, 'e_min': e_min,
        'e_med': float(np.nanmedian(err)), 'e_max': float(np.nanmax(err)),
        'fcut_selected': float(fcut_sel), 'e_at_selected': e_sel,
        'd_decades': float(np.log10(fcut_sel / fc_best)),
        'penalty': float(e_sel / e_min) if e_min > 0 else float('nan'),
        'fcut_selected_off': (float('nan') if fcut_off is None
                              else float(fcut_off)),
        'e_at_selected_off': e_off,
        'd_decades_off': (float('nan') if fcut_off is None
                          else float(np.log10(fcut_off / fc_best))),
        'penalty_off': (float(e_off / e_min) if e_min > 0
                        else float('nan')),
        'in_surviving': bool(surviving[i_best]),
        'region_lo': lo, 'region_hi': hi,
        'region_width_dec': (float(np.log10(hi / lo))
                             if lo and hi else float('nan')),
        'rel_pos': rel_pos,
        'trim_fired': bool(pd['cp_instab_removed'].any()),
        'excluded_by': excluded_by,
        'r2_at_best': float(r2[i_best]),
        'r2_at_selected': float(r2[i_sel]),
        'n_fail': int((~np.isfinite(err)).sum()),
        'consistent': consistent,
        }


def _worker(args: tuple) -> tuple[str, dict | None, str]:
    """
    Process-pool entry point: run one signal, never raise.

    Parameters
    ----------
    args : tuple
        ``(stem, sig_dir, truth_dir, cache_dir, diag_dir, stride)``.

    Returns
    -------
    stem : str
        The signal name.
    row : dict or None
        The summary row, or None on failure.
    error : str
        Empty on success, else the exception repr.

    """
    stem = args[0]
    try:
        return stem, diag_one(*args), ''
    except Exception as exc:
        return stem, None, repr(exc)


def main() -> None:
    """
    CLI entry point of the synthetic diagnostics runner.

    """
    parser = argparse.ArgumentParser(
        prog='synth_diag',
        description='score the fcut machinery on synthetic truth')
    parser.add_argument('dataset', help='directory from synth_dataset.py')
    parser.add_argument('--workers', type=int, default=8,
                        help='parallel signals (default: 8). The error '
                             'curve is serial per signal, so the useful '
                             'parallelism is over signals, not inside '
                             'the sweep')
    parser.add_argument('--stride', type=int, default=4,
                        help='fcut grid stride of the error curve '
                             '(default: 4, i.e. 0.019 decade)')
    parser.add_argument('--limit', type=int, default=None,
                        help='stop after N signals (pilot runs)')
    parser.add_argument('--pattern', default='*',
                        help='glob on the stem (default: all)')
    args = parser.parse_args()

    sig_dir = os.path.join(args.dataset, 'signals')
    truth_dir = os.path.join(args.dataset, 'truth')
    cache_dir = os.path.join(args.dataset, 'r2_cache')
    diag_dir = os.path.join(args.dataset, 'diag')
    os.makedirs(diag_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob(os.path.join(sig_dir,
                                              f'{args.pattern}.txt')))
    if args.limit:
        # Stride the sorted list rather than truncating it, so a pilot
        # spans every factor instead of one corner of the design.
        step = max(1, len(stems) // args.limit)
        stems = stems[::step][:args.limit]

    jobs = [(s, sig_dir, truth_dir, cache_dir, diag_dir, args.stride)
            for s in stems]
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            stem, row, err = fut.result()
            if row is None:
                failures.append((stem, err))
                print(f'[{k}/{len(jobs)}] {stem}: FAILED {err}')
                continue
            rows.append(row)
            print(f"[{k}/{len(jobs)}] {stem}: fc*={row['fc_best']:.3e} "
                  f"sel={row['fcut_selected']:.3e} "
                  f"d={row['d_decades']:+.3f} dec "
                  f"penalty={row['penalty']:.2f}")

    rows.sort(key=lambda r: r['stem'])
    out = os.path.join(args.dataset, 'diag_summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    if rows:
        d = np.array([r['d_decades'] for r in rows])
        pen = np.array([r['penalty'] for r in rows])
        n_in = sum(r['in_surviving'] for r in rows)
        n_bad = sum(not r['consistent'] for r in rows)
        print(f'\n{len(rows)} scored, {len(failures)} failed -> {out}')
        print(f'  optimum inside the surviving region: {n_in}/{len(rows)}')
        print(f'  d_decades  median {np.median(d):+.3f}  '
              f'p10 {np.percentile(d, 10):+.3f}  '
              f'p90 {np.percentile(d, 90):+.3f}')
        print(f'  penalty    median {np.median(pen):.2f}  '
              f'p90 {np.percentile(pen, 90):.2f}  '
              f'max {np.nanmax(pen):.2f}')
        if n_bad:
            print(f'  WARNING: {n_bad} signals where the re-derived '
                  f'selection disagrees with _fcutoff')
    if failures:
        print(f'{len(failures)} failures: {[s for s, _ in failures]}')


if __name__ == '__main__':
    main()
