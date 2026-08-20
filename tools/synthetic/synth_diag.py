#!/usr/bin/python3
"""
Diagnose the fcut machinery against synthetic ground truth.

For every signal of a dataset produced by ``tools/synthetic/synth_dataset.py``:

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

Accuracy is reported in units of the record's own noise, which is the
scale on which a difference in the corrected chromatogram is visible:

``target_noise = e_min / sigma``
    How far the best fit on the whole grid sits from the true baseline.
    Near 1, the grid holds a fit indistinguishable from the truth and
    ``fc_best`` is worth aiming at. Large, and no cutoff recovers this
    baseline: there is no optimum to find, and any accuracy figure for
    that signal describes the benchmark rather than the method.

``excess_noise = (e_at_selected - e_min) / sigma``
    What the selection costs over that best fit.

These replace ``d_decades`` and ``penalty`` as the reported figures.
``d_decades`` measures a distance along the parameter rather than in
the corrected signal, and two cutoffs two decades apart can move the
baseline by a few percent of the noise; ``penalty`` is a ratio that
diverges as ``e_min`` approaches zero, when both fits are far below the
noise. Both columns are still written, since the report battery reads
them, but neither should be quoted as accuracy.

``sigma`` is the generator's own noise array, so these figures exist on
synthetic data only.

Writes one ``diag/<stem>__diag.npz`` per signal and a summary CSV.

Usage
-----
python tools/synthetic/synth_diag.py DATASET_DIR [--workers 8] [--stride 4]
                           [--limit N] [--pattern GLOB]
"""

import argparse
import contextlib
import csv
import hashlib
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

# Endpoint window BEADS anchors its parabola on, as production sets it.
# See `error_curve` for why this is 3 and not `end_window(s)`.
_PARABOLA_LEN = 3

SUMMARY_FIELDS = [
    'stem', 'peak_case', 'baseline_case', 'noise_case', 'n_points',
    'replicate', 'n_used', 'snr',
    'fc_best', 'e_min', 'e_med', 'e_max', 'noise_sigma',
    'target_rmse', 'target_max', 'target_snr_db', 'target_area_pct',
    'selected_rmse', 'selected_max', 'selected_snr_db',
    'selected_area_pct',
    'target_rms_noise', 'target_max_noise',
    'selected_rms_noise', 'selected_max_noise',
    'target_noise', 'excess_noise',
    'fcut_selected', 'e_at_selected', 'd_decades', 'penalty',
    'fcut_selected_off', 'e_at_selected_off', 'd_decades_off',
    'penalty_off', 'excess_noise_off',
    'in_surviving', 'region_lo', 'region_hi', 'region_width_dec',
    'rel_pos', 'trim_fired', 'excluded_by',
    'r2_at_best', 'r2_at_selected', 'n_fail', 'consistent',
    ]


def error_curve(x: np.ndarray, s: np.ndarray, b_true: np.ndarray,
                fcuts: np.ndarray) -> np.ndarray:
    """
    Compute the true baseline-recovery error along the fcut grid.

    Mirrors the final-correction configuration of ``auto_beads``
    (custom_beads, per-region sampling, ``alpha=1``, ``parabola_len=3``,
    asymmetry 1). `auto_beads` only substitutes ``end_window(s)`` when a
    caller passes ``parabola_len=None``, which no caller does; that
    branch predates pybaselines exposing `parabola_len` on BEADS and is
    a fallback, not the production setting.

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
              'parabola_len': _PARABOLA_LEN}
    err = np.full(len(fcuts), np.nan)
    for k, fc in enumerate(fcuts):
        try:
            bl, _ = _custom_beads(fitter, s, freq_cutoff=fc, **kwargs)
            err[k] = float(np.sqrt(np.mean((bl - b_true) ** 2)))
        except Exception:
            pass
    return err


def _err_key(s: np.ndarray, x: np.ndarray, b_true: np.ndarray,
             fcuts: np.ndarray) -> str:
    """
    Key identifying an error curve, for reuse across runs.

    The curve depends on the signal, the true baseline, the grid, and
    the fit configuration `error_curve` uses -- which includes
    `_relevant_regions`, so a change there must invalidate it. It does
    NOT depend on the trimming, the selection or the reported metrics,
    which is what makes reuse worth having: those are what change from
    run to run.

    Arrays are hashed at float32 as in `baseline._r2_cache_key`, since
    neither `np.geomspace` nor the signal is bit-reproducible across
    numpy versions and platforms; a value within one ulp of a rounding
    boundary costs a recomputation, never a wrong curve.

    Parameters
    ----------
    s, x, b_true : array-like, shape (N,)
        Measured signal, time axis and known baseline.
    fcuts : array-like, shape (M,)
        The grid the curve is evaluated on.

    Returns
    -------
    key : str
        Hexadecimal digest.

    """
    regions, sampling, _ = _relevant_regions(s, x)
    sha = hashlib.sha1()
    for arr in (s, x, b_true, fcuts):
        sha.update(str(len(arr)).encode())
        sha.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    for arr in (regions, sampling):
        sha.update(repr(None if arr is None else np.asarray(arr).shape
                        ).encode())
        if arr is not None:
            sha.update(np.ascontiguousarray(arr).tobytes())
    sha.update(str(_PARABOLA_LEN).encode())
    return sha.hexdigest()[:12]


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


def _err_stats(x: np.ndarray, s: np.ndarray, b_true: np.ndarray,
               sig_true: np.ndarray, fcut: float | None) -> dict:
    """
    Departure of one fitted baseline from the true one, five ways.

    No source in the literature argues for a normalization; each simply
    picks one, so all of them are reported and the choice is left to the
    reader.

    ``rms``
        Plain RMSE against the known baseline, in the signal's units,
        dividing only by the point count. Niezen et al. (2022) Eq. (15).
    ``peak``
        Maximum absolute departure. No source uses this; it is here
        because the RMS averages a local defect over the whole record,
        so the same defect scores lower in a long run than a short one.
    ``rms_noise``, ``peak_noise``
        The two above in units of the record's noise. No source uses
        this either, and it misleads when the signal towers over the
        noise.
    ``snr_db``
        ``10 log10( energy(b_true) / energy(b_true - fit) )``. Ning,
        Selesnick & Duval (2014) §5.1, the BEADS paper, quoted there as
        "the energy of the generated baseline divided by the energy of
        the difference between the generated and the estimated
        baselines". NOTE: energy includes the baseline's offset, so a
        trace sitting far from zero scores high whatever the fit does.
        Their simulated baselines are near zero; ours are not.
    ``area_pct``
        Percent error in recovered peak area against the known one.
        Navarro-Huerta et al. (2017) §3.6; Niezen et al. (2022) §4.3.3
        uses the same measure. Taken
        over the whole record rather than per peak, since the region
        boundaries would otherwise be a choice of ours. NaN when the
        true area is too small to divide by, which is most blanks.

    Parameters
    ----------
    x, s, b_true, sig_true : array-like, shape (N,)
        Time axis, measured signal, known baseline, and known peak
        component.
    fcut : float or None
        The cutoff to evaluate. None returns all-NaN.

    Returns
    -------
    stats : dict
        The five figures above; NaN where the fit failed.

    """
    nan = float('nan')
    blank = {'rms': nan, 'peak': nan, 'snr_db': nan, 'area_pct': nan}
    if fcut is None:
        return blank
    peak_regions, sampling, _ = _relevant_regions(s, x)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bl, _ = _custom_beads(
                Baseline(x_data=x), s, freq_cutoff=float(fcut),
                regions=peak_regions, sampling=sampling, asymmetry=1.0,
                fit_parabola=True, alpha=1.0, parabola_len=_PARABOLA_LEN)
    except Exception:
        return blank
    d = bl - b_true
    e_diff = float(np.sum(d ** 2))
    e_true = float(np.sum(b_true ** 2))
    a_true = float(np.trapezoid(sig_true, x))
    a_fit = float(np.trapezoid(s - bl, x))
    # A blank has essentially no area; the ratio is then meaningless
    # rather than large, so it is not reported.
    rel = (100. * (a_fit - a_true) / a_true
           if abs(a_true) > 1e-9 * (abs(a_fit) + 1.) else nan)
    return {'rms': float(np.sqrt(np.mean(d ** 2))),
            'peak': float(np.abs(d).max()),
            'snr_db': (10. * np.log10(e_true / e_diff)
                       if e_diff > 0 and e_true > 0 else nan),
            'area_pct': rel}


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
             diag_dir: str, stride: int, reuse_err: bool = True,
             snr_threshold: float = 25.0) -> dict:
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
    reuse_err : bool, optional
        Reuse a stored error curve when its key matches. Default True.
        The key covers `_relevant_regions`, so a change there
        invalidates it.
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
    # The error curve is 250 BEADS fits and is unchanged by anything
    # downstream of it, so it is reused when its key still matches.
    npz = os.path.join(diag_dir, f'{stem}__diag.npz')
    key = _err_key(s, x, b_true, fcuts)
    err = None
    if reuse_err and os.path.isfile(npz):
        try:
            with np.load(npz) as d:
                if str(d.get('err_key', '')) == key:
                    err = d['err']
        except Exception:
            err = None
    if err is None:
        err = error_curve(x, s, b_true, fcuts)
    k_best = int(np.nanargmin(err))
    fc_best = float(fcuts[k_best])
    e_min = float(np.nanmin(err))

    e_sel = _err_at(x, s, b_true, fcut_sel)
    e_off = _err_at(x, s, b_true, fcut_off)

    # Both accuracy figures are expressed in units of the record's own
    # noise, which is the scale on which a difference in the corrected
    # chromatogram is visible at all.
    #
    # `target_noise` is how far the BEST fit on the grid sits from the
    # true baseline. Near 1 the grid contains a fit indistinguishable
    # from the truth, and `fc_best` is a target worth aiming at. Large,
    # and no cutoff recovers this baseline, so there is no optimum to
    # find and every accuracy figure below it measures the benchmark
    # rather than the method.
    #
    # `excess_noise` is what the selection costs over that best fit.
    # It replaces `d_decades`, which measures a distance along the
    # parameter rather than in the corrected signal: two cutoffs two
    # decades apart can move the baseline by a few percent of the
    # noise, and `d_decades` cannot tell that from a real miss. It also
    # replaces `penalty`, a ratio that diverges when `e_min` approaches
    # zero even though both fits are then far below the noise.
    #
    # The noise is the generator's own array, known exactly, so this is
    # available on synthetic data only.
    sigma = float(np.std(truth['noise']))
    sig_true = truth['signal']
    tg = _err_stats(x, s, b_true, sig_true, fc_best)
    sl = _err_stats(x, s, b_true, sig_true, fcut_sel)
    t_rms, t_max = tg['rms'], tg['peak']
    s_rms, s_max = sl['rms'], sl['peak']
    if sigma > 0:
        target_noise = e_min / sigma
        target_rms_noise, target_max_noise = t_rms / sigma, t_max / sigma
        selected_rms_noise = s_rms / sigma
        selected_max_noise = s_max / sigma
        excess_noise = (e_sel - e_min) / sigma
        excess_noise_off = ((e_off - e_min) / sigma if np.isfinite(e_off)
                            else float('nan'))
    else:
        target_noise = excess_noise = excess_noise_off = float('nan')
        target_rms_noise = target_max_noise = float('nan')
        selected_rms_noise = selected_max_noise = float('nan')

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
             fcuts=fcuts, err=err, err_key=key, n_used=n_used,
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
        'noise_sigma': sigma, 'target_noise': target_noise,
        'target_rmse': tg['rms'], 'target_max': tg['peak'],
        'target_snr_db': tg['snr_db'], 'target_area_pct': tg['area_pct'],
        'selected_rmse': sl['rms'], 'selected_max': sl['peak'],
        'selected_snr_db': sl['snr_db'],
        'selected_area_pct': sl['area_pct'],
        'target_rms_noise': target_rms_noise,
        'target_max_noise': target_max_noise,
        'selected_rms_noise': selected_rms_noise,
        'selected_max_noise': selected_max_noise,
        'excess_noise': excess_noise,
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
        'excess_noise_off': excess_noise_off,
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
        ``(stem, sig_dir, truth_dir, cache_dir, diag_dir, stride,
        reuse_err)``.

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
    parser.add_argument('--refresh-err', action='store_true',
                        help='recompute the error curve even when the '
                             'stored one still matches its key')
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

    jobs = [(s, sig_dir, truth_dir, cache_dir, diag_dir, args.stride,
             not args.refresh_err) for s in stems]
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
                  f"green-true rms={row['target_rms_noise']:.1f} "
                  f"max={row['target_max_noise']:.1f} sigma")

    rows.sort(key=lambda r: r['stem'])
    out = os.path.join(args.dataset, 'diag_summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    if rows:
        exc = np.array([r['excess_noise'] for r in rows])
        tgt = np.array([r['target_noise'] for r in rows])
        n_in = sum(r['in_surviving'] for r in rows)
        n_bad = sum(not r['consistent'] for r in rows)
        print(f'\n{len(rows)} scored, {len(failures)} failed -> {out}')
        print(f'  optimum inside the surviving region: {n_in}/{len(rows)}')
        # How far the best fit on the grid sits from the true baseline.
        # Where this is large the signal has no optimum to find, and the
        # accuracy line below it is measuring the benchmark.
        print('  green vs true, in noise units:')
        for key, lab in (('target_rms_noise', 'rms'),
                         ('target_max_noise', 'max')):
            v = np.array([r[key] for r in rows])
            print(f'    {lab:3s}  median {np.nanmedian(v):7.2f}  '
                  f'p90 {np.nanpercentile(v, 90):8.2f}  '
                  f'max {np.nanmax(v):9.2f}')
        print('  red vs true, in noise units:')
        for key, lab in (('selected_rms_noise', 'rms'),
                         ('selected_max_noise', 'max')):
            v = np.array([r[key] for r in rows])
            print(f'    {lab:3s}  median {np.nanmedian(v):7.2f}  '
                  f'p90 {np.nanpercentile(v, 90):8.2f}  '
                  f'max {np.nanmax(v):9.2f}')
        for thr in (3., 10.):
            n = int(np.nansum(tgt > thr))
            print(f'    beyond {thr:4.0f} sigma, no usable target : '
                  f'{n:3d}/{len(rows)}')
        print(f'  excess_noise   median {np.nanmedian(exc):7.2f}  '
              f'p90 {np.nanpercentile(exc, 90):7.2f}  '
              f'max {np.nanmax(exc):8.2f}   sigma')
        good = tgt <= 3.
        if good.any():
            e = exc[good]
            print(f'    on the {int(good.sum())} with a usable target: '
                  f'median {np.nanmedian(e):.2f}  '
                  f'p90 {np.nanpercentile(e, 90):.2f} sigma')
        if n_bad:
            print(f'  WARNING: {n_bad} signals where the re-derived '
                  f'selection disagrees with _fcutoff')
    if failures:
        print(f'{len(failures)} failures: {[s for s, _ in failures]}')


if __name__ == '__main__':
    main()
