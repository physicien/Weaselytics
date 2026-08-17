#!/usr/bin/python3
"""
Aggregate and plot the synthetic diagnostics of ``tools/synth_diag.py``.

Reads ``diag_summary.csv`` and the per-signal ``diag/*__diag.npz`` of a
dataset directory and answers the test battery: harness validity, the
accuracy of the current selector, containment of the true optimum, the
stiff-side instability trim with and without, and the candidate features
for stage 3.

It measures and plots. It fits nothing, and it proposes no constant: a
ratio refitted to this population would be exactly the kind of tuned
number the project rejects. Where a correlation is found it is reported
as a correlation.

Usage
-----
python tools/synth_report.py DATASET_DIR [-o OUT_DIR]
"""

import argparse
import csv
import os
from glob import glob

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# Detector quantisation step, mV. The truth baseline is not quantised
# while the signal is, so no estimator can score below roughly this.
ADC_STEP_MV = 0.008996

# Draws per signal for the null control of the local-minimum hypothesis.
NULL_DRAWS = 200


def load_summary(dataset: str) -> list[dict]:
    """
    Read ``diag_summary.csv``, coercing numeric columns.

    Parameters
    ----------
    dataset : str
        The dataset directory.

    Returns
    -------
    rows : list of dict
        One row per scored signal.

    """
    path = os.path.join(dataset, 'diag_summary.csv')
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v in ('', 'None'):
                    out[k] = None
                elif v in ('True', 'False'):
                    out[k] = v == 'True'
                else:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return rows


def col(rows: list[dict], key: str) -> np.ndarray:
    """
    One numeric column of the summary as an array, None becoming NaN.

    Parameters
    ----------
    rows : list of dict
        The summary rows.
    key : str
        Column name.

    Returns
    -------
    values : numpy.ndarray, shape (N,)

    """
    return np.array([np.nan if r[key] is None else r[key] for r in rows],
                    dtype=float)


def local_minima(values: np.ndarray, prominence_frac: float = 0.0,
                 scale: float | None = None) -> np.ndarray:
    """
    Indices of the local minima of a curve, edges excluded.

    Measured on the LINEAR values. A log axis stretches the settled
    floor of the sensitivity curve into rich-looking structure and
    manufactures minima that are not there, which is why the diagnostic
    draws that panel linearly.

    .. warning::
       With `prominence_frac` at 0 this counts **every** sample-to-
       sample wiggle, which on the sensitivity curve is 22 minima per
       surviving region at the median and up to 107. At that density a
       random cutoff is within 0.005 decade of some minimum and the
       question "is the optimum at a minimum?" is meaningless. A
       *visible* minimum -- the thing a person reads off the middle
       panel -- needs a prominence. No single value is adopted here;
       the caller sweeps it and the answer is reported as a function of
       it.

    Parameters
    ----------
    values : array-like, shape (N,)
        The curve.
    prominence_frac : float, optional
        Required prominence, as a fraction of `scale`. Default 0 keeps
        every wiggle.
    scale : float, optional
        Amplitude the prominence is measured against. Default None uses
        the peak-to-peak of `values`.

    Returns
    -------
    idx : numpy.ndarray
        Indices of the interior local minima.

    """
    v = np.asarray(values, dtype=float)
    if v.size < 3:
        return np.array([], dtype=int)
    if prominence_frac <= 0:
        idx, _ = find_peaks(-v)
        return idx
    amp = float(np.ptp(v)) if scale is None else float(scale)
    idx, _ = find_peaks(-v, prominence=prominence_frac * max(amp, 1e-30))
    return idx


def nearest_min_distance(fcut_range: np.ndarray, minima: np.ndarray,
                         fcut: float) -> float:
    """
    Distance in decades from a cutoff to the nearest local minimum.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The cutoff grid.
    minima : array-like
        Indices of the local minima.
    fcut : float
        The cutoff to measure from.

    Returns
    -------
    distance : float
        Absolute distance in decades, NaN when there is no minimum.

    """
    if len(minima) == 0:
        return float('nan')
    return float(np.min(np.abs(np.log10(fcut_range[minima])
                               - np.log10(fcut))))


def describe(name: str, v: np.ndarray, fmt: str = '.3f') -> str:
    """
    One-line summary of a distribution.

    Parameters
    ----------
    name : str
        Label.
    v : array-like
        Values; NaNs ignored.
    fmt : str, optional
        Number format.

    Returns
    -------
    line : str

    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f'{name:34s} (no finite values)'
    return (f'{name:34s} n={v.size:4d}  '
            f'p10 {np.percentile(v, 10):{fmt}}  '
            f'med {np.median(v):{fmt}}  '
            f'p90 {np.percentile(v, 90):{fmt}}  '
            f'max {np.max(v):{fmt}}')


def section_a(rows: list[dict], out: list[str]) -> None:
    """
    Harness validity. Nothing below is believable until this passes.

    Parameters
    ----------
    rows : list of dict
        Summary rows.
    out : list of str
        Report lines, appended in place.

    """
    out.append('\n=== A. HARNESS VALIDITY ===')
    n = len(rows)
    n_fail = col(rows, 'n_fail')
    e_min, e_med = col(rows, 'e_min'), col(rows, 'e_med')
    consistent = np.array([bool(r['consistent']) for r in rows])
    out.append(f'A1  signals scored (no ValueError)   {n}')
    out.append(f'A1b re-derivation matches _fcutoff   '
               f'{int(consistent.sum())}/{n}')
    out.append(f'A2  signals with zero failed fits    '
               f'{int((n_fail == 0).sum())}/{n}   '
               f'max failures on one signal {int(np.nanmax(n_fail))}')
    move = e_med / e_min
    out.append('A3  ' + describe('    e_med / e_min (metric moves)',
                                 move, '.1f'))
    out.append(f'A3  signals with e_med/e_min > 2     '
               f'{int((move > 2).sum())}/{n}')
    out.append('A4  ' + describe('    e_min, mV (quantisation floor)',
                                 e_min, '.5f'))
    out.append(f'A4  ADC step is {ADC_STEP_MV:.6f} mV; '
               f'{int((e_min < ADC_STEP_MV).sum())}/{n} signals reach '
               f'below it')


def section_b(rows: list[dict], out: list[str], outdir: str) -> None:
    """
    Accuracy of the current selector against the true optimum.

    Parameters
    ----------
    rows : list of dict
        Summary rows.
    out : list of str
        Report lines, appended in place.
    outdir : str
        Where figures are written.

    """
    out.append('\n=== B. DOES THE SELECTOR FIND THE OPTIMUM? ===')
    d = col(rows, 'd_decades')
    pen = col(rows, 'penalty')
    out.append(describe('d_decades (signed, sel-best)', d))
    out.append(describe('penalty  e(sel)/e(best)', pen, '.3f'))
    out.append(f'|d| <= 0.1 dec: {int((np.abs(d) <= 0.1).sum())}/{len(d)}'
               f'   penalty <= 1.10: {int((pen <= 1.10).sum())}/{len(pen)}')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(d[np.isfinite(d)], bins=40, color='tab:blue')
    ax[0].axvline(0, color='k', lw=1)
    ax[0].set_xlabel('log10(selected / optimal), decades')
    ax[0].set_title('selector bias and spread', fontsize=9)
    ax[1].hist(pen[np.isfinite(pen)], bins=40, color='tab:orange')
    ax[1].axvline(1, color='k', lw=1)
    ax[1].set_xlabel('penalty  e(selected)/e(optimal)')
    ax[1].set_title('cost of the selection', fontsize=9)
    nu = col(rows, 'n_used')
    ax[2].semilogx(nu, d, '.', ms=4, alpha=.6)
    ax[2].axhline(0, color='k', lw=1)
    ax[2].set_xlabel('n_used (sets the fundamental)')
    ax[2].set_ylabel('d_decades')
    ax[2].set_title('does accuracy depend on record length?', fontsize=9)
    for a in ax:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'B_selector_accuracy.png'), dpi=115)
    plt.close(fig)

    out.append('\n  by factor (median d_decades / median penalty):')
    for key in ('peak_case', 'baseline_case', 'noise_case'):
        levels = sorted({r[key] for r in rows})
        for lv in levels:
            m = np.array([r[key] == lv for r in rows])
            out.append(f'    {key:14s} {str(lv):16s} '
                       f'{np.nanmedian(d[m]):+.3f}   '
                       f'{np.nanmedian(pen[m]):.3f}   (n={int(m.sum())})')


def section_c(rows: list[dict], out: list[str]) -> None:
    """
    Containment: is the optimum inside the surviving region at all?

    Parameters
    ----------
    rows : list of dict
        Summary rows.
    out : list of str
        Report lines, appended in place.

    """
    out.append('\n=== C. CONTAINMENT ===')
    inside = np.array([bool(r['in_surviving']) for r in rows])
    out.append(f'true optimum inside the surviving region: '
               f'{int(inside.sum())}/{len(rows)}')
    if (~inside).any():
        out.append('  when it is not, the stage that excluded it:')
        reasons = [r['excluded_by'] for r in rows if not r['in_surviving']]
        for reason in sorted(set(reasons)):
            out.append(f'    {reason:16s} {reasons.count(reason)}')


def section_d(rows: list[dict], out: list[str], outdir: str) -> None:
    """
    The stiff-side instability trim, on against off.

    Parameters
    ----------
    rows : list of dict
        Summary rows.
    out : list of str
        Report lines, appended in place.
    outdir : str
        Where figures are written.

    """
    out.append('\n=== D. THE INSTABILITY TRIM, ON vs OFF ===')
    on, off = col(rows, 'penalty'), col(rows, 'penalty_off')
    fired = np.array([bool(r['trim_fired']) for r in rows])
    ok = np.isfinite(on) & np.isfinite(off)
    out.append(f'trim fired on {int(fired.sum())}/{len(rows)} signals')
    helped = ok & (on < off * 0.99)
    hurt = ok & (on > off * 1.01)
    same = ok & ~helped & ~hurt
    out.append(f'  helps   {int(helped.sum())}   '
               f'hurts   {int(hurt.sum())}   '
               f'no change {int(same.sum())}')
    out.append(describe('  penalty ON', on[ok], '.3f'))
    out.append(describe('  penalty OFF', off[ok], '.3f'))

    if helped.any() or hurt.any():
        out.append('\n  does any feature separate helped from hurt?')
        feats = ('n_used', 'snr', 'region_width_dec', 'rel_pos',
                 'r2_at_best', 'e_min')
        for fkey in feats:
            v = col(rows, fkey)
            a, b = v[helped], v[hurt]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size and b.size:
                ov = ('YES' if (a.min() < b.max() and b.min() < a.max())
                      else 'NO')
                out.append(f'    {fkey:18s} helped med {np.median(a):11.4g}'
                           f'   hurt med {np.median(b):11.4g}'
                           f'   overlap {ov}')

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].loglog(off[ok], on[ok], '.', ms=5, alpha=.6)
    lim = [np.nanmin(np.r_[on[ok], off[ok]]),
           np.nanmax(np.r_[on[ok], off[ok]])]
    ax[0].plot(lim, lim, 'k-', lw=1)
    ax[0].set_xlabel('penalty, trim OFF')
    ax[0].set_ylabel('penalty, trim ON')
    ax[0].set_title('below the line = the trim helps', fontsize=9)
    ratio = np.log10(on[ok] / off[ok])
    ax[1].hist(ratio, bins=40, color='tab:green')
    ax[1].axvline(0, color='k', lw=1)
    ax[1].set_xlabel('log10(penalty ON / penalty OFF)')
    ax[1].set_title('left = helps, right = hurts', fontsize=9)
    for a in ax:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'D_instability_trim.png'), dpi=115)
    plt.close(fig)


def section_e0(dataset: str, rows: list[dict], out: list[str],
               outdir: str) -> None:
    """
    Emmanuel's hypothesis: the optimum sits at a local minimum of the
    sensitivity curve.

    A local minimum of the sensitivity curve is a cutoff where the
    baseline is least sensitive to the cutoff, i.e. where the fit is
    most determined by the data. That is a mechanism rather than a
    ratio, which is what stage 3 needs.

    The null control is not optional: if the curve carries many minima,
    "the optimum is near a minimum" can be true by construction. The
    hypothesis is supported only if the observed distance is clearly
    below the distance from a random cutoff in the same region.

    Parameters
    ----------
    dataset : str
        The dataset directory, for the per-signal npz files.
    rows : list of dict
        Summary rows.
    out : list of str
        Report lines, appended in place.
    outdir : str
        Where figures are written.

    """
    out.append('\n=== E0. OPTIMUM AT A LOCAL MINIMUM OF SENSITIVITY? ===')
    out.append('Hypothesis (Emmanuel, 2026-08-16): the optimal cutoff sits')
    out.append('at a local minimum of the sensitivity curve -- the middle')
    out.append('panel of the r2 diagnostic. A minimum there is a cutoff')
    out.append('where the baseline is least sensitive to the cutoff, i.e.')
    out.append('most determined by the data. That is a mechanism, not a')
    out.append('ratio.')
    out.append('')
    out.append('Prominence is SWEPT, not chosen: at zero prominence every')
    out.append('numerical wiggle counts as a minimum, the region holds')
    out.append('tens of them, and a random cutoff is within 0.005 dec of')
    out.append('one -- the question would answer itself. A visible')
    out.append('minimum, the thing a person reads off the panel, needs a')
    out.append('prominence. No single value is adopted.')
    out.append('')
    rng = np.random.default_rng(0)
    cache = []
    for r in rows:
        f = os.path.join(dataset, 'diag', f"{r['stem']}__diag.npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            cache.append((r, z['fcut_range'].copy(), z['sensitivity'].copy(),
                          z['surviving'].copy()))

    # Two matched views. The null must be drawn over the SAME support
    # the minima are searched on, or the comparison is rigged: on this
    # data the optimum falls outside the surviving region on nearly half
    # the signals, so restricting the search to the region while drawing
    # the null inside it guarantees a large observed distance.
    views = (
        ('region, optimum inside only',
         [c for c in cache if c[0]['in_surviving']], True),
        ('whole curve, all signals', cache, False),
        )
    sweep = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
    best = {}
    for label, subset, in_region in views:
        out.append(f'\n  VIEW: {label}   (n={len(subset)})')
        out.append(f'  {"prom":>6} {"n_sig":>6} {"minima":>7} '
                   f'{"OBSERVED":>9} {"NULL":>9} {"obs/null":>9}  verdict')
        for frac in sweep:
            obs, null, nmin, rank_of, within = [], [], [], [], []
            for r, fr, sens, surv in subset:
                amp = float(np.ptp(sens[surv])) if surv.any() \
                    else float(np.ptp(sens))
                mins = local_minima(sens, frac, scale=amp)
                if in_region and mins.size:
                    mins = mins[surv[mins]]
                nmin.append(len(mins))
                if len(mins) == 0:
                    continue
                fc_best = r['fc_best']
                obs.append(nearest_min_distance(fr, mins, fc_best))
                step = abs(np.log10(fr[1] / fr[0]))
                within.append(obs[-1] <= 2 * step)
                j = int(np.argmin(np.abs(np.log10(fr[mins])
                                         - np.log10(fc_best))))
                rank_of.append(j / max(len(mins) - 1, 1))
                # the null is drawn over the same support as the search
                if in_region:
                    lo, hi = r['region_lo'], r['region_hi']
                else:
                    lo, hi = float(fr[0]), float(fr[-1])
                if lo and hi and hi > lo:
                    draws = 10 ** rng.uniform(np.log10(lo), np.log10(hi),
                                              NULL_DRAWS)
                    null.append(np.median([nearest_min_distance(fr, mins, d)
                                           for d in draws]))
            obs, null = np.array(obs), np.array(null)
            if obs.size == 0 or null.size == 0:
                out.append(f'  {frac:6.2f} {0:6d}       -         -    '
                           f'     -  no minima left')
                continue
            ratio = np.median(obs) / max(np.median(null), 1e-12)
            verdict = ('SUPPORTED' if ratio < 0.7 else
                       'weak' if ratio < 0.9 else 'not supported')
            out.append(f'  {frac:6.2f} {obs.size:6d} {np.median(nmin):7.1f} '
                       f'{np.median(obs):9.4f} {np.median(null):9.4f} '
                       f'{ratio:9.3f}  {verdict}')
            best[(label, frac)] = (obs, null, np.array(rank_of), within)

    out.append('')
    out.append('  OBSERVED / NULL below 1 means the optimum lies closer to')
    out.append('  a minimum than a random cutoff over the same support.')
    if best:
        ratios = {k: np.median(v[0]) / max(np.median(v[1]), 1e-12)
                  for k, v in best.items()}
        k_best = min(ratios, key=ratios.get)
        obs, null, rank_of, within = best[k_best]
        out.append(f'  strongest: {k_best[0]}, prominence {k_best[1]:.2f}'
                   f' -> ratio {ratios[k_best]:.3f}')
        out.append(f'  optimum within two grid steps of a minimum: '
                   f'{int(np.sum(within))}/{len(within)}')
        out.append(describe('  which minimum (0=low f, 1=high f)',
                            rank_of, '.3f'))

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for label, _, _ in views:
            fr_list = sorted(f for (lb, f) in ratios if lb == label)
            ax[0].plot(fr_list, [ratios[(label, f)] for f in fr_list],
                       'o-', label=label)
        ax[0].axhline(1.0, color='k', lw=1)
        ax[0].axhline(0.7, color='tab:green', lw=1, ls='--',
                      label='support threshold')
        ax[0].set_xlabel('minimum prominence, fraction of S range')
        ax[0].set_ylabel('observed / null distance')
        ax[0].legend(fontsize=6)
        ax[0].set_title('E0 swept over prominence', fontsize=9)
        bins = np.linspace(0, max(np.nanmax(obs), np.nanmax(null)), 40)
        ax[1].hist(obs, bins=bins, alpha=.65, label='observed')
        ax[1].hist(null, bins=bins, alpha=.65, label='null (random)')
        ax[1].set_xlabel('|distance| to nearest minimum, dec')
        ax[1].legend(fontsize=8)
        ax[1].set_title(f'{k_best[0]}, prom {k_best[1]:.2f}', fontsize=8)
        ax[2].hist(rank_of, bins=20, color='tab:purple')
        ax[2].set_xlabel('which minimum (0 = lowest freq, 1 = highest)')
        ax[2].set_title('if it is one, which one?', fontsize=9)
        for a in ax:
            a.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'E0_sensitivity_minima.png'),
                    dpi=115)
        plt.close(fig)


def section_e1(rows: list[dict], out: list[str], outdir: str) -> None:
    """
    Position of the optimum in the surviving region, and what predicts it.

    Parameters
    ----------
    rows : list of dict
        Summary rows.
    out : list of str
        Report lines, appended in place.
    outdir : str
        Where figures are written.

    """
    out.append('\n=== E1. POSITION IN THE REGION, AND FEATURES ===')
    p = col(rows, 'rel_pos')
    out.append(describe('rel_pos of the true optimum', p))
    out.append('  the shipped placeholder is 0.5; donnie measured '
               '0.65-0.74')
    out.append('  THIS IS A VALIDATION TARGET, NOT A RULE TO FIT.')

    feats = ('n_used', 'snr', 'region_width_dec', 'r2_at_best',
             'e_min', 'n_points')
    out.append('\n  Spearman correlation of rel_pos against features:')
    for fkey in feats:
        v = col(rows, fkey)
        m = np.isfinite(v) & np.isfinite(p)
        if m.sum() > 8:
            rho, pv = spearmanr(v[m], p[m])
            out.append(f'    {fkey:18s} rho {rho:+.3f}   p {pv:.2g}'
                       f'   (n={int(m.sum())})')

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax = ax.ravel()
    ax[0].hist(p[np.isfinite(p)], bins=30, color='tab:blue')
    for v, c, lab in ((0.5, 'k', 'placeholder'), (0.65, 'r', 'donnie lo'),
                      (0.74, 'r', 'donnie hi')):
        ax[0].axvline(v, color=c, lw=1.2, ls='--', label=lab)
    ax[0].legend(fontsize=7)
    ax[0].set_xlabel('rel_pos of the true optimum')
    for a, fkey in zip(ax[1:], feats):
        v = col(rows, fkey)
        a.semilogx(v, p, '.', ms=4, alpha=.6)
        a.axhline(0.5, color='k', lw=.8, ls='--')
        a.set_xlabel(fkey)
        a.set_ylabel('rel_pos')
    for a in ax:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'E1_position_features.png'), dpi=115)
    plt.close(fig)


def section_fgh(dataset: str, rows: list[dict], out: list[str],
                outdir: str) -> None:
    """
    Valley asymmetry, r2 at the optimum, and the SNR distribution.

    Parameters
    ----------
    dataset : str
        The dataset directory.
    rows : list of dict
        Summary rows.
    out : list of str
        Report lines, appended in place.
    outdir : str
        Where figures are written.

    """
    out.append('\n=== F. SHAPE OF THE COST (valley asymmetry) ===')
    offsets = (-0.6, -0.3, -0.1, 0.1, 0.3, 0.6)
    acc = {o: [] for o in offsets}
    for r in rows:
        f = os.path.join(dataset, 'diag', f"{r['stem']}__diag.npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            fcuts, err = z['fcuts'], z['err']
        good = np.isfinite(err)
        if good.sum() < 5:
            continue
        e0 = np.nanmin(err)
        lf = np.log10(fcuts)
        lb = np.log10(r['fc_best'])
        for o in offsets:
            j = int(np.argmin(np.abs(lf - (lb + o))))
            if good[j] and abs(lf[j] - (lb + o)) < 0.05 and e0 > 0:
                acc[o].append(err[j] / e0)
    out.append('  median penalty at a given offset from the optimum:')
    for o in offsets:
        v = np.array(acc[o])
        if v.size:
            side = 'rigid ' if o < 0 else 'flexible'
            out.append(f'    {o:+.1f} dec ({side})  '
                       f'median {np.median(v):6.2f}x   (n={v.size})')

    out.append('\n=== G. DOES r2 BRACKET THE OPTIMUM? ===')
    rb, rs = col(rows, 'r2_at_best'), col(rows, 'r2_at_selected')
    out.append(describe('r2 at the true optimum', rb, '.4f'))
    out.append(describe('r2 at the selected cutoff', rs, '.4f'))
    d = np.abs(rb - rs)
    out.append(describe('|r2(best) - r2(selected)|', d, '.5f'))
    pen = col(rows, 'penalty')
    out.append(f'  median |dr2| {np.nanmedian(d):.5f} while the median '
               f'penalty is {np.nanmedian(pen):.3f}')
    out.append('  (the standing claim: r2 moves ~0.003 while the error '
               'moves 10-13%, so r2 level cannot be the deciding feature)')

    out.append('\n=== H. _snr ON QUANTISED SYNTHETIC DATA ===')
    snr = col(rows, 'snr')
    out.append(describe('_snr', snr, '.1f'))
    fin = np.isfinite(snr)
    out.append(f'  non-finite: {int((~fin).sum())}/{len(snr)}')
    out.append(f'  gate _snr >= 25 fires on '
               f'{int((snr[fin] >= 25).sum())}/{int(fin.sum())}'
               f'  (real data: 336/339)')

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = [o for o in offsets if acc[o]]
    ys = [np.median(acc[o]) for o in xs]
    ax[0].plot(xs, ys, 'o-')
    ax[0].axvline(0, color='k', lw=.8)
    ax[0].set_xlabel('offset from the optimum, decades')
    ax[0].set_ylabel('median penalty')
    ax[0].set_title('F: err rigid or flexible?', fontsize=9)
    ax[1].hist(snr[fin], bins=40, color='tab:red')
    ax[1].axvline(25, color='k', lw=1.2, ls='--', label='gate')
    ax[1].set_xlabel('_snr')
    ax[1].legend(fontsize=8)
    ax[1].set_title('H: _snr distribution', fontsize=9)
    for a in ax:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'FGH_valley_r2_snr.png'), dpi=115)
    plt.close(fig)


def plot_valleys(dataset: str, rows: list[dict], outdir: str,
                 n: int = 12) -> None:
    """
    Draw the error curve of a sample of signals.

    Nothing is reported as numbers alone: this is where the shape of
    ``E(fcut)``, the true optimum and the selected cutoff can be judged
    by eye.

    Parameters
    ----------
    dataset : str
        The dataset directory.
    rows : list of dict
        Summary rows.
    outdir : str
        Where the figure is written.
    n : int, optional
        How many signals to draw. Default 12.

    """
    sel = rows[::max(1, len(rows) // n)][:n]
    ncol = 3
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.6 * nrow))
    for ax, r in zip(np.atleast_1d(axes).ravel(), sel):
        f = os.path.join(dataset, 'diag', f"{r['stem']}__diag.npz")
        with np.load(f) as z:
            fcuts, err = z['fcuts'], z['err']
        ax.loglog(fcuts, err, lw=.9, color='0.3')
        ax.axvline(r['fc_best'], color='tab:green', lw=1.3,
                   label='true optimum')
        ax.axvline(r['fcut_selected'], color='tab:red', lw=1.3,
                   label='selected')
        lo, hi = r['region_lo'], r['region_hi']
        if lo and hi:
            ax.axvspan(lo, hi, color='tab:blue', alpha=.12,
                       label='surviving')
        ax.axvline(1. / r['n_used'], color='k', ls=':', lw=.8,
                   label='fundamental')
        ax.set_title(f"{r['stem'][14:][:38]}\npenalty "
                     f"{r['penalty']:.2f}", fontsize=6.5)
        ax.tick_params(labelsize=6)
    np.atleast_1d(axes).ravel()[0].legend(fontsize=5.5)
    for ax in np.atleast_1d(axes).ravel()[len(sel):]:
        ax.axis('off')
    fig.suptitle('true error curve E(fcut): optimum, selection, region',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'valleys.png'), dpi=115)
    plt.close(fig)


def main() -> None:
    """
    CLI entry point of the diagnostics reporter.

    """
    parser = argparse.ArgumentParser(
        prog='synth_report',
        description='aggregate and plot the synthetic diagnostics')
    parser.add_argument('dataset', help='directory scored by synth_diag')
    parser.add_argument('-o', '--out', default=None,
                        help='output directory (default: DATASET/report)')
    args = parser.parse_args()
    outdir = args.out or os.path.join(args.dataset, 'report')
    os.makedirs(outdir, exist_ok=True)

    rows = load_summary(args.dataset)
    n_npz = len(glob(os.path.join(args.dataset, 'diag', '*__diag.npz')))
    out = ['Synthetic diagnostics report',
           f'dataset : {args.dataset}',
           f'signals : {len(rows)} scored, {n_npz} diag files']

    section_a(rows, out)
    section_b(rows, out, outdir)
    section_c(rows, out)
    section_d(rows, out, outdir)
    section_e0(args.dataset, rows, out, outdir)
    section_e1(rows, out, outdir)
    section_fgh(args.dataset, rows, out, outdir)
    plot_valleys(args.dataset, rows, outdir)

    text = '\n'.join(out)
    print(text)
    with open(os.path.join(outdir, 'report.txt'), 'w') as f:
        f.write(text + '\n')
    print(f'\n-> {outdir}')


if __name__ == '__main__':
    main()
