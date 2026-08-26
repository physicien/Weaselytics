"""
Quasi-optimality on the 165, against the corrected error curves.

Population: the trusted subset (`e_min <= 0.05`, the RMSE between the
baseline at the optimal cutoff and the true baseline) intersected with
the true optimum falling inside the **final surviving region**, which is
Emmanuel's condition and the only one under which a stage-3 rule can
reach the optimum at all.

That is NOT the harness's `in_candidates`, which comes from
`trim_candidates(fcut_range, segments, n_used)` with defaults: the
sub-fundamental clip and bridging only, no past-drop exclusion, no
instability trim, no dip fallback. Production selects from
`trim_plateaus(...)['surviving']`, which is narrower, so containment
there has to be computed rather than read off the summary.

The criterion is Bauer & Kindermann 2008 Definition 1.1 Eq. (6),
`argmin ||x_n - x_(n+1)||`. `_sensitivity_curve` is that functional up
to a positive per-signal constant, so its argmin is the same point. It
is taken INSIDE the surviving region, because stage 3 chooses there;
taking it over the whole support scores the trimming instead.

A null drawn uniformly in log over the same region is reported, without
which none of the ratios can be read.

Run: runs/SYNTH_2026-08-24, generated at parabola_len=3 (commit 1cd50fe).
"""

import csv
import glob
import os
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

sys.path.insert(0, '/home/esteban/Simulation/DFT/separation_part2/'
                   'Weaselytics')
import weaselytics.segmentation as S  # noqa: E402
from weaselytics.baseline import _snr  # noqa: E402
from weaselytics.parsers import ParsedData  # noqa: E402

ROOT = '/home/esteban/Simulation/DFT/separation_part2'
RUN = os.path.join(ROOT, 'runs/SYNTH_2026-08-24')
DS = os.path.join(ROOT, 'SYNTH_ERB_2026-08-18')
OUT = os.path.dirname(os.path.abspath(__file__))
RMSE_MAX = 0.05
NULL_DRAWS = 200
RNG = np.random.default_rng(20260824)


def population():
    """
    The trusted subset. Containment in the final surviving region is
    computed per signal in `one`, not taken from `in_candidates`.
    """
    p = os.path.join(RUN, 'diag', 'diag_summary.csv')
    return [r['stem'] for r in csv.DictReader(open(p))
            if float(r['e_min']) <= RMSE_MAX]


def one(stem):
    cache = glob.glob(os.path.join(RUN, 'r2_cache', f'{stem}__r2__*.npz'))
    dg = os.path.join(RUN, 'diag', f'{stem}__diag.npz')
    sig = os.path.join(DS, 'signals', f'{stem}.txt')
    if not (cache and os.path.exists(dg) and os.path.exists(sig)):
        return None
    c = np.load(cache[0])
    fcut, r2, sens = c['fcut_range'], c['r2_val'], c['sensitivity']
    d = np.load(dg)
    fcuts, err, n_used = d['fcuts'], d['err'], int(d['n_used'])
    ok = np.isfinite(err)
    if ok.sum() < 10:
        return None

    x, y = ParsedData(sig).data
    segs = S.classify_segments(S.segment_features(fcut, r2,
                                                  S.pelt_linear(r2)))
    trim = S.trim_plateaus(fcut, segs, S.detect_dips(fcut, r2), n_used,
                           exclude_past_drop=bool(_snr(y) >= 10.0),
                           sensitivity=sens)
    v = S.select_center(fcut, trim['surviving'])
    f_pkg = float(v) if v else np.nan

    mask = np.asarray(trim['surviving'], dtype=bool)
    if not mask.any():
        return None
    edges = np.flatnonzero(np.diff(mask.astype(int)))
    bounds = np.concatenate(([0], edges + 1, [len(mask)]))
    runs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])
            if mask[a] and b - a >= 3]
    if not runs:
        return None
    a, b = runs[-1]                      # select_center takes the last
    sr = sens[a:b]
    if not np.all(np.isfinite(sr)) or np.ptp(sr) <= 0:
        return None

    # 3-point running median before the argmin: the raw argmin lands
    # on a single anomalous sample on 20% of signals. Gallagher & Wise
    # 1981 Sec. II Def. 3 and Thm 1; see `despike` in qo_all.py.
    srf = median_filter(sr, size=3, mode='nearest')
    f_qo = float(fcut[a:b][int(np.argmin(srf))])
    ridx, _ = find_peaks(-srf)
    f_lo = float(fcut[a:b][ridx[0]]) if len(ridx) else np.nan
    f_hi = float(fcut[a:b][ridx[-1]]) if len(ridx) else np.nan

    lf, le = np.log(fcuts[ok]), err[ok]
    f_opt = float(fcuts[ok][int(np.argmin(le))])
    e_opt = float(np.min(le))

    def score(f):
        if f is None or not np.isfinite(f):
            return np.nan
        return float(np.interp(np.log(f), lf, le))

    draws = np.exp(RNG.uniform(np.log(fcut[a]), np.log(fcut[b - 1]),
                               NULL_DRAWS))
    return {'stem': stem, 'region': (int(a), int(b)),
            'in_surviving': bool(fcut[a] <= f_opt <= fcut[b - 1]),
            'n_reg_minima': int(len(ridx)),
            'f_opt': f_opt, 'e_opt': e_opt,
            'f_qo': f_qo, 'e_qo': score(f_qo),
            'f_lo': f_lo, 'e_lo': score(f_lo),
            'f_hi': f_hi, 'e_hi': score(f_hi),
            'f_pkg': f_pkg, 'e_pkg': score(f_pkg),
            'e_null': float(np.median([score(f) for f in draws]))}


def main():
    stems = population()
    rows = [r for r in (one(s) for s in stems) if r]
    print(f'trusted subset (e_min <= {RMSE_MAX}): {len(stems)} stems')
    print(f'scored (a surviving region exists):   {len(rows)}\n')

    ins = np.array([r['in_surviving'] for r in rows])
    print(f'**THE POPULATION**: optimum inside the FINAL surviving '
          f'region on {int(ins.sum())} of {len(rows)}.')
    print('For comparison the harness\'s `in_candidates`, the weaker')
    print('stage-1 mask, gave 165; the exclusions after it push the')
    print('optimum out of reach on the difference.\n')

    nm = np.array([r['n_reg_minima'] for r in rows])
    print(f'local minima of the sensitivity curve inside the region: '
          f'median {np.median(nm):.0f}, max {nm.max()}\n')

    eo = np.array([r['e_opt'] for r in rows])
    fo = np.array([r['f_opt'] for r in rows])
    keys = (('qo', 'quasi-optimality'), ('lo', '  lowest-f local min'),
            ('hi', '  highest-f local min'),
            ('pkg', 'package select_center'), ('null', 'null in region'))
    for label, sel in (('THE TEST: optimum INSIDE the surviving region',
                        ins),
                       ('optimum OUTSIDE it, stage 3 cannot reach it',
                        ~ins),
                       ('all scored, for reference only',
                        np.ones(len(rows), bool))):
        if not sel.any():
            continue
        print(f'--- {label}, n={int(sel.sum())} ---')
        print(f"{'rule':>24} {'med RMSE':>10} {'x optimum':>10} "
              f"{'p90 x':>8} {'<=2x':>7} {'med |dec|':>10}")
        for k, lab in keys:
            e = np.array([r[f'e_{k}'] for r in rows])
            g = sel & np.isfinite(e)
            if not g.any():
                continue
            ratio = e[g] / eo[g]
            if k == 'null':
                dec = np.nan
            else:
                f = np.array([r[f'f_{k}'] for r in rows])
                gg = g & np.isfinite(f)
                dec = (np.median(np.abs(np.log10(f[gg] / fo[gg])))
                       if gg.any() else np.nan)
            print(f'{lab:>24} {np.median(e[g]):10.4f} '
                  f'{np.median(ratio):10.2f} '
                  f'{np.percentile(ratio, 90):8.2f} '
                  f'{100*np.mean(ratio <= 2):6.0f}% {dec:10.3f}')
        print(f'{"optimum":>24} {np.median(eo[sel]):10.4f} {1.0:10.2f} '
              f'{1.0:8.2f} {100:6.0f}% {0.0:10.3f}\n')

    q = np.array([r['e_qo'] for r in rows])[ins]
    _p = np.array([r['e_pkg'] for r in rows])[ins]
    p = _p
    g = np.isfinite(q) & np.isfinite(p)
    print(f'quasi-optimality beats the package on '
          f'{int((q[g] < p[g]).sum())}/{int(g.sum())} '
          f'({100*np.mean(q[g] < p[g]):.0f}%)')
    np.save(os.path.join(OUT, 'qo2.npy'),
            np.array(rows, dtype=object), allow_pickle=True)
    print(f"\n{os.path.join(OUT, 'qo2.npy')}")


if __name__ == '__main__':
    main()
