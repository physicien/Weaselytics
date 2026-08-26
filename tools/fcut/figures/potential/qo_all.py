"""
Every generated signal, drawn once and filed under both classifications.

Renders the production merged figure for all 432 signals of
SYNTH_ERB_2026-08-18 against runs/SYNTH_2026-08-24, then files each
image twice:

  by_rmse/    Emmanuel's edges on `e_min`, the RMSE between the baseline
              at the optimal cutoff and the true baseline:
              <= 0.05 | 0.05 to 0.1 | 0.1 to 0.2 | the rest.
              The 13 signals whose fit failed have no error curve and so
              no RMSE to classify by; they go to `no_error_curve/`.

  by_reach/   whether the OPTIMAL fcut falls inside the FINAL
              surviving region, and where it went if not. Only there can
              a stage-3 rule reach the optimum at all.

The image is rendered once and copied, not drawn twice.

Three cutoffs on one figure: the package's `select_center`, the
quasi-optimal argmin of the sensitivity curve inside the surviving
region (Bauer & Kindermann 2008 Def. 1.1), and the OPTIMAL one, the
cutoff minimising the RMSE against the true baseline. On the failed
signals only the first two exist.

"optimal" is not a ground-truth parameter: the manifest records no
generating cutoff, so it is the best achievable choice, not a value the
generator used. Only `TRUE baseline` and `TRUE correction` on the right
panels are ground truth.
"""

import csv  # noqa: I001
import glob
import os
import shutil
import subprocess
import sys

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, '/home/esteban/Simulation/DFT/separation_part2/'
                   'Weaselytics')
import weaselytics.segmentation as S  # noqa: E402,I001
from pybaselines import Baseline  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from weaselytics.baseline import (  # noqa: E402
    _custom_beads, _log_transform, _relevant_regions, _snr)
from weaselytics.parsers import ParsedData  # noqa: E402
from weaselytics.plot import plot, r2_plots  # noqa: E402

ROOT = '/home/esteban/Simulation/DFT/separation_part2'
RUN = os.path.join(ROOT, 'runs/SYNTH_2026-08-24')
DS = os.path.join(ROOT, 'SYNTH_ERB_2026-08-18')
OUT = os.path.join(ROOT, 'MERGED/QUASIOPT_2026-08-24')
SCRATCH = os.path.dirname(os.path.abspath(__file__))
CQO, COPT = '#1f7a1f', '#7a3fa0'

EDGES = ((0.05, 'rmse_le_0.05'), (0.10, 'rmse_0.05_to_0.1'),
         (0.20, 'rmse_0.1_to_0.2'), (np.inf, 'rmse_gt_0.2'))


def despike(s):
    """
    Remove isolated single-sample outliers before taking an argmin.

    A 3-point running median. Tukey, Exploratory Data Analysis, 1977,
    p. 212, states the property for medians of 3: after the final
    smoothing the sequence is built from "steady upward, flat tops (of
    length at least 2), steady downward and flat bottoms (of length at
    least 2)", and those parts "remain untouched by further median
    smoothing by 3's". Gallagher & Wise 1981, IEEE Trans. ASSP 29(6),
    1136-1141, prove it. Their window is ``2N+1``; Definition 3 calls
    an impulse a run of "at least one, but no more than N points"
    between two constant neighbourhoods, and Theorem 1 requires every
    ``N+2`` consecutive points to be monotone for invariance. At
    ``N = 1`` that is a run of one point and monotonicity over three,
    so a width-3 median alters isolated samples and nothing wider.

    ``mode='nearest'`` is Tukey's "copying on" of the end values,
    p. 221 section 7D. Gallagher & Wise Section II append N samples
    equal in value to the first and last for the same purpose, and note
    the appended value is unimportant provided it is constant.

    Endpoints. The appending leaves the first and last samples of the
    interval unchanged, and the argmin lands on one of them on 85 of
    418 signals. On 82 of those the sensitivity curve falls to the
    boundary and the minimum is genuinely there; on 3 the endpoint is
    anomalous against its own interior. Tukey p. 221 gives end-value
    smoothing for this, taking the median of the end input value, the
    next-to-end smoothed value, and a straight-line extrapolation one
    step beyond. It is left out here because it extrapolates a sequence
    past its edge, while this interval ends where stage 2 cut it.
    Tukey's own condition for copying on being sufficient is that an
    end value "may have a special message", which a trimming boundary
    carries.

    Two further limits. A feature that is an "oscillation" in their
    Definition 4, neither constant neighbourhood, edge, nor impulse,
    survives one pass; Theorem 2 reaches a root only after repeated
    passes. And a minimum one sample wide is flattened like a spike,
    since their Corollary to Theorem 1 needs increase and decrease
    separated by at least ``N+1`` identical points. At 213 grid points
    per decade against features spanning a decade, no minimum here is
    one sample wide; that step is ours rather than theirs.
    """
    return median_filter(np.asarray(s, dtype=float), size=3,
                         mode='nearest')


def bucket(e_min):
    if e_min is None or not np.isfinite(e_min):
        return 'no_error_curve'
    for hi, name in EDGES:
        if e_min <= hi:
            return name
    return EDGES[-1][1]


class KeepOpen:
    """Suspend plt.close so a finished production figure can be added to."""

    def __enter__(self):
        self._c = plt.close
        plt.close = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        plt.close = self._c


def fit_at(x, y, f):
    """
    The baseline auto_beads returns at this cutoff: the RAW record over
    its full length (baseline.py:1259). The log transform belongs to
    cutoff SELECTION only; fitting it here and inverting with Eq. (11)
    gives a baseline that climbs into the peaks.
    """
    regions, sampling, _ = _relevant_regions(y, x)
    bl, _ = _custom_beads(Baseline(x_data=x), y, regions=regions,
                          sampling=sampling, freq_cutoff=f, asymmetry=1.0,
                          fit_parabola=True, alpha=1.0, parabola_len=3)
    return bl


def analyse(stem):
    """Cutoffs, region and classification for one signal."""
    cache = glob.glob(os.path.join(RUN, 'r2_cache', f'{stem}__r2__*.npz'))
    sig = os.path.join(DS, 'signals', f'{stem}.txt')
    if not (cache and os.path.exists(sig)):
        return None
    c = np.load(cache[0])
    fcut, r2, sens = c['fcut_range'], c['r2_val'], c['sensitivity']
    x, y = ParsedData(sig).data
    # The baseline the generator used, which `err` is measured against.
    # It was missing from the figures, so the ranking could not be
    # checked by eye against the thing it was scored on.
    b_true = np.load(os.path.join(DS, 'truth',
                                  f'{stem}__truth.npz'))['baseline']

    dg = os.path.join(RUN, 'diag', f'{stem}__diag.npz')
    if os.path.exists(dg):
        d = np.load(dg)
        n_used = int(d['n_used'])
        fcuts, err = d['fcuts'], d['err']
        ok = np.isfinite(err)
    else:
        # The fit failed, so there is no error curve. n_used is still
        # recoverable: it is the length of the log-transformed record.
        _, _, scut = _relevant_regions(y, x)
        n_used = len(_log_transform(y[:scut]))
        ok = np.zeros(0, dtype=bool)
        fcuts = err = np.zeros(0)

    segs = S.classify_segments(S.segment_features(fcut, r2,
                                                  S.pelt_linear(r2)))
    dips = S.detect_dips(fcut, r2)
    t = dict(S.trim_plateaus(fcut, segs, dips, n_used,
                             exclude_past_drop=bool(_snr(y) >= 10.0),
                             sensitivity=sens))
    flat = np.zeros(len(fcut), dtype=bool)
    for seg in segs:
        if seg['flat']:
            flat[seg['start']:seg['end']] = True
    t['flat'] = flat
    t['dips'] = S.dips_to_mask(fcut, dips)

    v = S.select_center(fcut, t['surviving'])
    f_pkg = float(v) if v else np.nan

    mask = np.asarray(t['surviving'], dtype=bool)
    f_qo, span = np.nan, None
    if mask.any():
        edges = np.flatnonzero(np.diff(mask.astype(int)))
        bounds = np.concatenate(([0], edges + 1, [len(mask)]))
        runs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])
                if mask[a] and b - a >= 3]
        if runs:
            a, b = runs[-1]
            sr = sens[a:b]
            if np.all(np.isfinite(sr)) and np.ptp(sr) > 0:
                span = (a, b)
                f_qo = float(fcut[a:b][int(np.argmin(despike(sr)))])

    if ok.any():
        f_opt = float(fcuts[ok][int(np.argmin(err[ok]))])
        e_min = float(np.min(err[ok]))
        reach = ('in_surviving' if span and fcut[span[0]] <= f_opt
                 <= fcut[span[1] - 1] else None)
        if reach is None:
            k = int(np.argmin(np.abs(fcut - f_opt)))
            reach = 'undetected'
            for name, tag in (('instab_removed', 'instab'),
                              ('snr_removed', 'removed'),
                              ('removed', 'removed')):
                if np.asarray(t[name], dtype=bool)[k]:
                    reach = tag
                    break
    else:
        f_opt, e_min, reach = np.nan, None, 'no_error_curve'

    return dict(stem=stem, x=x, y=y, b_true=b_true, fcut=fcut, r2=r2, sens=sens,
                n_used=n_used, t=t, dipc=S.dip_curve(r2),
                f_pkg=f_pkg, f_qo=f_qo, f_opt=f_opt, e_min=e_min,
                bucket=bucket(e_min), reach=reach)


def render(a, work):
    os.makedirs(work, exist_ok=True)
    stem = a['stem']
    path = os.path.join(work, f'{stem}.txt')
    fcut, r2, sens = a['fcut'], a['r2'], a['sens']
    k = int(np.argmin(np.abs(fcut - a['f_pkg']))) if np.isfinite(
        a['f_pkg']) else 0

    with KeepOpen():
        r2_plots(fcut, r2, a['dipc'], float(fcut[k]), float(r2[k]),
                 cp_flat=a['t']['flat'], cp_dips=a['t']['dips'],
                 cp_removed=a['t']['removed'],
                 cp_snr_removed=a['t']['snr_removed'],
                 cp_instab_removed=a['t']['instab_removed'],
                 sensitivity=sens, n_used=a['n_used'],
                 print_plot=False, path=path, output_dir=work)
        fig = plt.gcf()
        for ax in fig.axes:
            for f, col in ((a['f_qo'], CQO), (a['f_opt'], COPT)):
                if np.isfinite(f):
                    ax.axvline(f, color=col, lw=1.8, ls='--', zorder=6)
        h, lab = [], []
        for f, col, name in ((a['f_pkg'], 'tab:red', 'package'),
                             (a['f_qo'], CQO, 'quasi-optimal'),
                             (a['f_opt'], COPT, 'optimal')):
            if np.isfinite(f):
                h.append(plt.Line2D([], [], color=col, ls='--', lw=1.8))
                lab.append('%s %.4g' % (name, f))
        if not np.isfinite(a['f_opt']):
            h.append(plt.Line2D([], [], color='none'))
            lab.append('optimal: fit failed, no error curve')
        fig.axes[1].legend(h, lab, fontsize=7, loc='upper right',
                           framealpha=.9)
        p_r2 = os.path.join(work, f'{stem}_r2.png')
        fig.savefig(p_r2)
    plt.close(fig)

    # The 13 signals whose fit failed have no surviving region either,
    # so there is no cutoff of any kind and nothing to fit. Draw the raw
    # trace alone rather than handing BEADS a NaN, which it rejects with
    # "cutoff frequency must be between 0 and 0.5".
    x, y = a['x'], a['y']
    b_true = a['b_true']
    fits = []
    for f, col, name in ((a['f_pkg'], 'tab:red', 'package'),
                         (a['f_qo'], CQO, 'quasi-optimal'),
                         (a['f_opt'], COPT, 'optimal')):
        if np.isfinite(f):
            fits.append((fit_at(x, y, f), col, name, f))

    # Two panels on the right, not one: the baselines against the true
    # baseline, and the corrections against the true correction. They
    # live on different scales and overlaying them hides the residual.
    panels = []
    for kind in ('baseline', 'corrected'):
        with KeepOpen():
            plot(x, y, print_plot=False, path=path, output_dir=work)
            fig = plt.gcf()
            ax = fig.axes[0]
            if kind == 'baseline':
                ax.plot(x, b_true, lw=2.4, color='k', zorder=2,
                        label='TRUE baseline')
                for b, col, name, f in fits:
                    ax.plot(x, b, lw=1.3, color=col, ls='--', zorder=3,
                            label='%s, %.4g' % (name, f))
                ax.set_title('baselines against the true one',
                             fontsize=9, loc='left')
            else:
                # `plot` always draws the raw trace, and here its full
                # excursion sets the y-scale and flattens the very
                # curves this panel exists to show. Drop it and let the
                # axes rescale to the corrections.
                for ln in list(ax.lines):
                    if ln.get_label() == 'raw data':
                        ln.remove()
                ax.plot(x, y - b_true, lw=2.0, color='k', zorder=2,
                        label='TRUE correction')
                for b, col, name, f in fits:
                    ax.plot(x, y - b, lw=1.1, color=col, zorder=3,
                            alpha=.85, label='%s, %.4g' % (name, f))
                ax.relim()
                ax.autoscale_view()
                ax.set_title('corrected, y minus baseline',
                             fontsize=9, loc='left')
            ax.legend(fontsize=7, loc='best', framealpha=.9)
            p = os.path.join(work, f'{stem}_{kind}.png')
            fig.savefig(p)
            panels.append(p)
        plt.close(fig)

    right = os.path.join(work, f'{stem}_right.png')
    subprocess.run(['convert'] + panels + ['-append', right], check=True)
    out = os.path.join(work, f'{stem}.png')
    subprocess.run(['convert', p_r2, right, '+append', out], check=True)
    return out


def main():
    stems = sorted(os.path.basename(f)[:-4] for f in
                   glob.glob(os.path.join(DS, 'signals', '*.txt')))
    print(f'{len(stems)} generated signals')
    for d in ('by_rmse', 'by_reach'):
        shutil.rmtree(os.path.join(OUT, d), ignore_errors=True)
    work = os.path.join(SCRATCH, '.qo_all')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    rows, tally, failed = [], {}, []
    for i, stem in enumerate(stems, 1):
        # One bad signal must not cost the other four hundred renders.
        try:
            a = analyse(stem)
            if a is None:
                failed.append((stem, 'no r2 cache'))
                continue
            img = render(a, work)
        except Exception as exc:
            failed.append((stem, repr(exc)))
            plt.close('all')
            continue
        for group, name in (('by_rmse', a['bucket']),
                            ('by_reach', a['reach'])):
            d = os.path.join(OUT, group, name)
            os.makedirs(d, exist_ok=True)
            shutil.copy2(img, os.path.join(d, f'{stem}.png'))
            tally[f'{group}/{name}'] = tally.get(f'{group}/{name}', 0) + 1
        rows.append({'stem': stem,
                     'e_min': '' if a['e_min'] is None else
                              f"{a['e_min']:.6f}",
                     'bucket': a['bucket'], 'reach': a['reach'],
                     'f_pkg': a['f_pkg'], 'f_qo': a['f_qo'],
                     'f_opt': a['f_opt']})
        os.remove(img)
        if i % 50 == 0:
            print(f'  {i}/{len(stems)}', flush=True)

    with open(os.path.join(OUT, 'classification.csv'), 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(work, ignore_errors=True)
    print('\n'.join(f'  {k:<34} {v:4d}' for k, v in sorted(tally.items())))
    print(f'\n  rendered {len(rows)} of {len(stems)}')
    if failed:
        print(f'  FAILED {len(failed)}:')
        for stem, why in failed:
            print(f'    {stem}: {why}')
    print(f"\n{OUT}")


if __name__ == '__main__':
    main()
