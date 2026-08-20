#!/usr/bin/python3
"""
Render the per-signal review figures of a synthetic dataset.

Two figures per signal, for every signal rather than only the ones that
select a cutoff:

``r2_plots/<stem>_r2.png``
    The production autocorrelation figure of ``plot.r2_plots``, with the
    **true optimum** added as a green dashed line across every panel.
    Production has no slot for that marker and should not have one: the
    true optimum exists only where the baseline is known.

``images/<stem>.png``
    The chromatogram with the known baseline, the baseline the pipeline
    selected, and the baseline at the true optimum, over the
    corresponding corrected traces. When the selection raises, the
    selected baseline is absent and the figure says so; the true
    optimum is still drawn, so a failure can be read against what the
    method could have achieved.

Both are rebuilt from ``diag/<stem>__diag.npz`` and the ``r2_cache`` of
the same directory, so neither the autocorrelation sweep nor the error
curve is recomputed. Only the two final BEADS corrections are run per
signal, which are milliseconds. Signals with no diagnostic file (those
whose selection raised) have their curve read from the cache and their
error curve recomputed, which is the expensive path.

The cache directory is used as given and is expected to be the dataset's
own: on a miss ``_r2_array_cached`` deletes every entry sharing the stem
before writing.

Usage
-----
python tools/synthetic/synth_figs.py DATASET_DIR [--workers 8] [--pattern GLOB]
"""

import argparse
import contextlib
import fnmatch
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pybaselines import Baseline  # noqa: E402
from synth_diag import error_curve  # noqa: E402

from weaselytics.baseline import (  # noqa: E402
    _custom_beads,
    _log_transform,
    _r2_array_cached,
    _relevant_regions,
    _snr,
)
from weaselytics.parsers import ParsedData  # noqa: E402
from weaselytics.plot import r2_plots  # noqa: E402
from weaselytics.segmentation import (  # noqa: E402
    classify_segments,
    detect_dips,
    dip_curve,
    dips_to_mask,
    pelt_linear,
    segment_features,
    select_center,
    trim_plateaus,
)
from weaselytics.utils import end_window  # noqa: E402

SNR_THRESHOLD = 25.0
# Stride of the error-curve grid, matching `synth_diag` so a recomputed
# optimum is the one that function would have recorded.
STRIDE = 4
_TRUE = 'tab:green'
_SEL = 'tab:red'


def _beads_at(x, s, fcut, regions, sampling, parabola_len):
    """Fit one baseline at a given cutoff, or None when it fails."""
    if not np.isfinite(fcut):
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bl, _ = _custom_beads(
                Baseline(x_data=x), s, freq_cutoff=float(fcut),
                regions=(((None, None),) if regions is None else regions),
                sampling=(1 if regions is None else sampling),
                asymmetry=1.0, fit_parabola=True, alpha=1.0,
                parabola_len=parabola_len)
        return bl
    except Exception:
        return None


def _curve(stem, path, diag_dir, cache_dir, x, s, b_true, regions,
           sampling, scut):
    """
    Autocorrelation curve, sensitivity, grid and true optimum.

    Read from the stored diagnostic when there is one; otherwise
    recomputed, which is the path taken by the signals whose selection
    raised.
    """
    npz = os.path.join(diag_dir, f'{stem}__diag.npz')
    if os.path.isfile(npz):
        d = np.load(npz)
        if {'fcuts', 'err'} <= set(d.files):
            err = d['err']
            fcuts = d['fcuts']
            fc_best = (float(fcuts[np.nanargmin(err)])
                       if np.isfinite(err).any() else np.nan)
        else:
            fc_best = np.nan
    else:
        grid = np.geomspace(0.00001, 0.5, num=1000, endpoint=False)[::STRIDE]
        with contextlib.redirect_stdout(io.StringIO()):
            err = error_curve(x, s, b_true, grid)
        fc_best = (float(grid[np.nanargmin(err)])
                   if np.isfinite(err).any() else np.nan)

    fcut_range = np.geomspace(0.00001, 0.5, num=1000, endpoint=False)
    z = _log_transform(s[:scut])
    kwargs = {'asymmetry': 1.0, 'fit_parabola': True, 'alpha': 1.0,
              'parabola_len': 3, 'regions': regions, 'sampling': sampling}
    with contextlib.redirect_stdout(io.StringIO()):
        r2_val, sens = _r2_array_cached(
            _custom_beads, Baseline(x_data=x[:scut]), z, fcut_range,
            param='freq_cutoff', cache_dir=cache_dir, path=path, workers=1,
            return_sensitivity=True, **kwargs)
    return fcut_range, r2_val, sens, len(z), fc_best


def _chromatogram(x, s, b_true, bl_sel, bl_best, fcut, fc_best, scut,
                  stem, out_dir, stats, sigma):
    """Signal with the known, selected and best-achievable baselines."""
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(x, s, lw=.7, color='0.35', label='signal')
    ax[0].plot(x, b_true, lw=1.8, color='tab:blue', label='true baseline')
    if bl_best is not None:
        ax[0].plot(x, bl_best, lw=1.3, color=_TRUE, ls='--',
                   label=f'at the true optimum {fc_best:.3e}')
    if bl_sel is not None:
        ax[0].plot(x, bl_sel, lw=1.3, color=_SEL, ls='-.',
                   label=f'as selected {fcut:.3e}')
    ax[0].axvline(x[min(scut, len(x) - 1)], color='k', ls='dotted', lw=1.1,
                  label=f'scut ({scut})')
    ax[0].set_ylabel('mV')
    ax[0].legend(fontsize=8)
    # Accuracy in units of the record's own noise, which is the scale
      # on which a difference here is visible at all. The decade
      # distance is kept only as a secondary label: it does not
      # distinguish a harmless move from a damaging one.
    # Both reductions, because neither alone describes the curve: the
    # rms averages the departure over the whole record and so dilutes a
    # local defect by the record length, while the max is what the eye
    # reads off the plot.
    def _pair(rms, mx):
        if not (np.isfinite(rms) and np.isfinite(mx)):
            return 'n/a'
        return f'rms {rms:.1f}, max {mx:.1f} sigma'

    def _stats(bl):
        """Departure of a fitted curve from the truth, in noise units."""
        if bl is None or not np.isfinite(sigma) or sigma <= 0:
            return float('nan'), float('nan')
        d = bl - b_true
        return (float(np.sqrt(np.mean(d ** 2))) / sigma,
                float(np.abs(d).max()) / sigma)

    # A signal that selects nothing has no row in the summary, but its
    # optimum does not depend on the selection: fill green from the
    # curve already fitted here rather than leaving it blank.
    if not np.isfinite(stats['t_rms']):
        stats = dict(stats)
        stats['t_rms'], stats['t_max'] = _stats(bl_best)

    def _line(tag, pre, drawn):
        if drawn is None:
            return f'{tag}: nothing selected'
        rms, mx = stats[pre + '_rms'], stats[pre + '_max']
        out = f'{tag} vs true: {_pair(rms, mx)}'
        rmse, snr, area = (stats[pre + '_rmse'], stats[pre + '_snr'],
                           stats[pre + '_area'])
        if np.isfinite(rmse):
            out += f' | rmse {rmse:.4f} mV'
        if np.isfinite(snr):
            out += f' | SNR {snr:.1f} dB'
        if np.isfinite(area):
            out += f' | area {area:+.2f}%'
        return out

    ax[0].set_title(f'{stem}\n{_line("green", "t", bl_best)}\n'
                    f'{_line("red", "s", bl_sel)}', fontsize=8.5)

    ax[1].plot(x, s - b_true, lw=.7, color='tab:blue',
               label='corrected with the true baseline')
    if bl_best is not None:
        ax[1].plot(x, s - bl_best, lw=.7, color=_TRUE,
                   label='corrected at the true optimum')
    if bl_sel is not None:
        ax[1].plot(x, s - bl_sel, lw=.7, color=_SEL,
                   label='corrected as selected')
    ax[1].axhline(0, color='k', lw=.6)
    ax[1].set_xlabel('time (min)')
    ax[1].set_ylabel('corrected (mV)')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    outdir = os.path.join(out_dir, 'images')
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{stem}.png'), dpi=115)
    plt.close(fig)


def one(job):
    """Render both figures for a single signal."""
    stem, path, dataset, stats = job
    try:
        x, s = ParsedData(path).data
        b_true = np.load(os.path.join(dataset, 'truth',
                                      f'{stem}__truth.npz'))['baseline']
        regions, sampling, scut = _relevant_regions(s, x)
        fcut_range, r2_val, sens, n_used, fc_best = _curve(
            stem, path, os.path.join(dataset, 'diag'),
            os.path.join(dataset, 'r2_cache'), x, s, b_true, regions,
            sampling, scut)

        segments = classify_segments(
            segment_features(fcut_range, r2_val, pelt_linear(r2_val)))
        flat = np.zeros(len(fcut_range), dtype=bool)
        for seg in segments:
            if seg['flat']:
                flat[seg['start']:seg['end']] = True
        dips = detect_dips(fcut_range, r2_val)
        trim = trim_plateaus(fcut_range, segments, dips, n_used,
                             exclude_collapse=_snr(s) >= SNR_THRESHOLD,
                             sensitivity=sens)
        fcut = select_center(fcut_range, trim['surviving'])
        selected = np.nan if fcut is None else float(fcut)
        r2_at = (np.nan if fcut is None
                 else float(r2_val[int(np.argmin(
                     np.abs(fcut_range - selected)))]))

        real_savefig = plt.savefig

        def savefig(*a, **k):
            if np.isfinite(fc_best):
                for axis in plt.gcf().axes:
                    axis.axvline(x=fc_best, c=_TRUE, ls=(0, (6, 3)), lw=1.6,
                                 zorder=5)
                plt.gcf().axes[0].annotate(
                    f'true optimum {fc_best:.3e}', xy=(fc_best, 0.06),
                    xycoords=('data', 'axes fraction'), color=_TRUE,
                    fontsize=8, rotation=90, ha='right', va='bottom')
            return real_savefig(*a, **k)

        plt.savefig = savefig
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r2_plots(fcut_range, r2_val, dip_curve(r2_val), selected,
                         r2_at, cp_flat=flat,
                         cp_dips=dips_to_mask(fcut_range, dips),
                         cp_removed=trim['removed'],
                         cp_snr_removed=trim['snr_removed'],
                         cp_instab_removed=trim['instab_removed'],
                         sensitivity=sens, n_used=n_used,
                         print_plot=True, path=path, output_dir=dataset)
        finally:
            plt.savefig = real_savefig

        plen = end_window(s)
        # Both curves are fitted with the SAME configuration used for
        # the numbers in the title, so the plot cannot disagree with
        # its own labels.
        _chromatogram(x, s, b_true,
                      _beads_at(x, s, selected, regions, sampling, plen),
                      _beads_at(x, s, fc_best, regions, sampling, plen),
                      selected, fc_best, scut, stem, dataset, stats,
                      float(np.std(np.load(os.path.join(
                          dataset, 'truth',
                          f'{stem}__truth.npz'))['noise'])))
        return stem, (fcut is None), ''
    except Exception as exc:
        return stem, None, repr(exc)


def main() -> None:
    """CLI entry point of the figure rendering."""
    p = argparse.ArgumentParser(
        prog='synth_figs',
        description='per-signal review figures of a synthetic dataset')
    p.add_argument('dataset')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--pattern', default='*', help='glob on the stem')
    a = p.parse_args()

    # The figures quote the summary's own numbers rather than
    # recomputing them, so a plot can never disagree with the table.
    summary = {}
    spath = os.path.join(a.dataset, 'diag_summary.csv')
    if os.path.isfile(spath):
        import csv
        summary = {r['stem']: r for r in csv.DictReader(open(spath))}

    def _num(stem, key):
        r = summary.get(stem)
        try:
            return float(r[key])
        except (TypeError, KeyError, ValueError):
            return float('nan')

    jobs = []
    for q in sorted(glob(os.path.join(a.dataset, 'signals', '*.txt'))):
        stem = os.path.splitext(os.path.basename(q))[0]
        if fnmatch.fnmatch(stem, a.pattern):
            jobs.append((stem, q, a.dataset, {
                't_rms': _num(stem, 'target_rms_noise'),
                't_max': _num(stem, 'target_max_noise'),
                't_rmse': _num(stem, 'target_rmse'),
                't_snr': _num(stem, 'target_snr_db'),
                't_area': _num(stem, 'target_area_pct'),
                's_rms': _num(stem, 'selected_rms_noise'),
                's_max': _num(stem, 'selected_max_noise'),
                's_rmse': _num(stem, 'selected_rmse'),
                's_snr': _num(stem, 'selected_snr_db'),
                's_area': _num(stem, 'selected_area_pct')}))
    print(f'{len(jobs)} signals')

    crashed, errs = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            stem, no_plateau, err = f.result()
            if err:
                errs.append((stem, err))
                print(f'[{k}/{len(jobs)}] {stem}: ERROR {err}', flush=True)
                continue
            if no_plateau:
                crashed.append(stem)
            print(f'[{k}/{len(jobs)}] {stem}'
                  f'{"   no plateau" if no_plateau else ""}', flush=True)

    print(f'\n{len(jobs) - len(errs)} rendered, {len(crashed)} of them with '
          f'no surviving plateau, {len(errs)} errored')
    for stem, err in errs:
        print(f'  {stem}: {err}')
    print(f'figures -> {a.dataset}/r2_plots and {a.dataset}/images')


if __name__ == '__main__':
    main()
