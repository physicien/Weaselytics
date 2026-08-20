#!/usr/bin/python3
"""
Extract the baseline the production selection fits to every signal.

Runs `baseline.auto_beads` the way `weaselytics.weaselytics` runs it and
stores the fitted baseline itself, which no existing export gives: the
`-e` flag of the CLI writes ``params["signal"]``, that is
``s - baseline - noise``, so subtracting it from the raw trace returns
the baseline plus the noise rather than the baseline.

Written for comparing the baselines of real chromatograms against the
analytic ones used by the synthetic benchmark. One ``npz`` per signal
holding ``x``, ``y``, ``baseline`` and the selected cutoff, plus a
summary CSV.

The cache directory is used as given and is expected to be a COPY: on a
miss ``_r2_array_cached`` deletes every entry sharing the stem before
writing, so pointing this at a reference cache destroys it.

Usage
-----
python tools/fcut/extract_baselines.py DATA_DIR OUT_DIR [--cache DIR]
                                       [--workers 8] [--pattern GLOB]
"""

import argparse
import contextlib
import csv
import fnmatch
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np

from weaselytics.baseline import (
    _fcutoff,
    _relevant_regions,
    _snr,
    auto_beads,
)
from weaselytics.parsers import ParsedData
from weaselytics.utils import smooth_SG

FIELDS = ['stem', 'molecule', 'n_points', 'fcut', 'scut', 'snr',
          'baseline_range', 'baseline_min', 'baseline_max',
          'signal_range', 'noise_sigma', 'error']


def one(job: tuple) -> dict:
    """
    Fit one signal and return its summary row.

    Mirrors `weaselytics.weaselytics`, which corrects the raw trace
    unless `-sm` is given, so the baseline is the one production would
    return rather than one fitted to a different trace.

    Parameters
    ----------
    job : tuple
        ``(path, out_dir, cache_dir, smooth)``.

    Returns
    -------
    row : dict
        Summary keyed by `FIELDS`; ``error`` is empty on success.

    """
    path, out_dir, cache_dir, smooth = job
    stem = os.path.splitext(os.path.basename(path))[0]
    row = dict.fromkeys(FIELDS, '')
    row['stem'] = stem
    row['molecule'] = os.path.basename(os.path.dirname(path))
    try:
        x, y = ParsedData(path).data
        row['n_points'] = len(y)
        # `weaselytics.py` smooths only when -sm is passed, and it is
        # not passed by `tools/slurm_script`, so the default here is the
        # unsmoothed trace production actually corrects.
        ys = smooth_SG(y, 9, 0) if smooth else y
        with contextlib.redirect_stdout(io.StringIO()):
            baseline, params = auto_beads(
                ys, x, path=path, method='custom_beads',
                cache_dir=cache_dir, workers=1)
            # `auto_beads` returns what the BEADS call returns, which
            # does not carry the cutoff it selected. Re-derive it from
            # the same cached curve: a cache hit, so it costs nothing,
            # and it cannot disagree since the inputs are identical.
            regions, sampling, scut = _relevant_regions(ys, x)
            fcut, _ = _fcutoff(ys, x, scut, method='custom_beads',
                               cache_dir=cache_dir, path=path, workers=1,
                               asymmetry=1.0, fit_parabola=True,
                               alpha=1.0, parabola_len=3,
                               regions=regions, sampling=sampling)
        noise = params.get('noise')
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f'{stem}.npz'),
                 x=x, y=y, y_smoothed=ys, baseline=baseline,
                 signal=params['signal'], fcut=float(fcut), scut=scut)
        row['fcut'] = f'{float(fcut):.6e}'
        row['scut'] = scut
        row['snr'] = f'{_snr(y):.3f}'
        row['baseline_range'] = f'{baseline.max() - baseline.min():.6f}'
        row['baseline_min'] = f'{baseline.min():.6f}'
        row['baseline_max'] = f'{baseline.max():.6f}'
        row['signal_range'] = f'{y.max() - y.min():.6f}'
        row['noise_sigma'] = ('' if noise is None
                              else f'{float(np.std(noise)):.6f}')
    except Exception as exc:
        row['error'] = repr(exc)
    return row


def main() -> None:
    """
    CLI entry point of the baseline extraction.

    """
    p = argparse.ArgumentParser(
        prog='extract_baselines',
        description='store the baseline production fits to each signal')
    p.add_argument('data', help='directory holding Molecule/stem.txt')
    p.add_argument('out', help='output directory')
    p.add_argument('--cache', default=None,
                   help='r2 cache directory, a COPY (default: none)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--pattern', default='*', help='glob on the stem')
    p.add_argument('--smooth', action='store_true',
                   help='Savitzky-Golay the signal first, as the CLI does '
                        'with -sm (default: off, as production runs)')
    a = p.parse_args()

    bl_dir = os.path.join(a.out, 'baselines')
    jobs = []
    for q in sorted(glob(os.path.join(a.data, '*', '*.txt'))):
        stem = os.path.splitext(os.path.basename(q))[0]
        if fnmatch.fnmatch(stem, a.pattern):
            jobs.append((q, bl_dir, a.cache, a.smooth))
    print(f'{len(jobs)} signals', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            print(f"[{k}/{len(jobs)}] {row['stem']}: "
                  + (f"FAILED {row['error']}" if row['error']
                     else f"fcut={row['fcut']} "
                          f"range={row['baseline_range']} mV"), flush=True)

    rows.sort(key=lambda r: r['stem'])
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, 'baselines_summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r['error']]
    print(f'\n{len(rows) - len(bad)} extracted, {len(bad)} failed -> {out}')
    for r in bad:
        print(f"  {r['stem']}: {r['error']}")


if __name__ == '__main__':
    main()
