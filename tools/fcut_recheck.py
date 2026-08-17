#!/usr/bin/python3
"""
Compare the selected cutoff against a previous run, signal by signal.

Reads the ``Cutoff frequency:`` recorded in a previous sweep's per-signal
logs, re-runs the production selection with the current code, and reports
how far each cutoff moved in decades.

The cache directory is used as given and is expected to be a COPY: on a
miss ``_r2_array_cached`` deletes every entry with the same stem before
writing, so pointing this at a reference cache destroys it.

Usage
-----
python tools/fcut_recheck.py OLD_RUN_DIR DATA_DIR WORK_DIR [--workers 8]
"""

import argparse
import contextlib
import csv
import io
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np

from weaselytics.baseline import _fcutoff, _relevant_regions
from weaselytics.parsers import ParsedData

_CUTOFF = re.compile(r"Cutoff frequency:\s*([0-9.Ee+-]+)")


def old_cutoffs(run_dir: str) -> dict[str, float]:
    """
    Read the selected cutoff of every signal of a previous run.

    Parameters
    ----------
    run_dir : str
        Directory holding ``logs/<stem>.log``.

    Returns
    -------
    cutoffs : dict
        Stem to cutoff frequency.

    """
    out = {}
    for path in glob(os.path.join(run_dir, 'logs', '*.log')):
        with open(path) as f:
            m = _CUTOFF.search(f.read())
        if m:
            out[os.path.splitext(os.path.basename(path))[0]] = float(m[1])
    return out


def one(args: tuple) -> tuple[str, float | None, str]:
    """
    Re-run the selection for a single signal.

    Parameters
    ----------
    args : tuple
        ``(stem, path, cache_dir)``.

    Returns
    -------
    stem : str
        The signal name.
    fcut : float or None
        The newly selected cutoff, None on failure.
    error : str
        Empty on success, else the exception repr.

    """
    stem, path, cache_dir = args
    try:
        x, s = ParsedData(path).data
        peak_regions, sampling, scut = _relevant_regions(s, x)
        kwargs = {'asymmetry': 1.0, 'fit_parabola': True, 'alpha': 1.0,
                  'parabola_len': 3, 'regions': peak_regions,
                  'sampling': sampling}
        with contextlib.redirect_stdout(io.StringIO()):
            fcut, _ = _fcutoff(s, x, scut, method='custom_beads',
                               cache_dir=cache_dir, path=path, workers=1,
                               **kwargs)
        return stem, float(fcut), ''
    except Exception as exc:
        return stem, None, repr(exc)


def main() -> None:
    """
    CLI entry point of the cutoff re-check.

    """
    parser = argparse.ArgumentParser(
        prog='fcut_recheck',
        description='compare selected cutoffs against a previous run')
    parser.add_argument('old_run', help='previous run directory (logs/)')
    parser.add_argument('data', help='raw data directory (read-only)')
    parser.add_argument('work', help='working directory holding a COPY '
                                     'of the r2 cache')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    old = old_cutoffs(args.old_run)
    cache_dir = os.path.join(args.work, 'r2_cache')
    jobs = []
    for path in sorted(glob(os.path.join(args.data, '*', '*.txt'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in old:
            jobs.append((stem, path, cache_dir))
    print(f'{len(jobs)} signals with a recorded cutoff')

    rows, fails = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            stem, new, err = fut.result()
            if new is None:
                fails.append((stem, err))
                print(f'[{k}/{len(jobs)}] {stem}: FAILED {err}')
                continue
            o = old[stem]
            d = float(np.log10(new / o))
            rows.append({'stem': stem, 'old': o, 'new': new,
                         'd_decades': d})
            print(f'[{k}/{len(jobs)}] {stem}: {o:.4e} -> {new:.4e}  '
                  f'{d:+.4f} dec')

    rows.sort(key=lambda r: r['stem'])
    out = os.path.join(args.work, 'fcut_recheck.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stem', 'old', 'new',
                                          'd_decades'])
        w.writeheader()
        w.writerows(rows)

    d = np.array([r['d_decades'] for r in rows])
    print(f'\n{len(rows)} compared, {len(fails)} failed -> {out}')
    print(f'  identical (|d| < 1e-9)     : {int((np.abs(d) < 1e-9).sum())}')
    for thr in (0.01, 0.05, 0.1, 0.3):
        print(f'  |d| > {thr:<4}  decades       : '
              f'{int((np.abs(d) > thr).sum())}')
    print(f'  median {np.median(d):+.4f}   p10 {np.percentile(d, 10):+.4f}'
          f'   p90 {np.percentile(d, 90):+.4f}   max|d| {np.abs(d).max():.4f}')
    if fails:
        print(f'  failures: {[s for s, _ in fails]}')


if __name__ == '__main__':
    main()
