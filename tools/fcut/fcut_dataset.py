#!/usr/bin/python3
"""
Assemble the fcut-selection analysis table for issue #4.

Joins, one row per signal:

* the hand-labeled acceptable ``fcut`` ranges from a gallery directory
  (``index_<signal>.csv`` files written by ``tools/fcut/fcut_gallery.py``,
  with ``start`` / ``end`` marks in the ``label`` column);
* the candidate-region spans sampled by the gallery;
* ``Used points`` (and the legacy ``Cutoff frequency``, kept for
  reference only) parsed from the sweep run log;
* peak statistics computed from the raw chromatogram with the same
  detection recipe as ``weaselytics.baseline._relevant_regions``;
* optionally, the per-(mol, solvent) retention centers from a
  ``stats_v*.txt`` table.

Usage
-----
python tools/fcut/fcut_dataset.py GALLERY_DIR RUN_LOG DATA_DIR -o out.csv
    [--stats stats_v7.txt]
"""

import argparse
import csv
import os
import re
from glob import glob

import numpy as np
from scipy.ndimage import gaussian_filter1d

from weaselytics.parsers import ParsedData
from weaselytics.utils import peaks_params


def parse_signal_name(name: str) -> dict:
    """
    Split a signal name into its naming components.

    Parameters
    ----------
    name : str
        Signal name, e.g. ``2-Xylene__LPYE__60-100__3`` or
        ``3-Dichlorobenzene__LPYE__BLANK__7``.

    Returns
    -------
    parts : dict
        Keys ``solvent``, ``content`` and ``replicate``.

    """
    tokens = name.split('__')
    return {
        'solvent': tokens[0],
        'content': '__'.join(tokens[2:-1]),
        'replicate': int(tokens[-1]),
    }


def parse_labels(index_csv: str) -> tuple[list[tuple[float, float]],
                                          list[tuple[float, float]]]:
    """
    Extract the labeled ranges and candidate spans from a gallery index.

    Parameters
    ----------
    index_csv : str
        Path of an ``index_<signal>.csv`` file with columns
        ``k,region,fcut,r2,pos_in_region,image,label``.

    Returns
    -------
    ranges : list of (float, float)
        The labeled acceptable ``fcut`` ranges, in sweep order.
    regions : list of (float, float)
        The sampled span of each candidate region.

    Raises
    ------
    ValueError
        Raised when the ``start`` / ``end`` marks are not properly
        paired.

    """
    rows = []
    with open(index_csv, newline='') as f:
        for row in csv.DictReader(f):
            rows.append((int(row['k']), int(row['region']),
                         float(row['fcut']), row['label'].strip()))
    rows.sort()

    ranges = []
    lo = None
    for _, _, fcut, label in rows:
        if label == 'start':
            if lo is not None:
                raise ValueError(f'{index_csv}: unclosed start mark')
            lo = fcut
        elif label == 'end':
            if lo is None:
                raise ValueError(f'{index_csv}: end mark without start')
            ranges.append((lo, fcut))
            lo = None
        elif label:
            raise ValueError(f'{index_csv}: unknown label {label!r}')
    if lo is not None:
        raise ValueError(f'{index_csv}: unclosed start mark')

    regions = []
    for region in sorted({r for _, r, _, _ in rows}):
        fcuts = [f for _, r, f, _ in rows if r == region]
        regions.append((min(fcuts), max(fcuts)))
    return ranges, regions


def parse_run_log(log_path: str) -> dict[str, dict]:
    """
    Map each signal to its ``Used points`` and legacy cutoff frequency.

    Parameters
    ----------
    log_path : str
        Path of the sweep run log. Blocks start with the data-file path
        and contain ``Used points:`` and ``Cutoff frequency:`` lines.

    Returns
    -------
    info : dict
        ``{signal: {'n_used': int, 'fcut_legacy': float}}``. The legacy
        cutoff comes from the old derivative method and is kept for
        reference only.

    """
    info = {}
    current = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.endswith('.txt') and (os.sep in line):
                current = os.path.splitext(os.path.basename(line))[0]
                info[current] = {}
            elif line.startswith('Used points:') and current:
                info[current]['n_used'] = int(line.split()[-1])
            elif line.startswith('Cutoff frequency:') and current:
                info[current]['fcut_legacy'] = float(line.split()[-1])
    return info


def peak_stats(data_file: str, tol: float = 6.) -> dict:
    """
    Compute peak statistics of a chromatogram.

    Mirrors the detection recipe of
    ``weaselytics.baseline._relevant_regions``: weak smoothing, then
    ``peaks_params`` and the relevance filter on ``width / x[peak]``.

    Parameters
    ----------
    data_file : str
        Path of the two-column chromatogram file.
    tol : float, optional
        Relevance threshold on ``width / x[peak]``. Default is 6.

    Returns
    -------
    stats : dict
        Peak widths (FWHM, in points), positions (in ``x`` units) and
        signal geometry. Empty dict when no relevant peak is found.

    """
    x, s = ParsedData(data_file).data
    z = gaussian_filter1d(s, 3)
    peaks, widths = peaks_params(z, width=3, rel_prom_p=0.01, adapt=True,
                                 drop_enclosing=True)
    if len(peaks) == 0:
        return {'n_points': len(s), 'dt': float(np.median(np.diff(x)))}
    width_per_x = widths / x[peaks]
    exception = ((s[peaks] > 20) & (width_per_x < 11))
    keep = (width_per_x < tol) | exception
    rel_peaks = peaks[keep]
    rel_widths = widths[keep]
    if len(rel_peaks) == 0:
        return {'n_points': len(s), 'dt': float(np.median(np.diff(x)))}
    return {
        'n_points': len(s),
        'dt': float(np.median(np.diff(x))),
        'n_rel_peaks': int(len(rel_peaks)),
        'w_median': float(np.median(rel_widths)),
        'w_min': float(np.min(rel_widths)),
        'w_max': float(np.max(rel_widths)),
        'x_first_peak': float(x[rel_peaks[0]]),
        'x_last_peak': float(x[rel_peaks[-1]]),
        'h_max': float(np.max(s[rel_peaks])),
    }


def parse_stats_table(stats_path: str) -> dict[tuple[str, str], float]:
    """
    Read the Gaussian retention centers from a ``stats_v*.txt`` table.

    Parameters
    ----------
    stats_path : str
        Path of the whitespace-separated statistics table with columns
        ``mol solvent n_obs distribution x0_wav sigma_wav``.

    Returns
    -------
    x0 : dict
        ``{(mol, solvent): x0_wav}`` for the Gaussian rows.

    """
    x0 = {}
    pattern = re.compile(
        r'^\d+\s+(\S+)\s+(\S+)\s+\d+\s+Gaussian\s+(\S+)\s+\S+$')
    with open(stats_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                x0[(m.group(1), m.group(2))] = float(m.group(3))
    return x0


def main() -> None:
    """
    CLI entry point of the dataset builder.

    """
    parser = argparse.ArgumentParser(
        prog='fcut_dataset',
        description='assemble the fcut-selection analysis table')
    parser.add_argument('gallery', help='labeled gallery directory')
    parser.add_argument('log', help='sweep run log (Used points source)')
    parser.add_argument('data', help='raw chromatogram data directory')
    parser.add_argument('-o', '--output', default='fcut_dataset.csv',
                        help='output CSV path')
    parser.add_argument('--stats', default=None,
                        help='optional stats_v*.txt retention table')
    args = parser.parse_args()

    run_info = parse_run_log(args.log)
    x0 = parse_stats_table(args.stats) if args.stats else {}

    index_files = sorted(glob(os.path.join(
        args.gallery, '*', '*', 'index_*.csv')))
    print(f'{len(index_files)} labeled signals')

    fields = ['signal', 'molecule', 'solvent', 'content', 'replicate',
              'is_blank', 'n_ranges', 'lo1', 'hi1', 'lo2', 'hi2',
              'n_regions', 'cand_lo', 'cand_hi', 'regions',
              'n_used', 'fcut_legacy',
              'n_points', 'dt', 'n_rel_peaks', 'w_median', 'w_min',
              'w_max', 'x_first_peak', 'x_last_peak', 'h_max',
              't0_wav', 'x0_content']
    out_rows = []
    for index_csv in index_files:
        signal = os.path.basename(os.path.dirname(index_csv))
        molecule = os.path.basename(
            os.path.dirname(os.path.dirname(index_csv)))
        parts = parse_signal_name(signal)
        ranges, regions = parse_labels(index_csv)
        if len(ranges) > 2:
            raise ValueError(f'{signal}: {len(ranges)} labeled ranges')

        row = {
            'signal': signal,
            'molecule': molecule,
            'solvent': parts['solvent'],
            'content': parts['content'],
            'replicate': parts['replicate'],
            'is_blank': int(parts['content'] == 'BLANK'),
            'n_ranges': len(ranges),
            'n_regions': len(regions),
            'cand_lo': regions[0][0],
            'cand_hi': regions[-1][1],
            'regions': ';'.join(f'{lo:.6e}:{hi:.6e}'
                                for lo, hi in regions),
        }
        for k, (lo, hi) in enumerate(ranges, start=1):
            row[f'lo{k}'] = lo
            row[f'hi{k}'] = hi
        row.update(run_info.get(signal, {}))
        row.update(peak_stats(
            os.path.join(args.data, molecule, f'{signal}.txt')))
        row['t0_wav'] = x0.get(('t0', parts['solvent']), '')
        row['x0_content'] = x0.get((parts['content'], parts['solvent']),
                                   '')
        out_rows.append(row)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    n_ranges = sum(r['n_ranges'] for r in out_rows)
    print(f'{len(out_rows)} rows, {n_ranges} labeled ranges '
          f'-> {args.output}')


if __name__ == '__main__':
    main()
