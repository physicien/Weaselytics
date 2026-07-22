#!/usr/bin/python3
"""
Calibrate and validate the bracket of ``refine_candidates``.

Reproduces the constants hard-coded as the defaults of
``weaselytics.segmentation.refine_candidates`` from the hand-labeled
gallery, and measures what the bracket costs and buys:

* the geometry the constants come from — the width of the narrowest
  region a label ever touches (``min_width``), the log-relative
  position at which labels enter the first kept region (``left_cut``)
  and the position at which they leave the second one
  (``right_cut``);
* recall (the labeled span kept, over the labeled span the untrimmed
  candidates contained) and the resulting candidate span;
* a leave-one-solvent-family-out pass, where the constants are
  recalibrated on 13 families and evaluated on the held-out one, so
  the numbers are not self-reported.

Both labeled ranges of the double-range signals count: this is a
bracketing question, so every labeled point is worth keeping.

Usage
-----
python tools/fcut_bracket_calib.py fcut_dataset.csv
"""

import argparse

import numpy as np
import pandas as pd

from weaselytics.segmentation import refine_candidates

DEC = np.log(10)


def parse_regions(spec: str) -> list[tuple[float, float]]:
    """
    Decode the candidate-region column of the dataset table.

    Parameters
    ----------
    spec : str
        Regions serialized as ``lo:hi;lo:hi`` in ``fcut`` units.

    Returns
    -------
    regions : list of (float, float)
        The candidate-region spans, in natural log of ``fcut``.

    """
    out = []
    for token in spec.split(';'):
        lo, hi = token.split(':')
        out.append((np.log(float(lo)), np.log(float(hi))))
    return out


def labeled(row: pd.Series) -> list[tuple[float, float]]:
    """
    Return the labeled acceptable ranges of a signal.

    Parameters
    ----------
    row : pandas.Series
        A row of the dataset table.

    Returns
    -------
    ranges : list of (float, float)
        One or two ranges, in natural log of ``fcut``.

    """
    out = [(np.log(row['lo1']), np.log(row['hi1']))]
    if row['n_ranges'] == 2:
        out.append((np.log(row['lo2']), np.log(row['hi2'])))
    return out


def overlap(a: float, b: float, c: float, d: float) -> float:
    """
    Length of the intersection of two intervals.

    Parameters
    ----------
    a, b : float
        Bounds of the first interval.
    c, d : float
        Bounds of the second interval.

    Returns
    -------
    length : float
        The overlap, zero when the intervals are disjoint.

    """
    return max(0., min(b, d) - max(a, c))


def kept_regions(regions: list[tuple[float, float]],
                 min_width: float) -> list[tuple[float, float]]:
    """
    Drop the sliver regions, keeping the widest one as a fallback.

    Parameters
    ----------
    regions : list of (float, float)
        Candidate regions in log ``fcut``.
    min_width : float
        Minimum width in decades.

    Returns
    -------
    kept : list of (float, float)
        The regions at least `min_width` decades wide.

    """
    kept = [r for r in regions if (r[1] - r[0]) >= min_width * DEC]
    if not kept:
        kept = [max(regions, key=lambda r: r[1] - r[0])]
    return kept


def geometry(rows: list[dict], min_width: float) -> dict:
    """
    Measure where the labels sit relative to the candidate regions.

    Parameters
    ----------
    rows : list of dict
        Per-signal ``regions`` and ``labels`` in log ``fcut``.
    min_width : float
        Sliver threshold used to define the kept regions.

    Returns
    -------
    stats : dict
        Widths of the label-touched and untouched regions, and the
        entry/exit positions in the first and second kept regions.

    """
    touched, untouched, entry, exit_ = [], [], [], []
    for s in rows:
        for lo, hi in s['regions']:
            wdec = (hi - lo) / DEC
            if any(overlap(lo, hi, a, b) > 1e-9 for a, b in s['labels']):
                touched.append(wdec)
            else:
                untouched.append(wdec)
        kept = kept_regions(s['regions'], min_width)
        for a, b in s['labels']:
            for k, (lo, hi) in enumerate(kept):
                if overlap(lo, hi, a, b) <= 1e-9:
                    continue
                if k == 0:
                    entry.append(max(0., (a - lo) / (hi - lo)))
                elif k == 1:
                    exit_.append(min(1., (b - lo) / (hi - lo)))
    return {'touched': touched, 'untouched': untouched,
            'entry': entry, 'exit': exit_}


def evaluate(rows: list[dict], min_width: float, left_cut: float,
             right_cut: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Score a bracket: labeled span kept, and resulting candidate span.

    Parameters
    ----------
    rows : list of dict
        Per-signal ``regions`` and ``labels`` in log ``fcut``.
    min_width, left_cut, right_cut : float
        The bracket constants.

    Returns
    -------
    recall : numpy.ndarray, shape (M,)
        Labeled span kept over labeled span originally contained.
    span : numpy.ndarray, shape (M,)
        Width of the bracket, in decades.

    """
    recall, span = [], []
    for s in rows:
        kept = kept_regions(s['regions'], min_width)[:2]
        win = [(kept[0][0] + left_cut * (kept[0][1] - kept[0][0]),
                kept[0][1])]
        if len(kept) > 1:
            win.append((kept[1][0],
                        kept[1][0] + right_cut * (kept[1][1] - kept[1][0])))
        num = sum(overlap(a, b, c, d) for a, b in win for c, d in s['labels'])
        den = sum(overlap(a, b, c, d)
                  for a, b in s['regions'] for c, d in s['labels'])
        recall.append(num / den if den > 0 else 1.)
        span.append(sum(b - a for a, b in win) / DEC)
    return np.array(recall), np.array(span)


def main() -> None:
    """
    CLI entry point of the bracket calibration.

    """
    parser = argparse.ArgumentParser(
        prog='fcut_bracket_calib',
        description='calibrate and validate the refine_candidates bracket')
    parser.add_argument('dataset', help='CSV from tools/fcut_dataset.py')
    parser.add_argument('--min-width', type=float, default=0.5,
                        help='sliver threshold in decades (default: 0.5)')
    parser.add_argument('--left-cut', type=float, default=0.12,
                        help='fraction cut from the first region')
    parser.add_argument('--right-cut', type=float, default=0.55,
                        help='fraction kept of the second region')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    rows = [{'solvent': r['solvent'], 'regions': parse_regions(r['regions']),
             'labels': labeled(r)} for _, r in df.iterrows()]

    g = geometry(rows, args.min_width)
    print(f'{len(rows)} signals')
    print('\n--- geometry the constants come from ---')
    slivers = [w for w in g['untouched'] if w < min(g['touched'])]
    print(f"narrowest label-touched region: {min(g['touched']):.2f} dec "
          f"(next {sorted(g['touched'])[1]:.2f}); widest sliver below it: "
          f"{max(slivers):.2f} dec")
    print(f"label entry in the 1st kept region: min {min(g['entry']):.3f} "
          f"p1 {np.percentile(g['entry'], 1):.3f} "
          f"p5 {np.percentile(g['entry'], 5):.3f}")
    print(f"label exit in the 2nd kept region: max {max(g['exit']):.3f} "
          f"p99 {np.percentile(g['exit'], 99):.3f} (n={len(g['exit'])})")

    recall, span = evaluate(rows, args.min_width, args.left_cut,
                            args.right_cut)
    print(f'\n--- bracket min_width={args.min_width} '
          f'left_cut={args.left_cut} right_cut={args.right_cut} ---')
    print(f'recall >= 0.99 for {(recall >= 0.99).sum()}/{len(recall)}; '
          f'median {np.median(recall):.3f}; min {recall.min():.2f}')
    print(f'median span {np.median(span):.2f} dec')

    # the implementation must agree with the reference scoring above
    mism = 0
    grid = np.geomspace(1e-5, 0.5, 4000)
    for s, exp in zip(rows, span):
        mask = np.zeros(len(grid), dtype=bool)
        for lo, hi in s['regions']:
            mask |= (grid >= np.exp(lo)) & (grid <= np.exp(hi))
        ref = refine_candidates(grid, mask, args.min_width, args.left_cut,
                                args.right_cut)
        idx = np.flatnonzero(ref)
        if len(idx) == 0:
            mism += 1
            continue
        cuts = np.where(np.diff(idx) > 1)[0] + 1
        got = sum(np.log10(grid[r[-1]] / grid[r[0]])
                  for r in np.split(idx, cuts))
        if abs(got - exp) > 0.05:
            mism += 1
    print(f'refine_candidates agrees with this reference on '
          f'{len(rows) - mism}/{len(rows)} signals')

    print('\n--- leave-one-solvent-family-out ---')
    print('family                 min_w  left  right | recall>=0.99   span')
    for solvent in sorted({s['solvent'] for s in rows}):
        train = [s for s in rows if s['solvent'] != solvent]
        test = [s for s in rows if s['solvent'] == solvent]
        gt = geometry(train, args.min_width)
        sl = [w for w in gt['untouched'] if w < min(gt['touched'])]
        mw = 0.5 * (max(sl) + min(gt['touched']))
        lc = 0.8 * min(gt['entry'])
        rc = min(1., 1.2 * max(gt['exit'])) if gt['exit'] else 1.
        rec, sp = evaluate(test, mw, lc, rc)
        print(f'{solvent:20s} {mw:6.2f} {lc:5.3f} {rc:5.3f} | '
              f'{(rec >= 0.99).sum():3d}/{len(test):<3d}      '
              f'{np.median(sp):.2f}')


if __name__ == '__main__':
    main()
