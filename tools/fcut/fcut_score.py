#!/usr/bin/python3
"""
Score fcut selection rules against the labeled gallery ranges.

Reads the table built by ``tools/fcut/fcut_dataset.py`` and evaluates
candidate selection rules with the hit-rate metric: a rule scores a hit
on a signal when its predicted ``fcut`` falls inside one of the
hand-labeled acceptable ranges. Reports, for each rule family:

* the best constant found by grid search on the whole sample;
* the leave-one-solvent-out cross-validated hit rate (the honest
  number: the constant is refit without the evaluated family);
* a per-solvent breakdown of the cross-validated predictions.

Also fits a small log-log regression of the labeled range center on
peak/signal covariates and scores it with the same protocol.

Usage
-----
python tools/fcut/fcut_score.py fcut_dataset.csv [--target union|second]
"""

import argparse

import numpy as np
import pandas as pd

B_GRID = np.geomspace(1e-3, 1e2, 2001)


def parse_regions(spec: str) -> list[tuple[float, float]]:
    """
    Decode the candidate-region column of the dataset table.

    Parameters
    ----------
    spec : str
        Regions serialized as ``lo:hi;lo:hi`` (values in ``fcut``).

    Returns
    -------
    regions : list of (float, float)
        The candidate-region spans.

    """
    out = []
    for token in spec.split(';'):
        lo, hi = token.split(':')
        out.append((float(lo), float(hi)))
    return out


def labeled_ranges(row: pd.Series) -> list[tuple[float, float]]:
    """
    Return the labeled acceptable ranges of a signal.

    Parameters
    ----------
    row : pandas.Series
        A row of the dataset table.

    Returns
    -------
    ranges : list of (float, float)
        One or two ``fcut`` ranges.

    """
    ranges = [(row['lo1'], row['hi1'])]
    if row['n_ranges'] == 2:
        ranges.append((row['lo2'], row['hi2']))
    return ranges


def hits(pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Evaluate which predictions fall inside a labeled range.

    Parameters
    ----------
    pred : numpy.ndarray, shape (N,)
        Predicted ``fcut`` per signal.
    df : pandas.DataFrame
        The dataset table (same order as `pred`).

    Returns
    -------
    hit : numpy.ndarray of bool, shape (N,)
        True when the prediction is inside one of the signal's labeled
        ranges.

    """
    hit = np.zeros(len(df), dtype=bool)
    for i, (_, row) in enumerate(df.iterrows()):
        hit[i] = any(lo <= pred[i] <= hi
                     for lo, hi in labeled_ranges(row))
    return hit


def grid_best_b(feature: np.ndarray, df: pd.DataFrame,
                mask: np.ndarray) -> float:
    """
    Grid-search the constant of a ``fcut = b / feature`` rule.

    Parameters
    ----------
    feature : numpy.ndarray, shape (N,)
        The rule denominator (peak width, number of points, ...).
    df : pandas.DataFrame
        The dataset table.
    mask : numpy.ndarray of bool, shape (N,)
        Training subset selector.

    Returns
    -------
    b : float
        The constant maximizing the training hit rate. Ties resolve to
        the geometric middle of the best-scoring grid plateau.

    """
    sub = df[mask]
    feat = feature[mask]
    lows = np.full((len(sub), 2), np.nan)
    his = np.full((len(sub), 2), np.nan)
    for i, (_, row) in enumerate(sub.iterrows()):
        for k, (lo, hi) in enumerate(labeled_ranges(row)):
            lows[i, k] = lo
            his[i, k] = hi
    # b hits signal i iff lo*feat <= b <= hi*feat for one of its ranges
    b_lo = lows * feat[:, None]
    b_hi = his * feat[:, None]
    scores = np.zeros(len(B_GRID), dtype=int)
    for k in range(2):
        ok = ~np.isnan(b_lo[:, k])
        inside = ((B_GRID[:, None] >= b_lo[ok, k][None, :])
                  & (B_GRID[:, None] <= b_hi[ok, k][None, :]))
        scores += inside.sum(axis=1)
    best = np.flatnonzero(scores == scores.max())
    return float(np.sqrt(B_GRID[best[0]] * B_GRID[best[-1]]))


def lomo_rule(feature: np.ndarray, df: pd.DataFrame) -> tuple[
        np.ndarray, dict[str, float]]:
    """
    Cross-validate a ``fcut = b / feature`` rule leave-one-solvent-out.

    Parameters
    ----------
    feature : numpy.ndarray, shape (N,)
        The rule denominator.
    df : pandas.DataFrame
        The dataset table.

    Returns
    -------
    pred : numpy.ndarray, shape (N,)
        The cross-validated predictions.
    b_folds : dict
        The constant fitted for each held-out solvent.

    """
    pred = np.zeros(len(df))
    b_folds = {}
    for solvent in df['solvent'].unique():
        held = (df['solvent'] == solvent).to_numpy()
        b = grid_best_b(feature, df, ~held)
        b_folds[solvent] = b
        pred[held] = b / feature[held]
    return pred, b_folds


def chance_rate(df: pd.DataFrame) -> float:
    """
    Hit rate of a log-uniform random pick inside the candidate regions.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset table.

    Returns
    -------
    rate : float
        The expected hit rate, averaged over signals.

    """
    probs = []
    for _, row in df.iterrows():
        regions = parse_regions(row['regions'])
        total = sum(np.log(hi / lo) for lo, hi in regions)
        good = 0.
        for lo, hi in regions:
            for rlo, rhi in labeled_ranges(row):
                a, b = max(lo, rlo), min(hi, rhi)
                if b > a:
                    good += np.log(b / a)
        probs.append(good / total if total > 0 else 0.)
    return float(np.mean(probs))


def label_bounds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect the labeled range bounds as dense arrays.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset table.

    Returns
    -------
    lows, his : numpy.ndarray, shape (N, 2)
        Lower and upper bounds of the labeled ranges; NaN where a
        signal has a single range.

    """
    lows = np.full((len(df), 2), np.nan)
    his = np.full((len(df), 2), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        for k, (lo, hi) in enumerate(labeled_ranges(row)):
            lows[i, k] = lo
            his[i, k] = hi
    return lows, his


def _best_b(lows: np.ndarray, his: np.ndarray,
            feature: np.ndarray) -> tuple[float, int]:
    """
    Best constant of ``fcut = b * feature`` on a training subset.

    Parameters
    ----------
    lows, his : numpy.ndarray, shape (M, 2)
        Labeled bounds of the training signals (NaN-padded).
    feature : numpy.ndarray, shape (M,)
        The rule feature of the training signals.

    Returns
    -------
    b : float
        The constant maximizing the training hit count. Ties resolve
        to the geometric middle of the best-scoring grid plateau.
    score : int
        The training hit count at `b`.

    """
    b_lo = lows / feature[:, None]
    b_hi = his / feature[:, None]
    scores = np.zeros(len(B_GRID), dtype=int)
    for k in range(lows.shape[1]):
        ok = ~np.isnan(b_lo[:, k])
        inside = ((B_GRID[:, None] >= b_lo[ok, k][None, :])
                  & (B_GRID[:, None] <= b_hi[ok, k][None, :]))
        scores += inside.sum(axis=1)
    best = np.flatnonzero(scores == scores.max())
    b = float(np.sqrt(B_GRID[best[0]] * B_GRID[best[-1]]))
    return b, int(scores.max())


def lomo_power(df: pd.DataFrame, columns: list[str],
               exp_grids: list[np.ndarray]) -> np.ndarray:
    """
    Cross-validate a power-law rule leave-one-solvent-out.

    The rule is ``fcut = b * prod(col_i ^ a_i)``. Exponents and
    constant are refit on each training fold (nested selection), so
    the reported hit rate is honest.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset table.
    columns : list of str
        Dataset columns entering the product.
    exp_grids : list of numpy.ndarray
        Exponent grid for each column.

    Returns
    -------
    pred : numpy.ndarray, shape (N,)
        The cross-validated predictions.

    """
    import itertools

    lows, his = label_bounds(df)
    values = [df[c].to_numpy() for c in columns]
    pred = np.zeros(len(df))
    with np.errstate(invalid='ignore'):
        features = {
            exps: np.prod([v ** a for v, a in zip(values, exps)],
                          axis=0)
            for exps in itertools.product(*exp_grids)
        }
    for solvent in df['solvent'].unique():
        held = (df['solvent'] == solvent).to_numpy()
        best = (-1, None, None)
        for exps, feature in features.items():
            b, score = _best_b(lows[~held], his[~held], feature[~held])
            if score > best[0]:
                best = (score, exps, b)
        _, exps, b = best
        pred[held] = b * features[exps][held]
    return pred


def snap_to_regions(pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Snap predictions into the candidate regions of the segmentation.

    Predictions inside a candidate region are unchanged; the others
    move to the nearest region edge in log distance.

    Parameters
    ----------
    pred : numpy.ndarray, shape (N,)
        Predicted ``fcut`` per signal.
    df : pandas.DataFrame
        The dataset table.

    Returns
    -------
    snapped : numpy.ndarray, shape (N,)
        The snapped predictions.

    """
    out = np.array(pred)
    for i, (_, row) in enumerate(df.iterrows()):
        regions = parse_regions(row['regions'])
        p = pred[i]
        if not np.isfinite(p) or p <= 0:
            out[i] = np.sqrt(regions[0][0] * regions[-1][1])
            continue
        if any(lo <= p <= hi for lo, hi in regions):
            continue
        edges = [e for lo, hi in regions for e in (lo, hi)]
        out[i] = min(edges, key=lambda e: abs(np.log(e / p)))
    return out


def lomo_regression(df: pd.DataFrame, features: list[str],
                    target: np.ndarray) -> np.ndarray:
    """
    Cross-validate a log-log linear predictor leave-one-solvent-out.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset table.
    features : list of str
        Column names entering as ``log(column)`` regressors.
    target : numpy.ndarray, shape (N,)
        The ``fcut`` value the model is trained to reproduce.

    Returns
    -------
    pred : numpy.ndarray, shape (N,)
        The cross-validated predicted ``fcut``.

    """
    logx = np.column_stack(
        [np.ones(len(df))] + [np.log(df[c].to_numpy()) for c in features])
    logy = np.log(target)
    pred = np.zeros(len(df))
    for solvent in df['solvent'].unique():
        held = (df['solvent'] == solvent).to_numpy()
        coef, *_ = np.linalg.lstsq(logx[~held], logy[~held], rcond=None)
        pred[held] = np.exp(logx[held] @ coef)
    return pred


def family_table(df: pd.DataFrame, pred_map: dict[str, np.ndarray]
                 ) -> pd.DataFrame:
    """
    Build the per-solvent hit-count table of cross-validated rules.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset table.
    pred_map : dict
        ``{rule name: cross-validated predictions}``.

    Returns
    -------
    table : pandas.DataFrame
        Hits over totals per solvent and rule.

    """
    out = {}
    for name, pred in pred_map.items():
        hit = hits(pred, df)
        col = {}
        for solvent, sub in df.groupby('solvent'):
            col[solvent] = (f"{hit[df['solvent'] == solvent].sum():d}"
                            f"/{len(sub):d}")
        col['TOTAL'] = f'{hit.sum():d}/{len(df):d} = {hit.mean():.3f}'
        out[name] = col
    return pd.DataFrame(out)


def main() -> None:
    """
    CLI entry point of the rule scorer.

    """
    parser = argparse.ArgumentParser(
        prog='fcut_score',
        description='score fcut selection rules on the labeled sample')
    parser.add_argument('dataset', help='CSV from tools/fcut/fcut_dataset.py')
    parser.add_argument('--target', choices=['union', 'second'],
                        default='union',
                        help='labeled ranges used for scoring double-'
                             'range signals (default: union)')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    if args.target == 'second':
        two = df['n_ranges'] == 2
        df.loc[two, 'lo1'] = df.loc[two, 'lo2']
        df.loc[two, 'hi1'] = df.loc[two, 'hi2']
        df.loc[two, 'n_ranges'] = 1

    center = np.sqrt(df['lo1'] * df['hi1']).to_numpy()

    print(f"{len(df)} signals, {int(df['n_ranges'].sum())} ranges, "
          f"target={args.target}")
    print(f'chance (log-uniform in candidates): {chance_rate(df):.3f}')
    legacy = df['fcut_legacy'].to_numpy()
    print(f"legacy derivative method (reference only): "
          f"{hits(legacy, df).mean():.3f}\n")

    rules = {
        'b/w_median': df['w_median'].to_numpy(),
        'b/w_min': df['w_min'].to_numpy(),
        'b/n_used': df['n_used'].to_numpy(),
        'b/n_points': df['n_points'].to_numpy(),
    }
    pred_map = {}
    for name, feature in rules.items():
        b_all = grid_best_b(feature, df, np.ones(len(df), dtype=bool))
        fit_hit = hits(b_all / feature, df).mean()
        pred, b_folds = lomo_rule(feature, df)
        pred_map[name] = pred
        cv_hit = hits(pred, df).mean()
        spread = (min(b_folds.values()), max(b_folds.values()))
        print(f'{name:12s} best b={b_all:0.4g} fit={fit_hit:.3f} '
              f'CV={cv_hit:.3f} b-folds=[{spread[0]:0.4g},'
              f'{spread[1]:0.4g}]')

    a_grid = np.linspace(-1.6, 0.4, 41)
    c_grid = np.linspace(-0.6, 0.6, 25)
    power_rules = {
        'w^a': (['w_median'], [a_grid]),
        'w^a*h^c': (['w_median', 'h_max'], [a_grid, c_grid]),
    }
    print('\npower-law rules (nested LOMO CV):')
    for name, (columns, grids) in power_rules.items():
        pred = lomo_power(df, columns, grids)
        snapped = snap_to_regions(pred, df)
        inside = float((pred == snapped).mean())
        pred_map[name] = pred
        pred_map[f'snap({name})'] = snapped
        print(f'{name:12s} CV={hits(pred, df).mean():.3f} '
              f'inside candidates={inside:.3f} '
              f'snapped CV={hits(snapped, df).mean():.3f}')

    feature_pool = ['w_median', 'n_used', 'x_last_peak', 'n_rel_peaks']
    print('\nlog-log regression on labeled range centers (LOMO CV):')
    results = []
    for k in range(1, 2 ** len(feature_pool)):
        features = [f for i, f in enumerate(feature_pool)
                    if k & (1 << i)]
        pred = lomo_regression(df, features, center)
        results.append((hits(pred, df).mean(), features, pred))
    results.sort(key=lambda t: -t[0])
    for cv_hit, features, _ in results:
        print(f"  CV={cv_hit:.3f}  {' + '.join(features)}")
    best_hit, best_features, best_pred = results[0]
    pred_map[f"loglog({'+'.join(best_features)})"] = best_pred

    print('\nper-solvent hits (cross-validated):')
    print(family_table(df, pred_map).to_string())


if __name__ == '__main__':
    main()
