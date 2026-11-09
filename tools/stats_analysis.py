#!/usr/bin/python3
"""
Average the retention times per molecule and solvent, outliers removed.

Reads ``solvent_effect_data.csv`` from the current directory, as
``tools/data_aggregator.py`` writes it, rejects outliers with
Chauvenet's criterion within each (molecule, solvent, distribution)
group, and prints for each group the number of replicates kept, the
1/sigma²-weighted mean of ``x0`` and its weighted spread, for the
Gaussian, the skew-normal and the Pearson VII fits in turn, each
followed by the total count.

Usage
-----
python tools/stats_analysis.py
"""

import math

import numpy as np
import pandas as pd
from scipy import stats as st

DATAPATH = './solvent_effect_data.csv'
SOLV_CATEGORIES = ['Cyclohexane', 'Benzene', 'Chlorocyclohexane',
                   'Toluene', 'XylenesMix', '4-Xylene', '3-Xylene',
                   '2-Xylene', 'Chlorobenzene', '4-Chlorotoluene',
                   '3-Chlorotoluene', '2-Chlorotoluene', '3-Dichlorobenzene',
                   '2-Dichlorobenzene']
MOL_CATEGORIES = ['C60', 'C70', 'C90', 'C90m', 'C96', 'C100', 't0', 'CS2',
                  'n-Pentane']


def weight(df):
    """
    Weight of each fit, the inverse variance of its peak.

    Parameters
    ----------
    df : pandas.DataFrame
        Peak table with a ``sigma`` column.

    Returns
    -------
    w : pandas.Series
        ``sigma**-2``.

    """
    return df['sigma'].pow(-2)


def weighted_avg_and_std(values, weights):
    """
    Weighted mean and weighted population standard deviation.

    Parameters
    ----------
    values : array-like
        The values.
    weights : array-like
        Their weights, same shape, not all zero; only their ratios
        matter.

    Returns
    -------
    average : float
        ``sum(w x) / sum(w)``.
    std : float
        Square root of the weighted mean of ``(x - average)**2``, with no
        correction for the number of values.

    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


def avg_retention_time(df):
    """
    Weighted mean retention time of each group.

    Adds the weight column ``w`` to `df` in place.

    Parameters
    ----------
    df : pandas.DataFrame
        Peak table with ``mol``, ``solvent``, ``id``, ``distribution``,
        ``x0`` and ``sigma``.

    Returns
    -------
    groups : pandas.DataFrame
        One row per (``mol``, ``solvent``, ``distribution``), with
        ``n_obs``, ``x0_wav`` and ``sigma_wav`` from
        `weighted_avg_and_std`.

    """
    df['w'] = weight(df)
    df_restricted = df[['mol', 'solvent', 'id', 'distribution', 'x0', 'w']]
    gb = df_restricted.groupby(['mol', 'solvent', 'distribution'],
                               observed=True)[['x0', 'w']]
    df = gb.apply(lambda x: weighted_avg_and_std(x['x0'], x['w'])[0])\
        .reset_index(name='x0_wav')
    sigma = gb.apply(lambda x: weighted_avg_and_std(x['x0'], x['w'])[1])\
        .reset_index(name='sigma_wav')
    n_obs = gb.size().to_frame('n').reset_index()
    df.insert(2, 'n_obs', n_obs['n'])
    df['sigma_wav'] = sigma['sigma_wav']
    return df


def chauvenets_criterions(df):
    """
    Remove the outliers of each group with Chauvenet's criterion.

    A fit is rejected when the expected number of measurements at least
    as far from the group mean is below one half: with ``n`` the group
    size and ``t`` the distance in group spreads, when
    ``n * Prob(outside t sigma) < 1/2``, that is when ``t`` exceeds the
    normal quantile ``1 - 1/(4n)`` (Taylor 1997, §6.2, p. 167).

    Parameters
    ----------
    df : pandas.DataFrame
        Peak table as `avg_retention_time` takes it, with a default
        integer index.

    Returns
    -------
    kept : pandas.DataFrame
        The rows of `df` the criterion keeps, with a fresh index.

    Notes
    -----
    The group mean and spread are the 1/sigma²-weighted mean and the
    weighted population standard deviation of `weighted_avg_and_std`,
    where Taylor uses the plain mean and standard deviation of the
    measurements.

    The criterion fails, or is unreliable, when:

    - the group is small: the spread is a very uncertain estimate
      unless there are about 50 measurements (Taylor 1997, §6.3,
      p. 169), and groups here hold a few replicates;
    - a group holds several outliers: each fit is tested on its own
      against the same mean, where Taylor gives a separate procedure
      for several suspects (§6.3, p. 169).

    It is applied once, as Taylor recommends. A group of one has zero
    spread and is never rejected. Rows are read by position with
    `iloc` from the labels of `df.index`, so the index must be the
    default ``0..N-1``.

    References
    ----------
    Taylor, J. R. *An Introduction to Error Analysis*, 2nd ed.
    University Science Books, 1997, Chapter 6.

    """
    igb = avg_retention_time(df)
    reject_list = []
    for i in df.index:
        mol = df.iloc[i]['mol']
        solvent = df.iloc[i]['solvent']
        distribution = df.iloc[i]['distribution']
        group = ((igb['solvent'] == solvent) & (igb['mol'] == mol)
                 & (igb['distribution'] == distribution))
        n_obs = igb[group]['n_obs']
        x_bar = igb[group]['x0_wav']
        sigma_x = igb[group]['sigma_wav']
        x = df.iloc[i]['x0']

        # Probability represented by one tail of the normal distribution
        P_z = 1 - (1 / (4 * n_obs))
        # Maximum allowable deviation
        D_max = st.norm.ppf(P_z)
        z_score = abs(x - x_bar) / sigma_x
        reject = z_score > D_max
        reject_list.append(reject.iloc[0])
    df['reject'] = reject_list
    result = df[~df['reject']].drop(['w', 'reject'], axis=1).reset_index()
    return result.drop(['index'], axis=1)


def main() -> None:
    """CLI entry point of the retention-time statistics."""
    pd.set_option('display.max_rows', None)
    df = pd.read_csv(DATAPATH, comment='#')
    df.solvent = pd.Categorical(df.solvent, categories=SOLV_CATEGORIES,
                                ordered=True)
    df.mol = pd.Categorical(df.mol, categories=MOL_CATEGORIES, ordered=True)

    df_clean = chauvenets_criterions(df)
    test = avg_retention_time(df_clean)
    for k, dist in enumerate(('Gaussian', 'Skew-Normal', 'Pearson VII')):
        if k:
            print('')
        print(test[test['distribution'] == dist])
        print(test[test['distribution'] == dist]['n_obs'].sum())


if __name__ == '__main__':
    main()
