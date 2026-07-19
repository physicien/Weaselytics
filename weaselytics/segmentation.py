# coding: utf-8
"""
Changepoint-based plateau detection for autocorrelation plots.

Prototype for the strategy discussed in issue #4: instead of classifying
points of the autocorrelation curve one by one with absolute thresholds
on rolling statistics, the curve is segmented with a penalized
piecewise-linear model (optimal partitioning). Each segment is then
classified with scale-free criteria (slope and residual noise relative
to the geometry of the curve itself), and the optimal cutoff frequency
is selected from the surviving plateau candidates.
"""
from collections.abc import Callable

import numpy as np


def _linear_costs(y: np.ndarray) -> Callable[[int, np.ndarray], np.ndarray]:
    """
    Build the per-segment cost function of a piecewise-linear model.

    The cost of the segment ``y[i:j]`` is ``m * log(SSE / m)``, where
    ``SSE`` is the sum of squared residuals of the least-squares line
    fitted on the segment and ``m = j - i``. This is the Gaussian
    log-likelihood cost with a mean varying linearly and a variance
    fitted separately on each segment, so both slope changes and noise
    changes are detected.

    Parameters
    ----------
    y : array-like, shape (N,)
        The y-values of the data.

    Returns
    -------
    costs : Callable
        A function ``costs(j, i_arr)`` returning the array of segment
        costs of ``y[i:j]`` for every start index ``i`` in ``i_arr``,
        evaluated in O(1) per segment from cumulative sums.

    """
    n = len(y)
    t = np.arange(n, dtype=float)
    c1 = np.concatenate([[0.0], np.cumsum(np.ones(n))])
    ct = np.concatenate([[0.0], np.cumsum(t)])
    ct2 = np.concatenate([[0.0], np.cumsum(t * t)])
    cy = np.concatenate([[0.0], np.cumsum(y)])
    cy2 = np.concatenate([[0.0], np.cumsum(y * y)])
    cty = np.concatenate([[0.0], np.cumsum(t * y)])

    def costs(j: int, i_arr: np.ndarray) -> np.ndarray:
        m = j - i_arr
        s1 = c1[j] - c1[i_arr]
        st = ct[j] - ct[i_arr]
        st2 = ct2[j] - ct2[i_arr]
        sy = cy[j] - cy[i_arr]
        sy2 = cy2[j] - cy2[i_arr]
        sty = cty[j] - cty[i_arr]
        # Least-squares line y = a + b*t via the normal equations
        det = s1 * st2 - st * st
        b = (s1 * sty - st * sy) / det
        a = (sy - b * st) / s1
        sse = (sy2 - 2.0 * a * sy - 2.0 * b * sty + a * a * s1
               + 2.0 * a * b * st + b * b * st2)
        sse = np.maximum(sse, 1e-16)
        return m * np.log(sse / m)

    return costs


def pelt_linear(y: np.ndarray, penalty: float | None = None,
                min_size: int = 15) -> np.ndarray:
    """
    Segment the data with a penalized piecewise-linear model.

    Exact optimal partitioning (dynamic programming) of the data into
    contiguous segments, each fitted by a straight line with its own
    residual variance. The number of segments is controlled by a single
    penalty added per changepoint. Complexity is O(N^2), which is
    immediate for the typical N = 1000 of an autocorrelation plot.

    Parameters
    ----------
    y : array-like, shape (N,)
        The y-values of the data.
    penalty : float, optional
        Penalty added for each additional segment. Default is None,
        which uses the BIC-like value ``25 * log(N)``.
    min_size : int, optional
        Minimal length of a segment. Default is 15.

    Returns
    -------
    breakpoints : numpy.ndarray, shape (M,)
        The sorted end indices (exclusive) of the M segments. The last
        breakpoint is always N; segment ``k`` spans
        ``y[breakpoints[k-1]:breakpoints[k]]``, with an implicit start
        of 0 for the first segment.

    Raises
    ------
    ValueError
        Raised if the data is shorter than ``2 * min_size``.

    """
    n = len(y)
    if n < 2 * min_size:
        raise ValueError('data must contain at least 2 * min_size points')
    if penalty is None:
        penalty = 25.0 * np.log(n)

    costs = _linear_costs(y)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    prev = np.zeros(n + 1, dtype=int)
    for j in range(min_size, n + 1):
        i_arr = np.arange(0, j - min_size + 1)
        candidates = best[i_arr] + costs(j, i_arr) + penalty
        k = np.argmin(candidates)
        best[j] = candidates[k]
        prev[j] = i_arr[k]

    breakpoints = []
    j = n
    while j > 0:
        breakpoints.append(j)
        j = prev[j]
    return np.array(sorted(breakpoints))


def segment_features(fcut_range: np.ndarray, r2: np.ndarray,
                     breakpoints: np.ndarray) -> list[dict]:
    """
    Compute the descriptive features of each segment.

    The slope and the residual noise of each segment are also expressed
    relative to the geometry of the curve itself, so that the
    classification thresholds are scale-free: ``rel_slope`` is the
    segment slope in units of the mean slope that the total drop of `r2`
    would have if spread over one decade of `fcut`, and ``rel_noise`` is
    the residual standard deviation in units of the total drop.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies of the
        autocorrelation plot.
    r2 : array-like, shape (N,)
        The autocorrelation coefficients.
    breakpoints : array-like, shape (M,)
        The segment end indices returned by ``pelt_linear``.

    Returns
    -------
    segments : list of dict
        One dict per segment with keys ``start``, ``end``, ``mean``,
        ``slope``, ``resid_std``, ``rel_slope`` and ``rel_noise``.

    """
    n = len(r2)
    t = np.arange(n, dtype=float)
    drop = r2.max() - r2.min()
    decades = np.log10(fcut_range[-1] / fcut_range[0])
    points_per_decade = n / decades
    slope_scale = max(drop, 1e-16) / points_per_decade

    segments = []
    start = 0
    for end in breakpoints:
        seg = r2[start:end]
        tt = t[start:end]
        slope, intercept = np.polyfit(tt, seg, 1)
        resid = seg - (intercept + slope * tt)
        resid_std = float(np.std(resid))
        segments.append({
            'start': int(start),
            'end': int(end),
            'mean': float(np.mean(seg)),
            'slope': float(slope),
            'resid_std': resid_std,
            'rel_slope': float(abs(slope) / slope_scale),
            'rel_noise': float(resid_std / max(drop, 1e-16)),
        })
        start = end
    return segments


def classify_segments(segments: list[dict], rel_slope_max: float = 0.2,
                      rel_noise_max: float = 0.006,
                      rel_slope_loose: float = 0.6,
                      cliff_min: float = 1.0) -> list[dict]:
    """
    Mark each segment as flat (plateau candidate) or not.

    A segment is flat when its relative residual noise is small and its
    relative slope satisfies a two-tier (tight/loose) criterion, the
    dimensionless analogue of the tight and loose derivative tolerances
    of ``_fcutoff``:

    - tight: ``rel_slope < rel_slope_max`` — strictly flat on the scale
      of the whole curve;
    - loose: ``rel_slope < rel_slope_loose`` *and* the segment is
      bracketed by at least one cliff (``rel_slope > cliff_min``) on
      each side. This accepts the drifting shelves of staircase-shaped
      curves (blanks, multi-step programs), whose slope is substantial
      on the global scale but small compared to the cliffs surrounding
      them. On these morphologies the total drop is split over several
      steps, so a purely global slope criterion rejects every shelf.

    The residual-noise criterion is what separates the quiet plateaus
    from the regions where the baseline fit is unstable (e.g. the
    low-frequency instabilities of `p_ini` or the chaotic tail),
    replacing the rolling-standard-deviation and thresholding machinery
    of ``find_plateaus``.

    Parameters
    ----------
    segments : list of dict
        The segments returned by ``segment_features``. Modified in
        place: a boolean key ``flat`` is added to each dict.
    rel_slope_max : float, optional
        Maximum relative slope of a strictly flat segment. Default is
        0.2.
    rel_noise_max : float, optional
        Maximum relative residual noise of a flat segment. Default is
        0.006.
    rel_slope_loose : float, optional
        Maximum relative slope of a cliff-bracketed shelf. Default is
        0.6.
    cliff_min : float, optional
        Minimum relative slope of a segment counting as a cliff for the
        loose criterion. Default is 1.0.

    Returns
    -------
    segments : list of dict
        The input list with the added ``flat`` key.

    """
    slopes = [seg['rel_slope'] for seg in segments]
    for k, seg in enumerate(segments):
        tight = seg['rel_slope'] < rel_slope_max
        loose = (seg['rel_slope'] < rel_slope_loose
                 and any(s > cliff_min for s in slopes[:k])
                 and any(s > cliff_min for s in slopes[k + 1:]))
        seg['flat'] = (seg['rel_noise'] < rel_noise_max
                       and (tight or loose))
    return segments


def trim_candidates(fcut_range: np.ndarray, segments: list[dict],
                    n_used: int, c1: float = 0.5,
                    noise_floor: float = 4e-7,
                    cliff_min: float = 1.0,
                    bridge: bool = True) -> np.ndarray:
    """
    Trim the flat segments into a mask of candidate plateau regions.

    Reduces the flat set to the regions where the optimal cutoff
    frequency can actually lie, using only a-priori exclusions (no
    selection convention):

    - **sub-fundamental clip**: grid points below ``c1 / n_used`` are
      removed. The slowest oscillation representable on a record of
      `n_used` points has frequency ``1/n_used``, so every cutoff below
      it requests the same maximally rigid baseline; the region only
      contains duplicates of the solution at the fundamental. This is
      why the initial plateau of the autocorrelation plot always ends
      at ``~1/n_used``.
    - **frozen exclusion**: flat segments whose relative residual noise
      is at most `noise_floor` are removed. In the saturated far tail
      the baseline no longer responds to the cutoff frequency at all
      and the residual noise collapses by orders of magnitude below
      that of any genuine plateau.
    - **bridging** (optional): a non-flat segment sandwiched between
      candidate regions is absorbed when it is not a cliff
      (``rel_slope < cliff_min``), so drifting connectors do not split
      one plateau into several displayed pieces while genuine staircase
      steps still separate regions.

    On the 339-signal reference dataset, the trimmed mask contains the
    accepted cutoff frequency of every signal while covering half of
    the grid area of the untrimmed flat set, in 2-3 contiguous regions
    per signal.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    segments : list of dict
        The classified segments returned by ``classify_segments``.
    n_used : int
        Number of signal points used for the autocorrelation sweep
        (the length of the truncated, log-transformed signal).
    c1 : float, optional
        Safety factor of the sub-fundamental clip: grid points below
        ``c1 / n_used`` are excluded. Default is 0.5.
    noise_floor : float, optional
        Relative residual noise at or below which a flat segment is
        considered frozen. Default is 4e-7.
    cliff_min : float, optional
        Minimum relative slope of a segment acting as a real separation
        between candidate regions. Default is 1.0.
    bridge : bool, optional
        If True (default), absorb non-cliff segments lying between
        candidate regions.

    Returns
    -------
    candidates : numpy.ndarray, shape (N,), dtype bool
        Boolean mask of the grid points belonging to a candidate
        plateau region.

    """
    cand = [seg['flat'] and seg['rel_noise'] > noise_floor
            for seg in segments]
    if bridge:
        for k in range(1, len(segments) - 1):
            if (not cand[k] and segments[k]['rel_slope'] < cliff_min
                    and any(cand[:k]) and any(cand[k + 1:])):
                cand[k] = True
    candidates = np.zeros(len(fcut_range), dtype=bool)
    for seg, keep in zip(segments, cand):
        if keep:
            candidates[seg['start']:seg['end']] = True
    candidates[fcut_range < c1 / n_used] = False
    return candidates


def select_fcut(fcut_range: np.ndarray, r2: np.ndarray,
                penalty: float | None = None, min_size: int = 15,
                rel_slope_max: float = 0.2, rel_noise_max: float = 0.006,
                level_frac: float = 0.5
                ) -> tuple[float | None, list[dict], int | None]:
    """
    Select the optimal cutoff frequency from an autocorrelation plot.

    The curve is segmented with ``pelt_linear``, the segments are
    classified with ``classify_segments``, and the plateau containing
    the optimal `fcut` is chosen as the last flat segment before the
    first steep descending segment (the "knee" of the curve). The
    returned `fcut` is the right edge of that plateau, consistent with
    the behavior of the ``slope_thresh`` shift in ``_fcutoff``. If no
    flat segment precedes a steep drop, the flat segments whose mean is
    high enough (above ``level_frac`` of the total drop) are used as
    fallback candidates.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    r2 : array-like, shape (N,)
        The autocorrelation coefficients.
    penalty : float, optional
        Penalty per changepoint passed to ``pelt_linear``. Default is
        None (BIC-like ``25 * log(N)``).
    min_size : int, optional
        Minimal segment length passed to ``pelt_linear``. Default is 15.
    rel_slope_max : float, optional
        Maximum relative slope of a flat segment. Default is 0.2.
    rel_noise_max : float, optional
        Maximum relative residual noise of a flat segment. Default is
        0.006.
    level_frac : float, optional
        Fallback level criterion: fraction of the total drop of `r2`
        above which the mean of a candidate plateau must lie. Default
        is 0.5.

    Returns
    -------
    fcut : float or None
        The selected cutoff frequency, or None if no plateau candidate
        was found.
    segments : list of dict
        The classified segments, for diagnostics and plotting.
    chosen : int or None
        Index in `segments` of the selected plateau, or None.

    """
    breakpoints = pelt_linear(r2, penalty=penalty, min_size=min_size)
    segments = segment_features(fcut_range, r2, breakpoints)
    segments = classify_segments(segments, rel_slope_max=rel_slope_max,
                                 rel_noise_max=rel_noise_max)

    # First steep descending segment (start of the main drop)
    first_steep = len(segments)
    for k, seg in enumerate(segments):
        if seg['slope'] < 0 and seg['rel_slope'] > rel_slope_max:
            first_steep = k
            break

    candidates = [k for k in range(first_steep) if segments[k]['flat']]
    if not candidates:
        # Fallback: flat segments high enough above the global minimum
        drop = r2.max() - r2.min()
        level_min = r2.min() + level_frac * drop
        candidates = [k for k, seg in enumerate(segments)
                      if seg['flat'] and seg['mean'] >= level_min]
    if not candidates:
        return None, segments, None

    chosen = candidates[-1]
    fcut = float(fcut_range[segments[chosen]['end'] - 1])
    return fcut, segments, chosen
