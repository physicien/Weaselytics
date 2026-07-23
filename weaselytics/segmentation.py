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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from weaselytics.utils import _rolling_std


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


def detect_dips(fcut_range: np.ndarray, r2: np.ndarray, sigma: float = 8.0,
                min_prominence: float = 0.03, level_min: float = 0.08,
                level_max: float = 0.92, window: int = 3,
                rel_height: float = 0.5) -> list[dict]:
    """
    Detect proto-plateaus as dips of the rolling standard deviation.

    A plateau of the autocorrelation curve is a stretch of low local
    variation, so the rolling standard deviation of `r2` is small on the
    plateaus and large on the descents between them. A proto-plateau (a
    *relative* flattening that never becomes flat on the scale of the
    whole curve) therefore appears as a local **minimum** of the rolling
    standard deviation, sitting in the valley between two of its humps.
    This is the relative counterpart of the absolute slope criterion of
    ``classify_segments``: it finds the shelves whose slope is a local
    minimum even when it is above the absolute flat threshold, which the
    segment classifier misses.

    The rolling standard deviation is smoothed and normalised by its own
    maximum (the largest cliff), so the prominence of a valley is a
    scale-free fraction of that cliff rather than an absolute level. Only
    the descent is searched, from the first grid point up to the global
    minimum of `r2`; the rising tail beyond it is not a cutoff-frequency
    regime of interest.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    r2 : array-like, shape (N,)
        The autocorrelation coefficients.
    sigma : float, optional
        Standard deviation, in grid points, of the Gaussian smoothing
        applied to the rolling standard deviation before the valleys are
        located. On the log-uniform grid this is a fixed fraction of a
        decade. Default is 8.0.
    min_prominence : float, optional
        Minimum prominence of a valley, as a fraction of the largest
        cliff (the maximum of the normalised rolling standard deviation).
        Rejects the shallow wiggles of the noise floor. Default is 0.03.
    level_min : float, optional
        Lower bound on the level of a dip floor, as a fraction of the
        total drop of `r2` (0 = global minimum, 1 = curve maximum). Drops
        the flattening at the collapse floor, which is the saturated
        baseline rather than a plateau. Default is 0.08.
    level_max : float, optional
        Upper bound on the level of a dip floor, same units. Drops the
        micro-dips inside the initial plateau, already covered by the flat
        set. Default is 0.92.
    window : int, optional
        Window of the rolling standard deviation, passed to
        ``_rolling_std``. Default is 3, matching ``find_plateaus``.
    rel_height : float, optional
        Relative height at which the basin width of each valley is
        measured (``scipy.signal.peak_widths``); 0.5 gives the width at
        half prominence. Default is 0.5.

    Returns
    -------
    dips : list of dict
        One dict per accepted proto-plateau, sorted by cutoff frequency,
        with keys ``floor`` (grid index of the valley bottom), ``fcut``,
        ``r2``, ``level`` (fraction of the total drop), ``prominence``
        (fraction of the largest cliff), and ``start`` / ``end`` (grid
        indices bounding the basin at ``rel_height``).

    Notes
    -----
    The rolling-standard-deviation-dip criterion is a detection
    heuristic, not a result of Navarro-Huerta et al. (2017); it is the
    relative analogue of the absolute flat test. Its parameters are
    dimensionless (prominence relative to the largest cliff, level as a
    fraction of the drop, `sigma` in grid points) and were set by visual
    validation on the 339-signal reference gallery, not fitted to a
    selected cutoff. They are diagnostic tuning while the returned dips
    only feed the plateau overlay; they become load-bearing, and require
    grounding, if the dips are ever used to select the cutoff frequency.

    """
    n = len(r2)
    rss = gaussian_filter1d(_rolling_std(r2, window=window), sigma)
    peak = rss.max()
    if peak <= 0.0:
        return []
    norm = rss / peak
    imin = int(np.argmin(r2))
    drop = r2.max() - r2.min()
    if imin < 2 or drop <= 0.0:
        return []

    idx, props = find_peaks(-norm[:imin], prominence=min_prominence)
    if len(idx) == 0:
        return []
    prominences = props['prominences']
    _, _, lefts, rights = peak_widths(-norm[:imin], idx,
                                      rel_height=rel_height)

    dips = []
    for i, prom, lo, hi in zip(idx, prominences, lefts, rights):
        level = (r2[i] - r2.min()) / drop
        if not (level_min < level < level_max):
            continue
        dips.append({
            'floor': int(i),
            'fcut': float(fcut_range[i]),
            'r2': float(r2[i]),
            'level': float(level),
            'prominence': float(prom),
            'start': int(np.floor(lo)),
            'end': min(int(np.ceil(hi)), n - 1),
        })
    return dips


def dips_to_mask(fcut_range: np.ndarray, dips: list[dict]) -> np.ndarray:
    """
    Convert the dips of ``detect_dips`` into a boolean basin mask.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    dips : list of dict
        The dips returned by ``detect_dips``.

    Returns
    -------
    mask : numpy.ndarray, shape (N,), dtype bool
        True on the grid points covered by a proto-plateau basin.

    """
    mask = np.zeros(len(fcut_range), dtype=bool)
    for dip in dips:
        mask[dip['start']:dip['end'] + 1] = True
    return mask


def trim_candidates(fcut_range: np.ndarray, segments: list[dict],
                    n_used: int, c1: float = 0.5,
                    noise_floor: float = 4e-7,
                    cliff_min: float = 1.0,
                    bridge: bool = True,
                    exclude_collapse: bool = False,
                    collapse_level: float = 0.5) -> np.ndarray:
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
    - **collapse exclusion** (optional): when ``exclude_collapse`` is
      set, flat segments lying below ``collapse_level`` of the way up
      the total r2 drop are removed. Past the collapse the baseline has
      begun absorbing the analyte-correlated content, so for a signal
      that *has* analyte a cutoff there destroys peak area and the
      optimum cannot lie in it (Navarro-Huerta et al. 2017). Whether a
      signal has analyte is a property of the signal, not the curve, so
      the caller decides ``exclude_collapse`` from the signal-to-noise
      ratio (a split at SNR ~25 separates the two at 95-100% on the
      labeled and synthetic data); a blank leaves it off, and its low
      shelves survive as legitimate candidates.

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
    exclude_collapse : bool, optional
        If True, remove flat segments past the collapse (see the summary
        above). The caller sets this from the signal-to-noise ratio.
        Default is False.
    collapse_level : float, optional
        Relative level of the total r2 drop below which a flat segment
        is considered past the collapse, in [0, 1] (1 = plateau top,
        0 = collapse floor). Only used when ``exclude_collapse``.
        Default is 0.5.

    Returns
    -------
    candidates : numpy.ndarray, shape (N,), dtype bool
        Boolean mask of the grid points belonging to a candidate
        plateau region.

    """
    cand = [seg['flat'] and seg['rel_noise'] > noise_floor
            for seg in segments]
    if exclude_collapse and segments:
        means = [seg['mean'] for seg in segments]
        r2_max, r2_min = max(means), min(means)
        thr = r2_min + collapse_level * (r2_max - r2_min)
        cand = [c and (seg['mean'] >= thr)
                for c, seg in zip(cand, segments)]
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


def refine_candidates(fcut_range: np.ndarray, candidates: np.ndarray,
                      min_width: float = 0.5, left_cut: float = 0.12,
                      right_cut: float = 0.55) -> np.ndarray:
    """
    Refine the candidate regions with the label-calibrated bracket.

    Narrows the candidate mask of ``trim_candidates`` to the portions
    where the hand-labeled acceptable ranges of the 339-signal
    reference gallery (2026-07-20) actually lie. All rules are
    scale-free (region widths in decades, positions as log-relative
    fractions of a region). ``min_width`` and ``right_cut`` sit at a
    "labels never go there" extreme of the gallery with a margin;
    ``left_cut`` is a 1st-percentile quantile and does clip a small
    tail of the labels:

    - **sliver exclusion**: regions narrower than `min_width` decades
      are removed (the narrowest label-touched region spans 0.67
      decades); when every region is a sliver, the widest one is kept
      as a fallback;
    - **ordinal cut**: only the first two remaining regions are kept
      (no label touches a later one);
    - **left cut**: the first ``left_cut`` fraction of the first
      region is removed (the over-rigid edge; 99% of labels start
      beyond 0.12);
    - **right cut**: the second region is truncated beyond the
      ``right_cut`` fraction (labels enter it only from the left,
      never beyond 0.41 of its width).

    On the reference gallery this keeps the labeled range (recall at
    least 0.99) for 336/339 signals while shrinking the median
    candidate span from 2.33 to 1.84 decades.

    The constants are **provisional**: the gallery was labeled before
    the coarse detrend of ``_relevant_regions`` (commit 6a1a380),
    which changes the peak regions or ``scut`` — and therefore the
    r2 curve these regions are derived from — for 161 of the 339
    reference signals. They also come from one instrument and one
    labeling session, and the labels are censored by the candidate
    set (only cutoffs inside the candidates were ever rendered), so
    this function can narrow ``trim_candidates`` but cannot validate
    its outer boundaries. See ``segmentation.md`` §4c and
    ``tools/fcut_bracket_calib.py``.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    candidates : array-like, shape (N,), dtype bool
        Candidate mask returned by ``trim_candidates``.
    min_width : float, optional
        Minimum width of a real plateau region, in decades. Default
        is 0.5.
    left_cut : float, optional
        Log-relative fraction removed from the left of the first kept
        region. Default is 0.12.
    right_cut : float, optional
        Log-relative fraction of the second kept region beyond which
        it is truncated. Default is 0.55.

    Returns
    -------
    refined : numpy.ndarray, shape (N,), dtype bool
        The refined candidate mask.

    """
    log_f = np.log10(fcut_range)
    idx = np.flatnonzero(candidates)
    if len(idx) == 0:
        return np.zeros(len(fcut_range), dtype=bool)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    regions = [(run[0], run[-1]) for run in np.split(idx, splits)]

    widths = [log_f[b] - log_f[a] for a, b in regions]
    kept = [r for r, w in zip(regions, widths) if w >= min_width]
    if not kept:
        kept = [regions[int(np.argmax(widths))]]
    kept = kept[:2]

    refined = np.zeros(len(fcut_range), dtype=bool)
    a, b = kept[0]
    lo = log_f[a] + left_cut * (log_f[b] - log_f[a])
    refined[a:b + 1] = log_f[a:b + 1] >= lo
    if len(kept) > 1:
        a, b = kept[1]
        hi = log_f[a] + right_cut * (log_f[b] - log_f[a])
        refined[a:b + 1] = log_f[a:b + 1] <= hi
    return refined


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
