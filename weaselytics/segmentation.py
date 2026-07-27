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
    low-frequency instabilities of `p_ini` or the chaotic tail). It
    replaced a detector built on a rolling standard deviation and a
    stack of thresholds, since removed.

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


def dip_curve(r2: np.ndarray, window: int = 3,
              sigma: float = 8.0) -> np.ndarray:
    """
    The curve the proto-plateau detector actually reads.

    The rolling standard deviation of `r2`, Gaussian-smoothed and
    scaled to its own maximum. ``detect_dips`` finds proto-plateaus as
    local MINIMA of this curve, so this — not the raw rolling standard
    deviation, which is only an unsmoothed precursor — is what the
    detection sees. Exposed so the diagnostic can draw the same array
    the detector uses, rather than a lookalike that can drift from it.

    Parameters
    ----------
    r2 : array-like, shape (N,)
        The autocorrelation coefficients.
    window : int, optional
        Window of the rolling standard deviation. Default is 3.
    sigma : float, optional
        Standard deviation of the Gaussian smoothing. Default is 8.0.

    Returns
    -------
    curve : numpy.ndarray, shape (N,)
        The normalised curve, in [0, 1]; all zeros if `r2` is constant.

    """
    rss = gaussian_filter1d(_rolling_std(r2, window=window), sigma)
    peak = float(rss.max())
    if peak <= 0.0:
        return np.zeros_like(rss)
    return rss / peak


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
        ``_rolling_std``. Default is 3.
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
    norm = dip_curve(r2, window=window, sigma=sigma)
    if not norm.any():
        return []
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
                    n_used: int, c1: float = 1.0,
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
        ``c1 / n_used`` are excluded. Default is 1.0, the fundamental
        itself: below it the data cannot determine the baseline at all,
        and on the 51 hand-labeled reference signals no labeled optimum
        has its stiff edge below 0.9x the fundamental (median 8.4x), so
        clipping at the fundamental removes no usable cutoff.
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


def sensitivity_dispersion(fcut_range: np.ndarray, sensitivity: np.ndarray,
                         win_dec: float = 0.2) -> np.ndarray:
    """
    Local dispersion of the baseline-sensitivity curve.

    The interquartile range of `sensitivity` inside a sliding window of
    `win_dec` decades of cutoff frequency. Where the BEADS fit is
    undetermined the baseline swings between adjacent cutoffs, so the
    sensitivity values do not merely run high, they *scatter*: dispersion
    separates that from a baseline that moves steadily, which is what
    happens on the flexible side approaching the collapse, where the
    values are high but tightly ordered.

    The interquartile range rather than a standard deviation, and a
    window rather than a smoothing kernel, because the excursions are
    the signal here and must set the level they belong to instead of
    being averaged away.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    sensitivity : array-like, shape (N,)
        The baseline-sensitivity curve (``baseline._sensitivity_curve``).
    win_dec : float, optional
        Width of the window, in decades of cutoff frequency. Default is
        0.2.

    Returns
    -------
    dispersion : numpy.ndarray, shape (N,)
        The windowed interquartile range of `sensitivity`.

    """
    per_dec = (len(fcut_range) - 1) / np.log10(fcut_range[-1] / fcut_range[0])
    width = max(5, int(round(win_dec * per_dec)) | 1)
    half = width // 2
    padded = np.pad(np.asarray(sensitivity, dtype=float), half, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    q75, q25 = np.percentile(windows, [75, 25], axis=-1)
    return q75 - q25


def instability_boundary(fcut_range: np.ndarray, sensitivity: np.ndarray,
                         n_used: int, trigger: float = 0.10,
                         settled: float = 0.05,
                         win_dec: float = 0.2) -> float | None:
    """
    Cutoff frequency up to which the fit is undetermined, or None.

    Implements the rule that when the record's fundamental falls inside
    a region where the baseline is flailing, that region is unusable up
    to the point where the oscillations become small again. The test is
    made at the fundamental itself, so it is local to the signal and
    needs no prior classification of the curve.

    Below the fundamental the data cannot constrain the baseline at all
    and ``trim_candidates`` already clips there; this extends the
    exclusion upward over the frequencies where the fit is still
    swinging, which the sub-fundamental clip cannot reach.

    .. warning::
       `trigger` and `settled` are **not grounded**. They are amplitudes
       of the sensitivity curve, which is dimensionless (rms baseline
       change as a fraction of the signal range, per decade), so they
       read as physical statements about tolerable baseline movement
       rather than as instrument constants — but no reference fixes
       where that tolerance lies. The values are adopted provisionally
       from a review of all 339 reference signals. `settled` is the
       sensitive one: it sets how far the exclusion reaches. See the
       README TO DO.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    sensitivity : array-like, shape (N,)
        The baseline-sensitivity curve (``baseline._sensitivity_curve``).
    n_used : int
        Number of signal points used for the autocorrelation sweep; its
        reciprocal is the fundamental.
    trigger : float, optional
        Dispersion at the fundamental above which the fit counts as
        flailing there. Default is 0.10.
    settled : float, optional
        Dispersion below which the oscillations count as small enough.
        Default is 0.05.
    win_dec : float, optional
        Window of ``sensitivity_dispersion``, in decades. Default is 0.2.

    Returns
    -------
    boundary : float or None
        The cutoff frequency up to which the fit is undetermined, or
        None when the fundamental is not inside a flailing region.

    """
    dispersion = sensitivity_dispersion(fcut_range, sensitivity,
                                        win_dec=win_dec)
    fundamental = 1.0 / n_used
    start = int(np.argmin(np.abs(fcut_range - fundamental)))
    if dispersion[start] <= trigger:
        return None
    end = start
    while end < len(fcut_range) and dispersion[end] >= settled:
        end += 1
    return float(fcut_range[min(end, len(fcut_range) - 1)])


def _trim_masks(fcut_range: np.ndarray, segments: list[dict],
                dips: list[dict], n_used: int, exclude_collapse: bool,
                c1: float, collapse_level: float,
                sensitivity: np.ndarray | None) -> dict[str, np.ndarray]:
    """
    Apply the stage-1 exclusions to a given detected selection.

    Split out of ``trim_plateaus`` so that the flat channel can be run
    through the identical chain on its own, which is what decides
    whether the proto-plateaus are needed at all.

    """
    cp_flat = np.zeros(len(fcut_range), dtype=bool)
    for seg in segments:
        if seg['flat']:
            cp_flat[seg['start']:seg['end']] = True
    cp_dips = dips_to_mask(fcut_range, dips)
    union = cp_flat | cp_dips
    sub_fund = fcut_range >= c1 / n_used

    # No bridging: bridging absorbs the non-cliff connector between two
    # candidate regions, which on a gentle (e.g. blank) descent merges a
    # plateau and a lower shelf into one region, so the sampling then
    # lands on the descent *between* the plateaus. The surviving set must
    # keep the plateaus separate.
    flat_12 = trim_candidates(fcut_range, segments, n_used, c1=c1,
                              bridge=False, exclude_collapse=False)
    trimmed_12 = flat_12 | (cp_dips & sub_fund)
    removed = union & ~trimmed_12

    if exclude_collapse:
        flat_123 = trim_candidates(fcut_range, segments, n_used, c1=c1,
                                   bridge=False, exclude_collapse=True,
                                   collapse_level=collapse_level)
        dips_123 = dips_to_mask(
            fcut_range, [d for d in dips if d['level'] >= collapse_level])
        trimmed_123 = flat_123 | (dips_123 & sub_fund)
    else:
        trimmed_123 = trimmed_12
    snr_removed = trimmed_12 & ~trimmed_123

    # Stiff-side instability exclusion. Applied last, on whatever has
    # survived: it removes the frequencies at which the fit is still
    # undetermined above the fundamental, which the sub-fundamental clip
    # cannot reach.
    instab_removed = np.zeros(len(fcut_range), dtype=bool)
    if sensitivity is not None:
        boundary = instability_boundary(fcut_range, sensitivity, n_used)
        if boundary is not None:
            surviving = trimmed_123 & (fcut_range > boundary)
            instab_removed = trimmed_123 & ~surviving
            trimmed_123 = surviving

    return {'surviving': trimmed_123, 'removed': removed,
            'snr_removed': snr_removed, 'instab_removed': instab_removed}


def trim_plateaus(fcut_range: np.ndarray, segments: list[dict],
                  dips: list[dict], n_used: int,
                  exclude_collapse: bool = False, c1: float = 1.0,
                  collapse_level: float = 0.5,
                  sensitivity: np.ndarray | None = None
                  ) -> dict[str, np.ndarray]:
    """
    Stage-1 trimming of the detected plateau selection.

    The detected selection is the union of the flat set
    (``classify_segments``, the strong and initial plateaus) and the
    proto-plateau basins (``detect_dips``, the relative flattenings the
    flat test misses). This applies the a-priori exclusions to it:

    - the sub-fundamental clip and the frozen tail, always;
    - the SNR-gated collapse exclusion, only when ``exclude_collapse``.
      Past the collapse a cutoff destroys analyte peak area
      (Navarro-Huerta et al. 2017), so for a signal that *has* analyte
      the plateaus below ``collapse_level`` of the drop are removed; a
      blank keeps them. Whether a signal has analyte is a property of
      the signal, not the curve, so the caller sets this from the SNR.
    - the stiff-side instability exclusion, when `sensitivity` is given.

    **The proto-plateaus are a fallback, not a peer of the flat set.**
    ``detect_dips`` exists to catch the relative flattenings the
    absolute flat test misses, so it earns its place only when that
    test has yielded nothing: the flat channel is run through the
    identical chain on its own, and the dips contribute only if it
    leaves no surviving region. Without this the dip detector also
    fires on the descent past the collapse, where a momentary easing of
    the slope is a local minimum of the rolling standard deviation
    while still being an order of magnitude steeper than any plateau —
    those shelves then survive as a spurious second region. On the 339
    reference signals the rule keeps the dips of exactly the 5 signals
    that have no flat region at all, and drops them everywhere else,
    without leaving any signal with nothing surviving.

    This is the single source of the trimming: both the diagnostic
    overlay (``baseline.auto_beads``) and the fcut gallery use it, so
    the surviving regions cannot drift between them.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    segments : list of dict
        The classified segments from ``classify_segments``.
    dips : list of dict
        The proto-plateau dips from ``detect_dips``.
    n_used : int
        Number of signal points used for the autocorrelation sweep.
    exclude_collapse : bool, optional
        If True, apply the SNR-gated collapse exclusion. Default False.
    c1 : float, optional
        Safety factor of the sub-fundamental clip. Default is 1.0 (the
        fundamental itself); see ``trim_candidates``.
    collapse_level : float, optional
        Relative level of the total drop below which a plateau is past
        the collapse, in [0, 1]. Only used when ``exclude_collapse``.
        Default is 0.5.
    sensitivity : array-like, shape (N,), optional
        The baseline-sensitivity curve. When given, the stiff-side
        instability exclusion of ``instability_boundary`` is applied on
        top of the others — note its thresholds are not grounded.
        Default is None, which disables that exclusion.

    Returns
    -------
    masks : dict of numpy.ndarray, dtype bool
        ``surviving`` — regions that survive every applied exclusion;
        ``removed`` — detected regions cut by the clip and frozen tail,
        and the proto-plateaus dropped as unnecessary by the fallback
        rule above;
        ``snr_removed`` — the extra cut by the collapse exclusion;
        ``instab_removed`` — the extra cut by the instability
        exclusion (all False when `sensitivity` is None).

    """
    masks = _trim_masks(fcut_range, segments, dips, n_used,
                        exclude_collapse, c1, collapse_level, sensitivity)
    if not dips:
        return masks
    # The dips are only wanted when the flat channel, put through the
    # same exclusions, has nothing left to offer.
    flat_only = _trim_masks(fcut_range, segments, [], n_used,
                            exclude_collapse, c1, collapse_level,
                            sensitivity)
    if not flat_only['surviving'].any():
        return masks

    # The flat channel suffices, so the surviving set is its own. The
    # removal masks, however, must still account for the WHOLE detected
    # selection: `flat_only` was computed without the dips, so on its
    # own it leaves the dip basins in no mask at all and they would be
    # drawn nowhere on the diagnostic. Attribute every detected point
    # that does not survive, whatever removed it.
    cp_flat = np.zeros(len(fcut_range), dtype=bool)
    for seg in segments:
        if seg['flat']:
            cp_flat[seg['start']:seg['end']] = True
    detected = cp_flat | dips_to_mask(fcut_range, dips)
    surviving = flat_only['surviving']
    snr_removed = flat_only['snr_removed']
    instab_removed = flat_only['instab_removed']
    removed = detected & ~surviving & ~snr_removed & ~instab_removed
    return {'surviving': surviving, 'removed': removed,
            'snr_removed': snr_removed, 'instab_removed': instab_removed}


def select_center(fcut_range: np.ndarray,
                  surviving: np.ndarray) -> float | None:
    """
    Cutoff frequency at the centre of the surviving plateau.

    Preliminary stage-3 selection: the surviving mask of
    ``trim_plateaus`` is reduced to a single cutoff by taking the centre
    of its region. The centre is **geometric** — the midpoint in
    log(fcut) — because the sweep grid is geometric and every position
    in this package is expressed in decades; an arithmetic mean of
    frequencies would be pulled to the high-frequency end of the region
    and is not the same point.

    The returned cutoff is a **grid point**: the midpoint is taken on
    the index axis, which on a geometric grid is the same thing up to
    half a step. That keeps the answer on a frequency the sweep actually
    evaluated, so its r2 can be read from the cached curve instead of
    costing another baseline fit, and the value reported is the same one
    the diagnostic plots rather than a re-fit that may differ slightly.
    The cost is at most half a grid step — 0.0024 decades on the
    production grid, against a valley where being 0.3 decades off costs
    about 1.8x the optimal baseline error.

    When several regions survive, the **last** one is used, per
    Navarro-Huerta: the optimum lies on the last step of the stepped
    ``y - b`` curve. On the 339-signal reference set the trimming now
    leaves exactly one region per signal, so this only matters for
    robustness.

    Note the departure from the reference: it places the optimum near
    the centre of that step but recommends **slightly below** the
    centre in practice, to reduce baseline flexibility. This takes the
    centre itself, which is the preliminary rule asked for; the offset
    is deliberately not invented here.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    surviving : array-like, shape (N,), dtype bool
        The surviving mask from ``trim_plateaus``.

    Returns
    -------
    fcut : float or None
        The selected cutoff frequency, or None when nothing survives.

    References
    ----------
    Navarro-Huerta et al. (2017), J. Chromatogr. A 1507, 1-10, §3.4.

    """
    idx = np.flatnonzero(surviving)
    if idx.size == 0:
        return None
    splits = np.where(np.diff(idx) > 1)[0] + 1
    region = np.split(idx, splits)[-1]
    centre = int(round(0.5 * (region[0] + region[-1])))
    return float(fcut_range[centre])
