# coding: utf-8
"""
Changepoint-based plateau detection for autocorrelation plots.

The autocorrelation curve is cut into contiguous segments by a penalized
piecewise-linear model (optimal partitioning), each segment is classified
by criteria expressed relative to the geometry of the curve itself, the
segments that cannot hold the answer are removed, and the cutoff
frequency is read off what survives. This is the production path of
``baseline.auto_beads``.

The method, its parameters and the questions still open are documented in
``tools/fcut/segmentation.md``.

References
----------
.. [1] Jackson, B. et al. An algorithm for optimal partitioning of data
   on an interval. IEEE Signal Process. Lett. 12(2), 105-108 (2005).
   doi:10.1109/LSP.2001.838216
.. [2] Killick, R., Fearnhead, P. and Eckley, I. A. Optimal detection
   of changepoints with a linear computational cost. J. Am. Stat.
   Assoc. 107(500), 1590-1598 (2012). doi:10.1080/01621459.2012.737745
.. [3] Schwarz, G. Estimating the dimension of a model. Ann. Statist.
   6(2), 461-464 (1978). doi:10.1214/aos/1176344136
.. [4] Truong, C., Oudre, L. and Vayatis, N. Selective review of
   offline change point detection methods. Signal Process. 167, 107299
   (2020). doi:10.1016/j.sigpro.2019.107299
.. [5] Navarro-Huerta, J. A. et al. Assisted baseline subtraction in
   complex chromatograms using the BEADS algorithm. J. Chromatogr. A
   1507, 1-10 (2017). doi:10.1016/j.chroma.2017.05.057
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
    log-likelihood cost with a linearly varying mean and a variance
    fitted separately on each segment, so a boundary is found at a change
    of slope and at a change of noise level alike.

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

    Notes
    -----
    The cumulative sums make each segment cost O(1), which is what
    keeps the exact partitioning affordable, and they are also where
    the accuracy goes: ``SSE`` is recovered by expanding
    ``sum((y - a - b t)^2)`` into moments that are each far larger than
    their difference. The relative error is around 1e-10 on the sizes
    this package sweeps, which is immaterial against a penalty of order
    ``log(N)``.

    **On a segment that is exactly straight the cost is set by rounding
    rather than by the data.** The true residual sum is zero, the
    expansion leaves a positive remainder some orders of magnitude
    above the ``1e-16`` clamp, and that remainder is what the logarithm
    sees. The clamp therefore does not govern this case, and the cost
    of a perfect fit is not reproducible across builds that round
    differently. Real curves carry noise, so this is reached by
    synthetic or already-fitted input rather than by a swept
    autocorrelation.

    **A one-point segment divides by zero**, the normal-equation
    determinant vanishing when the segment cannot define a slope.
    `pelt_linear` never asks for one, holding segments to `min_size`.

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
        """Cost of ``y[i:j]`` for every start `i`, from the cumsums."""
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
    Detect the changepoints of the data with a penalized
    piecewise-linear model.

    Exact optimal partitioning (dynamic programming) of the data into
    contiguous segments, each fitted by a straight line with its own
    residual variance. The number of segments is controlled by a single
    penalty charged per changepoint. Complexity is O(N^2), which is
    immediate for the typical N = 1000 of an autocorrelation plot.

    Parameters
    ----------
    y : array-like, shape (N,)
        The y-values of the data.
    penalty : float, optional
        Penalty charged for each changepoint. Default is None, which
        uses ``3 * log(N)``, the Schwarz penalty for this model: Killick
        et al. [2]_ give ``beta = p log n`` with `p` the parameters a
        changepoint adds, and a segment here carries a slope, an
        intercept and its own variance.
    min_size : int, optional
        Minimal length of a segment. Default is 15, about 1.5% of the
        1000-point grid the autocorrelation sweep produces.

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

    Notes
    -----
    The penalty is the price of a changepoint. Splitting a segment gains
    ``m*log(v) - m1*log(v1) - m2*log(v2)``, so what a changepoint can
    offer is its length times the per-point log improvement in residual
    variance, and it is kept only while that total clears the price.
    Killick et al. [2]_ state the criterion directly and their
    Theorem 3.2 (A4) writes the same balance as a condition on the mean
    segment length. Truong et al. [3]_ describe the consequence: a small
    penalty favours many regimes, a large one discards most
    changepoints.

    The recursion is the optimal partitioning of Jackson et al. [1]_
    §II: the best segmentation ending at ``j`` is the best ending at
    some earlier ``i``, plus the cost of ``y[i:j]``, plus the penalty.
    ``best[0]`` is seeded at ``-penalty`` so the first segment is not
    charged for a changepoint that does not exist.

    Timing is about a tenth of a second at ``N = 1000`` and a few
    seconds by ``N = 8000``, so the exact search is comfortable at
    sweep sizes.

    **The model is piecewise linear, so a curve that bends inside a
    segment is fitted by a chord.** Where the curvature is gradual
    there is no boundary in the data to find, and the partition places
    one where it most reduces the residual variance, which makes the
    segment edges a property of the penalty as much as of the curve.

    The minimum is exact over all partitions, not greedy. The pruning of
    Killick et al. [2]_ would bring the cost down to expected O(N), and
    ``ruptures`` [3]_ offers equivalent implementations; the pure-NumPy
    version here avoids the dependency and is fast enough at the sizes
    involved.

    References
    ----------
    .. [1] Jackson, B.; Scargle, J.D.; et al. An algorithm for optimal
           partitioning of data on an interval. IEEE Signal Processing
           Letters, 2005, 12(2), 105-108, §II (the recursion implemented
           here).
    .. [2] Killick, R.; Fearnhead, P.; Eckley, I.A. Optimal detection of
           changepoints with a linear computational cost. Journal of the
           American Statistical Association, 2012, 107(500), 1590-1598,
           §2 (``beta = p log n``, following Schwarz, G. The Annals of
           Statistics, 1978, 6(2), 461-464) and Theorem 3.2.
    .. [3] Truong, C.; Oudre, L.; Vayatis, N. Selective review of offline
           change point detection methods. Signal Processing, 2020, 167,
           107299. The reference to read first for the approach as a
           whole: cost functions, search methods, and the penalties that
           set the number of changepoints.

    """
    n = len(y)
    if n < 2 * min_size:
        raise ValueError('data must contain at least 2 * min_size points')
    if penalty is None:
        penalty = 3.0 * np.log(n)

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

    Notes
    -----
    ``rel_slope = 1`` is a segment that falls by the whole drop of the
    curve within one decade, which is the unit `cliff_min` names. A
    curve descending uniformly across the entire sweep sits at one over
    its span in decades.

    Slopes are fitted against the sample index rather than against
    `fcut`. On a geometric grid the index is proportional to the
    logarithm of the cutoff, so this is a slope per decade up to the
    constant that `points_per_decade` removes.

    **Both scales are global.** `drop` is the range of the whole curve,
    so a segment is judged against the total descent rather than
    against its neighbours. Where the descent is split across several
    cliffs, an intermediate shelf falls by a large share of the global
    scale and reads as steep even when it is flat beside the cliffs
    bounding it; the loose tier of `classify_segments` exists for that
    case.

    **A curve with no drop is clamped**, the divisor being held at
    1e-16, which sends the ratios up and leaves nothing marked flat.

    **A segment of a single point yields NaN features**, from a
    straight-line fit through one point, and NaN compares false against
    every threshold, so the segment is silently classified as not flat
    rather than raising. `pelt_linear` keeps segments to `min_size`, so
    this is reachable only by passing breakpoints from elsewhere.

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

    A segment is flat when its relative residual noise is below
    `rel_noise_max` and its relative slope passes a two-tier criterion:

    - tight: ``rel_slope < rel_slope_max``, flat on the scale of the
      whole curve;
    - loose: ``rel_slope < rel_slope_loose`` *and* the segment has at
      least one cliff (``rel_slope > cliff_min``) on each side. This
      accepts the drifting shelves of staircase curves, where the total
      drop is split over several steps so every shelf is steep on the
      global scale.

    The residual-noise criterion separates quiet plateaus from the
    regions where the baseline fit is unstable.

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

    Notes
    -----
    Both criteria are ratios against the total drop of the curve, so
    they carry no units and do not depend on the grid density or the
    signal length. They fail on a curve with no drop: `segment_features`
    clamps the divisor, the ratios become large, and no segment is
    marked flat.

    **None of the four thresholds is grounded.** Each cuts a continuous
    quantity where no gap in the distribution marks a boundary, so the
    dividing point is chosen rather than found.

    A significance or likelihood criterion would not settle them
    either. The curve carries almost no independent noise at the grid
    scale, so a segment's residual is leftover curvature rather than
    sampling scatter, and a test built on a null distribution answers a
    question the data cannot pose. Flatness here is geometric.

    The noise criterion is ANDed with the slope criterion rather than
    folded into it, so it can reject a segment that either tier
    accepts.

    The first and the last segment can never qualify under the loose
    tier, which requires a cliff on each side and finds nothing beyond
    them.

    The loose tier changes the answer rarely, and on the staircase
    morphologies it was introduced for it may not fire at all.

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
    Curve read by the proto-plateau detector.

    The rolling standard deviation of `r2`, Gaussian-smoothed and scaled
    to its own maximum. ``detect_dips`` marks proto-plateaus at its local
    minima. Public so the diagnostic overlay can draw the same array the
    detector reads.

    Parameters
    ----------
    r2 : array-like, shape (N,)
        The autocorrelation coefficients.
    window : int, optional
        Window of the rolling standard deviation. Default is 3.
    sigma : float, optional
        Standard deviation of the Gaussian smoothing, in grid points.
        Default is 8.0.

    Returns
    -------
    curve : numpy.ndarray, shape (N,)
        The normalised curve, in [0, 1]; all zeros if `r2` is constant.

    Notes
    -----
    `sigma` is a width in grid points, so what it means on the cutoff
    axis depends on how densely the sweep samples the range: the same
    value smooths half as much of the axis if the grid is doubled. It
    is the one constant in this chain tied to grid density, where the
    classification thresholds are ratios and are not.

    The normalisation divides by the largest value anywhere on the
    curve, including the rebound past the collapse, which
    ``detect_dips`` does not search. A tall feature outside the
    searched range therefore compresses every prominence inside it.

    The rolling standard deviation is computed over a centred window,
    so its first and last samples come from fewer points and read high;
    the smoothing spreads that edge effect inward.

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
    Where ``classify_segments`` asks whether a segment is flat on the
    scale of the whole curve, this asks whether it is flatter than its
    neighbours, so it catches shelves the segment classifier rejects.

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
        applied to the rolling standard deviation before the valleys
        are located. What it covers on the cutoff axis depends on the
        grid density; see `dip_curve`. Default is 8.0.
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
    The prominence and level thresholds are dimensionless, prominence
    relative to the largest cliff and level as a fraction of the drop.
    `sigma` is not: it is a count of grid points, so it spans a
    different stretch of the cutoff axis at a different sweep density.
    All of them were set by visual validation rather than derived, and
    no source fixes them.

    ``trim_plateaus`` admits these dips wherever the flat channel
    survives nothing, and on those signals the parameters below decide
    the cutoff.

    Only the descent is searched, up to the global minimum of `r2`, so
    a valley whose floor lies at or past that minimum is not found and
    one straddling it is truncated. Nothing is returned when the
    minimum falls within the first two grid points, or when the curve
    has no drop.

    Prominences are measured on the curve `dip_curve` normalises by its
    largest value anywhere, including the unsearched tail, so a tall
    feature outside the search range raises the bar inside it.

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

    Notes
    -----
    **A dip's ``end`` is inclusive, where a segment's ``end`` is
    exclusive.** ``detect_dips`` clamps it to the last valid index and
    this function slices with ``end + 1``, while the segments of
    ``segment_features`` are sliced as ``[start:end]`` throughout. The
    two dictionaries look alike and are indexed differently.

    Overlapping dips are absorbed into one another, since the mask
    records only whether a grid point is covered.

    """
    mask = np.zeros(len(fcut_range), dtype=bool)
    for dip in dips:
        mask[dip['start']:dip['end'] + 1] = True
    return mask


def collapse_floor(segments: list[dict]) -> int:
    """
    Index at which the collapse floor begins.

    The floor is the segment of lowest mean r2, the bottom of the
    collapse. Everything at a higher cutoff is past it, on the rebound
    where r2 climbs back towards Nyquist while the baseline is fitting
    the noise; a cutoff there is inadmissible whatever the signal, which
    is why the clip built on this index carries no options.

    The segment mean locates the floor rather than the raw minimum of
    r2, because the segments are all this function is given. The two
    need not coincide: a mean taken over a sloping segment can fall
    below the mean of the segment that actually contains the minimum,
    and the floor is then placed at the start of the sloping one, which
    may lie below the minimum and clip candidates that are admissible.

    Parameters
    ----------
    segments : list of dict
        The segments returned by ``segment_features``.

    Returns
    -------
    index : int
        Start index of the lowest-mean segment, or a large value when
        `segments` is empty, so that slicing with it clips nothing.

    Warnings
    --------
    Only meaningful on a curve that collapses inside the swept range.
    Two degenerate cases return a sentinel that clips nothing rather
    than a floor: an empty segment list, and a lowest-mean segment that
    is the first one. The latter covers the curve the segmentation sees
    as a single piece, whose only segment starts at index 0; clipping
    from there would remove every candidate on the curve.

    On a curve that is flat throughout, the lowest mean is picked among
    near-identical values and the index is arbitrary. Such a curve has
    no plateau structure to select from in the first place, but the
    arbitrariness is real and this function cannot detect it.

    """
    if not segments:
        return np.iinfo(np.intp).max
    means = [seg['mean'] for seg in segments]
    k = int(np.argmin(means))
    if k == 0:
        return np.iinfo(np.intp).max
    return segments[k]['start']


def trim_candidates(fcut_range: np.ndarray, segments: list[dict],
                    n_used: int, c1: float = 1.0,
                    cliff_min: float = 1.0,
                    bridge: bool = False,
                    exclude_past_drop: bool = False,
                    drop_level: float = 0.5) -> np.ndarray:
    """
    Trim the flat segments into a mask of candidate plateau regions.

    Reduces the flat set to the regions where the optimal cutoff
    frequency can lie, using only a-priori exclusions:

    - **sub-fundamental clip**: grid points below ``c1 / n_used`` are
      removed. The slowest oscillation representable on a signal of
      `n_used` points has frequency ``1/n_used``, so every cutoff below
      it requests a baseline the data cannot constrain and returns an
      overly rigid one. No cutoff in that region is admissible. This is
      why the initial plateau of the autocorrelation plot always ends
      at ``~1/n_used``.
    - **rebound clip**: grid points at or above the collapse floor, the
      segment of lowest mean r2, are removed. Past the floor the
      baseline has absorbed the analyte-correlated content and r2
      climbing back towards Nyquist reports on the noise it is now
      fitting, not on the quality of the fit. No property of the signal
      makes a cutoff there admissible, so unlike the past-drop exclusion
      below this clip is unconditional.
    - **bridging** (optional): a non-flat segment sandwiched between
      candidate regions is absorbed when it is not a cliff
      (``rel_slope < cliff_min``), so drifting connectors do not split
      one plateau into several displayed pieces while genuine staircase
      steps still separate regions.
    - **past-drop exclusion** (optional): when ``exclude_past_drop`` is
      set, flat segments lying below ``drop_level`` of the way up
      the total r2 drop are removed. Approaching the collapse the
      baseline has begun absorbing the analyte-correlated content, so
      for a signal that carries analyte a cutoff there destroys peak
      area and the optimum cannot lie in it (Navarro-Huerta et al.
      2017, §3.4). This reaches the descent *before* the floor, which the
      rebound clip does not; the two are complementary and neither
      subsumes the other. Whether a signal carries analyte is a property
      of the signal, not of the curve, so the caller decides
      ``exclude_past_drop`` from the signal-to-noise ratio. A weak signal
      leaves it off, and its low shelves survive as legitimate
      candidates.

    Bridging is off by default. See ``bridge`` below.

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
        itself: below it the data cannot determine the baseline at all.
    cliff_min : float, optional
        Minimum relative slope of a segment acting as a real separation
        between candidate regions. Default is 1.0.
    bridge : bool, optional
        If True, absorb non-cliff segments lying between candidate
        regions. Default is False, which is what the production path
        uses: on a gentle descent, bridging merges a plateau and a lower
        shelf into one region and the sampling then lands on the descent
        between them.
    exclude_past_drop : bool, optional
        If True, remove flat segments past the drop (see the summary
        above). The caller sets this from the signal-to-noise ratio.
        Default is False.
    drop_level : float, optional
        Relative level of the total r2 drop below which a flat segment
        is considered past the drop, in [0, 1] (1 = plateau top,
        0 = collapse floor). Only used when ``exclude_past_drop``.
        Default is 0.5.

    Returns
    -------
    candidates : numpy.ndarray, shape (N,), dtype bool
        Boolean mask of the grid points belonging to a candidate
        plateau region.

    Notes
    -----
    **``drop_level`` names two different scales.** Here it is
    compared against the level of a flat segment on the total r2 drop.
    The same value is also compared against ``dip['level']`` from
    ``detect_dips``, which is computed from the raw curve rather than
    from segment means. One number, two references.

    """
    cand = [seg['flat'] for seg in segments]
    if exclude_past_drop and segments:
        means = [seg['mean'] for seg in segments]
        r2_max, r2_min = max(means), min(means)
        thr = r2_min + drop_level * (r2_max - r2_min)
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
    candidates[collapse_floor(segments):] = False
    return candidates


def sensitivity_dispersion(fcut_range: np.ndarray, sensitivity: np.ndarray,
                         win_dec: float = 0.2) -> np.ndarray:
    """
    Local dispersion of the baseline-sensitivity curve.

    The interquartile range of `sensitivity` inside a sliding window of
    `win_dec` decades of cutoff frequency.

    Where the fit is undetermined the baseline swings between adjacent
    cutoffs, so the sensitivity values scatter. Dispersion distinguishes
    that from a baseline moving steadily, which is what happens on the
    flexible side approaching the collapse: there the values are high but
    tightly ordered. A quantile range keeps the excursions at the level
    they belong to, where a standard deviation or a smoothing kernel
    would spread them into their neighbours.

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

    Notes
    -----
    `win_dec` is converted through the grid's own points per decade, so
    the window covers the same stretch of the cutoff axis whatever the
    sweep density. The width is forced odd and to at least five points.

    **An interquartile range cannot see a small number of large
    excursions.** A quarter of the window may be arbitrary without
    moving it, which is what makes it robust to a steadily moving
    baseline, and also what makes it blind to a handful of isolated
    swings. If the instability being looked for is sparse rather than
    sustained, this statistic will not report it.

    The two ends are padded by repeating the edge value, so the
    dispersion is damped over the first and last half-window.

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

    When the signal's fundamental falls inside a region where the
    baseline is flailing, that region is unusable up to the point where
    the oscillations become small again. The test is made at the
    fundamental itself, so it is local to the signal and needs no prior
    classification of the curve.

    ``trim_candidates`` already clips below the fundamental, where the
    data cannot constrain the baseline at all. This extends the exclusion
    upward, over frequencies the clip cannot reach.

    Warnings
    --------
    `trigger` and `settled` are **not grounded**. They are amplitudes of
    the sensitivity curve, which is dimensionless (rms baseline change as
    a fraction of the signal range, per decade), so they read as
    statements about tolerable baseline movement, but no source fixes
    where that tolerance lies. They were adopted provisionally from a
    review of a signal collection. `settled` is the sensitive one,
    setting how far the exclusion reaches; `trigger` only changes how
    many signals it touches. See the README TO DO.

    This exclusion is also the least reproducible stage of the chain
    across library versions, because it reads the sensitivity curve at
    low cutoff where the fit is ill-conditioned.

    Because the test is an interquartile range, it answers to sustained
    scatter and not to isolated swings; see `sensitivity_dispersion`.

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
                dips: list[dict], n_used: int, exclude_past_drop: bool,
                c1: float, drop_level: float,
                sensitivity: np.ndarray | None) -> dict[str, np.ndarray]:
    """
    Apply the stage-1 exclusions to a given detected selection.

    Separate from ``trim_plateaus`` so the flat channel can be run
    through the identical chain on its own, which is what decides whether
    the proto-plateaus are needed at all.

    Parameters
    ----------
    fcut_range : array-like, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    segments : list of dict
        The classified segments from ``classify_segments``.
    dips : list of dict
        The proto-plateau dips from ``detect_dips``; pass ``[]`` to run
        the flat channel alone.
    n_used : int
        Number of signal points used for the autocorrelation sweep.
    exclude_past_drop : bool
        Whether to apply the past-drop exclusion.
    c1 : float
        Safety factor of the sub-fundamental clip.
    drop_level : float
        Level of the total drop below which a plateau is past the
        collapse, in [0, 1].
    sensitivity : array-like, shape (N,), or None
        The baseline-sensitivity curve; None disables the instability
        exclusion.

    Returns
    -------
    masks : dict of numpy.ndarray, dtype bool
        Keys ``surviving``, ``removed``, ``snr_removed`` and
        ``instab_removed``, as documented on ``trim_plateaus``.

    """
    cp_flat = np.zeros(len(fcut_range), dtype=bool)
    for seg in segments:
        if seg['flat']:
            cp_flat[seg['start']:seg['end']] = True
    cp_dips = dips_to_mask(fcut_range, dips)
    union = cp_flat | cp_dips
    sub_fund = fcut_range >= c1 / n_used
    # The rebound clip applies to the proto-plateaus too. `trim_candidates`
    # only ever sees the flat segments, so the dip channel is clipped here
    # rather than there.
    pre_floor = np.ones(len(fcut_range), dtype=bool)
    pre_floor[collapse_floor(segments):] = False
    cp_dips = cp_dips & pre_floor

    # No bridging: bridging absorbs the non-cliff connector between two
    # candidate regions, which on a gentle descent merges a
    # plateau and a lower shelf into one region, so the sampling then
    # lands on the descent *between* the plateaus. The surviving set must
    # keep the plateaus separate.
    flat_12 = trim_candidates(fcut_range, segments, n_used, c1=c1,
                              bridge=False, exclude_past_drop=False)
    trimmed_12 = flat_12 | (cp_dips & sub_fund)
    removed = union & ~trimmed_12

    # Stiff-side instability boundary. Independent of the collapse
    # exclusion, so it is computed once and reused by both branches
    # below.
    boundary = (None if sensitivity is None
                else instability_boundary(fcut_range, sensitivity, n_used))

    def _with_past_drop(active: bool) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
        """Surviving set and its two removal masks, collapse on or off."""
        if active:
            flat_123 = trim_candidates(fcut_range, segments, n_used, c1=c1,
                                       bridge=False, exclude_past_drop=True,
                                       drop_level=drop_level)
            dips_123 = dips_to_mask(
                fcut_range,
                [d for d in dips if d['level'] >= drop_level])
            kept = flat_123 | (dips_123 & sub_fund)
        else:
            kept = trimmed_12
        snr_rm = trimmed_12 & ~kept
        # The instability exclusion is applied last, on whatever has
        # survived: it removes the frequencies at which the fit is still
        # undetermined above the fundamental, which the sub-fundamental
        # clip cannot reach.
        instab_rm = np.zeros(len(fcut_range), dtype=bool)
        if boundary is not None:
            surv = kept & (fcut_range > boundary)
            instab_rm = kept & ~surv
            kept = surv
        return kept, snr_rm, instab_rm

    surviving, snr_removed, instab_removed = _with_past_drop(exclude_past_drop)
    # The past-drop exclusion narrows the choice among surviving regions;
    # it is not a veto on selecting at all. When the sub-fundamental
    # clip and the instability boundary have already removed
    # everything outside it, applying it as well leaves nothing
    # and the signal yields no cutoff, which is a worse answer than the
    # one it was refining. Fall back to the set without it in that case.
    if exclude_past_drop and not surviving.any():
        alt, alt_snr, alt_instab = _with_past_drop(False)
        if alt.any():
            surviving, snr_removed, instab_removed = (alt, alt_snr,
                                                      alt_instab)

    return {'surviving': surviving, 'removed': removed,
            'snr_removed': snr_removed, 'instab_removed': instab_removed}


def trim_plateaus(fcut_range: np.ndarray, segments: list[dict],
                  dips: list[dict], n_used: int,
                  exclude_past_drop: bool = False, c1: float = 1.0,
                  drop_level: float = 0.5,
                  sensitivity: np.ndarray | None = None
                  ) -> dict[str, np.ndarray]:
    """
    Stage-1 trimming of the detected plateau selection.

    The detected selection is the union of the flat set
    (``classify_segments``, the strong and initial plateaus) and the
    proto-plateau basins (``detect_dips``, the relative flattenings the
    flat test misses). This applies the a-priori exclusions to it:

    - the sub-fundamental clip, always;
    - the SNR-gated past-drop exclusion, only when ``exclude_past_drop``.
      Past the collapse a cutoff destroys analyte peak area
      (Navarro-Huerta et al. 2017 §3.4), so on a signal carrying analyte
      the plateaus below ``drop_level`` of the drop are removed,
      while a weak signal keeps them. Whether a signal carries analyte is a
      property of the signal, not of the curve, so the caller decides it
      from the signal-to-noise ratio;
    - the stiff-side instability exclusion, when `sensitivity` is given.

    The proto-plateaus are a **fallback**: the flat channel is run
    through the identical chain on its own, and the dips contribute only
    where it leaves nothing. Unioned unconditionally they also fire on
    the descent past the drop, where an easing of the slope is a
    local minimum of the rolling standard deviation while still an order
    of magnitude steeper than a plateau. The rule keeps the dips only
    for signals with no flat region, and drops them everywhere else.

    Applying the past-drop exclusion never leaves the caller with nothing:
    if it would empty the surviving set, the set without it is used
    instead. No cutoff at all is a worse answer than a poor one.

    Both the diagnostic overlay and the fcut gallery read the trimming
    from here, so the surviving regions are the same in both.

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
    exclude_past_drop : bool, optional
        If True, apply the SNR-gated past-drop exclusion. Default False.
    c1 : float, optional
        Safety factor of the sub-fundamental clip. Default is 1.0 (the
        fundamental itself); see ``trim_candidates``.
    drop_level : float, optional
        Relative level of the total drop below which a plateau is past
        the collapse, in [0, 1]. Only used when ``exclude_past_drop``.
        Default is 0.5.
    sensitivity : array-like, shape (N,), optional
        The baseline-sensitivity curve. When given, the stiff-side
        instability exclusion of ``instability_boundary`` is applied on
        top of the others; its thresholds are not grounded. Default is
        None, which disables that exclusion.

    Returns
    -------
    masks : dict of numpy.ndarray, dtype bool
        ``surviving``, the regions surviving every applied exclusion;
        ``removed``, the detected regions cut by the sub-fundamental
        clip, together with the proto-plateaus the fallback rule dropped
        as unnecessary; ``snr_removed``, the extra cut by the collapse
        exclusion; and ``instab_removed``, the extra cut by the
        instability exclusion, all False when `sensitivity` is None.

    Notes
    -----
    The past-drop exclusion is gated on the signal-to-noise ratio, and
    that gate discriminates only where the population straddles the
    threshold. On data whose values sit far above it the exclusion is
    applied to everything and the gate is a constant, which also means
    such data cannot validate the threshold. See ``baseline._snr``.

    """
    masks = _trim_masks(fcut_range, segments, dips, n_used,
                        exclude_past_drop, c1, drop_level, sensitivity)
    if not dips:
        return masks
    # The dips are only wanted when the flat channel, put through the
    # same exclusions, has nothing left to offer.
    flat_only = _trim_masks(fcut_range, segments, [], n_used,
                            exclude_past_drop, c1, drop_level,
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

    Preliminary stage-3 selection. The centre is geometric, the midpoint
    in log(fcut), matching the geometric sweep grid and the decades every
    position in this package is expressed in. It is taken on the index
    axis, so the answer is a grid point the sweep evaluated and its r2
    can be read from the cached curve; the snap costs at most half a
    grid step, on a grid of a thousand points.

    Where several regions survive, the last one is used, following
    Navarro-Huerta et al. (2017) §3.4: the optimum lies on the last step
    of the stepped ``y - b`` curve.

    Warnings
    --------
    **The 0.5 is a placeholder.** It is the value that sits equally
    well against both ends of the plateau, which is what recommends it
    while the boundaries themselves still move. Navarro-Huerta et al.
    (2017) §3.4 advise a point between the beginning and the centre of
    the region, so the midpoint is the far end of their range, and
    against baselines that are known exactly the optimum sits well
    inside the region rather than at its midpoint.

    Where it sits is a validation target, not a rule: substituting one
    fixed fraction for another leaves it just as arbitrary. The
    replacement has to come from a feature of the signal or of the
    curves. The r2 level is an unlikely candidate, since it barely moves
    between the selected and the optimal cutoff while the baseline error
    moves substantially.

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
