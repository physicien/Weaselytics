# coding: utf-8
"""
Helper functions to perform various signal preprocessing operations.

Peak detection and width measurement, the Durbin-Watson autocorrelation
the cutoff sweep is read on, interval merging, and the endpoint-outlier
prototype that preceded pybaselines issue #70.

References
----------
.. [1] Durbin, J. and Watson, G. S. Testing for serial correlation in
   least squares regression. I. Biometrika 37(3/4), 409-428 (1950).
   doi:10.1093/biomet/37.3-4.409
.. [2] Durbin, J. and Watson, G. S. Testing for serial correlation in
   least squares regression. III. Biometrika 58(1), 1-19 (1971).
   doi:10.1093/biomet/58.1.1
.. [3] Navarro-Huerta, J. A. et al. Assisted baseline subtraction in
   complex chromatograms using the BEADS algorithm. J. Chromatogr. A
   1507, 1-10 (2017). doi:10.1016/j.chroma.2017.05.057
.. [4] pybaselines issue #70, on rejecting an outlying endpoint before
   the parabola fit. https://github.com/derb12/pybaselines/issues/70
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter


def end_window(data: np.ndarray, window_min: int = 3,
               window_max: int = 20) -> int:
    """
    Length of the window at each end of a signal, scaled to its length.

    One percent of the signal length, clamped to
    ``[window_min, window_max]``.

    Parameters
    ----------
    data : numpy.ndarray
        The signal the window is sized for. Only its length is used.
    window_min : int, optional
        Lower clamp. Default is 3.
    window_max : int, optional
        Upper clamp. Default is 20.

    Returns
    -------
    size : int
        The window length in samples.

    Notes
    -----
    The one percent and the two clamps are ungrounded.

    This is the window-sizing half of the endpoint-outlier prototype
    that preceded pybaselines issue #70, kept because its behaviour is
    not the one that was implemented upstream. Production passes the
    result as ``parabola_len``, so the length-scaled window here is
    combined with pybaselines' own criterion, which compares an
    endpoint against two standardized median absolute deviations of its
    edge rather than against a fraction of the signal range.

    Reached only when a caller passes ``parabola_len=None`` to
    `auto_beads`; the shipped default is 3, pybaselines' own.

    """
    size = int(round(0.01*len(data)))
    if size < window_min:
        size = window_min
    if size > window_max:
        size = window_max
    return size


def rm_ends_outliers(data: np.ndarray, window_min: int = 5,
                     window_max: int = 100) -> np.ndarray:
    """
    Replace an outlying first or last point with its local median.

    **Not called anywhere in the package.** Kept as the full prototype
    of the endpoint-outlier handling that preceded pybaselines issue
    #70, whose criterion differs from what was implemented upstream and
    may still be wanted: the bar here is a fraction of the signal's
    full range, where pybaselines compares an endpoint against two
    standardized median absolute deviations of its own edge. A peak
    anywhere in the signal therefore raises this bar and can leave a
    bad endpoint in place, which is the behaviour to weigh before
    reaching for it. `end_window` is its window-sizing half and is
    live.

    An endpoint counts as an outlier when it departs from the median of
    its own window by more than one percent of the signal's full range.
    The window is one percent of the signal length, clamped to
    ``[window_min, window_max]``.

    Parameters
    ----------
    data : numpy.ndarray
        The signal to be tested.
    window_min : int, optional
        Lower clamp on the window. Default is 5.
    window_max : int, optional
        Upper clamp on the window. Default is 100.

    Returns
    -------
    s : numpy.ndarray
        A copy of `data` with either endpoint replaced where it was
        found to be an outlier.

    Notes
    -----
    The one percent used for the window, the two clamps, and the one
    percent used for the criterion are ungrounded.

    Only the single first and last samples are tested, so a run of bad
    samples at either end is left in place, and a bad sample just
    inside the endpoint enters the median it is compared against.

    """
    s = np.copy(data)
    size = round(0.01*len(data))
    if size < window_min:
        size = window_min
    if size > window_max:
        size = window_max
    ymax = 0.01*np.abs(np.max(data)-np.min(data))
    y0_med = np.median(data[:size])
    diff0 = np.abs(data[0]-y0_med)
    y1_med = np.median(data[-size:])
    diff1 = np.abs(data[-1]-y1_med)

    if diff0 > ymax:
        s[0] = y0_med
    if diff1 > ymax:
        s[-1] = y1_med
    return s

def _durbin_watson(resids: np.ndarray, axis: int = 0) -> np.ndarray:
    r"""
    Durbin-Watson statistic of a series.

    Parameters
    ----------
    resids : array-like
        Series to measure. Named for the least squares residuals the
        statistic was built for; here it is a baseline-corrected
        chromatogram.
    axis : int, optional
        Axis to reduce when the input has more than one dimension.
        Default is 0.

    Returns
    -------
    dw : float or array-like
        The statistic, between 0 and 4.

    Notes
    -----
    Durbin and Watson [1]_ §4 p. 424 adopt

    .. math::

       d = \sum_{i=2}^{n}(z_i - z_{i-1})^2 \Big/ \sum_{i=1}^{n} z_i^2

    with ``z`` the residuals from a least squares fit (§2 p. 411). The
    value is approximately ``2 (1 - r)`` with ``r`` the lag-1
    autocorrelation: unrelated neighbours give 2, strong positive
    correlation drives it towards 0, strong negative correlation
    towards 4. The upper bound follows from
    ``(a - b)^2 <= 2a^2 + 2b^2``. It is read here as a descriptive
    measure of how much correlated structure a residual still carries.

    **Fails on a series that is not centred on zero.** The denominator
    sums squares about zero, so an offset inflates it and drives the
    statistic down: white noise displaced by two and a half times its
    own standard deviation reads about 0.3, the value of a strongly
    correlated series. Durbin and Watson assume throughout that the
    design matrix carries a column of ones ([2]_ p. 1), which is to say
    that a constant has been fitted and the residuals are centred. A
    channel that can carry an offset, such as a corrected signal whose
    fit does not bisect the noise, is read as more structured than it
    is.

    The expression follows the implementation in
    ``statsmodels.stats.stattools``.

    References
    ----------
    .. [1] Durbin, J. and Watson, G. S. Testing for serial correlation
       in least squares regression. I. *Biometrika* **37**, 409-428
       (1950).
    .. [2] Durbin, J. and Watson, G. S. Testing for serial correlation
       in least squares regression. III. *Biometrika* **58**, 1-19
       (1971).
    """
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids**2, axis=axis) / np.sum(resids**2, axis=axis)
    return dw

def r2_dw(s: np.ndarray) -> float:
    r"""
    Squared lag-1 autocorrelation of a series, from its Durbin-Watson
    statistic.

    This is the quantity the cutoff sweep is read on: near 1 where the
    residual still carries correlated structure, near 0 where it has
    been reduced to white noise.

    Parameters
    ----------
    s : array-like
        Series to measure, in practice a baseline-corrected
        chromatogram.

    Returns
    -------
    r2 : float
        The squared autocorrelation level, between 0 and 1.

    Notes
    -----
    Navarro-Huerta et al. [1]_ §3.2 Eq. (5) give

    .. math::

       r^2 \approx (2 - DW)^2 / 4

    which follows from their Eq. (4), ``DW = 2 - 2r``. They take the
    square as the more convenient measure in practice.

    **The approximation is exact only when the first and last points of
    the series match**, ``d_1 = d_n`` ([1]_ §3.2). Otherwise a boundary
    term is dropped and the value carries that error.

    **Squaring discards the sign.** ``DW`` runs from 0 at perfect
    positive correlation to 4 at perfect negative correlation, and both
    ends map to ``r2 = 1``, so a high value does not say which kind of
    structure remains.

    Undefined for an all-zero series, where the Durbin-Watson
    denominator vanishes ([1]_ §3.2).

    Inherits the centring assumption of `_durbin_watson`: a series with
    a non-zero mean is read as more correlated than it is.

    References
    ----------
    .. [1] Navarro-Huerta, J. A., Torres-Lapasió, J. R., López-Ureña,
       S. and García-Alvarez-Coque, M. C. Assisted baseline subtraction
       in complex chromatograms using the BEADS algorithm.
       *J. Chromatogr. A* **1507**, 1-10 (2017).
    """
    r2 = ((2-_durbin_watson(s))**2)/4
    return r2

def smooth_SG(x: np.ndarray, window_length: int,
              polyorder: int) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter to an array.

    **Slated for removal; see the TO DO in the README.** The wrapper is
    an artefact of the abandoned derivative approach, when smoothing
    the signal was thought to help. Its one production caller passes
    ``polyorder=0``, which fits a constant to each window and so is a
    plain moving average rather than the polynomial filter the name
    promises.

    Parameters
    ----------
    x : array-like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int
        The order of the polynomial used to fit the samples. `polyorder` must
        be less than `window_length`.

    Returns
    -------
    smooth_data : ndarray, same shape as x
        The filtered data.

    """
    smooth_data = savgol_filter(x,window_length,polyorder)
    return smooth_data

def peaks_params(s: np.ndarray, rel_prom_p: float = 0.05,
                 rel_prom_n: float = 0.8,
                 rel_height_p: float = 0.5, rel_height_n: float = 0.5,
                 width: int | None = None,
                 adapt: bool = False,
                 drop_enclosing: bool = False
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate every peak of a signal, positive and negative, with its
    width.

    Positive and negative features are detected separately, each gated
    on prominence relative to the strongest feature of its own sign, so
    the threshold does not depend on where the baseline sits.

    Parameters
    ----------
    s : numpy.ndarray
        A signal with peaks.
    rel_prom_p : float, optional
        Prominence a positive peak needs, as a fraction of the largest
        positive prominence in `s`. Default is 0.05. Overridden by
        `adapt`; see Notes.
    rel_prom_n : float, optional
        Prominence a negative peak needs, as a fraction of the deepest
        negative prominence in `s`. Default is 0.8.
    rel_height_p, rel_height_n : float, optional
        Height at which a peak's width is measured, as a fraction of
        its prominence: 1.0 gives the width at the lowest contour
        level, 0.5 at half the prominence. Default is 0.5 for both,
        which is ``scipy.signal.peak_widths``' own default, so every
        width returned here is a half-prominence width unless a caller
        says otherwise.
    width : number or ndarray or sequence, optional
        Minimum peak width in samples, or a ``(min, max)`` pair, passed
        to ``scipy.signal.find_peaks``. Default is `None`.
    adapt : bool, optional
        If True, set `rel_prom_p` from the signal instead of using the
        value passed; see Notes. Default is False.
    drop_enclosing : bool, optional
        If True, discard any feature whose width encloses a taller
        peak. See `_drop_enclosing`. Default is False.

    Returns
    -------
    peaks : numpy.ndarray
        Indices of the peaks satisfying every condition, sorted.
    widths : numpy.ndarray
        Their widths in samples, in the same order.

    Notes
    -----
    **The `adapt` ladder.** With ``adapt=True`` the fraction is taken
    from the largest positive prominence ``m`` in the signal: 0.5 for
    ``m <= 1``, 0.08 for ``m <= 2.5``, five times the passed value for
    ``m <= 10``, and the passed value above that. A signal whose
    tallest feature is small must therefore clear a larger share of it.
    Nothing fixes the three fractions or the two breakpoints, and the
    ladder is to be investigated.

    **The ladder is discontinuous and not monotonic.** The absolute bar
    is the fraction times ``m``, while the fraction changes in steps,
    so two signals whose tallest features differ negligibly can be
    gated very differently: crossing ``m = 1`` the bar falls by about
    six, and crossing ``m = 10`` it falls by five. Detection can change
    abruptly between signals that look alike.

    **A signal with no genuine negative peak still yields one, and
    this should be treated as a defect rather than a limitation.** The
    negative gate is a fraction of the deepest negative prominence
    present, so when no real negative excursion exists the reference
    becomes the noise floor and the deepest noise dip clears the bar. A
    caller that takes a returned negative peak to be real will act on
    noise.

    Detection and width measurement are ``scipy.signal.find_peaks`` and
    ``scipy.signal.peak_widths``.

    """
    _, raw_params_p = find_peaks(s,prominence=0.0)
    _, raw_params_n = find_peaks(-s,prominence=0.0)
    max_prom_p = (raw_params_p["prominences"].max()
                  if len(raw_params_p["prominences"]) > 0 else 0.0)
    max_prom_n = (raw_params_n["prominences"].max()
                  if len(raw_params_n["prominences"]) > 0 else 0.0)
    if adapt:
        if max_prom_p <= 1:
            rel_prom_p = 0.5
        elif max_prom_p <= 2.5:
            rel_prom_p = 0.08
        elif max_prom_p <= 10.0:
            rel_prom_p = 5*rel_prom_p
    prom_p = rel_prom_p * max_prom_p
    prom_n = rel_prom_n * max_prom_n

    peaks_p, _ = find_peaks(s, prominence=prom_p, width=width)
    peaks_n, _ = find_peaks(-s, prominence=prom_n, width=width)
    widths_p = peak_widths(s, peaks_p, rel_height=rel_height_p)[0]
    widths_n = peak_widths(-s, peaks_n, rel_height=rel_height_n)[0]

    unsorted_peaks = np.append(peaks_p, peaks_n)
    unsorted_widths = np.append(widths_p, widths_n)
    index_array = np.argsort(unsorted_peaks)

    peaks = unsorted_peaks[index_array]
    widths = unsorted_widths[index_array]
    if drop_enclosing and len(peaks) > 1:
        abs_dev = np.abs(s[peaks] - np.median(s))
        peaks, widths = _drop_enclosing(peaks, widths, abs_dev)
    return peaks, widths


def _drop_enclosing(peaks: np.ndarray, widths: np.ndarray,
                    abs_dev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Discard features that enclose a taller peak.

    A chromatographic peak can carry a smaller shoulder inside its own
    half-width, but never a taller peak. Structure that does, a drift
    feature with peaks on it or a gap between two peaks, is not a peak,
    and the peaks it encloses are what should define the region.

    Compared on deviation about the signal median rather than on
    prominence: a valley inherits the prominence of the peaks bounding
    it, so a shallow gap between two tall peaks outranks them.

    Parameters
    ----------
    peaks : array-like, shape (M,)
        Peak indices, sorted.
    widths : array-like, shape (M,)
        Their widths in samples.
    abs_dev : array-like, shape (M,)
        Their absolute deviations from the signal median, in the same
        order.

    Returns
    -------
    peaks, widths : numpy.ndarray
        The surviving peaks and widths.

    Notes
    -----
    A feature encloses another when the other's index falls strictly
    inside ``peaks +- widths / 2``. The deviations are magnitudes, so a
    deep negative excursion can outrank a positive peak and remove the
    feature that contains it.

    **Every verdict is decided in one pass, against the deviations of
    all the features given, so a feature that is itself discarded can
    still cause a discard.** Given three nested features of rising
    deviation, the outermost goes because the middle one is larger and
    inside it, while the middle one goes in the same pass because the
    innermost is larger and inside that. The outermost is therefore
    judged against something the function has already rejected, and
    only the innermost survives.

    """
    lo = peaks - widths / 2.
    hi = peaks + widths / 2.
    inside = ((peaks[None, :] > lo[:, None])
              & (peaks[None, :] < hi[:, None]))
    np.fill_diagonal(inside, False)
    taller = abs_dev[None, :] > abs_dev[:, None]
    keep = ~(inside & taller).any(axis=1)
    return peaks[keep], widths[keep]


def merge_intervals(intervals: list[list[int]]) -> np.ndarray:
    """
    Merge overlapping intervals.

    Parameters
    ----------
    intervals : array-like, shape (N,2)
        Start and stop indices of each interval. Need not be sorted.

    Returns
    -------
    merged_intervals : numpy.ndarray, shape (M,2) for M <= N
        Start and stop indices of the non-overlapping intervals, sorted
        by start. Empty input returns an empty array of shape ``(0,)``
        rather than ``(0,2)``.

    Notes
    -----
    Intervals that merely touch are merged: a stop equal to the next
    start counts as an overlap.

    **The input is modified in place.** The surviving rows are the
    caller's own objects, not copies, and extending one writes the new
    stop index through to the caller. This holds for a list of lists
    and for a two-dimensional array alike. Pass a copy if the original
    is needed afterwards.

    """
    sortedIntervals = sorted(intervals, key=lambda x: x[0])
    merged = []

    for interval in sortedIntervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)
        else:
            merged[-1][1] = max(interval[1], merged[-1][1])

    merged_intervals = np.array(merged)
    return merged_intervals

def _rolling_std(x: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Rolling standard deviation of a signal, centred on each sample.

    Parameters
    ----------
    x : array-like, shape (N,)
        Input signal.
    window : int, optional
        Size of the rolling window. Default is 3.

    Returns
    -------
    rolling_std : numpy.ndarray, shape (N,)
        The rolling standard deviation, same length as `x`.

    Notes
    -----
    The window is centred, so the value at a sample describes the
    scatter around it rather than behind it, and the output is the same
    length as the input.

    This is the sample standard deviation, dividing by one less than
    the number of points in the window, which is pandas' default. At
    the default window of three the divisor is two, so the values run
    about 22 percent above the population figure.

    Near the two ends the window is truncated and the value comes from
    fewer points, which raises the scatter there on an otherwise
    uniform signal: the first and last samples of a three-point window
    are computed from two points.

    **Returns all NaN for ``window=1``**, where the sample standard
    deviation of a single point is undefined.

    """
    data = {'value': x}
    df = pd.DataFrame(data)
    df['rolling_std'] = df['value'].rolling(window=window,
                                            center=True,
                                            min_periods=1
                                            ).std()
    rolling_std = df['rolling_std'].to_numpy()
    return rolling_std
