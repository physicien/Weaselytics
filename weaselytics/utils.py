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
.. [5] Kim, S.-J., Koh, K., Boyd, S. and Gorinevsky, D. l1 trend
   filtering. SIAM Rev. 51(2), 339-360 (2009). doi:10.1137/070690274
.. [6] Selesnick, I. W., Graber, H. L., Pfeil, D. S. and Barbour, R. L.
   Simultaneous low-pass filtering and total variation denoising. IEEE
   Trans. Signal Process. 62(5), 1109-1124 (2014).
   doi:10.1109/TSP.2014.2298836
.. [7] Chapel, S., Rouvière, F., Peppermans, V., Desmet, G. and Heinisch,
   S. A comprehensive study on the phenomenon of total breakthrough in
   liquid chromatography. J. Chromatogr. A 1653, 462399 (2021).
   doi:10.1016/j.chroma.2021.462399
"""
import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded
from scipy.ndimage import convolve1d
from scipy.signal import find_peaks, peak_widths


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


# Kept on purpose: no caller, but its endpoint-outlier criterion may still be
# wanted.
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

def _runs(mask: np.ndarray, length: int) -> np.ndarray:
    """
    Keep the runs of consecutive True of at least a given length.

    Parameters
    ----------
    mask : numpy.ndarray of bool, shape (N,)
        The mask.
    length : int
        Shortest run to keep.

    Returns
    -------
    runs : numpy.ndarray of bool, shape (N,)
        `mask`, with every shorter run set to False.

    """
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    runs = np.zeros(len(mask), dtype=bool)
    for start, stop in zip(starts, stops):
        runs[start:stop] = stop - start >= length
    return runs


def noise_level(s: np.ndarray, window: int = 20, min_run: int = 3,
                max_iter: int = 100) -> float:
    """
    Standard deviation of the noise of a signal.

    Measured on the raw signal, in the windows free of peaks. The peaks
    are located on the difference between the signal and its rolling
    mean.

    Parameters
    ----------
    s : numpy.ndarray, shape (N,)
        The signal.
    window : int, optional
        Number of samples of the rolling mean and of the windows.
        Default is 20, a chosen value.
    min_run : int, optional
        Shortest run of samples that counts as a peak. Default is 3.
    max_iter : int, optional
        Largest number of passes over the difference. Default is 100.

    Returns
    -------
    noise : float
        The noise, in the units of `s`. Zero when more than half of the
        windows kept hold a straight line.

    Raises
    ------
    ValueError
        If no window is free of peaks.

    Notes
    -----
    A run of at least `min_run` samples of the difference, all above or
    all below three times its root mean square, belongs to a peak. The
    root mean square is computed again without the runs, until they no
    longer change. The factor of 3 is chosen. A shorter excursion is
    noise, such as a jump by one step of the digitizer.

    The windows follow each other from the first sample, and the
    samples left at the end are unused. In each window without a run, a
    straight line is removed from the raw signal and the standard
    deviation of what is left is taken. The noise is the median of
    these.

    The noise slower than `window` samples is not counted. A peak much
    wider than `window` barely differs from its rolling mean. Its
    windows are kept, and the estimate is too large when they outnumber
    the flat ones, or when the peaks leave few flat windows.

    """
    s = np.asarray(s, dtype=float)
    # Centred mean: an even window has half a sample at each end.
    weights = np.ones(window)
    if window % 2 == 0:
        weights = np.concatenate(([0.5], np.ones(window - 1), [0.5]))
    diff = s - convolve1d(s, weights / window)
    peaks = np.zeros(len(s), dtype=bool)
    for _ in range(max_iter):
        limit = 3. * np.sqrt(np.mean(diff[~peaks]**2))
        new_peaks = (_runs(diff > limit, min_run)
                     | _runs(diff < -limit, min_run))
        if np.array_equal(new_peaks, peaks):
            break
        peaks = new_peaks

    n = len(s) // window
    free = ~peaks[:n * window].reshape(n, window).any(axis=1)
    if not free.any():
        raise ValueError("no window of the signal is free of peaks, so "
                         "its noise cannot be measured")
    segments = s[:n * window].reshape(n, window)[free]
    # Offset removed first: the mean of equal samples is not exact.
    segments = segments - segments[:, :1]
    t = np.arange(window) - (window - 1) / 2.
    slopes = segments @ t / (t @ t)
    rest = segments - segments.mean(axis=1, keepdims=True)
    rest = rest - np.outer(slopes, t)
    return float(np.median(np.sqrt(np.sum(rest**2, axis=1)
                                   / (window - 2))))


def l1_trend_filter(s: np.ndarray, lam: float, noise: float,
                    tol: float = 1e-3, max_iter: int = 200) -> np.ndarray:
    r"""
    Smooth a signal with straight segments, keeping its sharp peaks.

    Minimizes ``0.5 * ||s - z||^2 + lam * ||D z||_1``, Kim et al. [5]_,
    with ``D`` the second difference. The :math:`\ell_1` penalty makes
    most of the second differences zero, so ``z`` is made of straight
    segments that bend where the signal does.

    Parameters
    ----------
    s : numpy.ndarray, shape (N,)
        The signal to smooth.
    lam : float
        Strength of the smoothing, in the units of `s`.
    noise : float
        Standard deviation of the noise of `s`, which sets the scale
        below which the iterations stop and a second difference counts
        as zero.
    tol : float, optional
        Largest change between two iterations, in units of `noise`, at
        which the iterations stop. Default is 1e-3.
    max_iter : int, optional
        Largest number of iterations. Default is 200.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The smoothed signal. A copy of `s` when `lam` or `noise` is
        zero, or when `s` has fewer than 3 samples.

    Notes
    -----
    Solved by majorization-minimization, which Selesnick et al. [6]_ use
    for total variation: each iteration replaces the absolute values by
    quadratics that touch them at the current solution, and solves the
    pentadiagonal system that results. A constant added to the absolute
    values keeps the weights finite.

    The filter is not linear. No sample moves by more than ``4 * lam``
    [5]_. A peak far above the noise keeps its height and its position
    whatever its width, and a peak close to the noise loses height, more
    so when it is narrow.

    It fails on noise that is correlated over several samples: the
    segments follow it, since a bump of correlated noise and a small
    peak look the same.

    """
    n = len(s)
    if lam <= 0 or noise <= 0 or n < 3:
        return np.array(s, dtype=float)
    z = np.array(s, dtype=float)
    eps = 1e-3 * noise
    bands = np.zeros((3, n))
    for _ in range(max_iter):
        w = lam / (np.abs(np.diff(z, 2)) + eps)
        # I + D.T @ diag(w) @ D in upper banded form
        bands[2] = 1.
        bands[2, :-2] += w
        bands[2, 1:-1] += 4. * w
        bands[2, 2:] += w
        bands[1] = 0.
        bands[1, 1:-1] -= 2. * w
        bands[1, 2:] -= 2. * w
        bands[0, 2:] = w
        z_new = solveh_banded(bands, s)
        change = np.max(np.abs(z_new - z))
        z = z_new
        if change <= tol * noise:
            break
    return z


def peaks_params(s: np.ndarray, rel_prom_p: float = 0.05,
                 rel_prom_n: float = 0.8,
                 rel_height_p: float = 0.5, rel_height_n: float = 0.5,
                 width: int | None = None,
                 min_prom: float = 0.,
                 drop_enclosing: bool = False,
                 negative: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate the positive and negative peaks of a signal, with their
    widths.

    Positive and negative features are detected separately, each gated
    on prominence relative to the strongest feature of its own sign and
    on an absolute prominence, so the threshold does not depend on where
    the baseline sits.

    Parameters
    ----------
    s : numpy.ndarray
        A signal with peaks.
    rel_prom_p : float, optional
        Prominence a positive peak needs, as a fraction of the largest
        positive prominence in `s`. Default is 0.05.
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
    min_prom : float, optional
        Prominence any peak needs, positive or negative, in the units of
        `s`. Default is 0.
    drop_enclosing : bool, optional
        If True, discard any feature whose width encloses a taller
        peak. See `_drop_enclosing`. Default is False.
    negative : bool, optional
        If False, only the positive peaks are located. Default is True.

    Returns
    -------
    peaks : numpy.ndarray
        Indices of the peaks satisfying every condition, sorted.
    widths : numpy.ndarray
        Their widths in samples, in the same order.

    Notes
    -----
    A signal with no genuine negative peak can still yield one. The
    negative gate is a fraction of the deepest negative prominence
    present, and the prominence of a negative feature runs up to the top
    of the lower of the two peaks beside it. A flat gap between two tall
    peaks is therefore a deep negative feature, and the deepest one is
    kept whether or not the signal holds a real negative peak. A caller
    that takes a returned negative peak to be real will act on such a
    gap. See GitHub issue #18.

    Detection and width measurement are ``scipy.signal.find_peaks`` and
    ``scipy.signal.peak_widths``.

    """
    _, raw_params_p = find_peaks(s,prominence=0.0)
    _, raw_params_n = find_peaks(-s,prominence=0.0)
    max_prom_p = (raw_params_p["prominences"].max()
                  if len(raw_params_p["prominences"]) > 0 else 0.0)
    max_prom_n = (raw_params_n["prominences"].max()
                  if len(raw_params_n["prominences"]) > 0 else 0.0)
    prom_p = max(rel_prom_p * max_prom_p, min_prom)
    prom_n = max(rel_prom_n * max_prom_n, min_prom)

    peaks_p, _ = find_peaks(s, prominence=prom_p, width=width)
    peaks_n, _ = find_peaks(-s, prominence=prom_n, width=width)
    if not negative:
        peaks_n = peaks_n[:0]
    widths_p = peak_widths(s, peaks_p, rel_height=rel_height_p)[0]
    widths_n = peak_widths(-s, peaks_n, rel_height=rel_height_n)[0]

    unsorted_peaks = np.append(peaks_p, peaks_n)
    unsorted_widths = np.append(widths_p, widths_n)
    index_array = np.argsort(unsorted_peaks)

    peaks = unsorted_peaks[index_array]
    widths = unsorted_widths[index_array]
    if drop_enclosing and len(peaks) > 1:
        peaks, widths = _drop_enclosing(peaks, widths, s[peaks])
    return peaks, widths


def _drop_enclosing(peaks: np.ndarray, widths: np.ndarray,
                    heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Discard features that enclose a taller peak.

    A chromatographic peak can carry a smaller shoulder inside its own
    half-width, but never a taller peak. Structure that does, such as a
    wave of the baseline with peaks on it, is not a peak, and the peaks
    it encloses are what should define the region. A large injection in
    a strong solvent makes such structure, between the breakthrough peak
    near the dead time and the retained peak [7]_.

    Parameters
    ----------
    peaks : array-like, shape (M,)
        Peak indices, sorted.
    widths : array-like, shape (M,)
        Their widths in samples.
    heights : array-like, shape (M,)
        Their heights, in the same order.

    Returns
    -------
    peaks, widths : numpy.ndarray
        The surviving peaks and widths.

    Notes
    -----
    A feature encloses another when the other's index falls strictly
    inside ``peaks +- widths / 2``.

    Every verdict is decided in one pass, against the heights of all the
    features given, so a feature that is itself discarded can still
    cause a discard. Given three nested features of rising height, the
    outermost goes because the middle one is taller and inside it, while
    the middle one goes in the same pass because the innermost is taller
    and inside that. The outermost is therefore judged against something
    the function has already rejected, and only the innermost survives.

    """
    lo = peaks - widths / 2.
    hi = peaks + widths / 2.
    inside = ((peaks[None, :] > lo[:, None])
              & (peaks[None, :] < hi[:, None]))
    np.fill_diagonal(inside, False)
    taller = heights[None, :] > heights[:, None]
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
        by start. Empty input returns an empty array of shape ``(0, 2)``.

    Notes
    -----
    Intervals that merely touch are merged: a stop equal to the next
    start counts as an overlap.

    """
    sortedIntervals = sorted(intervals, key=lambda x: x[0])
    merged = []

    for interval in sortedIntervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(list(interval))
        else:
            merged[-1][1] = max(interval[1], merged[-1][1])

    if not merged:
        return np.empty((0, 2), dtype=int)
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
