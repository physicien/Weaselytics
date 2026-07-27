# coding: utf-8
"""
Helper functions to perform various signal preprocessing operations.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter


def end_window(data: np.ndarray, window_min: int = 3,
               window_max: int = 20) -> int:
    """
    Calculate the size of the local window used to detect endpoint outliers.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be tested.
    window_min : int, optional
        Minimum width of the window. Default is 3.
    window_max : int, optional
        Maximum width of the window. Default is 20.

    Returns
    -------
    size : int
        Size of the window.

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
    Check whether the first and last elements of the input data are outliers.
    If either of them is classified as an outlier, substitute it with the
    median computed from a local window of data points whose size is
    ``window_min <= round(0.01*len(s)) <= window_max``.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be tested.
    window_min : int, optional
        Minimum width of the window. Default is 5.
    window_max : int, optional
        Maximum width of the window. Default is 100.

    Returns
    -------
    s : numpy.ndarray
        The data with outliers removed from both ends.

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
    Calculate the Durbin-Watson statistic.

    Parameters
    ----------
    resids : array-like
        Data for which to compute the Durbin-Watson statistic. Usually
        regression model residuals.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    dw : float, array-like
        The Durbin-Watson statistic.

    Notes
    -----
    The null hypothesis of the test is that there is no serial correlation
    in the residuals.
    The Durbin-Watson test statistic is defined as:

    .. math::

       \sum_{t=2}^T((e_t - e_{t-1})^2)/\sum_{t=1}^Te_t^2

    The test statistic is approximately equal to 2*(1-r) where ``r`` is the
    sample autocorrelation of the residuals. Thus, for r == 0, indicating no
    serial correlation, the test statistic equals 2. This statistic will
    always be between 0 and 4. The closer to 0 the statistic, the more
    evidence for positive serial correlation. The closer to 4, the more
    evidence for negative serial correlation.

    Based on the implementation found in ``statsmodels.stats.stattools``.
    """
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids**2, axis=axis) / np.sum(resids**2, axis=axis)
    return dw

def r2_dw(s: np.ndarray) -> float:
    """
    Compute the squared values of `r`, the Durbin-Watson (DW) autocorrelation
    level.

    Parameters
    ----------
    s : array-like
        Data for which to compute the squared DW autocorrelation level. Usually
        regression model residuals.

    Returns
    -------
    r2 : float
        The squared values of the DW autocorrelation level.

    """
    r2 = ((2-_durbin_watson(s))**2)/4
    return r2

def smooth_SG(x: np.ndarray, window_length: int,
              polyorder: int) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter to an array.

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
                 rel_prom_n: float = 0.8, height_n: float = 0.1,
                 rel_height_p: float = 0.5, rel_height_n: float = 0.5,
                 width: int | None = None,
                 adapt: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the center and width for every peak of the chromatogram (including
    the negative ones).

    Parameters
    ----------
    s : numpy.ndarray
        A signal with peaks.
    rel_prom_p : float, optional
        Required prominence of positive peaks relative to the highest positive
        peak. Default is 0.05.
    rel_prom_n : float, optional
        Required prominence of negative peaks relative to the deepest negative
        peak. Default is 0.5.
    height_n : float, optional
        Required height of negative peaks. Either a number, ``None``, an array
        matching x or a 2-element sequence of the former. The first element is
        always interpreted as the minimal and the second, if supplied, as the
        maximal required height. Default is 0.1.
    rel_height_p : float, optional
        Selects the relative height at which the width of a positive peak is
        determined, expressed as a fraction of its prominence. A value of 1.0
        measures the peak's width at its lowest contour level, whereas 0.5
        measures it at half the prominence height. The value must be at
        least 0. Default is 0.5.
    rel_height_n : float, optional
        Selects the relative height at which the width of a negative peak is
        determined, expressed as a fraction of its prominence. A value of 1.0
        measures the peak's width at its lowest contour level, whereas 0.5
        measures it at half the prominence height. The value must be at
        least 0. Default is 0.5.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, `None`, an array
        matching x or a 2-element sequence of the former. The first element is
        always interpreted as the minimal and the second, if supplied, as the
        maximal required width. Default is `None`.
    adapt : bool, optional
        If True, lets the function change the value of `rel_prom_p` according
        the the maximum prominence of the data.

    Returns
    -------
    peaks : numpy.ndarray
        Indices of peaks in `s` that satisfy all given conditions.
    widths : numpy.ndarray
        The widths for each peak in `s`.

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
    peaks_n, _ = find_peaks(-s, prominence=prom_n, height=height_n,
                             width=width)
    widths_p = peak_widths(s, peaks_p, rel_height=rel_height_p)[0]
    widths_n = peak_widths(-s, peaks_n, rel_height=rel_height_n)[0]

    unsorted_peaks = np.append(peaks_p, peaks_n)
    unsorted_widths = np.append(widths_p, widths_n)
    index_array = np.argsort(unsorted_peaks)

    peaks = unsorted_peaks[index_array]
    widths = unsorted_widths[index_array]
    return peaks, widths


def merge_intervals(intervals: list[list[int]]) -> np.ndarray:
    """
    Merge overlapping intervals.

    Parameters
    ----------
    intervals : array-like, shape (N,2)
        The two dimensional array containing the start and stop indices for
        each intervals of interest.

    Returns
    -------
    merged_intervals : numpy.ndarray, shape (M,2) for M <= N
        The two dimensional array containing the start and stop indices for
        each non-overlapping intervals.

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
    Compute the rolling standard deviation of the data.

    Parameters
    ----------
    x : array-like, shape (N,)
       Input array of the data.
    window : int, optional
        Size of the rolling window. Default is 3.

    Returns
    -------
    rolling_std : array-like, shape (N,)
        The rolling standard deviation.

    """
    data = {'value': x}
    df = pd.DataFrame(data)
    df['rolling_std'] = df['value'].rolling(window=window,
                                            center=True,
                                            min_periods=1
                                            ).std()
    rolling_std = df['rolling_std'].to_numpy()
    return rolling_std
