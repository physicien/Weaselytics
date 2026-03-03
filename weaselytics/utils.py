#!/usr/bin/python
# coding: utf-8
"""
Helper functions to perform various signal preprocessin operations.
"""
import numpy as np
from scipy.signal import savgol_filter
#from statsmodels.stats.stattools import durbin_watson as dwtest

def rm_ends_outliers(s, window_min=5, window_max=100):
    """
    Checks whether the first and last elements of the input data are outliers.
    If either boundary value is classified as an outlier, substitute it with
    the median computed from a local window of data points whose size is
    ``round(0.01*len(s))``, ensuring that the window size lies between
    `window_min` and `window_max` (inclusive). 

    Parameters
    ----------
    s : numpy.ndarray
        The data to be tested.
    window_min : int, optional
        Minimum width of the window. Default is 5.
    window_max : int, optional
        Maximum width of the window. Default is 100.

    Returns
    -------
    _signal : numpy.ndarray
        The data with outliers removed from both ends.

    """
    _signal = np.copy(s)
    _x0_len = round(0.01*len(s))
    if _x0_len < window_min:
        _x0_len = window_min
    if _x0_len > window_max:
        _x0_len = window_max
    _y_max = 0.01*np.abs(np.max(s)-np.min(s))
    _y0_med = np.median(s[:_x0_len])
    _y0_gap = np.abs(s[0]-_y0_med)
    _y1_med = np.median(s[-_x0_len:])
    _y1_gap = np.abs(s[-1]-_y1_med)

    if _y0_gap > _y_max:
        _signal[0] = _y0_med
    if _y1_gap > _y_max:
        _signal[-1] = _y1_med
    return _signal

def durbin_watson(resids, axis=0):
    """
    Calculates the Durbin-Watson statistic.

    Parameters
    ----------
    resids : array_like
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
    """
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids**2, axis=axis) / np.sum(resids**2, axis=axis)
    return dw

def r2_fct(s):
    """
    Computes the squared values of `r`, the Durbin-Watson (DW) autocorrelation
    level.

    Parameters
    ----------
    s : array-like
        Data for which to compute the squared DW autocorrelation level. Usually
        regression model residuals.

    Returns
    -------
    _r2 : numpy.ndarray
        The squared values of the DW autocorrelation level.

    """
    _r2 = ((2-durbin_watson(s))**2)/4
    return _r2

def smooth_SG(x,window_lenght,polyorder):
    """
    Applies a Savitzky-Golay filter to an array.

    Parameters
    ----------
    x : array-like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_lenght : int
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int
        The order of the polynomial used to fit the samples. `polyorder` must
        be less than `window_lenght`.

    Returns
    -------
    y : ndarray, same shape as x
        The filtered data.

    """
    smooth_data = savgol_filter(x,window_lenght,polyorder)
    return smooth_data

def continuous_ranges(x):
    """
    Separates an array of integers into continuous segments.

    Parameters
    ----------
    x : array-like 
        The array.

    Returns
    -------
    c_range : list of ndarrays
        A list of continuous sub-arrays.
    """
    c_range = np.split(x, np.where(x[1:] != x[:-1] +1)[0] +1)
    return c_range

def find_plateaus(x, include_tol, exclude_tol=0, mode='absolute'):
    """
    Finds the plateaus of an array according to a certain threshold. A second
    threshold can also be used to exclude certain regions from the plateaus.

    Parameters
    ----------
    x : array-like
        The array on which to find plateaus.
    include_tol : float
        The cutoff threshold of the values to be included in the plateaus.
    exclude_tol : float, optional
        A second threshold to exclude values from the plateaus. Its value
        should be smaller than that of `include_tol`. Default is 0.
    mode : str, optional
        One of the following string values.
        'absolute' (default)
            Takes the absolute value of the array.
        'signed'
            Takes signed values of the array.
        
    Returns
    -------
    plateaus : array-like
        The array containing the position of the plateau regions of `x`.

    Raises
    ------
    ValueError
        Raised if `exclude_tol` in not smaller than `include_tol`, or if the 
        `mode` being passed is not allowed.

    """
    # Make sure that the mode being passed is allowed
    allowed_modes = ["absolute", "signed"]
    if mode not in allowed_modes:
        raise  ValueError(f"mode '{mode}' is not supported")

    if mode == "absolute":
        array = np.absolute(x)
    elif mode == "signed":
        array = x

    if exclude_tol > include_tol:
        raise ValueError("exclude_tol must be smaller than include_tol")
    include_condition = ((array < include_tol) & (array > exclude_tol))
    plateaus = np.where(include_condition)[0]
    return plateaus

def merge_intervals(intervals):
    """
    Merges overlapping intervals of indices.

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
