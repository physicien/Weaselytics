#!/usr/bin/python
# coding: utf-8
"""
Helper functions to perform various signal preprocessin operations.
"""
import numpy as np
from scipy.signal import savgol_filter

def rm_ends_outliers(s, window_min=5, window_max=100):
    """
    Check whether the first and last elements of the input data are outliers.
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


def smooth_SG_data(x,window_lenght,polyorder):
    """
    Apply a Savitzky-Golay filter to an array.

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

