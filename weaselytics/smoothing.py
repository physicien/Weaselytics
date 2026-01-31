#!/usr/bin/python
# coding: utf-8
"""
Functions to perform signal smoothing.
"""

from scipy.signal import savgol_filter

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

