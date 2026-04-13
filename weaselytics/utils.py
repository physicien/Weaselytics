#!/usr/bin/python
# coding: utf-8
"""
Helper functions to perform various signal preprocessin operations.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation
from skimage.filters import threshold_triangle, threshold_sauvola
#from scipy.ndimage import gaussian_filter1d
from diptest import diptest

def end_window(data, window_min=3, window_max=20):
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


def rm_ends_outliers(data, window_min=5, window_max=100):
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

def _durbin_watson(resids, axis=0):
    """
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

def r2_dw(s):
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

def smooth_SG(x, window_lenght, polyorder):
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
    smooth_data : ndarray, same shape as x
        The filtered data.

    """
    smooth_data = savgol_filter(x,window_lenght,polyorder)
    return smooth_data

def continuous_ranges(x):
    """
    Separate an array of integers into continuous segments.

    Parameters
    ----------
    x : array-like 
        The array to split.

    Returns
    -------
    continuous : list of ndarrays
        A list of continuous sub-arrays.
    """
    continuous = np.split(x, np.where(x[1:] != x[:-1] +1)[0] +1)
    return continuous

def find_plateaus(x, include_tol, exclude_tol=0, mode='absolute'):
    """
    Find the plateaus of an array according to a certain threshold. A second
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

def _rolling_std(x, window=3):
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

def _rolling_mad(x, window=3):
    """
    Compute the rolling median absolute deviation of the data.

    Parameters
    ----------
    x : array-like, shape (N,)
       Input array of the data.
    window : int, optional
        Size of the rolling window. Default is 3.

    Returns
    -------
    rolling_mad : array-like, shape (N,)
        The rolling median absolute deviation of the data.
    
    """
    data = {'value': x}
    df = pd.DataFrame(data)
    df['rolling_mad'] = df['value'].rolling(window=window,
                                            center=True,
                                            min_periods=1
                                            ).apply(median_abs_deviation)
    rolling_mad = df['rolling_mad'].to_numpy()
    return rolling_mad

def _long_plateaus(x, min_len=10):
    """
    Eliminate, in a discontinuous boolean array, the continuous segments of
    `True` values that are shorter than `min_len`.

    Parameters
    ----------
    x : array-like, shape (N,)
        The discontinuous boolean array.
    min_len : int, optional
        Minimal length of a continuous segment. Default is 10.

    Returns
    -------
    long_plateaus : array-like, shape (N,)
        The boolean array in which every contiguous segment of `True` values
        has a length of at least `min_len`.

    """
    seg_list = []
    segments = []
    args_ini = np.nonzero(x)[0]
    for seg in continuous_ranges(args_ini):
        if len(seg) >= min_len:
            seg_list.append(seg)
    if seg_list:
        segments = np.concatenate(seg_list)
    long_plateaus = np.zeros(len(x), dtype=bool)
    long_plateaus[segments] = True
    return long_plateaus


def find_plateaus2(x, window=3, nbins=256, pval_cutoff=0.002):  #0.05 ?
    """
    NOTE: CHANGE pval_cutoff to 0.05 later
    """
    # TODO: Try to minimize the rolling MAD. Also, plot both rolling STD and
    #       MAD side-by-side keeping in mind the idea of minimization.

    # Rolling statistics
    rolling_std = _rolling_std(x, window=window)
    rolling_mad = _rolling_mad(x, window=window)
    diff_std_mad = rolling_std - rolling_mad

    local_threshold = threshold_sauvola(rolling_std)
    corrected = rolling_std - local_threshold
    
    # Test if the distribution is unimodal (p=1)  
    _, pval = diptest(rolling_std)
    print(f"{'pval:':<20}{pval:0.4f}")

    # Find the threshold value
    if pval < pval_cutoff:
        # In case of significant multimodality
        threshold = threshold_triangle(corrected, nbins=nbins)
        plateaus = corrected < threshold
    else:
        threshold = threshold_triangle(rolling_std, nbins=nbins)
        plateaus = rolling_std < threshold
    
#    crossings = np.where(np.diff(np.sign(corrected)))[0]
#    plateaus[crossings] = False

#    test = diff_std_mad < 5.0E-05
#    plateaus = np.logical_and(plateaus, test)

#    not_too_flat = rolling_std > 1.0E-06
#    plateaus = np.logical_and(plateaus, not_too_flat)

    # Discard shorter plateaus
    plateaus = _long_plateaus(plateaus)
#    print(continuous_ranges(np.where(plateaus)[0]))

    return plateaus, rolling_std, diff_std_mad#rolling_mad#local_threshold
