#!/usr/bin/python
# coding: utf-8
"""
Functions to perform Peak fitting.
"""

import numpy as np
from scipy.special import erf
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import least_squares

def gauss(x, params):
    """
    Generate a Gaussian distribution based on `params`.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    params : array-like with shape (n,)
        `params` with the following fields defined:

        amp : float
            The maximum height of the distribution.
        x0 : float
            The center of the distribution.
        sigma : float
            The standard deviation of the distribution.

    Returns
    -------
    dist : numpy.ndarray
        The Gaussian distribution evaluated with x.

    Raises
    ------
    ValueError
        Raised if `sigma` is not greater than 0.

    """
    amp, x0, sigma = params

    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")

    dist = amp*np.exp(-0.5*((x-x0)**2)/sigma**2)
    return dist

def skew_norm(x,params):
    """
    Generate a Skew normal distribution based on `params`.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    params : array-like with shape (n,)
        `params` with the following fields defined:

        amp : float
            The maximum height of the distribution.
        loc :  float
            The location parameter of the distribution.
        scale : float
            The scale parameter of the distribution.
        alpha : float
            The shape parameter of the distribution.

    Returns
    -------
    dist : numpy.ndarray
        The Skew normal distribution evaluated with x.

    """
    amp, loc, scale, alpha = params

    z = alpha*(x-loc)/scale
    norm = np.sqrt(2*np.pi*scale**2)**-1* np.exp(
            -((x-loc)**2)/(2*scale**2)
            )
    cdf = 0.5*(1+erf(z/np.sqrt(2)))

    dist = amp*2*norm*cdf
    return dist

def lsq_eq(p,fct,x,y):
    """
    Compute the vector of residuals in order to solve the least-squares
    problem.

    Parameters
    ----------
    p : array-like with shape (n,)
        Set of independent variables defining the function.
    fct : callable
        Function used to solve the least-squares problem.
    x : numpy.ndarray
        Range on the x-axis to fit `fct`.
    y : numpy.ndarray
        Values on which to fit `fct` for each point of the x-axis range.

    Returns
    -------
    callable
        A function to feed to the `scipy.optimize.least_squares` method.

    """
    return fct(x,p) - y

def peaks_params(s, rel_prom_p=0.05, rel_prom_n=0.8, height_n=0.1,
                 rel_height_p=0.5, rel_height_n=0.5, width=None, adapt=False):
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
        measures the peak’s width at its lowest contour level, whereas 0.5
        measures it at half the prominence height. The value must be at 
        least 0. Default is 0.5.
    rel_height_n : float, optional
        Selects the relative height at which the width of a negative peak is
        determined, expressed as a fraction of its prominence. A value of 1.0
        measures the peak’s width at its lowest contour level, whereas 0.5
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
    max_prom_p = raw_params_p["prominences"].max()
    max_prom_n = raw_params_n["prominences"].max()
    # In case of low noisy signal
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

def lsq_gauss_fit(x,y):
    """
    Use non-linear least squares to fit a Gaussian distribution to data. The
    procedure was made robust by assuming that inlier residuals remain below
    0.1. For further information, see [1].
    
    Parameters
    ----------
    x : numpy.ndarray
        The independent variable where the data is measured.
    y : numpy.ndarray
        The dependent variable where the data is measured.

    Returns
    -------
    s : ndarray with shape (3,)
        Solution found, `s`, with the following fields defined:

        amp : float
            The maximum height of the distribution.
        x0 : float
            The center of the distribution.
        sigma : float
            The standard deviation of the distribution.

    References
    ----------
    [1] https://scipy-cookbook.readthedocs.io/items/robust_regression.html

    """
    peaks, widths = peaks_params(y)
    main_index = np.absolute(y[peaks]).argmax()
    peak = peaks[main_index]

    A0 = y[peak]
    tau0 = x[peak]
    sigma0 = x[peak + int(widths[main_index]/2)] - x[peak]
    p0 = [A0, tau0, sigma0]
    if A0 < 0:
        bA = [-np.inf,0]
    else:
        bA = [0,np.inf]
    bounds = ([bA[0],tau0-0.1,0],[bA[1],tau0+0.1,np.inf])

    res_robust = least_squares(lsq_eq, p0, loss="soft_l1",
                              f_scale=0.1, args=(gauss,x,y),
                               bounds=bounds)
    s = res_robust.x
    return s

def lsq_skew_norm_fit(x,y):
    """
    Use non-linear least squares to fit a Skew normal distribution to data. The
    procedure was made robust by assuming that inlier residuals remain below
    0.1. For further information, see [1].
    
    Parameters
    ----------
    x : numpy.ndarray
        The independent variable where the data is measured.
    y : numpy.ndarray
        The dependent variable where the data is measured.

    Returns
    -------
    s : ndarray with shape (4,)
        Solution found, `s`, with the following fields defined:

        amp : float
            The maximum height of the distribution.
        loc :  float
            The location parameter of the distribution.
        scale : float
            The scale parameter of the distribution.
        alpha : float
            The shape parameter of the distribution.

    References
    ----------
    [1] https://scipy-cookbook.readthedocs.io/items/robust_regression.html

    """
    peaks, widths = peaks_params(y)
    main_index = np.absolute(y[peaks]).argmax()
    peak = peaks[main_index]

    A0 = y[peak]
    tau0 = x[peak]
    sigma0 = x[peak + int(widths[main_index]/2)] - x[peak]
    p0 = [A0, tau0, sigma0, 0]
    if A0 < 0:
        bA = [-np.inf,0]
    else:
        bA = [0,np.inf]
    bounds = ([bA[0],tau0-sigma0,0,-np.inf],[bA[1],tau0+sigma0,np.inf,np.inf])

    res_robust = least_squares(lsq_eq, p0, loss="soft_l1",
                               f_scale=0.1, args=(skew_norm,x,y),
                               bounds=bounds)
    s = res_robust.x
    return s
