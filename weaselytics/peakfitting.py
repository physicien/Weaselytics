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
    Generate a Gaussian distribution based on params.

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
    numpy.ndarray
        The Gaussian distribution evaluated with x.

    Raises
    ------
    ValueError
        Raised if `sigma` is not greater than 0.

    """
    amp, x0, sigma = params
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")
    return amp*np.exp(-0.5*((x-x0)**2)/sigma**2)

def skew_norm(x,params):
    """
    Generate a Skew normal distribution based on params.

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
    numpy.ndarray
        The Skew normal distribution evaluated with x.

    """
    amp, loc, scale, alpha = params
    _x = alpha*(x-loc)/scale
    norm = np.sqrt(2*np.pi*scale**2)**-1* np.exp(
            -((x-loc)**2)/(2*scale**2)
            )
    cdf = 0.5*(1+erf(_x/np.sqrt(2)))
    return amp*2*norm*cdf

def lsq_eq(p,fct,x,y):
    """
    Function which compute the vector of residuals in order to solve the
    least-squares problem.

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

def peaks_params(s):
    """
    Function which find the center and width for every peak of the
    chromatogram (including the negative ones).

    Parameters
    ----------

    s : numpy.ndarray
        A signal with peaks.

    Returns
    -------
    _peaks : numpy.ndarray
        Indices of peaks in `s` that satisfy all given conditions.
    _widths : numpy.ndarray
        The widths for each peak in `s`.

    """
    _prom_p = 0.05*s.max()
    _prom_n = 0.5*(-s).max()
    _peaks_p, _ = find_peaks(s,prominence=_prom_p)
    _peaks_n, _ = find_peaks(-s,prominence=_prom_n,height=0.1)
    _widths_p = peak_widths(s, _peaks_p, rel_height=0.5)[0]
    _widths_n = peak_widths(-s, _peaks_n, rel_height=0.5)[0]
    _peaks = np.append(_peaks_p,_peaks_n)
    _widths = np.append(_widths_p,_widths_n)
    return [_peaks,_widths]

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
    x : ndarray-like with shape (3,)
        Solution found, `x`, with the following fields defined:

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
    _peaks, _widths = peaks_params(y)
    main_peak_i = np.absolute(y[_peaks]).argmax()
    _i = _peaks[main_peak_i]
    A0 = y[_i]
    tau0 = x[_i]
    sigma0 = x[_i + int(_widths[main_peak_i]/2)] - x[_i]
    p0 = [A0, tau0, sigma0]
    if A0 < 0:
        bA = [-np.inf,0]
    else:
        bA = [0,np.inf]
    bounds = ([bA[0],tau0-0.1,0],[bA[1],tau0+0.1,np.inf])
    res_robust = least_squares(lsq_eq, p0, loss="soft_l1",
                              f_scale=0.1, args=(gauss,x,y),
                               bounds=bounds)
    return res_robust.x

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
    x : ndarray-like with shape (4,)
        Solution found, `x`, with the following fields defined:

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
    _peaks, _widths = peaks_params(y)
    main_peak_i = np.absolute(y[_peaks]).argmax()
    _i = _peaks[main_peak_i]
    A0 = y[_i]
    tau0 = x[_i]
    sigma0 = x[_i + int(_widths[main_peak_i]/2)] - x[_i]
    p0 = [A0, tau0, sigma0, 0]
    if A0 < 0:
        bA = [-np.inf,0]
    else:
        bA = [0,np.inf]
#    bounds = ([bA[0],tau0-0.1,0,-np.inf],[bA[1],tau0+0.1,np.inf,np.inf])
    bounds = ([bA[0],tau0-sigma0,0,-np.inf],[bA[1],tau0+sigma0,np.inf,np.inf])
    res_robust = least_squares(lsq_eq, p0, loss="soft_l1",
                               f_scale=0.1, args=(skew_norm,x,y),
                               bounds=bounds)
    return res_robust.x
