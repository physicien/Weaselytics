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

def peaks_params(s, rel_prom_p=0.05, rel_prom_n=0.8, height_n=0.1,
                 rel_height_p=0.5, rel_height_n=0.5, width=None):
    """
    Function which find the center and width for every peak of the
    chromatogram (including the negative ones).

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

    Returns
    -------
    _peaks : numpy.ndarray
        Indices of peaks in `s` that satisfy all given conditions.
    _widths : numpy.ndarray
        The widths for each peak in `s`.

    """
    # @EB remove height_n? Seems to be useful...
    _, _raw_params_p = find_peaks(s,prominence=0.0)
    _, _raw_params_n = find_peaks(-s,prominence=0.0)
    _max_prom_p = _raw_params_p["prominences"].max()
    _max_prom_n = _raw_params_n["prominences"].max()
    if _max_prom_p <= 1:
        rel_prom_p = 0.5
    _prom_p = rel_prom_p * _max_prom_p
    _prom_n = rel_prom_n * _max_prom_n
#    _prom_p = rel_prom_p*s.max()            # @EB heuristic
#    _prom_n = rel_prom_n*(-s).max()         # @EB heuristic
    _peaks_p, _ = find_peaks(s, prominence=_prom_p, width=width)
    _peaks_n, _ = find_peaks(-s, prominence=_prom_n, height=height_n,
                             width=width)
#    print(_peaks_p)
#    print(_peaks_n)
    _widths_p = peak_widths(s, _peaks_p, rel_height=rel_height_p)[0]
    _widths_n = peak_widths(-s, _peaks_n, rel_height=rel_height_n)[0]
    _peaks = np.append(_peaks_p, _peaks_n)
    _widths = np.append(_widths_p, _widths_n)
    index_array = np.argsort(_peaks)
    return [_peaks[index_array],_widths[index_array]]

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
