#!/usr/bin/python
# coding: utf-8
"""
Functions to perform the baseline correction.
"""
import os
import numpy as np
from pybaselines import Baseline
from scipy.signal import argrelmin, argrelmax#, medfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import time                             #@EB temporary?

from peakfitting import peaks_params
from utils import (r2_dw, continuous_ranges, find_plateaus, merge_intervals,
                   end_window
                   )
from plot import r2_plots

def _relevant_regions(s, x, tol=6.):
    """
    Divide the signal into regions maximizing the contribution of the signal in
    the calculation of the autocorrelation plot. in order to find the optimal
    cutoff frequency for the BEADS algorithm.

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    tol : float, optional
        Threshold on the ratio of a peak’s width as a function of its location
        in `x`.

    Returns
    -------
    peak_regions : array-like, shape (M,2)
        The two dimensional array containing the start and stop indices for
        each region containing a relevant peak. Each region is defined as
        ``data[start:stop]``. Default is ((None, None),), which will use all
        points.
    sampling : int or array-like of shape (M,2)
        The sampling step size for each region defined in `peak_regions`.
    scut : int
        Index of the last data point in `s` (signal cutoff) relevant to the
        calculation of the autocorrelation.

    """
    # NOTE: A weak smoothing helps to avoid peak detection in noisy region of
    #       the signal by:
    #           1) removing most of the spurious features in the raw signal
    #           2) sligntly enlarging features relevant for peaks detection
    z = gaussian_filter1d(s,3)
    peaks, widths = peaks_params(z, height_n=0.50, width=3, rel_prom_p=0.01,
                                   adapt=True)

    # TODO: Find a way to make this part of the code more robust.
    width_per_x = widths/x[peaks]
    # In case of very tall and large peaks (see acetonitrile)
    exception = ((s[peaks] > 20) & (width_per_x < 11))
    # Signal splitting
    rel_peaks = peaks[((width_per_x < tol) | exception)]
    rel_widths = widths[((width_per_x < tol) | exception)]
    ratio_w = rel_widths/np.min(rel_widths)

    # Peak full width
    # NOTE: Assuming that `rel_widths` is the FWHM and that the peak is
    #       gaussian, `buffer` is equal to half of the full peak width.
    buffer = np.ceil(0.85*rel_widths).astype(int)
    left_lim = rel_peaks - buffer
    right_lim = rel_peaks + buffer
    full_widths = np.array([left_lim,right_lim]).T

    # Peak regions and sampling
    large_peaks = full_widths[ratio_w > 1]      # Ignore the narrowest peak
    peak_regions = merge_intervals(np.copy(large_peaks))
    if len(peak_regions) == 0:
        peak_regions = ((None, None),)
        sampling = 1
    else:
        # Because values in regions must be less than len(data)
        if peak_regions[-1,-1] >= len(s):
            peak_regions[-1,-1] = len(s) - 1
        sampling = np.ceil((peak_regions[:,1]-peak_regions[:,0])/2
                           /np.min(rel_widths)).astype(int)
        #@EB Why divided by 2?

    # Signal cutoff
    # TODO: Maybe it would be a good idea to cut the less relevant starting
    #       segment too.
    arg_last_peak = rel_peaks.argmax()
    pos_last_peak = rel_peaks[arg_last_peak]
    buffer_last_peak = int(np.ceil(2*rel_widths)[arg_last_peak])
    pos_max = pos_last_peak + buffer_last_peak
    if len(s) > pos_max:
        scut = pos_max
    else:
        scut = len(s)
    return peak_regions, sampling, scut

def _log_transform(s, epsilon=1):
    """
    Log transformation used in the calculation of the autocorrelation for the
    BEADS algorithm. For further information, see [1].

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal to transform.
    epsilon : float, optional
        An arbitrary positive offset. The larger the offset, the less
        aggressive the pre-treatment. Default is 1 (see Notes).

    Returns
    -------
    log_s : numpy.ndarray
        The log transformed data.

    Notes
    -----
    The default value of ``epsilon = 1`` was originally suggested by
    Navarro-Huerta et al. [1] for two reasons:
        1) It was judged appropriate regarding the magnitude of the signals
        reaching maxima around 500-10000.
        2) If ``yi = min(y)``, then ``log(yi - min(y) + 1) = log 1 = 0``.

    However, it seems uncertain whether choosing ``epsilon = 1`` is optimal
    for a signal whose maximum value is below 500. My impression is that the
    500-10000 guideline reported by Navarro-Huerta et al. [1] was most likely
    derived purely from the particular signals they had at their disposal,
    rather than being grounded in a substantive theoretical argument.
    
    References
    ----------
    [1] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex
        chromatograms using the BEADS algorithm. Journal of Chromatography A,
        2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057.

    """
    log_s = np.log10(s - np.min(s) + epsilon)
    return log_s

def beads(baseline_fitter, s, freq_cutoff=0.005, asymmetry=1.0,
          fit_parabola=True, alpha=1.0, parabola_len=3, **kwargs):
    """
    Baseline estimation and denoising with sparsity (BEADS).

    Parameters
    ----------
    baseline_fitter : `Baseline` object
        Contains the x-values of the signal to baseline correct and all
        available baseline correction algorithms in pybaselines.
    s :  array-like, shape (N,)
        The y-values of the signal.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is 0.005.
    asymmetry : float, optional
        A number greater than 0 that determines the weighting of negative
        values compared to positive values in the cost function. For example,
        if is 6.0, it will give negative values six times more impact on the
        cost function that positive values. If set to 1 (the default), the
        cost function is symmetric, and a value less than 1 will weigh positive
        values more.
    fit_parabola : bool, optional
        If True (default), will fit a parabola to the data and subtract it
        before performing the BEADS fit as suggested in [2]. This ensures the
        endpoints of the fit data are close to 0, which is required by BEADS.
        If the data is already close to 0 on both endpoints, set `fit_parabola`
        to False (but it does not change anything in reality).
    alpha : float, optional
        #@EB will change in pybaselines. Default is 1.0.
    parabola_len : int, optional
        Size of the window used, at each ends of the data, to prevent issues
        in fitting a parabola before the baseline correction[2] when the first
        and/or last point is an outlier. Default is 3.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    bl : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'signal': numpy.ndarray, shape (N,)
            The pure signal portion of the input `data` without noise or the
            baseline.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for each
            iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.
        * 'fidelity': float
            The fidelity term of the final fit, given as
            :math:`0.5 * ||H(y - s)||_2^2`.
        * 'penalty' : tuple[float, float, float]
            The penalty terms of the final fit before multiplication with the
            `lam_d` terms. These correspond to
            :math:`\sum\limits_{i}^{N} \theta(s_i)`,
            :math:`\sum\limits_{i}^{N - 1} \phi(\Delta^1 s_i)`, and
            :math:`\sum\limits_{i}^{N - 2} \phi(\Delta^2 s_i)`, respectively.

    References
    ----------
    .. [1] Ning, X., et al. Chromatogram baseline estimation and denoising
        using sparsity (BEADS). Chemometrics and Intelligent Laboratory
        Systems, 2014, 139, 156-167.
    .. [2] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in
        complex chromatograms using the BEADS algorithm. Journal of
        Chromatography A, 2017, 1507, 1-10.

    """
    bl, params = baseline_fitter.beads(
            s,
            freq_cutoff=freq_cutoff,
            fit_parabola=fit_parabola,
            asymmetry=asymmetry,
            alpha=alpha,
            parabola_len=parabola_len
            )
    return bl, params

def custom_beads(baseline_fitter, s, regions=((None,None),), sampling=1,
                 freq_cutoff=0.005, asymmetry=1.0, fit_parabola=True,
                 alpha=1.0, parabola_len=3, **kwargs):
    """
    Customized variant of BEADS for fine tuned stiffness of the baseline in
    specific regions.

    Parameters
    ----------
    baseline_fitter : `Baseline` object
        Contains the x-values of the signal to baseline correct and all
        available baseline correction algorithms in pybaselines.
    s :  array-like, shape (N,)
        The y-values of the signal.
    regions : array-line, shape (M,2), optional
        The two dimensional array containing the start and stop indices for
        each region containing a relevant peak. Each region is defined as
        ``data[start:stop]``. Default is ((None, None),), which will use all
        points.
    sampling : int or array-like, optional
        The sampling step size for each region defined in `regions`. Default
        is 1.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is 0.005.
    asymmetry : float, optional
        A number greater than 0 that determines the weighting of negative
        values compared to positive values in the cost function. For example,
        if is 6.0, it will give negative values six times more impact on the
        cost function that positive values. If set to 1 (the default), the
        cost function is symmetric, and a value less than 1 will weigh positive
        values more.
    fit_parabola : bool, optional
        If True (default), will fit a parabola to the data and subtract it
        before performing the BEADS fit as suggested in [2]. This ensures the
        endpoints of the fit data are close to 0, which is required by BEADS.
        If the data is already close to 0 on both endpoints, set `fit_parabola`
        to False (but it does not change anything in reality).
    alpha : float, optional
        #@EB will change in pybaselines. Default is 1.0.
    parabola_len : int, optional
        Size of the window used, at each ends of the data, to prevent issues
        in fitting a parabola before the baseline correction [2] when the first
        and/or last point is an outlier. Default is 3.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    bl : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'signal': numpy.ndarray, shape (N,)
            The pure signal portion of the input `data` without noise or the
            baseline.
        * 'x_fit': numpy.ndarray, shape (P,)
            The truncated x-values used for fitting the baseline.
        * 'y_fit': numpy.ndarray, shape (P,)
            The truncated y-values used for fitting the baseline.
        * 'baseline_fit': numpy.ndarray, shape (P,)
            The truncated baseline before interpolating from `P` points to `N`
            points.
        * 'method_params': dict
            A dictionary containing the output parameters for the fit using the
            selected `method`.

    References
    ----------
    .. [1] Ning, X., et al. Chromatogram baseline estimation and denoising
        using sparsity (BEADS). Chemometrics and Intelligent Laboratory
        Systems, 2014, 139, 156-167.
    .. [2] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in
        complex chromatograms using the BEADS algorithm. Journal of
        Chromatography A, 2017, 1507, 1-10.
    .. [3] Liland, K., et al. Customized baseline correction. Chemometrics and
        Intelligent Laboratory Systems, 2011, 109(1), 51-56.

    """
    beads_kwargs = {'freq_cutoff': freq_cutoff,
                    'fit_parabola': fit_parabola,
                    'asymmetry': asymmetry,
                    'alpha': alpha,
                    'parabola_len': parabola_len
                    }

    bl, params = baseline_fitter.custom_bc(
            s,
            method="beads",
            regions=regions,
            sampling=sampling,
            lam=None,
            method_kwargs=beads_kwargs
            )
    
    noise_fit = (params['y_fit'] - params['baseline_fit']
                 - params['method_params']['signal'])
    params['noise'] = np.interp(baseline_fitter.x, params['x_fit'], noise_fit)
    params['signal'] = s - bl - params['noise']
    return bl, params

def _r2(algo, baseline_fitter, y, p, param="freq_cutoff", **kwargs):
    """
    Calculate the autocorrelation, based on the Durbin-Watson statistics, of
    the baseline corrected signal for a given value of a given parameter used
    for the substraction of the baseline.

    Parameters
    ----------
    algo : Callable
       The callable method corresponding to the input string.
    baseline_fitter : `Baseline` object
        Contains the x-values of the signal to baseline correct and all
        available baseline correction algorithms in pybaselines.
    y : array-like, shape (N,)
        The y-values of the signal.
    p : float
        Value or `param` at which r2 is evaluated.
    param : str, optional
        Label of the parameter to correlate with the value of r2. Default is
        "freq_cutoff".
    **kwargs
        Additional keyword arguments.
        
    Returns
    -------
    r2 : float
        The autocorrelation of the baseline corrected signal for `param`=`p`.

    """
    kwargs[param] = p
    _, params = algo(baseline_fitter, y, **kwargs)
    y_corr = params["signal"]
    r2 = r2_dw(y_corr)
    return r2

def _r2_array(algo, baseline_fitter, signal, param_range,
              param="freq_cutoff", **kwargs):
    """
    Calculate the array of `r2`, the Durbin-Watson autocorrelation of the
    baseline corrected signal, relative to a parameter on a specific range.

    Parameters
    ----------
    algo : Callable
       The callable method corresponding to the input string.
    baseline_fitter : `Baseline` object
        Contains the x-values of the signal to baseline correct and all
        available baseline correction algorithms in pybaselines.
    signal : array-like, shape (N,)
        The y-values of the signal.
    param_range : array-like, shape (M,)
        Range of values taken by `param` and at which r2 is evaluated.
    param : str, optional
        Label of the parameter to correlate with the value of r2. Default is
        "freq_cutoff".
    **kwargs
        Additional keyword arguments.
    
    Returns
    -------
    vr2 : numpy.ndarray, shape (M,)
        The calculated array of r2.

    """
    r2_func = lambda x: _r2(algo, baseline_fitter, signal, x, param=param,
                            **kwargs)
    vr2_func = np.vectorize(r2_func)
    vr2 = vr2_func(param_range)
    return vr2

def _fcutoff(s, x, scut, smoothing_window=15, slope_thresh=5.0E-05,
            tol0=1.0E-03, tol1_0=1.0E-05, tol1_1=5.0E-04, tol2=2.0E-06,
            num=1000, show_plot=False, print_plot=False, path="./file.txt",
            method="beads", param="freq_cutoff", **kwargs):
    """
    Find the optimal cutoff frequency.

    ###EXPERIMENTAL###
    Since this function is still under development and very unreliable, it is
    best to explain the general idea behind it rather than the details of the
    current implementation. This is done in fcut.md
    ##################

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    scut : int
        Index of the last data point in `s` (signal cutoff) relevant to the
        calculation of the autocorrelation.
    smoothing_window : int, optional
        Standard deviation for Gaussian kernel used to smooth the signal.
    slope_thresh : float, optional
        Threshold on the value of `smooth_d1` for the final shift of the
        frequency cutoff. Default is 5.0E-05.
    tol0 : float, optional
        Threshold used to find the first plateau on the autocorrelation plot.
        Default is 1.0E-03.
    tol1_0 : float, optional
        Tight threshold used to find plateaus on the first derivative of the
        smoothed autocorrelation plot. Default is 1.0E-05.
    tol1_1 : float, optional
        Loose threshold used to find plateaus on the first derivative of the
        smoothed autocorrelation plot. Default is 5.0E-04.
    tol2 : float, optional
        Threshold used to find plateaus on the second derivative of the
        smoothed autocorrelation plot. Default is 2.0E-06.
    num : int, optional
        Number of x-values spanning the frequency range to evaluate r2.
        Default is 1000.
    show_plot : bool, optional
        If True, the plot will be shown to the screen. Default is False.
    print_plot : bool, optional
        If True, the plot will be exported as an image. Default is False.
    path : str, optional
        Path of the data file.
    method : Callable
       The callable method corresponding to the input string.
    param : str, optional
        Label of the parameter to correlate with the value of r2. Default is
        "freq_cutoff".
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    fcut : float
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5.
    case : int,
        The case rule from which `fcut` have been selected. Not necessarily
        useful in the current implementation, but it is advisable to keep it
        until proven otherwise.

    """
    tic = time.perf_counter()
 
    # Make sure that the method being passed is allowed
    allowed_methods = {"beads": beads, "custom_beads": custom_beads}
    if method not in allowed_methods:
        raise ValueError("method '{method}' is not implemented")

    algo = allowed_methods[method]
 
    baseline_fitter = Baseline(x_data=x[:scut])

    # log transform of the signal
    z = _log_transform(s[:scut])
    print(f"{'Used points:':<20}{len(z):d}")

    fcut_range = np.geomspace(0.00001, 0.5, num=num, endpoint=False)
 
    # y-data
    r2_val = _r2_array(algo, baseline_fitter, z, fcut_range, param=param,
                       **kwargs)

    ##########################################################################
    # Smoothed data and derivatives
    smooth_d0 = gaussian_filter1d(r2_val,smoothing_window)
    #smooth_d0 = medfilt(r2_val, smoothing_window)
    smooth_d1 = np.gradient(smooth_d0)
    smooth_d2 = np.gradient(smooth_d1)
    min_d1 = argrelmin(smooth_d1)[0]
    max_d1 = argrelmax(smooth_d1)[0]
    d1_min = np.argmin(smooth_d1)
    #@EB not general at all...
    lim_d1_drop = np.where(smooth_d1 < -1E-03)[0][0] 

    # Proto-plateaus from d1 and d2
    tight_d1_flats = find_plateaus(smooth_d1, tol1_0)
    loose_d1_flats = find_plateaus(smooth_d1, tol1_1)
    d2_flats = np.where(np.absolute(smooth_d2) < tol2)[0]

    # Find initial plateau
    tight_continuous = continuous_ranges(tight_d1_flats)
    starting_r2 = np.mean(smooth_d0[tight_continuous[0]])
    starting_end = np.where(
            np.absolute(starting_r2 - r2_val[:d1_min]) < tol0)[0][-1]
    starting_plateau = np.arange(starting_end+1)

    # Remove final plateau if it is tight
    last_r2 = num - 1
    if np.isin(tight_continuous[-1], last_r2).any():
        last_r2 = tight_continuous[-1][0]

    # Plateaus
    plateaus = loose_d1_flats[(loose_d1_flats > starting_plateau[-1]) &
                              (loose_d1_flats < last_r2)]
    secondary_plateaus = np.intersect1d(plateaus, d2_flats)

    # Anchors
    sec_max_d1 = np.intersect1d(secondary_plateaus,max_d1)
    if len(sec_max_d1) == 0:
        p2_start = secondary_plateaus[0]
    else:
        # Make sure this is not on the tail of the initial plateau (p1) by
        # starting p2 at the first max of d1 on the secondary plateaus.
        p2_start = sec_max_d1[0]
    anchors = secondary_plateaus[((secondary_plateaus < lim_d1_drop) &
                                  (secondary_plateaus > p2_start))]

    # Differents cases
    if len(anchors) == 0:
        case = 1
        arg_l = continuous_ranges(secondary_plateaus)[0][-1]
        # Not needed if slope_arg is well chosen?
        slope_thresh = tol1_1*0.5    #@EB temporary?
    else:
        case = 2
        arg_l = anchors[np.argmin(np.absolute(smooth_d1[anchors]))]
    
    ##########################################################################
    # Shift relative to the chosen anchor
    slope_arg = np.where(np.absolute(smooth_d1) >= slope_thresh)[0]
    try:
        cutoff = slope_arg[slope_arg >= arg_l][0]
    except:
        print("WARNING: slope_arg < arg_l.")
        cutoff = arg_l

    fcut = fcut_range[cutoff]
    ##########################################################################

    print(f"Case {case:d}")
    toc = time.perf_counter()
    print(f"Autocorrelation in {toc-tic:0.4f} seconds")
    fi_r2_val = _r2(algo, baseline_fitter, z, fcut, param=param, **kwargs)
    print(f"{'r2 value:':<20}{fi_r2_val:0.4f}")

    # r2 plot
    if show_plot or print_plot:
        r2_plots(fcut_range, r2_val, smooth_d0, smooth_d1, smooth_d2, min_d1,
                 max_d1, starting_plateau[-1], secondary_plateaus, tol1_0,
                 tol1_1,tol2, fcut, fi_r2_val, case=case, show_plot=show_plot,
                 print_plot=print_plot, path=path)

    return fcut, case

###############################################################################
#BEADS baseline correction
def auto_beads(s, x, freq_cutoff=None, show_plot=False, print_plot=False,
               path="./file.txt", method="beads", asymmetry=1.0,
               fit_parabola=True, alpha=None, parabola_len=3):
    """
    Automatic implementation of the Baseline estimation and denoising with
    sparsity (BEADS) algorithm.

    Decomposes the input data into baseline and pure, noise-free signal by
    modeling the baseline as a low pass filter and by considering the signal
    and its derivatives as sparse [1].

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    show_plot : bool, optional
        If True, the plot will be shown to the screen. Default is False.
    print_plot : bool, optional
        If True, the plot will be exported as an image. Default is False.
    path : str, optional
        Path of the data file.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is None, which will calculate its
        value based on the autocorrelation plot of the log-transform from
        Navarro-Huerta [2].
    asymmetry : float, optional
        A number greater than 0 that determines the weighting of negative
        values compared to positive values in the cost function. For example,
        if is 6.0, it will give negative values six times more impact on the
        cost function that positive values. If set to 1 (default), the cost
        function is symmetric, and a value less than 1 will weigh positive
        values more.
    fit_parabola : bool, optional
        If True (default), will fit a parabola to the data and subtract it
        before performing the BEADS fit as suggested in [2]. This ensures the
        endpoints of the fit data are close to 0, which is required by BEADS.
        If the data is already close to 0 on both endpoints, set `fit_parabola`
        to False (but it does not change anything in reality).
    alpha : float, optional
        #@EB will change in pybaselines. If None (default), will automatically
        ajust the value (always to 1 for now).
    parabola_len : int, optional
        Size of the window used, at each ends of the data, to prevent issues
        in fitting a parabola before the baseline correction[2] when the first
        and/or last point is an outlier. If None, will be ajusted to the length
        of the data.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    p : dict
        A dictionary with the various parameters depending of the method used.
    case : int,
        The case rule from which `fcut` have been selected. Not necessarily
        useful in the current implementation, but it is advisable to keep it
        until proven otherwise.

    References
    ----------
    .. [1] Ning, X., et al. Chromatogram baseline estimation and denoising
        using sparsity (BEADS). Chemometrics and Intelligent Laboratory
        Systems, 2014, 139, 156-167.
    .. [2] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in
        complex chromatograms using the BEADS algorithm. Journal of
        Chromatography A, 2017, 1507, 1-10.

    """
    if asymmetry <= 0:
        raise ValueError('asymmetry must be greater than 0')
    
    # Make sure that the method being passed is allowed
    allowed_methods = {"beads": beads, "custom_beads": custom_beads}
    if method not in allowed_methods:
        raise ValueError("method '{method}' is not implemented")
    algo = allowed_methods[method]

    # Limits the range and splits the signal
    peak_regions, sampling, scut= _relevant_regions(s, x)

    # NOTE: The value of `alpha` doesn't need to change when looking for the
    #       best r**2 because of the log transform
    # NOTE: The default setting of `parabola_len=3` is suitable to determine
    #       `fcut` since the signal is log-transformed beforehand.
    method_kwargs = {
            "asymmetry": asymmetry,
            "fit_parabola": fit_parabola,
            "alpha": 1.0,
            "parabola_len": 3,
            "regions": peak_regions,
            "sampling": sampling
            }

    print(f"{'Data points:':<20}{len(s):d}")

    # Cutoff frequency
    if freq_cutoff is None:
        fcut, case = _fcutoff(s, x, scut,
                             show_plot=show_plot, print_plot=print_plot,
                             path=path, method=method, **method_kwargs)
    else:
        if ((freq_cutoff <= 0) or (freq_cutoff >= 0.5)):
            raise ValueError("cutoff frequency must be 0 < freq_cutoff < 0.5")
        fcut = freq_cutoff
        case = 0
    method_kwargs["freq_cutoff"] = fcut

    # Change alpha for the final baseline correction
    if alpha is None:
        alpha=1.0
        method_kwargs.update({"alpha": alpha})  #@EB TO CHANGE WHEN I KNOW HOW

    # Change parabola_len for the final baseline correction
    if parabola_len is None:
        parabola_len=end_window(s)
        method_kwargs.update({"parabola_len": parabola_len})

    print(f"{'Cutoff frequency:':<20}{fcut:E}")
    print(f"{'Asymmetry:':<20}{asymmetry:0.1f}")
    print(f"{'Fit parabola:':<20}{str(fit_parabola):s}")
    print(f"{'alpha:':<20}{alpha:0.2f}")
    print(f"{'parabola_len:':<20}{parabola_len:d}")

    # Final baseline correction
    tic = time.perf_counter()                               #@TEMP

    baseline_fitter = Baseline(x_data=x)
    baseline, p = algo(baseline_fitter, s, **method_kwargs)

    toc = time.perf_counter()                               #@TEMP

    print(f"Baseline correction in {toc-tic:0.4f} seconds") #@TEMP
    return baseline, p, case

