# coding: utf-8
"""
Functions to perform the baseline correction.
"""
import time  #@EB temporary?
from collections.abc import Callable

import numpy as np
from pybaselines import Baseline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax, argrelmin  #, medfilt

from weaselytics.plot import r2_plots
from weaselytics.utils import (
    continuous_ranges,
    end_window,
    find_flat,
    find_plateaus,
    merge_intervals,
    peaks_params,
    r2_dw,
)


def _relevant_regions(
    s: np.ndarray, x: np.ndarray, tol: float = 6.
) -> tuple[np.ndarray | None, np.ndarray, int]:
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
    peak_regions : array-like, shape (M,2), or None
        The two dimensional array containing the start and stop indices for
        each region containing a relevant peak. Each region is defined as
        ``data[start:stop]``. `None` means no relevant peaks found.
    sampling : array-like of shape (M,)
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
    #       For a Gaussian: FWHM ≈ 2.355σ. To capture ~95 % of the peak
    #       area we need ±2σ from the center. That's 4σ total =
    #       4 / 2.355 ≈ 1.7 × FWHM, so each side gets 0.85 × FWHM.
    _FWHM_TO_HALF_PEAK = 0.85
    buffer = np.ceil(_FWHM_TO_HALF_PEAK * rel_widths).astype(int)
    left_lim = rel_peaks - buffer
    right_lim = rel_peaks + buffer
    full_widths = np.array([left_lim,right_lim]).T

    # Peak regions and sampling
    large_peaks = full_widths[ratio_w > 1]      # Ignore the narrowest peak
    peak_regions = merge_intervals(large_peaks)
    if len(peak_regions) == 0:
        peak_regions = None
        sampling = np.array([1])
    else:
        # Because values in regions must be less than len(data)
        if peak_regions[-1,-1] >= len(s):
            peak_regions[-1,-1] = len(s) - 1
        # The region width covers 2×buffer per peak (= 2 × 0.85 × FWHM),
        # so dividing by 2 converts it to buffer units. Combined with the
        # division by rel_widths (FWHM), an isolated peak yields sampling≈1
        # (keep all points). Without the /2, sampling would be ≈2, making
        # the baseline unnecessarily stiff for isolated peaks.
        sampling = np.ceil((peak_regions[:,1]-peak_regions[:,0])/2
                           /np.min(rel_widths)).astype(int)

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

def _log_transform(s: np.ndarray, epsilon: float = 1) -> np.ndarray:
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

def _beads(baseline_fitter: Baseline, s: np.ndarray,
           freq_cutoff: float = 0.005, asymmetry: float = 1.0,
           fit_parabola: bool = True, alpha: float = 1.0,
           parabola_len: int = 3, **kwargs
           ) -> tuple[np.ndarray, dict]:
    r"""Wrap ``pybaselines.Baseline.beads``.

    See `auto_beads` for parameter details.
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

def _custom_beads(baseline_fitter: Baseline, s: np.ndarray,
                  regions: tuple | np.ndarray | None = None,
                  sampling: int | np.ndarray | None = None,
                  freq_cutoff: float = 0.005, asymmetry: float = 1.0,
                  fit_parabola: bool = True, alpha: float = 1.0,
                  parabola_len: int = 3, **kwargs
                  ) -> tuple[np.ndarray, dict]:
    """Customized BEADS with per-region stiffness control.

    Extends `_beads` by splitting the signal into regions, each with its
    own sampling density. See `auto_beads` for common parameter details.

    Parameters
    ----------
    regions : array-like, shape (M,2), optional
        Start and stop indices for each region containing a relevant peak.
        ``data[start:stop]``. If `None` (default), uses all points.
    sampling : int or array-like, optional
        Sampling step size for each region in `regions`. Default 1.

    Returns
    -------
    bl : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        See `auto_beads` for details.
    """
    if regions is None:
        regions = ((None, None),)
    if sampling is None:
        sampling = 1

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

def _r2(
    algo: Callable[..., tuple[np.ndarray, dict]],
    baseline_fitter: Baseline, y: np.ndarray,
    p: float, param: str = "freq_cutoff", **kwargs
) -> float:
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

def _r2_array(
    algo: Callable[..., tuple[np.ndarray, dict]],
    baseline_fitter: Baseline,
    signal: np.ndarray, param_range: np.ndarray,
    param: str = "freq_cutoff", **kwargs
) -> np.ndarray:
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
    def _r2_wrapper(x):
        return _r2(algo, baseline_fitter, signal, x, param=param, **kwargs)
    vr2_func = np.vectorize(_r2_wrapper)
    vr2 = vr2_func(param_range)
    return vr2

def _fcutoff(s: np.ndarray, x: np.ndarray, scut: int,
            smoothing_window: int = 15, slope_thresh: float = 5.0E-05,
            tol0: float = 1.0E-03, tol1_0: float = 1.0E-05,
            tol1_1: float = 5.0E-04, tol2: float = 2.0E-06,
            num: int = 1000,
            method: str = "beads", param: str = "freq_cutoff", **kwargs
            ) -> tuple[float, int, dict]:
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
        Default is 15.
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
    plot_data : dict
        Dictionary of internal variables needed to produce the r2 diagnostic
        plot. Empty dict if no plotting was requested.

    """
    tic = time.perf_counter()

    # Make sure that the method being passed is allowed
    allowed_methods = {"beads": _beads, "custom_beads": _custom_beads}
    if method not in allowed_methods:
        raise ValueError(f"method '{method}' is not implemented")

    algo = allowed_methods[method]

    baseline_fitter = Baseline(x_data=x[:scut])

    # log transform of the signal
    z = _log_transform(s[:scut])
    print(f"{'Used points:':<20}{len(z):d}")

    fcut_range = np.geomspace(0.00001, 0.5, num=num, endpoint=False)

    # y-data
    r2_val = _r2_array(algo, baseline_fitter, z, fcut_range, param=param,
                       **kwargs)
    #####
    test_plateaus, ends, test, test3 = find_plateaus(r2_val)

    #####

    ##########################################################################
    # Smoothed data and derivatives
    smooth_d0 = gaussian_filter1d(r2_val,smoothing_window)
    #smooth_d0 = medfilt(r2_val, smoothing_window)
    smooth_d1 = np.gradient(smooth_d0)
    smooth_d2 = np.gradient(smooth_d1)
    min_d1 = argrelmin(smooth_d1)[0]
    max_d1 = argrelmax(smooth_d1)[0]
    d1_min = np.argmin(smooth_d1)
    #EB not general at all...
    lim_d1_drop = np.where(smooth_d1 < -1E-03)[0][0]

    # Proto-plateaus from d1 and d2
    tight_d1_flats = find_flat(smooth_d1, tol1_0)
    loose_d1_flats = find_flat(smooth_d1, tol1_1)
    d2_flats = np.where(np.absolute(smooth_d2) < tol2)[0]

    # Find initial plateau
    tight_continuous = continuous_ranges(tight_d1_flats)
    starting_r2 = np.mean(smooth_d0[tight_continuous[0]])
    starting_end = np.where(
            np.absolute(starting_r2 - r2_val[:d1_min]) < tol0)[0][-1]
    starting_plateau = np.arange(starting_end+1)

    # Remove final plateau if it is tight
    last_r2 = num - 1
    if np.isin(last_r2, tight_continuous[-1]).any():
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
    except IndexError:
        print("WARNING: slope_arg < arg_l.")
        cutoff = arg_l

    fcut = fcut_range[cutoff]
    ##########################################################################

    print(f"Case {case:d}")
    toc = time.perf_counter()
    print(f"Autocorrelation in {toc-tic:0.4f} seconds")
    fi_r2_val = _r2(algo, baseline_fitter, z, fcut, param=param, **kwargs)
    print(f"{'r2 value:':<20}{fi_r2_val:0.4f}")

    plot_data = {
        "fcut_range": fcut_range,
        "r2_val": r2_val,
        "smooth_d0": smooth_d0,
        "test": test,
        "test3": test3,
        "min_d1": min_d1,
        "max_d1": max_d1,
        "ends": ends,
        "secondary_plateaus": secondary_plateaus,
        "test_plateaus": test_plateaus,
        "tol1_1": tol1_1,
        "tol2": tol2,
        "fcut": fcut,
        "fi_r2_val": fi_r2_val,
        "case": case,
    }
    return fcut, case, plot_data

###############################################################################
#BEADS baseline correction
def auto_beads(s: np.ndarray, x: np.ndarray,
               freq_cutoff: float | None = None, show_plot: bool = False,
               print_plot: bool = False, path: str = "./file.txt",
               output_dir: str = "results",
               method: str = "beads", asymmetry: float = 1.0,
               fit_parabola: bool = True, alpha: float | None = None,
               parabola_len: int | None = 3
               ) -> tuple[np.ndarray, dict, int]:
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
        adjust the value (always to 1 for now).
    parabola_len : int, optional
        Size of the window used, at each ends of the data, to prevent issues
        in fitting a parabola before the baseline correction[2] when the first
        and/or last point is an outlier. If None, will be adjusted to the length
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
    allowed_methods = {"beads": _beads, "custom_beads": _custom_beads}
    if method not in allowed_methods:
        raise ValueError(f"method '{method}' is not implemented")
    algo = allowed_methods[method]

    # Limits the range and splits the signal.
    # Only needed when freq_cutoff is auto-selected (needs scut) or when
    # using custom_beads (needs regions/sampling).
    if method == "custom_beads" or freq_cutoff is None:
        peak_regions, sampling, scut = _relevant_regions(s, x)
    else:
        peak_regions = None
        sampling = None
        scut = None

    # NOTE: The value of `alpha` doesn't need to change when looking for the
    #       best r**2 because of the log transform
    # NOTE: The default setting of `parabola_len=3` is suitable to determine
    #       `fcut` since the signal is log-transformed beforehand.
    #       regions and sampling are only relevant for custom_beads.
    method_kwargs = {
            "asymmetry": asymmetry,
            "fit_parabola": fit_parabola,
            "alpha": 1.0,
            "parabola_len": 3,
            }
    if method == "custom_beads":
        method_kwargs.update(regions=peak_regions, sampling=sampling)

    print(f"{'Data points:':<20}{len(s):d}")

    # Cutoff frequency
    if freq_cutoff is None:
        fcut, case, plot_data = _fcutoff(
            s, x, scut, method=method, **method_kwargs)
    else:
        if ((freq_cutoff <= 0) or (freq_cutoff >= 0.5)):
            raise ValueError("cutoff frequency must be 0 < freq_cutoff < 0.5")
        fcut = freq_cutoff
        case = 0
        plot_data = {}
    if show_plot or print_plot:
        r2_plots(
            plot_data["fcut_range"], plot_data["r2_val"],
            plot_data["smooth_d0"], plot_data["test"],
            plot_data["test3"], plot_data["min_d1"],
            plot_data["max_d1"], plot_data["ends"],
            plot_data["secondary_plateaus"], plot_data["test_plateaus"],
            plot_data["tol1_1"], plot_data["tol2"],
            plot_data["fcut"], plot_data["fi_r2_val"],
            case=plot_data["case"],
            show_plot=show_plot, print_plot=print_plot,
            path=path, output_dir=output_dir,
        )
    method_kwargs = {**method_kwargs, "freq_cutoff": fcut}

    # Change alpha for the final baseline correction
    # @EB TO CHANGE WHEN I KNOW HOW TO DO IT...
    if alpha is None:
        alpha=1.0
        method_kwargs["alpha"] = alpha

    # Change parabola_len for the final baseline correction
    if parabola_len is None:
        parabola_len=end_window(s)
        method_kwargs["parabola_len"] = parabola_len

    print(f"{'Cutoff frequency:':<20}{fcut:0.4E}")
    print(f"{'Asymmetry:':<20}{asymmetry:0.1f}")
    print(f"{'Fit parabola:':<20}{str(fit_parabola):s}")
    print(f"{'alpha:':<20}{alpha:0.2f}")
    print(f"{'parabola_len:':<20}{parabola_len:d}")

    # Final baseline correction
    tic = time.perf_counter()                               #@TEMP

    baseline_fitter = Baseline(x_data=x)
    baseline, params = algo(baseline_fitter, s, **method_kwargs)

    toc = time.perf_counter()                               #@TEMP

    print(f"Baseline correction in {toc-tic:0.4f} seconds") #@TEMP
    return baseline, params, case

