#!/usr/bin/python
# coding: utf-8
"""
Functions to perform the baseline correction.
"""
import os
import numpy as np
from pybaselines import Baseline
from scipy.signal import argrelmin, argrelmax
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import time                             #@EB temporary?

from peakfitting import peaks_params
from utils import (r2_dw, rm_ends_outliers, continuous_ranges, find_plateaus,
                   merge_intervals
                   )

def relevant_range(s, x, tol=6.):
    """
    Limits the signal to the relevant range in order to find the optimal
    cutoff frequency for the BEADS algorithm.

    Parameters
    ----------
    s : array-like
        The signal to limit.
    x : array-like
        The x of the signal.......
    tol : float, optional
        Threshold 

    Returns
    -------
    _last_arg : int
        Index of the last relevant data point of in the signal ``s``.

    """
    _s = gaussian_filter1d(s,3)
    _peaks, _widths = peaks_params(_s, height_n=0.50, width=3, rel_prom_p=0.01,
                                   adapt=True)

    #@EB Signal splitting
    width_per_x = _widths/x[_peaks]
    # In case of very tall and large peaks (see acetonitrile)
    exception = ((s[_peaks] > 20) & (width_per_x < 11))
#    exception = False
    rel_peaks = _peaks[((width_per_x < tol) | exception)]
    rel_widths = _widths[((width_per_x < tol) | exception)]
    ratio_w = rel_widths/np.min(rel_widths)

    print("===========================")
    print(np.round(x[rel_peaks],2))
    print(rel_peaks)
    print(np.round(rel_widths,2))
    print(np.round(ratio_w,2))
#    print(np.round(rel_widths/x[rel_peaks],2))
    print("===========================")
    
    # Assuming that `rel_widths` is the FWHM and that the peak is gaussian,
    # `buffer` is equal to the full peak width
    buffer = np.ceil(0.85*rel_widths).astype(int)
    lim_inf = rel_peaks - buffer
    lim_sup = rel_peaks + buffer
    limits = np.array([lim_inf,lim_sup]).T

    large_lim = limits[ratio_w > 1] # All peaks or only the larger ones?
    merged_lim = merge_intervals(np.copy(large_lim))
    if len(merged_lim) == 0:
        merged_lim = ((None, None),)
        sampling = 1
    else:
        # Because values in regions must be less than len(data)
        if merged_lim[-1,-1] >= len(s):
            merged_lim[-1,-1] = len(s) - 1
        #@EB Why divide by "2"?
        sampling = np.ceil((merged_lim[:,1]-merged_lim[:,0])/2.0
                           /np.min(rel_widths)).astype(int)
    print(sampling)
    print(merged_lim)

    _arg_last_peak = rel_peaks.argmax()
    _pos_last_peak = rel_peaks[_arg_last_peak]
    _last_buffer = int(np.ceil(2*rel_widths)[_arg_last_peak])
    _limmax = _pos_last_peak + _last_buffer
    if len(s) > _limmax:
        _last_arg = _limmax
    else:
        _last_arg = len(s)
    return _last_arg, merged_lim, sampling

def log_transform(s,epsilon):
    """
    Log transformation used in the calculation of BEADS frequency cutoff. For
    further information, see [1].

    Parameters
    ----------
    s : array-like
        The data to transform.
    epsilon : float
        An arbitrary positive offset. The larger the offset, the less
        aggressive the pre-treatment. Default is 1. This value is appropriate
        regarding the magnitude of the signals being processed, which reach
        maxima around 500–10,000. Another reason for selecting ``epsilon = 1``
        is because if ``yi = min (y)``, then 
        ``log (yi − min(y) + 1) = log 1 = 0``.

    Returns
    -------
    log_s : numpy.ndarray
        The log transformed data.
    
    References
    ----------
    [1] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex
        chromatograms using the BEADS algorithm. Journal of Chromatography A,
        2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057.

    """
    log_s = np.log10(s - np.min(s) + epsilon)
    return log_s

def beads(s, freq_cutoff, bl_fitter, asymmetry=1.0, fit_parabola=True,
          alpha=1.0, **kwargs):
    """
    Baseline correction with the BEADS algorithm.
    """
    bl, params = bl_fitter.beads(
            s,
            freq_cutoff=freq_cutoff,
            fit_parabola=fit_parabola,
            asymmetry=asymmetry,
            alpha=alpha
            )
    return bl, params

def custom_beads(s, freq_cutoff, bl_fitter, regions=((None,None),),
                 sampling=1, asymmetry=1.0, fit_parabola=True, alpha=1.0,
                 **kwargs):
    """
    Baseline correction with the custom BEADS algorithm.
    """
    beads_kwargs = {'freq_cutoff': freq_cutoff,
                    'fit_parabola': fit_parabola,
                    'asymmetry': asymmetry,
                    'alpha': alpha
                    }

    bl, params = bl_fitter.custom_bc(
            s,
            method="beads",
            regions=regions,
            sampling=sampling,
            lam=None,
            method_kwargs=beads_kwargs
            )
    
    noise_fit = (params['y_fit'] - params['baseline_fit']
                 - params['method_params']['signal'])
    params['noise'] = np.interp(bl_fitter.x, params['x_fit'], noise_fit)
    params['signal'] = s - bl - params['noise']
#    params['signal'] = np.interp(bl_fitter.x, params['x_fit'],
#                                 params['method_params']['signal'])
    return bl, params

def r2_fcut(fcut, s, bl_fitter, algo, **kwargs):
    """
    Compute the squared of the autocorrelation level `r` for a given cutoff
    frequency used for the substraction of the baseline.

    Parameters
    ----------

    Returns
    -------

    """
    _, params = algo(s, fcut, bl_fitter, **kwargs)
    s_corr = params["signal"]
    r2 = r2_dw(s_corr)
    return r2

def r2_fcut_array(x, y, baseline_fitter, fcut_range, algo, **kwargs):
    """
    Define a vectorized function 
    """
    r2_func = lambda x: r2_fcut(x, y, baseline_fitter, algo, **kwargs)
    vr2_func = np.vectorize(r2_func)
    return vr2_func(fcut_range)

def r2_plots(x, r2, sm_d0, sm_d1, sm_d2, pl_thresh, pl_ext_thresh, freq_cutoff,
             final_r2, case=0, show_plot=False, print_plot=False,
             path="./file.txt"):
    """
    """
    pos_min_d1 = argrelmin(sm_d1)[0]
    pos_max_d1 = argrelmax(sm_d1)[0]
    infls = np.where(np.diff(np.sign(sm_d2)))[0]

    #@EB
#    fig = plt.figure(figsize=[6.4,9.6],num="Autocorrelation plots")
    fig = plt.figure(figsize=[9.4,9.6],num="Autocorrelation plots")
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].semilogx(x, r2, marker='.', ls='',label=r'$r^2$',ms=3)
    axs[0].semilogx(x, sm_d0, marker='', ls='-',
                    label=r'$r^2_\text{smooth}$',ms=3)

    axs[1].fill_between(x, 0, 1,
                        where=np.absolute(sm_d1) < pl_thresh,
                        color="none", ec="white", alpha=0.3, fc="purple", 
                        hatch="//", hatch_linewidth=4,
                        transform=axs[1].get_xaxis_transform())
    axs[1].fill_between(x, 0, 1,
                        where=np.absolute(sm_d1) < pl_ext_thresh,
                        color='orange', alpha=0.3,
                        transform=axs[1].get_xaxis_transform())

    axs[1].semilogx(x, sm_d1, label='First Derivative')
    axs[2].semilogx(x, sm_d2, label='Second Derivative')
    for ax in axs.flat:
        for i, infl in enumerate(infls, 1):
            ax.axvline(x=x[infl], c='k', lw=0.5)#, label=f'Inflection Point {i}')
        ax.axvline(x=freq_cutoff,c='tab:red',ls='dashed'),
        ax.label_outer()
    for md1 in pos_min_d1:
        axs[1].axvline(x=x[md1],ymax=0.5,c='tab:pink',ls='dashed')
    for md1 in pos_max_d1:
        axs[1].axvline(x=x[md1],ymin=0.5,c='tab:green',ls='dashed')
    axs[0].annotate(f'{final_r2:0.4f}',
                    xy=(freq_cutoff,1.01),
                    xycoords=("data","axes fraction"),
                    ha='center',
                    color='tab:red'
                    )
    axs[0].annotate(f"{'Case:'}{case:>3d}",
                xy=(0.00,1.01),
                xycoords=("axes fraction"),
                ha='left',
                color='tab:red'
                )
    axs[2].set_xlabel('Cutoff frequency')
    axs[0].set_ylabel(r'$r^2_{y-b}$')
    axs[1].set_ylabel(r"$r^2_{y-b}$'")
    axs[2].set_ylabel(r"$r^2_{y-b}$''")

    # How do we find the right inflection point?
    # @EB Ajuster le calcul suivant?
    infl_min = np.argmin(sm_d1[infls])
    r2_ymin = r2[infls[infl_min-1]]-0.05  #only for the r2 plot limit
    axs[0].set_ylim(r2_ymin,1.0)
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
    axs[2].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
    axs[0].legend()
    plt.tight_layout()
    if show_plot:
        plt.show()
    if print_plot:
        # @EB temporaty
        _filename = os.path.splitext(os.path.basename(path))[0]
        plt.savefig(f"r2_plots/{_filename}_r2.png")
    plt.close()

#Frequency cutoff for BEADS
def fcutoff(s, x, last_pt, smoothing_window=15,
            slope_thresh=5.0E-05, tol0=3.0E-05, tol1=2.6E-04, show_plot=False,
            print_plot=False, path="./file.txt", method="beads", **kwargs):
    """

    """
    tic = time.perf_counter()
 
    # Make sure that the method being passed is allowed
    allowed_methods = {"beads": beads, "custom_beads": custom_beads}
    if method not in allowed_methods:
        raise ValueError("method '{method}' is not implemented")

    algo = allowed_methods[method]
 
    bl_fitter = Baseline(x_data=x[:last_pt])

    # log transform of the signal
    z = log_transform(s[:last_pt],1)
    print(f"{'Used points:':<20}{len(z):d}")

    fcut_range = np.geomspace(0.00001, 0.5, num=1000, endpoint=False)
 
    # y-data
    r2_val = r2_fcut_array(x, z, bl_fitter, fcut_range, algo, **kwargs)

##############################################################################
    smooth_d0 = gaussian_filter1d(r2_val,smoothing_window)
    smooth_d1 = np.gradient(smooth_d0)
    smooth_d2 = np.gradient(smooth_d1)
    pos_min_d1 = argrelmin(smooth_d1)[0]
    pos_max_d1 = argrelmax(smooth_d1)[0]

#    d0_drops = np.ediff1d(smooth_d0[pos_max_d1])
#    arg_d0_drops = (d0_drops<-0.01).nonzero()
#    rel_max_d1 = pos_max_d1[arg_d0_drops]

    tight_plateaus = find_plateaus(smooth_d1, tol0)
#    loose_plateaus = find_plateaus(smooth_d1, tol1,
#                                   tol0)
    loose_plateaus = find_plateaus(smooth_d1, tol1)

    tight_cont_reg = continuous_ranges(tight_plateaus)
    starting_r2 = np.mean(smooth_d0[tight_cont_reg[0]])
    starting_plateau = np.where(
            np.absolute(starting_r2 - smooth_d0) < 2E-03)[0]

    secondary_plateaus = loose_plateaus[loose_plateaus > starting_plateau[-1]]
    #@EB not general at all...
    lim_r2 = 0.6
    min_r2 = np.min(r2_val)
    if  lim_r2 < min_r2:
        lim_r2 = min_r2
    lim_d0_drop = np.where(r2_val <= lim_r2)[0][0]
    lim_d1_drop = np.where(smooth_d1 < -1E-03)[0][0]
    
    test = np.intersect1d(secondary_plateaus,pos_max_d1)
    if len(test) == 0:
        test2 = secondary_plateaus[0]
    else:
        test2 = test[0]

    pot_anchors = secondary_plateaus[((secondary_plateaus < lim_d0_drop)
                                      & (secondary_plateaus < lim_d1_drop)
                                      & (secondary_plateaus > test2))]

    # Differents cases
    if len(pot_anchors) == 0:
        case = 1
        arg_l = np.intersect1d(starting_plateau, pos_max_d1)[-1] 
    else:
        case = 2
        arg_l = pot_anchors[np.argmin(np.absolute(smooth_d1[pot_anchors]))]
        drop = starting_r2 - smooth_d0[arg_l]
        print(f"{'drop:':<20}{drop:E}")
        if drop > 8E-02:
            case = 3
            arg_l = np.intersect1d(starting_plateau, pos_max_d1)[-1] 
#            arg_l = np.intersect1d(tight_plateaus[tight_plateaus < test2],
#                                   pos_max_d1[pos_max_d1 <= test2])[-1]
    
##############################################################################
    slope_arg = np.where(np.absolute(smooth_d1) >= slope_thresh)[0]
    try:
        cutoff = slope_arg[slope_arg >= arg_l][0]
    except:
        print("WARNING: slope_arg < arg_l.")
        cutoff = arg_l
    fcut = fcut_range[cutoff]
##############################################################################

    print(f"Case {case:d}")
    toc = time.perf_counter()
    print(f"Autocorrelation in {toc-tic:0.4f} seconds")
    fi_r2_val = r2_fcut(fcut, z, bl_fitter, algo, **kwargs)
    print(f"{'r2 value:':<20}{fi_r2_val:0.4f}")

    # r2 plot
    if show_plot or print_plot:
        r2_plots(fcut_range, r2_val, smooth_d0, smooth_d1, smooth_d2,
                 tol0, tol1, fcut, fi_r2_val, case=case, show_plot=show_plot,
                 print_plot=print_plot, path=path)

    return fcut, case

###############################################################################
#BEADS baseline correction
def auto_beads(s, x, freq_cutoff=None, show_plot=False, print_plot=False,
               path="./file.txt", method="beads", asymmetry=1.0,
               fit_parabola=True, alpha=1.0):
    """
    Automatic implementation of the Baseline estimation and denoising with
    sparsity (BEADS) algorithm.

    Decomposes the input data into baseline and pure, noise-free signal by
    modeling the baseline as a low pass filter and by considering the signal
    and its derivatives as sparse [1].

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the measured signal, with N data points.
    x : array-like, shape (N,)
        The x-values of the measured signal, with N data points.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is None, which will calculate its
        value based on the autocorrelation plot of the log-transform from
        Navarro-Huerta [2].
    asymmetry : float, optional
        A number greater than 0 that determines the weighting of negative
        values compared to positive values in the cost function. For example,
        if is 6.0, it will give negative values six times more impact on the
        cost function that positive values. If set to 1 (the default) for a
        symmetric cost function, or a value less than 1 to weigh positive
        values more.
    fit_parabola : bool, optional
        If True (default), will fit a parabola to the data and subtract it
        before performing the BEADS fit as suggested in [2]. This ensures the
        endpoints of the fit data are close to 0, which is required by BEADS.
        If the data is already close to 0 on both endpoints, set `fit_parabola`
        to False (but it does not change anything in reality).
    alpha : float, optional
        ###########################  TO CONTINUE  ############################

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)

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

    # Takes care of possible outliers at both ends of the signal
    signal = rm_ends_outliers(s)
    # Limits the range and splits the signal
    last_pt, peaks_range, sampling = relevant_range(signal,x)

    method_kwargs = {
            "asymmetry": asymmetry,
            "fit_parabola": fit_parabola,
            "alpha": alpha,
            "regions": peaks_range,
            "sampling": sampling
            }

    print(f"{'Data points:':<20}{len(signal):d}")

    if freq_cutoff is None:
        fcut, case = fcutoff(signal, x, last_pt,
                             show_plot=show_plot, print_plot=print_plot,
                             path=path, method=method, **method_kwargs)
    else:
        if ((freq_cutoff <= 0) or (freq_cutoff >= 0.5)):
            raise ValueError("cutoff frequency must be 0 < freq_cutoff < 0.5")
        fcut = freq_cutoff
        case = 0

    print(f"{'Cutoff frequency:':<20}{fcut:E}")
    print(f"{'Asymmetry:':<20}{asymmetry:0.1f}")
    print(f"{'Fit parabola:':<20}{str(fit_parabola):s}")
    print(f"{'alpha:':<20}{alpha:0.2f}")

    # Final baseline correction
    tic = time.perf_counter()                               #@TEMP

    baseline_fitter = Baseline(x_data=x)
    baseline, p = algo(signal, fcut, baseline_fitter, **method_kwargs)

    toc = time.perf_counter()                               #@TEMP

    print(f"Baseline correction in {toc-tic:0.4f} seconds") #@TEMP
    return baseline, p, case

