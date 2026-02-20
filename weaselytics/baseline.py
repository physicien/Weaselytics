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
from utils import r2_fct, rm_ends_outliers

def relevant_range(s):
    """
    Limits the signal to the relevant range in order to find the optimal
    cutoff frequency for the BEADS algorithm.

    Parameters
    ----------
    s : array-like
        The signal to limit.

    Returns
    -------
    _last_arg : int
        Index of the last relevant data point of in the signal ``s``.

    """
#    _s = gaussian_filter1d(s,10)
#    window_size = 5
#    _s = np.convolve(s, np.ones(window_size)/window_size, mode='valid')
    # No smoothing, but specific height_n value in case of noisy signal.
    _peaks, _widths = peaks_params(s, height_n=0.01)

    # @EB Signal splitting
#    print(_peaks)
#    print(_widths)
#    print("===========================")
#    print(_peaks/_widths)
#    print("===========================")

#    print(_peaks[np.argmin(_widths)])
#    print(np.min(_widths))
#    print(_widths/np.min(_widths))  # Width outliers?

    _arg_last_peak = _peaks.argmax()
    _last_peak = _peaks[_arg_last_peak]
    _buffer = int(3*np.ceil(_widths)[_arg_last_peak])
    _limmax = _last_peak + _buffer
    if len(s) > _limmax:
        _last_arg = _limmax
    else:
        _last_arg = len(s)
    return _last_arg

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
    _log_s : numpy.ndarray
        The log transformed data.
    
    References
    ----------
    [1] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex
        chromatograms using the BEADS algorithm. Journal of Chromatography A,
        2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057.

    """
    _log_s = np.log10(s-np.min(s)+epsilon)
    return _log_s


def r2_beads(f_cut, s, bl_fitter, asym=1.0, fp=True, hw=None, alpha=1.0):
    """
    Minimal baseline correction with the BEADS algorithm. Used to compute
    the autocorrelation plot.

    Parameters
    ----------

    Returns
    -------

    """
    _bl, _p = bl_fitter.beads(
            s,
            freq_cutoff = f_cut,
            fit_parabola = fp,
            asymmetry = asym,
            smooth_half_window = hw,
            alpha = alpha
            )
    _s_corr = _p["signal"]
    _r2 = r2_fct(_s_corr)
    return _r2

def r2_beads_array(x, y, baseline_fitter, alpha, frequency_range):
    """
    """
    _r2_func = lambda x: r2_beads(x, y, baseline_fitter, alpha)
    _vr2_func = np.vectorize(_r2_func)
    return _vr2_func(frequency_range)

#Frequency cutoff for BEADS
def fcutoff_beads(s, x, args, alpha=1.0, smoothing_window=25, 
                  slope_thresh=-1.0E-04, plateau_thresh=1.0E-04,
                  drop_thresh=0.90):
    """

    """
    tic = time.perf_counter()
   
    path = args.filename    # @EB temporaty

    _last_pt = relevant_range(s)            # @EB split signal here?
    _bl_fitter = Baseline(x_data=x[:_last_pt])
    # log transform of the signal
    _z = log_transform(s[:_last_pt],1)
    print(f"{'Used points:':<20}{len(_z):d}")

    _freq_cutoff_range = np.geomspace(0.00001, 0.5, num=1000, endpoint=False)
    
    # y-data
    r2_val = r2_beads_array(x,_z,_bl_fitter,alpha,_freq_cutoff_range)

    smooth_d0 = gaussian_filter1d(r2_val,smoothing_window)
    smooth_d1 = np.gradient(smooth_d0)
    smooth_d2 = np.gradient(smooth_d1)
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    pos_min_d1 = argrelmin(smooth_d1)[0]
    pos_max_d1 = argrelmax(smooth_d1)[0]
    d1_min = np.argmin(smooth_d1[pos_min_d1])
    # @EB Ajusting some problematic "Case 2"...
#    if ((pos_max_d1[-1] < pos_min_d1[-1]) and
#        (pos_min_d1[-1] == pos_min_d1[d1_min])):    # @EB not general yet...
#        _last_point = np.array(len(_freq_cutoff_range)-1)
#        pos_max_d1 = np.append(pos_max_d1,_last_point)

    # How do we find the right inflection point?
    infl_min = np.argmin(smooth_d1[infls])

#    d0_drops = np.ediff1d(smooth_d0[pos_max_d1])
#    arg_d0_drops = (d0_drops<-0.01).nonzero()
#    rel_max_d1 = pos_max_d1[arg_d0_drops]

    plateau = (np.absolute(smooth_d1) < plateau_thresh)
    arg_plateau = np.where(plateau)[0]                          # @EB plateau
#    patate = pos_max_d1[smooth_d1[pos_max_d1] < 0]
#    patate = pos_max_d1[smooth_d1[pos_max_d1] < plateau_thresh]
#    patate = pos_max_d1[((smooth_d1[pos_max_d1] < 0) &
    patate = pos_max_d1[((smooth_d1[pos_max_d1] < plateau_thresh) &
                          (smooth_d0[pos_max_d1] > drop_thresh))]
    print(pos_max_d1)
    print(patate)
    patate2 = np.intersect1d(arg_plateau,patate)
    print(patate2)
    # Differents cases
    if len(patate2) == 0:
        case = 1
        arg_l = pos_max_d1[pos_max_d1 < patate[0]][-1]
        print(arg_l)
    else:
        case = 2
        arg_l = patate2[-1]


    # Differents cases
#    if len(rel_max_d1) == 0:
#        case = 1
#        arg_l = pos_max_d1[-1]
#    elif len(rel_max_d1) == 1:
#        case = 2
#        arg_l = rel_max_d1[0]
#        arg_r = pos_min_d1[pos_min_d1 > arg_l][0]
#    else:
#        case = 3
#        print(smooth_d0[pos_max_d1])
#        rel_drop_values = d0_drops[arg_d0_drops]
#        tot_drop = np.cumsum(rel_drop_values)

#        optimal_max_d1 = np.argmin(tot_drop[tot_drop>-0.50])

        # @EB -0.08 or -0.095?
#        if ((optimal_max_d1 == 0) and (rel_drop_values[0]>-0.095)):
#            case = 4
#            optimal_max_d1 += 1
#        arg_l = rel_max_d1[optimal_max_d1]
#        arg_r = pos_min_d1[pos_min_d1 > arg_l][0]
##        print("==========")
##        print("TEST")
##        print(tot_drop)
##        print(optimal_max_d1)
##        print(tot_drop[optimal_max_d1])
##        print("==========")

    # @EB
    r2_lim_l = r2_val[arg_l]

    slope_arg = np.where(smooth_d1 <= slope_thresh)[0]
    _arg_cutoff = slope_arg[slope_arg >= arg_l][0]
    _freq_cutoff = _freq_cutoff_range[_arg_cutoff]

    print(f"Case {case:d}")
    # @EB Ajuster le calcul suivant?
    r2_ymin = r2_val[infls[infl_min-1]]-0.05  #only for the r2 plot limit

    toc = time.perf_counter()
    print(f"Autocorrelation in {toc-tic:0.4f} seconds")
    fi_r2_val = r2_beads(_freq_cutoff,_z,_bl_fitter)
    print(f"{'r2 value:':<20}{fi_r2_val:0.4f}")

    
    if args.show or args.print:
        xx = _freq_cutoff_range
        yy = r2_val
#        fig = plt.figure(figsize=[6.4,9.6],num="Autocorrelation plots")    #@EB
        fig = plt.figure(figsize=[9.4,9.6],num="Autocorrelation plots")
        gs = fig.add_gridspec(3, hspace=0)
        axs = gs.subplots(sharex=True)
        axs[0].semilogx(xx, yy, marker='.', ls='',label=r'$r^2$',ms=3)
        axs[0].semilogx(xx, smooth_d0, marker='', ls='-',
                        label=r'$r^2_\text{smooth}$',ms=3)
        axs[1].semilogx(xx, smooth_d1, label='First Derivative')

        axs[1].fill_between(xx, 0, 1,
                            where=np.absolute(smooth_d1) < plateau_thresh,
                            color='green', alpha=0.3, 
                            transform=axs[1].get_xaxis_transform())

        axs[2].semilogx(xx, smooth_d2, label='Second Derivative')
        for ax in axs.flat:
            for i, infl in enumerate(infls, 1):
                ax.axvline(x=xx[infl], c='k')#, label=f'Inflection Point {i}')
            ax.axvline(x=_freq_cutoff,c='tab:red',ls='dashed'),
            ax.label_outer()
        for md1 in pos_min_d1:
            axs[1].axvline(x=xx[md1],ymax=0.5,c='tab:pink',ls='dashed')
        for md1 in pos_max_d1:
            axs[1].axvline(x=xx[md1],ymin=0.5,c='tab:green',ls='dashed')
        axs[0].annotate(f'{fi_r2_val:0.4f}',
                        xy=(_freq_cutoff,1.01),
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
        axs[0].set_ylim(r2_ymin,1.0)
        axs[1].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
        axs[0].legend()
        plt.tight_layout()
        if args.show:
            plt.show()
        if args.print:
            filename = os.path.splitext(os.path.basename(path))[0]
            plt.savefig(f"r2_plots/{filename}_r2.png")
        plt.close()
    return [_freq_cutoff,case]

###############################################################################
#BEADS baseline correction
def beads(s, x, args, asym=1.0, fp=True, hw=None, alpha=1.):
    """

    """
    # Read Navarro-Huerta et al (2017)
    # Section 3.2: Monitoring the autocorrelation to explore the BEADS
    #              working parameters
    # 3.3.2. Chromatograms involving peaks with extremely different magnitude
    # Section 3.4: Autocorrelation plot using the baseline-corrected signal
    # Section 3.5: Application of the assisted BEADS

    _baseline_fitter = Baseline(x_data=x)
    _signal = rm_ends_outliers(s)
    print(f"{'Data points:':<20}{len(_signal):d}")
    _fcut, _case = fcutoff_beads(_signal, x, args)
    print(f"{'Cutoff frequency:':<20}{_fcut:E}")
    print(f"{'Asymmetry:':<20}{asym:0.1f}")
    print(f"{'Fit parabola:':<20}{str(fp):s}")
    print(f"{'Half window:':<20}{str(hw):s}")
    print(f"{'alpha:':<20}{alpha:0.2f}")

    tic = time.perf_counter()
    _bl, _p = _baseline_fitter.beads(
            _signal,
            freq_cutoff=_fcut,
            fit_parabola=fp,
            asymmetry=asym,
            smooth_half_window=hw,
            alpha=alpha
            )
    toc = time.perf_counter()
    print(f"Baseline correction in {toc-tic:0.4f} seconds")
    return [_bl,_p,_case]

