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


def r2_beads(f_cut, s, bl_fitter, asym=1.0, fit_parabola=True,
             half_window=None, alpha=1.0):
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
            fit_parabola = fit_parabola,
            asymmetry = asym,
            smooth_half_window = half_window,
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


def r2_plots(x, r2, sm_d0, sm_d1, sm_d2, pl_thresh, pl_ext_thresh, freq_cutoff,
             final_r2, case=0, show_plot=False, print_plot=False,
             path="./file.txt"):
    """
    """
#    x = _freq_cutoff_range
#    r2 = r2_val
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
def fcutoff_beads(s, x, alpha=1.0, smoothing_window=15, slope_thresh=-1.0E-04,
                  plateau_thresh=5.0E-05, plateau_ext_thresh=1.1E-04,
                  drop_thresh=0.55, show_plot=False, print_plot=False,
                  path="./file.txt"):
    """

    """
    tic = time.perf_counter()
   
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
    pos_max_d1 = argrelmax(smooth_d1)[0]

#    d0_drops = np.ediff1d(smooth_d0[pos_max_d1])
#    arg_d0_drops = (d0_drops<-0.01).nonzero()
#    rel_max_d1 = pos_max_d1[arg_d0_drops]

    plateau = (np.absolute(smooth_d1) < plateau_thresh)
    arg_plateau = np.where(plateau)[0]                          # @EB plateau
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

# @EB HERE
#        next_patate = np.setdiff1d(patate,patate2)
        next_patate = np.setdiff1d(pos_max_d1,patate2)
        if len(next_patate) != 0:
            next_plateau = next_patate[0]
            print(f"{'r2[arg_l]':<20}{smooth_d0[arg_l]:0.4f}")
            print(f"{'d1[next_plateau]':<20}{smooth_d1[next_plateau]:E}")
            if ((smooth_d0[arg_l] >= 0.995) and
                (np.absolute(smooth_d1[next_plateau]) < plateau_ext_thresh)):
                case = 3
                arg_l = next_plateau
            elif smooth_d0[arg_l] >= 0.998:
                case = 4
                arg_l = next_plateau

    # @EB
#    r2_lim_l = r2_val[arg_l]

    slope_arg = np.where(smooth_d1 <= slope_thresh)[0]
    try:
        _arg_cutoff = slope_arg[slope_arg >= arg_l][0]
    except:
        print("WARNING: slope_arg < arg_l.")
        _arg_cutoff = arg_l
    _freq_cutoff = _freq_cutoff_range[_arg_cutoff]

    print(f"Case {case:d}")

    toc = time.perf_counter()
    print(f"Autocorrelation in {toc-tic:0.4f} seconds")
    fi_r2_val = r2_beads(_freq_cutoff,_z,_bl_fitter)
    print(f"{'r2 value:':<20}{fi_r2_val:0.4f}")
    print("=================================")
    max_d1_val = smooth_d1[arg_l]
    print(f"{'arg_l d1 value:':<20}{max_d1_val:E}")
    r2_d1_val = smooth_d1[_arg_cutoff]
    print(f"{'cutoff d1 value:':<20}{r2_d1_val:E}")
    r2_d2_val = smooth_d2[_arg_cutoff]
    print(f"{'cutoff d2 value:':<20}{r2_d2_val:E}")
    print("=================================")

    # r2 plot
    if show_plot or print_plot:
        r2_plots(_freq_cutoff_range, r2_val, smooth_d0, smooth_d1, smooth_d2,
                 plateau_thresh, plateau_ext_thresh, _freq_cutoff, fi_r2_val,
                 case=case, show_plot=show_plot, print_plot=print_plot,
                 path=path)

    return [_freq_cutoff,case]

###############################################################################
#BEADS baseline correction
def auto_beads(s, x, freq_cutoff=None, asym=1.0, fit_parabola=True,
               half_window=None, alpha=1., show_plot=False, print_plot=False,
               path="./file.txt"):
    """
    Automatic implementation of the Baseline estimation and denoising with
    sparsity (BEADS) algorithm.

    Decomposes the input data into baseline and pure, noise-free signal by
    modeling the baseline as a low pass filter and by considering the signal
    and its derivatives as sparse [1].

    Parameters
    ----------

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
    if freq_cutoff is None:
        _freq_cutoff, _case = fcutoff_beads(_signal, x, show_plot=show_plot,
                                            print_plot=print_plot, path=path)
    else:
        if ((freq_cutoff <= 0) or (freq_cutoff >= 0.5)):
            raise ValueError("cutoff frequency must be 0 < freq_cutoff < 0.5")
        _freq_cutoff = freq_cutoff
        _case = 0
    print(f"{'Cutoff frequency:':<20}{_freq_cutoff:E}")
    print(f"{'Asymmetry:':<20}{asym:0.1f}")
    print(f"{'Fit parabola:':<20}{str(fit_parabola):s}")
    print(f"{'Half window:':<20}{str(half_window):s}")
    print(f"{'alpha:':<20}{alpha:0.2f}")

    tic = time.perf_counter()                               #@TEMP
    _bl, _p = _baseline_fitter.beads(
            _signal,
            freq_cutoff=_freq_cutoff,
            fit_parabola=fit_parabola,
            asymmetry=asym,
            smooth_half_window=half_window,
            alpha=alpha
            )
    toc = time.perf_counter()                               #@TEMP
    print(f"Baseline correction in {toc-tic:0.4f} seconds") #@TEMP
    return [_bl,_p,_case]


# @EB 2026-02-23 mask
def auto_fabc(s, x):
    _baseline_fitter = Baseline(x_data=x)
    _signal = rm_ends_outliers(s)
    _, _pp = _baseline_fitter.fabc(_signal,min_length=5)
    return _pp
