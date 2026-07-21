# coding: utf-8
"""
Plotting functions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot(x: np.ndarray, y: np.ndarray, y_sm: np.ndarray | None = None,
         s: np.ndarray | None = None, bl: np.ndarray | None = None,
         x_fit: np.ndarray | None = None,
         y_fit_g: np.ndarray | None = None,
         y_fit_sn: np.ndarray | None = None, case: int = 0,
         show_plot: bool = False, print_plot: bool = False,
         path: str = "./file.txt",
         output_dir: str = "results") -> None:
    """
    Plot the signal and its various modified variations.

    Parameters
    ----------
    x : array-like, shape (N,)
        The x-values of the signal.
    y : array-like, shape (N,)
        The raw y-values of the signal.
    y_sm : array-like, shape (N,), optional
        The smoothed y-values of the signal. If set to None (default), will
        not be plotted.
    s : array-like, shape (N,), optional
        The baseline corrected y-values of the signal. If set to None
        (default), will not be plotted.
    bl : array-like, shape (N,), optional
        The baseline obtained from the baseline correction algorithm. If set
        to None (default), will not be plotted.
    x_fit : array-like, shape (M,), optional
        The x-values used to fit a peak. If set to None (default), will not
        be plotted.
    y_fit_g : array-like, shape (M,), optional
        The y-values of the Gaussian distribution fitted on a peak. If set to
        None (default), will not be plotted.
    y_fit_sn : array-like, shape (M,), optional
        The y-values of the Skew-Normal distribution fitted on a peak. If set
        to None (default), will not be plotted.
    case : int, optional
        The case rule from which `fcut` have been selected. Not necessarily
        useful in the current implementation, but it is advisable to keep it
        until proven otherwise.
    show_plot : bool, optional
        If True, the plot will be shown to the screen. Default is False.
    print_plot : bool, optional
        If True, the plot will be exported as an image. Default is False.
    path : str, optional
        Path of the data file.

    Returns
    -------
    None

    """
    palette = sns.color_palette("colorblind")
    sns.set_palette(palette)

    plt.figure(num="Chromatogram")
    plt.plot(x, y, marker='.', ls='', c=palette[7], label='raw data', ms=3)
    if y_sm is not None:
        plt.plot(x, y_sm, ls='-.', c=palette[2], lw=1.5,
                 label='smoothed data')
    if s is not None:
        plt.plot(x, s, ls='-', c=palette[5], lw=1.5, label='adjusted data')
    if bl is not None:
        plt.plot(x, bl, ls='--', c=palette[0], lw=2.0, label='baseline')
    if (x_fit is not None) and (y_fit_g is not None):
        plt.plot(x_fit, y_fit_g, ls='--', c=palette[2], lw=2.0,
                 label='robust gaussian fit')
    if (x_fit is not None) and (y_fit_sn is not None):
        plt.plot(x_fit, y_fit_sn, ls='-.', c=palette[3], lw=2.0,
                 label='robust skew-normal fit')

    plt.annotate(f"{'# data pts:'}{len(x):>6d}",
                 xy=(1.0,1.01),
                 xycoords=("axes fraction"),
                 ha="right",
                 color="tab:red"
                 )
    if case != 0:
        plt.annotate(f"{'Case:'}{case:>3d}",
                    xy=(0.00,1.01),
                    xycoords=("axes fraction"),
                    ha="left",
                    color="tab:red"
                    )
    plt.legend()
    plt.xlabel('Time (min.)')
    plt.ylabel('Potential (mV)')
    plt.tight_layout()
    if show_plot:
        plt.show()
    if print_plot:
        filename = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.join(output_dir, "images")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, filename + ".png"))
    plt.close()
    return None

def r2_plots(x: np.ndarray, r2: np.ndarray, sm_d0: np.ndarray,
             sm_d1: np.ndarray, sm_d2: np.ndarray, min_d1: np.ndarray,
             max_d1: np.ndarray, ends: np.ndarray, sec_p: np.ndarray,
             tol1_0: np.ndarray, tol1_1: float, tol2: float,
             freq_cutoff: float, fcut_r2: float, case: int = 0,
             cp_candidates: np.ndarray | None = None,
             cp_refined: np.ndarray | None = None,
             show_plot: bool = False, print_plot: bool = False,
             path: str = "./file.txt",
             output_dir: str = "results") -> None:
    """
    Plot the autocorrelation and its first two derivatives.

    Parameters
    ----------
    x : array-like, shape (N,)
        The x-values of the parameter on which depend the autocorrelation.
    r2 : array-like, shape (N,)
        The y-values of the autocorrelation.
    sm_d0 : array-like, shape (N,)
        The y-values of the smoothed autocorrelation.
    sm_d1 : array-like, shape (N,)
        The y-values of the smoothed first derivation of the autocorrelation.
    sm_d2 : array-like, shape (N,)
        The y-values of the smoothed second derivation of the autocorrelation.
    min_d1 : array-like, shape (u,)
        Indices of the `sm_d1` minimums.
    max_d1 : array-like, shape (v,)
        Indices of the `sm_d1` maximums.
    ends : array-like, shape (N,), dtype bool
        Boolean mask of the first plateau region of `r2`.
    sec_p : array-like, shape (M,), dtype int
        Indices of the points assigned to secondary plateaus.
    tol1_0 : array-like, shape (N,), dtype bool
        Boolean mask of the first plateau region of `sm_d1`.
    tol1_1 : float, optional
        Loose threshold used to find plateaus on the first derivative of the
        smoothed autocorrelation plot. Default is 5.0E-04.
    tol2 : float, optional
        Threshold used to find plateaus on the second derivative of the
        smoothed autocorrelation plot. Default is 2.0E-06.
    freq_cutoff : float
        Frequency cutoff.
    fcut_r2 : float
        Value of `r2` at `freq_cutoff`.
    case : int, optional
        The case rule from which `fcut` have been selected. Not necessarily
        useful in the current implementation, but it is advisable to keep it
        until proven otherwise. Default is 0.
    cp_candidates : array-like, shape (N,), dtype bool, optional
        Mask of the candidate plateau regions returned by
        ``segmentation.trim_candidates``, overlaid on the
        autocorrelation panel as hatched purple regions. Default is None,
        which disables the overlay.
    cp_refined : array-like, shape (N,), dtype bool, optional
        Mask of the label-calibrated bracket returned by
        ``segmentation.refine_candidates``, overlaid as orange hatching
        (opposite direction to the candidate hatching) on top of the
        candidate regions. Default is None, which disables the overlay.
    show_plot : bool, optional
        If True, the plot will be shown to the screen. Default is False.
    print_plot : bool, optional
        If True, the plot will be exported as an image. Default is False.
    path : str, optional
        Path of the data file.
    output_dir : str, optional
        Output directory for the exported plots. Default is "results".

    """
    #TODO: Cleanup this function...
    accepted = np.zeros(len(x))
    accepted[sec_p] = 1

    #@EB
    #fig = plt.figure(figsize=[6.4,9.6],num="Autocorrelation plots")
    fig = plt.figure(figsize=[9.4,9.6],num="Autocorrelation plots")
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].fill_between(x, 0, 1,
                        where= ends,
                        color='red', alpha=0.1,
                        transform=axs[0].get_xaxis_transform())
    axs[0].fill_between(x, 0, 1,
                        where= accepted,
                        color='green', alpha=0.3,
                        transform=axs[0].get_xaxis_transform())
    axs[0].semilogx(x, r2, marker='.', ls='',label=r'$r^2$',ms=3)

    # Changepoint prototype overlay (issue #4): trimmed candidate
    # plateau regions
    if cp_candidates is not None:
        axs[0].fill_between(x, 0, 1,
                            where=cp_candidates,
                            color="none", ec="tab:purple", alpha=0.3,
                            hatch="\\\\", hatch_linewidth=2,
                            label='CP candidates',
                            transform=axs[0].get_xaxis_transform())
    if cp_refined is not None:
        axs[0].fill_between(x, 0, 1,
                            where=cp_refined,
                            color="none", ec="tab:orange", alpha=0.6,
                            hatch="//", hatch_linewidth=2,
                            label='CP bracket',
                            transform=axs[0].get_xaxis_transform())

    axs[1].fill_between(x, 0, 1,
                        where=tol1_0,
                        color="none", ec="white", alpha=0.3, fc="purple",
                        hatch="//", hatch_linewidth=4,
                        transform=axs[1].get_xaxis_transform())
    axs[1].semilogx(x, sm_d1, ls='-', label=r'corrected')
    axs[1].semilogx(x, sm_d2, ls='-', label=r'smooth')

    for ax in axs.flat:
        ax.axvline(x=freq_cutoff, c='tab:red', ls='dashed')
        ax.label_outer()
    #for md1 in min_d1:
    #    axs[1].axvline(x=x[md1],ymax=0.5,c='tab:pink',ls='dashed')
    #for md1 in max_d1:
    #    axs[1].axvline(x=x[md1],ymin=0.5,c='tab:green',ls='dashed')
    axs[0].annotate(f'{fcut_r2:0.4f}',
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
#    axs[2].set_xlabel('Cutoff frequency')
    axs[0].set_ylabel(r'$r^2_{y-b}$')
    axs[1].set_ylabel(r"Rolling Std($r^2_{y-b}$)")
#    axs[1].set_ylabel(r"$r^2_{y-b}$'")
#    axs[2].set_ylabel(r"$r^2_{y-b}$''")

    # How do we find the right inflection point?
    # @EB Ajuster le calcul suivant?
#    infl_min = np.argmin(sm_d1[infls])
#    r2_ymin = r2[infls[infl_min-1]]-0.05  #only for the r2 plot limit
    p1_ymax = 2E-3#np.max(sm_d1[tol1_0])*2.50
    p1_ymin = -1E-4#-0.05*p1_ymax
#    axs[0].set_ylim(r2_ymin,1.0)
    axs[1].set_ylim(bottom=p1_ymin, top=p1_ymax)
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
    #axs[2].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
    axs[0].legend()
    plt.tight_layout()
    if show_plot:
        plt.show()
    if print_plot:
        _filename = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.join(output_dir, "r2_plots")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, _filename + "_r2.png"))
    plt.close()
    return None

