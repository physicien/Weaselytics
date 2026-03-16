#!/usr/bin/python
# coding: utf-8
"""
Plotting functions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot(x, y, y_sm=None, s=None, bl=None, x_fit=None, y_fit_g=None,
         y_fit_sn=None, case=0, show_plot=False, print_plot=False,
         path="./file.txt"):
    """
    Plot the signal and its various modified variations.

    Parameters
    ----------
    x : array-like, shape (N,)
        The x-values of the signal.
    y : array-like, shape (N,)
        The raw y-values of the signal.
    y_sm : array-like, shape (N,), optional
        The smoothed y-values of the signal. If set to `None` (default), will
        not be plotted.
    s : array-like, shape (N,), optional
        The baseline corrected y-values of the signal. If set to `None`
        (default), will not be plotted.
    bl : array-like, shape (N,), optional
        The baseline obtained from the baseline correction algorithm. If set
        to `None` (defautl), will not be plotted.
    x_fit : array-like, shape (M,), optional
        The x-values used to fit a peak. If set to `None` (default), will not
        be plotted.
    y_fit_g : array-like, shape (M,), optional
        The y-values of the Gaussian distribution fitted on a peak. If set to
        `None` (default), will not be plotted.
    y_fit_sn : array-like, shape (M,), optional
        The y-values of the Skew-Normal distribution fitted on a peak. If set
        to `None` (default), will not be plotted.
    case : int, optional
        The case rule from which ``fcut`` have been selected.
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
        plt.plot(x, s, ls='-', c=palette[5], lw=1.5, label='ajusted data')
    if bl is not None:
        plt.plot(x, bl, ls='--', c=palette[0], lw=2.0, label='baseline')
    if ((x_fit is not None) & (y_fit_g is not None)):
        plt.plot(x_fit, y_fit_g, ls='--', c=palette[2], lw=2.0,
                 label='robust gaussian fit')
    if ((x_fit is not None) & (y_fit_sn is not None)):
        plt.plot(x_fit, y_fit_sn, ls='-.', c=palette[3], lw=2.0,
                 label='robust skew-normal fit')

    plt.annotate(f"{'# data pts:'}{len(x):>6d}",
                 xy=(1.0,1.01),
                 xycoords=("axes fraction"),
                 ha="right",
                 color="tab:red"
                 )
    if case:
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
        plt.savefig(f"images/{filename}.png")
    plt.close()
    return None

def r2_plots(x, r2, sm_d0, sm_d1, sm_d2, min_d1, max_d1, last_start, sec_p,
             tol1_0, tol1_1, tol2, freq_cutoff, final_r2, case=0,
             show_plot=False, print_plot=False, path="./file.txt"):
    """
    """
    infls = np.where(np.diff(np.sign(sm_d2)))[0]
    accepted = np.zeros(len(x))
    accepted[sec_p] = 1

    #@EB
#    fig = plt.figure(figsize=[6.4,9.6],num="Autocorrelation plots")
    fig = plt.figure(figsize=[9.4,9.6],num="Autocorrelation plots")
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].fill_between(x, 0, 1,
                        where= x <= x[last_start],
                        color='red', alpha=0.1,
                        transform=axs[0].get_xaxis_transform())
    axs[0].fill_between(x, 0, 1,
                        where= accepted,
                        color='green', alpha=0.3,
                        transform=axs[0].get_xaxis_transform())
    axs[0].semilogx(x, r2, marker='.', ls='',label=r'$r^2$',ms=3)
    axs[0].semilogx(x, sm_d0, marker='', ls='-',
                    label=r'$r^2_\text{smooth}$',ms=3)

    axs[1].fill_between(x, 0, 1,
                        where=np.absolute(sm_d1) < tol1_0,
                        color="none", ec="white", alpha=0.3, fc="purple", 
                        hatch="//", hatch_linewidth=4,
                        transform=axs[1].get_xaxis_transform())
    axs[1].fill_between(x, 0, 1,
                        where=np.absolute(sm_d1) < tol1_1,
                        color='orange', alpha=0.3,
                        transform=axs[1].get_xaxis_transform())
    axs[1].semilogx(x, sm_d1, label='First Derivative')

    axs[2].fill_between(x, 0, 1,
                        where=np.absolute(sm_d2) < tol2,
                        color='blue', alpha=0.1,
                        transform=axs[2].get_xaxis_transform())
    axs[2].semilogx(x, sm_d2, label='Second Derivative')
    for ax in axs.flat:
#        for i, infl in enumerate(infls, 1):
#            ax.axvline(x=x[infl], c='k', lw=0.5)#, label=f'Inflection Point {i}')
        ax.axvline(x=freq_cutoff,c='tab:red',ls='dashed'),
        ax.label_outer()
#    for md1 in min_d1:
#        axs[1].axvline(x=x[md1],ymax=0.5,c='tab:pink',ls='dashed')
#    for md1 in max_d1:
#        axs[1].axvline(x=x[md1],ymin=0.5,c='tab:green',ls='dashed')
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
    return None

