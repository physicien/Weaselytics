# coding: utf-8
"""
Plotting functions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator

#: Colour of the baseline-stability curve. Kept clear of every other
#: element of the r2 figure: blue is r2, red is the selected cutoff and
#: the trimmed fill, orange the proto-plateaus, purple the flat set.
_STABILITY_COLOR = "darkslateblue"

#: Fixed top of the baseline-stability panel. Fixed rather than
#: data-scaled so that the panel means the same thing on every figure
#: and the curves can be compared across signals; the instability
#: spikes run orders of magnitude higher, so the true peak is annotated
#: whenever it leaves the frame.
_STABILITY_YMAX = 1.5

#: Spacing of the stability panel's y ticks, in the same units as
#: `_STABILITY_YMAX`: labelled majors, and unlabelled minors giving the
#: finer reference scale. Both are absolute steps, so raising the
#: ceiling without raising them crowds the axis.
_STABILITY_YTICK_MAJOR = 0.5
_STABILITY_YTICK_MINOR = 0.1

#: TEMPORARY: horizontal reference grid on the stability panel, to read
#: levels off the curve by eye while the stiff-side trim is being
#: designed. Remove this and its use below once that work is settled.
_STABILITY_GRID = True


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
             cp_flat: np.ndarray | None = None,
             cp_dips: np.ndarray | None = None,
             cp_removed: np.ndarray | None = None,
             cp_snr_removed: np.ndarray | None = None,
             cp_instab_removed: np.ndarray | None = None,
             stability: np.ndarray | None = None,
             n_used: int | None = None,
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
    cp_flat : array-like, shape (N,), dtype bool, optional
        Mask of the full flat set from ``segmentation.classify_segments``
        (before any trimming), drawn as a light solid blue fill beneath
        the candidate hatching so the two can be compared. Default is
        None, which disables the overlay.
    cp_dips : array-like, shape (N,), dtype bool, optional
        Mask of the proto-plateau basins from ``segmentation.detect_dips``
        (the relative flattenings the flat test misses), drawn as an
        orange fill. Together with ``cp_flat`` this shows the detected
        plateau selection (their union) with its provenance. Default is
        None, which disables the overlay.
    cp_removed : array-like, shape (N,), dtype bool, optional
        Mask of the detected plateaus/proto-plateaus removed by the
        stage-1 trimming, drawn as a red fill. Default is None, which
        disables the overlay.
    cp_snr_removed : array-like, shape (N,), dtype bool, optional
        Mask of the additional regions the SNR-gated collapse exclusion
        would remove (beyond ``cp_removed``), drawn as a dark-red
        cross-hatch. A preview only; it does not affect the selection.
        Default is None, which disables the overlay.
    cp_instab_removed : array-like, shape (N,), dtype bool, optional
        Mask of the regions removed by the stiff-side instability
        exclusion (``segmentation.instability_boundary``), drawn as a
        solid fill in the colour of the stability curve, so the cut can
        be read directly against the panel that produced it. Default is
        None, which disables the overlay.
    stability : array-like, shape (N,), optional
        Baseline-stability curve from ``baseline._stability_curve``: the
        rms change of the fitted baseline between adjacent cutoff
        frequencies, relative to the signal range, per decade. Drawn on
        its own middle panel on a linear y-axis (the quantity is linear,
        and a log axis stretches the settled floor into structure that
        is not there). Large and erratic where the BEADS fit is
        unstable, settling where it becomes reliable. Default is None,
        which leaves the panel empty.
    n_used : int, optional
        Number of signal points used by the sweep. Its reciprocal is the
        record's fundamental frequency — the slowest baseline the data
        can constrain — marked on the stability panel. Default is None,
        which omits the marker.
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
    #@EB
    #fig = plt.figure(figsize=[6.4,9.6],num="Autocorrelation plots")
    fig = plt.figure(figsize=[9.4,9.6],num="Autocorrelation plots")
    # Three stacked panels at the unchanged figure size: r2 takes half
    # the height, the baseline-stability curve and the rolling-std
    # panel split the other half evenly.
    gs = fig.add_gridspec(3, hspace=0, height_ratios=[2.0, 1.0, 1.0])
    axs = gs.subplots(sharex=True)
    # Red fill: the detected plateaus/proto-plateaus removed by the
    # stage-1 trimming (sub-fundamental clip and frozen tail).
    if cp_removed is not None:
        axs[0].fill_between(x, 0, 1,
                            where=cp_removed,
                            color='red', alpha=0.15,
                            label='trimmed',
                            transform=axs[0].get_xaxis_transform())
    # Dark-red cross-hatch: what the SNR-gated collapse exclusion (#3)
    # would additionally remove. A preview; it does not affect selection.
    # Only drawn (and labelled) when it actually removes something, so
    # blanks do not carry a phantom legend entry.
    if cp_snr_removed is not None and np.any(cp_snr_removed):
        axs[0].fill_between(x, 0, 1,
                            where=cp_snr_removed,
                            color="none", ec="darkred", alpha=0.9,
                            hatch="xxx", hatch_linewidth=1.0,
                            label='SNR-trimmed',
                            transform=axs[0].get_xaxis_transform())
    # The stiff-side instability cut, in the colour of the stability
    # curve below: this region is removed because that curve is still
    # flailing there, and drawing them alike lets the two be read
    # together.
    if cp_instab_removed is not None and np.any(cp_instab_removed):
        axs[0].fill_between(x, 0, 1,
                            where=cp_instab_removed,
                            color=_STABILITY_COLOR, alpha=0.30,
                            label='unstable',
                            transform=axs[0].get_xaxis_transform())
    axs[0].semilogx(x, r2, marker='.', ls='',label=r'$r^2$',ms=3)

    # Changepoint prototype overlay (issue #4): the full flat set from
    # classify_segments, hatched purple.
    if cp_flat is not None:
        axs[0].fill_between(x, 0, 1,
                            where=cp_flat,
                            color="none", ec="tab:purple", alpha=0.3,
                            hatch="\\\\", hatch_linewidth=2,
                            label='CP flat',
                            transform=axs[0].get_xaxis_transform())
    # Proto-plateaus from detect_dips, orange fill: the union of this and
    # the flat set is the detected plateau selection.
    if cp_dips is not None:
        axs[0].fill_between(x, 0, 1,
                            where=cp_dips,
                            color="tab:orange", alpha=0.25,
                            label='proto-plateau',
                            transform=axs[0].get_xaxis_transform())
    # trimmed candidate plateau regions
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

    # Middle panel: the baseline-stability curve, on a LINEAR y-axis.
    # The quantity is linear, and a log axis turns the settled floor
    # into structure that is not there.
    if stability is not None:
        axs[1].semilogx(x, stability, ls='-', lw=0.8,
                        color=_STABILITY_COLOR)
        # The fundamental, 1/n_used: the slowest baseline the record can
        # constrain. Below it the fit has nothing to fix the baseline
        # against, which is where the instability lives.
        if n_used:
            axs[1].axvline(x=1.0 / n_used, c='k', ls='dotted', lw=1.2,
                           label='fundamental')
            axs[1].legend(fontsize=7, loc='upper right')
        # Fixed limits, with a little headroom below zero so the settled
        # floor reads as a curve resting on zero rather than as the axis
        # itself. The instability spikes run orders of magnitude above
        # the frame, so the true peak is annotated below and a clipped
        # spike is never silent.
        axs[1].set_ylim(bottom=-0.06 * _STABILITY_YMAX,
                        top=_STABILITY_YMAX)
        finite = np.asarray(stability)[np.isfinite(stability)]
        if finite.size:
            peak = float(np.max(finite))
            if peak > _STABILITY_YMAX:
                axs[1].annotate(f'peak {peak:.3g}',
                                xy=(0.995, 0.80),
                                xycoords='axes fraction',
                                ha='right', va='top', fontsize=7,
                                color=_STABILITY_COLOR)
        # Ticks on fixed absolute steps, so a given level always sits at
        # the same height and can be compared between figures. Only the
        # majors are labelled; the minors carry the finer scale.
        # `arange` stops before the ceiling, dropping the tick that would
        # otherwise land on the boundary shared with the panel above.
        axs[1].yaxis.set_major_locator(FixedLocator(
            np.arange(0.0, _STABILITY_YMAX, _STABILITY_YTICK_MAJOR)))
        axs[1].yaxis.set_minor_locator(
            MultipleLocator(_STABILITY_YTICK_MINOR))
        axs[1].tick_params(axis='y', labelsize=8)
        axs[1].tick_params(axis='y', which='minor', length=2)
        # TEMPORARY reference grid, see `_STABILITY_GRID`. Drawn under
        # the curve so it stays readable.
        if _STABILITY_GRID:
            axs[1].grid(axis='y', which='major', color='0.55',
                        lw=0.6, alpha=0.9)
            axs[1].grid(axis='y', which='minor', color='0.78',
                        lw=0.4, alpha=0.8)
            axs[1].set_axisbelow(True)

    axs[2].fill_between(x, 0, 1,
                        where=tol1_0,
                        color="none", ec="white", alpha=0.3, fc="purple",
                        hatch="//", hatch_linewidth=4,
                        transform=axs[2].get_xaxis_transform())
    axs[2].semilogx(x, sm_d1, ls='-', label=r'corrected')
    axs[2].semilogx(x, sm_d2, ls='-', label=r'smooth')

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
    # rms change of the fitted baseline between adjacent cutoffs, as a
    # fraction of the range of the log-transformed signal, per decade.
    axs[1].set_ylabel(r"rms $\Delta b$" "\n" r"/range/dec", fontsize=9)
    axs[2].set_ylabel(r"Rolling Std($r^2_{y-b}$)")
#    axs[1].set_ylabel(r"$r^2_{y-b}$'")
#    axs[2].set_ylabel(r"$r^2_{y-b}$''")

    # How do we find the right inflection point?
    # @EB Ajuster le calcul suivant?
#    infl_min = np.argmin(sm_d1[infls])
#    r2_ymin = r2[infls[infl_min-1]]-0.05  #only for the r2 plot limit
    p1_ymax = 2E-3#np.max(sm_d1[tol1_0])*2.50
    p1_ymin = -1E-4#-0.05*p1_ymax
#    axs[0].set_ylim(r2_ymin,1.0)
    axs[2].set_ylim(bottom=p1_ymin, top=p1_ymax)
    axs[2].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
    # With hspace=0 the scientific offset text is drawn just above this
    # panel, i.e. inside the stability panel; shrink it so it stops
    # colliding with that panel's own ticks.
    axs[2].yaxis.get_offset_text().set_fontsize(7)
    # Its top tick sits exactly on the boundary shared with the
    # stability panel, where it collides with that panel's zero. Prune
    # it so the stability floor stays labelled.
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
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

