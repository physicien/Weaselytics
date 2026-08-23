# coding: utf-8
"""
Plotting functions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator

#: Colour of the baseline-sensitivity curve. Kept clear of every other
#: element of the r2 figure: blue is r2, red is the selected cutoff and
#: the trimmed fill, orange the proto-plateaus, purple the flat set.
_SENSITIVITY_COLOR = "darkslateblue"

#: _HATCH_NOTE -- why the hatched overlays set `facecolor`/`edgecolor`
#: explicitly instead of `color="none", ec=..., alpha=...`.
#:
#: matplotlib 3.11 made the hatch colour a property of its own, and
#: changed the default of `rcParams["hatch.color"]` from ``"black"`` to
#: ``"edge"``. With `color="none"` the hatch colour then resolves
#: against that "none" rather than against the `ec=` given afterwards,
#: and the hatch is not drawn at all -- silently, with no warning and
#: with every artist property still reporting the expected value
#: (`get_hatch()`, `get_edgecolor()` and `get_linewidth()` are identical
#: on 3.10 and 3.11). The only visible symptom is the missing hatch in
#: the rendered figure.
#:
#: Passing `facecolor="none"` with an RGBA `edgecolor` carrying the
#: alpha renders identically on both versions -- verified by pixel count
#: on 3.10.1 and 3.11.0. Do not "simplify" it back to `color=`/`alpha=`.

#: Fixed top of the baseline-sensitivity panel. Fixed rather than
#: data-scaled so that the panel means the same thing on every figure
#: and the curves can be compared across signals; the instability
#: spikes run orders of magnitude higher, so the true peak is annotated
#: whenever it leaves the frame.
_SENSITIVITY_YMAX = 1.5

#: Spacing of the sensitivity panel's y ticks, in the same units as
#: `_SENSITIVITY_YMAX`: labelled majors, and unlabelled minors giving the
#: finer reference scale. Both are absolute steps, so raising the
#: ceiling without raising them crowds the axis.
_SENSITIVITY_YTICK_MAJOR = 0.5
_SENSITIVITY_YTICK_MINOR = 0.1

#: TEMPORARY: horizontal reference grid on the sensitivity panel, to read
#: levels off the curve by eye while the stiff-side trim is being
#: designed. Remove this and its use below once that work is settled.
_SENSITIVITY_GRID = True


def plot(x: np.ndarray, y: np.ndarray, y_sm: np.ndarray | None = None,
         s: np.ndarray | None = None, bl: np.ndarray | None = None,
         x_fit: np.ndarray | None = None,
         y_fit_g: np.ndarray | None = None,
         y_fit_sn: np.ndarray | None = None,
         y_fit_p7: np.ndarray | None = None,
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
    y_fit_p7 : array-like, shape (M,), optional
        The y-values of the modified Pearson VII distribution fitted on a
        peak. If set to None (default), will not be plotted.
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
    if (x_fit is not None) and (y_fit_p7 is not None):
        plt.plot(x_fit, y_fit_p7, ls=':', c=palette[4], lw=2.0,
                 label='robust Pearson VII fit')

    plt.annotate(f"{'# data pts:'}{len(x):>6d}",
                 xy=(1.0,1.01),
                 xycoords=("axes fraction"),
                 ha="right",
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

def r2_plots(x: np.ndarray, r2: np.ndarray, dip_curve: np.ndarray,
             freq_cutoff: float, fcut_r2: float,
             cp_flat: np.ndarray | None = None,
             cp_dips: np.ndarray | None = None,
             cp_removed: np.ndarray | None = None,
             cp_snr_removed: np.ndarray | None = None,
             cp_instab_removed: np.ndarray | None = None,
             sensitivity: np.ndarray | None = None,
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
    dip_curve : array-like, shape (N,)
        The curve the proto-plateau detector reads
        (``segmentation.dip_curve``): the rolling standard deviation of
        `r2`, Gaussian-smoothed and normalised to its own maximum.
        ``detect_dips`` marks proto-plateaus at its local minima, so the
        orange basins above line up with the dips of this panel.
    freq_cutoff : float
        Frequency cutoff.
    fcut_r2 : float
        Value of `r2` at `freq_cutoff`.
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
        solid fill in the colour of the sensitivity curve, so the cut can
        be read directly against the panel that produced it. Default is
        None, which disables the overlay.
    sensitivity : array-like, shape (N,), optional
        Baseline-sensitivity curve from ``baseline._sensitivity_curve``: the
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
        can constrain — marked on the sensitivity panel. Default is None,
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
    # the height, the baseline-sensitivity curve and the rolling-std
    # panel split the other half evenly.
    gs = fig.add_gridspec(3, hspace=0, height_ratios=[2.0, 1.0, 1.0])
    axs = gs.subplots(sharex=True)
    # Red fill: the detected plateaus/proto-plateaus removed by the
    # stage-1 trimming (the sub-fundamental clip).
    if cp_removed is not None:
        axs[0].fill_between(x, 0, 1,
                            where=cp_removed,
                            color='red', alpha=0.15,
                            label='trimmed',
                            transform=axs[0].get_xaxis_transform())
    # Dark-red cross-hatch: what the SNR-gated collapse exclusion (#3)
    # would additionally remove. A preview; it does not affect selection.
    # Only drawn (and labelled) when it actually removes something, so
    # a signal it never fires on carries no phantom legend entry.
    if cp_snr_removed is not None and np.any(cp_snr_removed):
        # facecolor/edgecolor set explicitly, with the alpha carried in
        # the RGBA edge colour rather than passed as `alpha`. See
        # `_HATCH_NOTE` below: `color="none"` silently kills the hatch
        # from matplotlib 3.11 on.
        axs[0].fill_between(x, 0, 1,
                            where=cp_snr_removed,
                            facecolor="none",
                            edgecolor=to_rgba("darkred", 0.9),
                            hatch="xxx", hatch_linewidth=1.0,
                            label='SNR-trimmed',
                            transform=axs[0].get_xaxis_transform())
    # The stiff-side instability cut, in the colour of the sensitivity
    # curve below: this region is removed because that curve is still
    # flailing there, and drawing them alike lets the two be read
    # together.
    if cp_instab_removed is not None and np.any(cp_instab_removed):
        axs[0].fill_between(x, 0, 1,
                            where=cp_instab_removed,
                            color=_SENSITIVITY_COLOR, alpha=0.30,
                            label='unstable',
                            transform=axs[0].get_xaxis_transform())
    axs[0].semilogx(x, r2, marker='.', ls='',label=r'$r^2$',ms=3)

    # Changepoint prototype overlay (issue #4): the full flat set from
    # classify_segments, hatched purple.
    if cp_flat is not None:
        axs[0].fill_between(x, 0, 1,
                            where=cp_flat,
                            facecolor="none",
                            edgecolor=to_rgba("tab:purple", 0.3),
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
    # Middle panel: the baseline-sensitivity curve, on a LINEAR y-axis.
    # The quantity is linear, and a log axis turns the settled floor
    # into structure that is not there.
    if sensitivity is not None:
        axs[1].semilogx(x, sensitivity, ls='-', lw=0.8,
                        color=_SENSITIVITY_COLOR)
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
        axs[1].set_ylim(bottom=-0.06 * _SENSITIVITY_YMAX,
                        top=_SENSITIVITY_YMAX)
        finite = np.asarray(sensitivity)[np.isfinite(sensitivity)]
        if finite.size:
            peak = float(np.max(finite))
            if peak > _SENSITIVITY_YMAX:
                axs[1].annotate(f'peak {peak:.3g}',
                                xy=(0.995, 0.80),
                                xycoords='axes fraction',
                                ha='right', va='top', fontsize=7,
                                color=_SENSITIVITY_COLOR)
        # Ticks on fixed absolute steps, so a given level always sits at
        # the same height and can be compared between figures. Only the
        # majors are labelled; the minors carry the finer scale.
        # `arange` stops before the ceiling, dropping the tick that would
        # otherwise land on the boundary shared with the panel above.
        axs[1].yaxis.set_major_locator(FixedLocator(
            np.arange(0.0, _SENSITIVITY_YMAX, _SENSITIVITY_YTICK_MAJOR)))
        axs[1].yaxis.set_minor_locator(
            MultipleLocator(_SENSITIVITY_YTICK_MINOR))
        axs[1].tick_params(axis='y', labelsize=8)
        axs[1].tick_params(axis='y', which='minor', length=2)
        # TEMPORARY reference grid, see `_SENSITIVITY_GRID`. Drawn under
        # the curve so it stays readable.
        if _SENSITIVITY_GRID:
            axs[1].grid(axis='y', which='major', color='0.55',
                        lw=0.6, alpha=0.9)
            axs[1].grid(axis='y', which='minor', color='0.78',
                        lw=0.4, alpha=0.8)
            axs[1].set_axisbelow(True)

    axs[2].semilogx(x, dip_curve, ls='-', lw=0.9, color='tab:orange')

    for ax in axs.flat:
        ax.axvline(x=freq_cutoff, c='tab:red', ls='dashed')
        ax.label_outer()
    axs[0].annotate(f'{fcut_r2:0.4f}',
                    xy=(freq_cutoff,1.01),
                    xycoords=("data","axes fraction"),
                    ha='center',
                    color='tab:red'
                    )
#    axs[2].set_xlabel('Cutoff frequency')
    axs[0].set_ylabel(r'$r^2_{y-b}$')
    # rms change of the fitted baseline between adjacent cutoffs, as a
    # fraction of the range of the log-transformed signal, per decade.
    axs[1].set_ylabel(r"rms $\Delta b$" "\n" r"/range/dec", fontsize=9)
    axs[2].set_ylabel("dip curve\n(norm. rolling std)", fontsize=9)
#    axs[1].set_ylabel(r"$r^2_{y-b}$'")
#    axs[2].set_ylabel(r"$r^2_{y-b}$''")

    # The curve is normalised to its own maximum, so the scale is fixed
    # and comparable between figures. A little headroom below zero keeps
    # the floor readable, as on the sensitivity panel.
    axs[2].set_ylim(bottom=-0.06, top=1.05)
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
    axs[2].tick_params(axis='y', labelsize=8)
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

