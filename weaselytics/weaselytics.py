#!/usr/bin/python3

import argparse
import logging
import os
import sys

import numpy as np

from weaselytics.baseline import auto_beads
from weaselytics.export import export_csv, export_txt
from weaselytics.parsers import ParsedData
from weaselytics.peakfitting import fit_peak
from weaselytics.plot import plot
from weaselytics.utils import smooth_SG

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """
    Send the package's progress messages to stdout.

    Configures the ``weaselytics`` logger rather than the root logger:
    ``logging.basicConfig`` is a no-op once the root logger has any
    handler, so it silently produces no output when the CLI runs inside
    a host that has already configured logging (pytest, for one). Owning
    a single package-level handler also means importing the library
    never installs one, which is what keeps library use quiet.

    Returns
    -------
    None

    """
    pkg_logger = logging.getLogger("weaselytics")
    pkg_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.INFO)
    # The messages are the CLI's own output; do not hand them to the
    # root logger as well, or a configured host prints them twice.
    pkg_logger.propagate = False


def main() -> None:
    """
    CLI entry point for the Weaselytics workflow.

    Raises
    ------
    FileNotFoundError
        Raised if the input path does not point to an existing file.
    ValueError
        Raised if ``--start-x`` and ``--end-x`` are equal, reversed order, or
        negative.

    Notes
    -----
    Progress messages are emitted through the ``logging`` module by the
    library code, and are silent on import. This entry point is what
    turns them on, so running the CLI prints what it always printed
    while ``import weaselytics`` stays quiet.
    """
    _configure_logging()
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog='weaselytics',
        description='Parse and analyse chromatographic data from .txt file'
    )

    parser.add_argument("path", help="the .txt data file")
    parser.add_argument('-s', '--show', action='store_true',
                        help='show the plot windows')
    parser.add_argument('-p', '--print', action='store_true',
                        help='print the plots')
    parser.add_argument('-e', '--export-bldata', action='store_true',
                        help=(
                            'export the baseline corrected data'
                            ' to filename_bl.txt'
                        ))
    parser.add_argument('-o', '--output-csv', action='store_true',
                        help='output data to <ARG>.csv')
    parser.add_argument('-os', '--output-stats', type=str,
                        help='output stats to filename_<ARG>.csv')
    parser.add_argument('-n', '--nofit', action='store_true',
                        help='do not fit the chromatogram')
    parser.add_argument('-nb', '--nobaseline', action='store_true',
                        help='do not correct the baseline')
    parser.add_argument('-sm', '--dosmoothing', action='store_true',
                        help='do not smooth the signal')
    parser.add_argument('-x0', '--start-x', type=float,
                        help='start fitting the gaussian at x min')
    parser.add_argument('-x1', '--end-x', type=float,
                        help='end fitting the gaussian at x min')
    parser.add_argument('-od', '--output-dir', default='results',
                        help='output directory (default: results)')
    parser.add_argument('-cd', '--cache-dir', default=None,
                        help=(
                            'directory where the autocorrelation curves are'
                            ' cached (default: no caching)'
                        ))
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help=(
                            'number of worker processes for the'
                            ' autocorrelation sweep (default: 1, serial)'
                        ))
    parser.add_argument('-fc', '--freq-cutoff', type=float, default=None,
                        help=(
                            'bypass the automatic selection and use this'
                            ' cutoff frequency, 0 < freq_cutoff < 0.5'
                            ' (default: automatic selection)'
                        ))
    parser.add_argument('-snr', '--snr-threshold', type=float, default=10.0,
                        help=(
                            'signal-to-noise ratio above which the'
                            ' collapsed (past-drop) plateaus are excluded'
                            ' from the candidate cutoff regions: a high-SNR'
                            ' signal has analyte whose area a cutoff there'
                            ' would destroy. Below it they are'
                            ' kept. Default: 10, the limit of quantitation'
                        ))

    args: argparse.Namespace = parser.parse_args()

    fit_data: bool = not args.nofit
    do_bl: bool = not args.nobaseline
    do_sm: bool = args.dosmoothing

    if args.start_x is not None and args.end_x is not None:
        if args.start_x == args.end_x:
            raise ValueError("x0 and x1 are equal.")
        if args.start_x > args.end_x:
            raise ValueError("x1 is larger than x0.")

    if args.start_x and args.start_x < 0:
        raise ValueError("x0 < 0.")

    if args.end_x and args.end_x < 0:
        raise ValueError("x1 < 0.")

    path: str = args.path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")
    logger.info(path)
    parsed: ParsedData = ParsedData(path)
    xdata: np.ndarray
    ydata: np.ndarray
    xdata, ydata = parsed.data

    output_dir: str = args.output_dir

    if do_sm:
        ydata_sm: np.ndarray = smooth_SG(ydata, 9, 0)
        mod_ydata: np.ndarray = ydata_sm
    else:
        mod_ydata = ydata

    if do_bl:
        baseline, params = auto_beads(
            mod_ydata, xdata, freq_cutoff=args.freq_cutoff,
            show_plot=args.show, print_plot=args.print, path=path,
            output_dir=output_dir,
            method="custom_beads",
            cache_dir=args.cache_dir,
            workers=args.workers,
            snr_threshold=args.snr_threshold
        )
        signal = params["signal"]
        mod_ydata = signal

    if args.export_bldata and do_bl:
        export_txt(xdata, mod_ydata, path=path, output_dir=output_dir)

    if args.output_csv:
        export_csv(xdata, ydata, path=path, output_dir=output_dir)

    if fit_data:
        x_robust, y_robust_g, y_robust_sn, y_robust_p7 = fit_peak(
            mod_ydata, xdata,
            x0=args.start_x, x1=args.end_x,
            mol=args.output_stats, path=path,
            output_dir=output_dir,
        )

    if args.show or args.print:
        plot_kwargs: dict = {
            "show_plot": args.show,
            "print_plot": args.print,
            "path": path,
            "output_dir": output_dir,
        }
        if do_sm:
            plot_kwargs["y_sm"] = ydata_sm
        if do_bl:
            plot_kwargs.update(bl=baseline, s=signal)
        if fit_data:
            plot_kwargs.update(
                x_fit=x_robust, y_fit_g=y_robust_g,
                y_fit_sn=y_robust_sn, y_fit_p7=y_robust_p7,
            )
        plot(xdata, ydata, **plot_kwargs)


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        logger.error("Error: %s", e)
        sys.exit(1)
