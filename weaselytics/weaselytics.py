#!/usr/bin/python3

import sys
import argparse
import numpy as np

from weaselytics.parsers import ParsedData
from weaselytics.peakfitting import gauss, skew_norm, fit_peak
from weaselytics.utils import smooth_SG
from weaselytics.baseline import auto_beads
from weaselytics.export import export_txt, export_csv
from weaselytics.plot import plot


def main():
    parser = argparse.ArgumentParser(
        prog='weaselytics',
        description='Parse and analyse chromatographic data from .txt file'
    )

    parser.add_argument("path", help="the .txt data file")
    parser.add_argument('-s', '--show', default=0, action='store_true',
                        help='show the plot windows')
    parser.add_argument('-p', '--print', default=0, action='store_true',
                        help='print the plots')
    parser.add_argument('-e', '--export_bldata', default=0, action='store_true',
                        help='export the baseline corrected data to filename_bl.txt')
    parser.add_argument('-o', '--output_csv', default=0, action='store_true',
                        help='output data to <ARG>.csv')
    parser.add_argument('-os', '--output_stats', type=str,
                        help='output stats to filename_<ARG>.csv')
    parser.add_argument('-n', '--nofit', default=1, action='store_false',
                        help='do not fit the chromatogram')
    parser.add_argument('-nb', '--nobaseline', default=1, action='store_false',
                        help='do not correct the baseline')
    parser.add_argument('-sm', '--dosmoothing', default=0, action='store_true',
                        help='do not smooth the signal')
    parser.add_argument('-x0', '--startx', type=float,
                        help='start fitting the gaussian at x min')
    parser.add_argument('-x1', '--endx', type=float,
                        help='end fitting the gaussian at x min')

    args = parser.parse_args()

    fit_data = args.nofit
    do_bl = args.nobaseline
    do_sm = args.dosmoothing

    if args.startx is not None and args.endx is not None and args.startx == args.endx:
        print("Warning. x0 and x1 are equal. Exit.")
        sys.exit(1)

    if args.startx is not None and args.endx is not None and args.startx > args.endx:
        print("Warning. x1 is larger than x0. Exit.")
        sys.exit(1)

    if args.startx:
        if args.startx < 0:
            print("Warning. x0 < 0. Exit.")
            sys.exit(1)

    if args.endx:
        if args.endx < 0:
            print("Warning. x1 < 0. Exit.")
            sys.exit(1)

    path = args.path
    print(path)
    parsed = ParsedData(path)
    xdata, ydata = parsed.data

    if args.show or args.print:
        plot_kwargs = {
            "show_plot": args.show,
            "print_plot": args.print,
            "path": path
        }

    if do_sm:
        ydata_sm = smooth_SG(ydata, 9, 0)
        if args.show or args.print:
            plot_kwargs['y_sm'] = ydata_sm
        mod_ydata = ydata_sm
    else:
        mod_ydata = ydata

    if do_bl:
        baseline, params, case = auto_beads(
            mod_ydata, xdata, freq_cutoff=None,
            show_plot=args.show, print_plot=args.print, path=args.path,
            method="custom_beads"
        )
        signal = params["signal"]
        if args.show or args.print:
            plot_kwargs['bl'] = baseline
            plot_kwargs['case'] = case
            plot_kwargs['s'] = signal
        mod_ydata = signal

    if args.export_bldata and do_bl:
        export_txt(xdata, mod_ydata, path=path)

    if args.output_csv:
        export_csv(xdata, ydata, path=path)

    if fit_data:
        x_robust, y_robust_g, y_robust_sn = fit_peak(
            mod_ydata, xdata,
            x0=args.startx, x1=args.endx,
            mol=args.output_stats, path=args.path
        )
        if args.show or args.print:
            plot_kwargs['x_fit'] = x_robust
            plot_kwargs['y_fit_g'] = y_robust_g
            plot_kwargs['y_fit_sn'] = y_robust_sn

    if args.show or args.print:
        plot(xdata, ydata, **plot_kwargs)
        print("")


if __name__ == "__main__":
    main()
