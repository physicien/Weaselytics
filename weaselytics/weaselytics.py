#!/usr/bin/python3

import re
import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from parsers import ParsedData
from peakfitting import (gauss, skew_norm, lsq_gauss_fit, lsq_skew_norm_fit,
                         fit_peak)
from utils import *
from baseline import log_transform, relevant_range, auto_beads

#GLOBAL LIST
header1 = ["time","potential"]

###############################################################################
#PARSER
#Create parser
parser = argparse.ArgumentParser(prog='hplc_parser',\
        description='Parse data from .txt file')

#File is required
parser.add_argument("path",
        help="the .inp data file")

parser.add_argument('-s','--show',
        default=0, action='store_true',
        help='show the plot windows')

parser.add_argument('-p','--print',
        default=0, action='store_true',
        help='print the plots')

parser.add_argument('-e','--export_bldata',
        default=0, action='store_true',
        help='export the baseline corrected data to filename_bl.txt')

parser.add_argument('-o','--output_csv',
        default=0, action='store_true',
#        type=str,
        help='output data to <ARG>.csv')

parser.add_argument('-os','--output_stats',
        type=str,
        help='output stats to filename_<ARG>.csv')

parser.add_argument('-n','--nofit',
        default=1, action='store_false',
        help='do not fit the chromatogram')

parser.add_argument('-nb','--nobaseline',
        default=1, action='store_false',
        help='do not correct the baseline')

parser.add_argument('-sm','--dosmoothing',
        default=0, action='store_true',
        help='do not smooth the signal')

parser.add_argument('-x0','--startx',
        type=float,
        help='start fitting the gaussian at x min')

parser.add_argument('-x1','--endx',
        type=float,
        help='end fitting the gaussian at x min')

#Parse arguments
args = parser.parse_args()

#change values according to arguments
fit_data = args.nofit
do_bl = args.nobaseline
do_sm = args.dosmoothing

#check if startx and endx are equal - exif if true.
if args.startx is not None and args.endx is not None and args.startx == args.endx:
    print("Warning. x0 and x1 are equal. Exit.")
    sys.exit(1)

#check if endx is smaller than startx - exif if true.
if args.startx is not None and args.endx is not None and args.startx > args.endx:
    print("Warning. x1 is larger than x0. Exit.")
    sys.exit(1)

#check if startx < 0 - exit if true.
if args.startx:
    if args.startx < 0:
        print("Warning. x0 < 0. Exit.")
        sys.exit(1)

#check if endx < 0 - exit if true.
if args.endx:
    if args.endx < 0:
        print("Warning. x1 < 0. Exit.")
        sys.exit(1)

#Data processing            #@EB write a good parser and use it here
path = args.path
print(path)
#data =  np.loadtxt(path,skiprows=7)
#xdata = data[:,0]
#ydata = data[:,1]
parsed = ParsedData(path)
xdata, ydata = parsed.data

#smoothing
if do_sm:
    ydata_sm = smooth_SG(ydata,9,0) #9,0
else:
    ydata_sm = ydata

#baseline correction
if do_bl:
    baseline, params, case = auto_beads(ydata_sm, xdata, freq_cutoff=None,
                                        show_plot=args.show,
                                        print_plot=args.print, path=args.path,
                                        method="custom_beads")
    ydata_bl = params["signal"]
#    ydata_bl = ydata_sm - baseline
else:
    ydata_bl = ydata_sm

signal = ydata_bl

#if export_bldata is given - txt generation of the bl corrected chromatogram
if args.export_bldata and do_bl:
    ajusted_data = np.array([xdata,signal]).T
    filename = os.path.splitext(os.path.basename(path))[0]
    line1 = "Baseline corrected chromatogram of:\n"
    header = line1 + filename +"\n\n\n\n\n" 
    np.savetxt(filename+"_bl.txt", ajusted_data,
               delimiter = ' ',
               header=header
              )

#if output_csv is given - csv generation of the chromatogram
if args.output_csv:
    filename = os.path.splitext(os.path.basename(path))[0]
    outdata = np.array([xdata, ydata]).T
    df = pd.DataFrame(outdata)
    df.to_csv(filename+".csv", index=False, header=header1)

#Curve fit with data
if fit_data:
    fitted_peak =fit_peak(signal, xdata, x0=args.startx, x1=args.endx,
                          mol=args.output_stats, path=args.path)
    x_robust, y_robust_g, y_robust_sn = fitted_peak

#Prepare plot
if args.show or args.print:
    palette = sns.color_palette("colorblind")
    sns.set_palette(palette)

    plt.figure(num="Chromatogram")
    plt.plot(xdata, ydata, marker='.', ls='', c=palette[7],
             label='raw data',ms=3)
    if do_sm:
        plt.plot(xdata, ydata_sm, ls='-.',c=palette[2], lw=1.5,
                label='smoothed data')
    if do_bl:
        plt.plot(xdata, signal, ls='-',c=palette[5], lw=1.5,
                label='ajusted data')
        plt.plot(xdata, baseline, ls='--',c=palette[0], lw=2.0,
                label='baseline')

    if fit_data:
        plt.plot(x_robust, y_robust_g, ls='--', c=palette[2], lw=2.0,
                 label='robust gaussian fit')
        plt.plot(x_robust, y_robust_sn, ls='-.', c=palette[3], lw=2.0,
                 label='robust skew-normal fit')

    plt.annotate(f"{'# data pts:'}{len(xdata):>6d}",
                 xy=(1.0,1.01),
                 xycoords=("axes fraction"),
                 ha='right',
                 color='tab:red'
                )
    if do_bl:
        plt.annotate(f"{'Case:'}{case:>3d}",
                    xy=(0.00,1.01),
                    xycoords=("axes fraction"),
                    ha='left',
                    color='tab:red'
                    )
    plt.legend()
    plt.xlabel('Time (min.)')
    plt.ylabel('Potential (mV)')
    plt.tight_layout()
    if args.show:
        plt.show()
    if args.print:
        filename = os.path.splitext(os.path.basename(path))[0]
        plt.savefig(f"images/{filename}.png")
    plt.close()
    print("") # Why?
