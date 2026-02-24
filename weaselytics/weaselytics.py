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

from peakfitting import gauss, skew_norm, lsq_gauss_fit, lsq_skew_norm_fit
from utils import *
from baseline import log_transform, relevant_range, auto_beads

#GLOBAL LIST
header1 = ["time","potential"]
header2 = ["mol","solvent","distribution","A","x0","sigma","alpha"]
mol_list = list()

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
        help='output stats to <ARG>.csv')

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
data =  np.loadtxt(path,skiprows=7)
xdata = data[:,0]

ydata = data[:,1]

#smoothing
if do_sm:
    ydata_sm = smooth_SG(ydata,9,0) #9,0
else:
    ydata_sm = ydata

#baseline correction
if do_bl:
    baseline, params, case = auto_beads(ydata_sm, xdata, freq_cutoff=None,
                                        show_plot=args.show,
                                        print_plot=args.print, path=args.path)
    ydata_bl = params["signal"]
#    from baseline import auto_fabc                  #@EB 2026-02-23 mask
#    from scipy.ndimage import gaussian_filter1d     #@EB 2026-02-23 mask
#    patate = auto_fabc(gaussian_filter1d(ydata,15), xdata) #@EB 2026-02-23 mask
else:
    ydata_bl = ydata_sm

signal = ydata_bl
ajusted_data = np.array([xdata,signal]).T

#if export_bldata is given - txt generation of the bl corrected chromatogram
if args.export_bldata and do_bl:
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
    df = pd.DataFrame(data)
    df.to_csv(filename+".csv", index=False, header=header1)

#@EB make a function to fit from here
#if startx argument is given - x-axis range
if args.startx:
    xmin_fit = args.startx
else:
    xmin_fit = min(xdata)

#if endx argument is given - x-axis range
if args.endx:
    xmax_fit = args.endx
else:
    xmax_fit = max(xdata)

data_fit = ajusted_data[(xdata>xmin_fit) & (xdata<xmax_fit)]
xdata_fit = data_fit[:,0]
ydata_fit = data_fit[:,1]

#Curve fit with data
if fit_data:
    x_robust = np.arange(xdata_fit.min()-0.1, xdata_fit.max()+0.1, 0.001)
    #Gaussian curve fit
    p_lsq_g = lsq_gauss_fit(xdata_fit,ydata_fit)
    y_robust_g = gauss(x_robust,p_lsq_g)
    A_g, x0_g, sigma_g = p_lsq_g
    sigma_g = abs(sigma_g)

#    FWHM = 2.35482*sigma_g
    print('The Amplitude of the gaussian fit is', A_g)
    print('The center of the gaussian fit is', x0_g)
    print('The sigma of the gaussian fit is', sigma_g,"\n")
#    print('The FWHM of the gaussian fit is', FWHM)

    #Skew-Normal curve fit
    p_lsq_sn = lsq_skew_norm_fit(xdata_fit,ydata_fit)
    y_robust_sn = skew_norm(x_robust,p_lsq_sn)
    A_sn, x0_sn, sigma_sn, alpha_sn = p_lsq_sn
    sigma_sn = abs(sigma_sn)
    
    print('The Amplitude of the skew-normal fit is', A_sn)
    print('The center of the skew-normal fit is', x0_sn)
    print('The sigma of the skew-normal fit is', sigma_sn)
    print('The skew parameter of the skew-normal fit is', alpha_sn)

    #if output_stats is given - csv generation
    if args.output_stats:
        mol = args.output_stats
        path = args.path
        solv_pattern = r"(^.+)__LPYE"
        filename = os.path.basename(path)
        outname = re.match(r"(^.+).txt",filename).group(1)
        solvent = re.match(solv_pattern,filename).group(1)
        data_gauss = {
                "mol": mol,
                "solvent": solvent,
                "distribution":"Gaussian",
                "A": A_g,
                "x0": x0_g,
                "sigma": sigma_g,
                "alpha": 0
                }
        data_skew_norm = {
                "mol": mol,
                "solvent": solvent,
                "distribution":"Skew-Normal",
                "A": A_sn,
                "x0": x0_sn,
                "sigma": sigma_sn,
                "alpha": alpha_sn
                }
        mol_list.append(data_gauss)
        mol_list.append(data_skew_norm)
        df = pd.DataFrame(mol_list)
        df.to_csv(outname+"_"+mol+".csv", index=False, header=header2)

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
#        mask = patate["mask"]                   # @EB 2026-02-23 mask
#        plt.plot(xdata[~mask], ydata[~mask],"ms", ms=3 )# @EB 2026-02-23 mask

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
