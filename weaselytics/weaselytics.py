#!/usr/bin/python3

import re
import sys
import os
import argparse
#import numba
import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import argrelmin, argrelmax
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.stattools import durbin_watson as dwtest
import matplotlib.pyplot as plt
import seaborn as sns
import time

from peakfitting import gauss, skew_norm, lsq_gauss_fit, lsq_skew_norm_fit
from utils import *

#GLOBAL LIST
header1 = ["time","potential"]
header2 = ["mol","solvent","distribution","A","x0","sigma","alpha"]
mol_list = list()

#FUNCTIONS
##Check if the first and last data points are outlier
#def rm_ends_outliers(s):
#    _signal = np.copy(s)
#    _x0_len = round(0.01*len(s))
#    _y_max = 0.01*np.abs(np.max(s)-np.min(s))
#    _y0_med = np.median(s[:_x0_len])
#    _y0_gap = np.abs(s[0]-_y0_med)
#    _y1_med = np.median(s[-_x0_len:])
#    _y1_gap = np.abs(s[-1]-_y1_med)
#
#    if _y0_gap > _y_max:
#        _signal[0] = _y0_med
#    if _y1_gap > _y_max:
#        _signal[-1] = _y1_med
#    return _signal

###############################################################################
#Autocorrelation 
def r2_fct(s):
    _r2 = ((2-dwtest(s))**2)/4
    return _r2

#Basic BEADS for the autocorrelation plot
def r2_beads(f_cut,s):
    _asym =1.0 
    _fp = True
    _hw = None

    _bl, _p = baseline_fitter.beads(
            s,
            freq_cutoff=f_cut,
            fit_parabola=_fp,
            asymmetry=_asym,
            smooth_half_window=_hw
            )
    _s_corr = s - _bl
    _r2 = r2_fct(_s_corr)
    return _r2

#Log transformation used in the calculation of BEADS frequency cutoff
def log_transform(s,epsilon):
    return np.log10(s-np.min(s)+epsilon)

#Frequency cutoff for BEADS
def fcutoff_beads(s):
    tic = time.perf_counter()
    
    # log transform of the signal
    _z = log_transform(s,1)

    _freq_cutoff_range = np.geomspace(0.00001, 0.5, num=1000, endpoint=False)
    
    r2_func = lambda x: r2_beads(x,_z)
    vr2_func = np.vectorize(r2_func)
    r2_val = vr2_func(_freq_cutoff_range)   # y-data

    smooth = gaussian_filter1d(r2_val,25)
    smooth_d1 = np.gradient(smooth)
    smooth_d2 = np.gradient(np.gradient(smooth))
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    rel_min_d1 = argrelmin(smooth_d1)[0]
    rel_max_d1 = argrelmax(smooth_d1)[0]
    # How do we find the right inflection point?
    d1_min = np.argmin(smooth_d1[infls])
    if d1_min == 0:
        if len(infls) == 2:
            print("Case 1")  #TEST
            infl_plateau = _freq_cutoff_range[infls[d1_min]]
            _freq_cutoff = 0.10*infl_plateau    #0.25
        else:
            print("Case 2")  #TEST
            infl_plateau = _freq_cutoff_range[infls[d1_min+1]]
            _freq_cutoff = 0.75*infl_plateau
    else:
        thresh_d1 = smooth_d1[infls[d1_min-1]]
        print(f"{'thresh_d1 value:':<20}{thresh_d1:0.4E}")  # TEST
        if ((thresh_d1 < -4E-04) and (d1_min > 2)):     # -2E-04 vs -4E-04
            print("Case 3")  #TEST
            infl_plateau = _freq_cutoff_range[infls[d1_min-3]]
            infl_min = _freq_cutoff_range[infls[d1_min-2]]
            shift_factor = 0.20#0.1/np.log(len(s))*np.log(20)
        else:
            print("Case 4")  #TEST
            infl_plateau = _freq_cutoff_range[infls[d1_min-1]]
            infl_min = _freq_cutoff_range[infls[d1_min]]
            shift_factor = 0.05#0.1/np.log(len(s))*np.log(20)
        infl_shift = shift_factor*(infl_min-infl_plateau)
        _freq_cutoff = infl_plateau + infl_shift #1.50*infl_plateau
    r2_ymin = r2_val[infls[d1_min-1]]-0.05  #only for the r2 plot limit

    toc = time.perf_counter()
    print(f"Autocorrelation in {toc-tic:0.4f} seconds")
    fi_r2_val = r2_beads(_freq_cutoff,_z)
    print(f"{'r2 value:':<20}{fi_r2_val:0.4f}")

    
    if args.show or args.print:
        xx = _freq_cutoff_range
        yy = r2_val
        fig = plt.figure(figsize=[6.4,9.6])
        gs = fig.add_gridspec(3, hspace=0)
        axs = gs.subplots(sharex=True)
        axs[0].semilogx(xx, yy, marker='.', ls='',label=r'$r^2$',ms=3)
        axs[0].semilogx(xx, smooth, marker='', ls='-',
                        label=r'$r^2_\text{smooth}$',ms=3)
        axs[1].semilogx(xx, smooth_d1, label='First Derivative')
        axs[2].semilogx(xx, smooth_d2, label='Second Derivative')
        for ax in axs.flat:
            for i, infl in enumerate(infls, 1):
                ax.axvline(x=xx[infl], c='k')#, label=f'Inflection Point {i}')
            ax.axvline(x=_freq_cutoff,c='tab:red',ls='dashed'),
            ax.label_outer()
        for md1 in rel_min_d1:
            axs[1].axvline(x=xx[md1],ymax=0.5,c='tab:pink',ls='dashed')
        for md1 in rel_max_d1:
            axs[1].axvline(x=xx[md1],ymin=0.5,c='tab:green',ls='dashed')
        axs[0].annotate(f'{fi_r2_val:0.4f}',
                        xy=(_freq_cutoff,1.01),
                        xycoords=("data","axes fraction"),
                        ha='center',
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
    return _freq_cutoff

###############################################################################
#BEADS baseline correction
def beads(s):
    # Read Navarro-Huerta et al (2017)
    # Section 3.2: Monitoring the autocorrelation to explore the BEADS
    #              working parameters
    # 3.3.2. Chromatograms involving peaks with extremely different magnitude
    # Section 3.4: Autocorrelation plot using the baseline-corrected signal
    # Section 3.5: Application of the assisted BEADS

    print(f"{'Data points:':<20}{len(s):d}")

    _fcut = fcutoff_beads(s)#0.005*2000/len(s)
    print(f"{'Cutoff frequency:':<20}{_fcut:E}")

    _asym =1.0 
    print(f"{'Asymmetry:':<20}{_asym:0.1f}")

    _fp = True
    print(f"{'Fit parabola:':<20}{str(_fp):s}")

    _hw = None
    print(f"{'Half window:':<20}{str(_hw):s}")

    tic = time.perf_counter()
    _bl, _p = baseline_fitter.beads(
            s,
            freq_cutoff=_fcut,
            fit_parabola=_fp,
            asymmetry=_asym,
            smooth_half_window=_hw
            )
    toc = time.perf_counter()
    print(f"Baseline correction in {toc-tic:0.4f} seconds")
    return [_bl,_p]

###############################################################################
#PARSER
#Create parser
parser = argparse.ArgumentParser(prog='hplc_parser',\
        description='Parse data from .txt file')

#File is required
parser.add_argument("filename",
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

parser.add_argument('-ns','--nosmoothing',
        default=1, action='store_false',
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
do_sm = args.nosmoothing

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

#Data processing
path = args.filename
print(path)
data =  np.loadtxt(path,skiprows=7)
xdata = data[:,0]
ydata_ini = data[:,1]
signal = rm_ends_outliers(ydata_ini)

#smoothing  #BEFORE OR AFTER baseline correction?
if do_sm:
    ydata_s = smooth_SG_data(signal,9,0) #9,0
else:
    ydata_s = signal

#baseline correction
if do_bl:
    baseline_fitter = Baseline(x_data=xdata)
    baseline, params = beads(ydata_s)
    ydata = params["signal"]
#    ydata = ydata_s - baseline
else:
    ydata = ydata_s

ajusted_data = np.array([xdata,ydata]).T

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
        path = args.filename
        filename = os.path.basename(path)
        outname = re.match(r"(^.+).txt",filename).group(1)
        solvent = re.match(r"(^.+)__LPYE",filename).group(1)
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

    plt.plot(xdata, ydata_ini, marker='.', ls='', c=palette[7],
             label='raw data',ms=3)
    if do_sm:
        plt.plot(xdata, ydata_s, ls='-.',c=palette[2], lw=1.5,
                label='smooth data')
    if do_bl:
        plt.plot(xdata, ydata, ls='-',c=palette[5], lw=1.5,
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
    plt.legend()
    plt.xlabel('Time (min.)')
    plt.ylabel('Intensity (a.u.)')
    plt.tight_layout()
    if args.show:
        plt.show()
    if args.print:
        filename = os.path.splitext(os.path.basename(path))[0]
        plt.savefig(f"images/{filename}.png")
    print("") # Why?
