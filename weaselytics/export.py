#!/usr/bin/python
# coding: utf-8
"""
Functions to export data to various file formats.
"""

import os
import re
import numpy as np
import pandas as pd

def export_txt(x, y, path="./file.txt"):
    """
    Export the data to a txt file after the baseline correction.

    Parameters
    ----------
    x : array-like, shape (N,)
        The x data.
    y : array-like, shape (N,)
        The y data.
    path : str, optional
        File path of the original data.

    Returns
    -------
    None

    """
    line = "Baseline corrected chromatogram of: "
    ajusted_data = np.array([x, y]).T
    filename = os.path.splitext(os.path.basename(path))[0]
    header = line + filename + "\n\n\n\n\n\n"
    np.savetxt(filename+"_bl.txt", ajusted_data, delimiter=' ', header=header)
    return None

def export_csv(x, y, path="./file.txt"):
    """
    Export the data to a csv file.

    Parameters
    ----------
    x : array-like, shape (N,)
        The x data.
    y : array-like, shape (N,)
        The y data.
    path : str, optional
        File path of the original data.

    Returns
    -------
    None

    """
    header = ["time","potential"]
    filename = os.path.splitext(os.path.basename(path))[0]
    outdata = np.array([x, y]).T
    df = pd.DataFrame(outdata)
    df.to_csv(filename+".csv", index=False, header=header)
    return None

def export_dist(mol, g_fit, sn_fiti, path):
    """
    Export the statistics of the fitted distribution for a peak to a csv file.

    Parameters
    ----------
    mol : str
        Label (molecule name) of the fitted peak.
    g_fit : ndarray with shape (3,)
        Parameters for a Gaussian distribution with the following fields
        defined:
        
        amp : float
            The maximum height of the distribution.
        x0 : float
            The center of the distribution.
        sigma : float
            The standard deviation of the distribution.
    sn_fit : ndarray with shape (4,)
        Parameters for a Skew-Normal distribution with the following fields
        defined:

        amp : float
            The maximum height of the distribution.
        loc : float
            The location parameter of the distribution.
        scale : float
            The scale parameter of the distribution.
        alpha : float
            The shape parameter of the distribution.
    path : str
        File path of the original data.

    Returns
    -------
    None

    """
    solv_pattern = r"(^.+)__LPYE"   # not general...
    filename = os.path.basename(path)
    outname = re.match(r"(^.+).txt", filename).group(1)
    solvent = re.match(solv_pattern, filename).group(1)
    data_gauss = {
            "mol": mol,
            "solvent": solvent,
            "distribution": "Gaussian",
            "A": g_fit[0],
            "x0": g_fit[1],
            "sigma": abs(g_fit[2]),
            "alpha": 0
            }
    data_skew_norm = {
            "mol": mol,
            "solvent": solvent,
            "distribution": "Gaussian",
            "A": sn_fit[0],
            "x0": sn_fit[1],
            "sigma": abs(sn_fit[2]),
            "alpha": sn_fit[3]
            }
    mol_list = list()
    mol_list.append(data_gauss)
    mol_list.append(data_skew_norm)
    df = pd.DataFrame(mol_list)
    header = ["mol","solvent","distribution","A","x0","sigma","alpha"]
    df.to_csv(outname+"_"+mol+".csv", index=False, header=header)
    return None
