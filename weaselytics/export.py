# coding: utf-8
"""
Functions to export data to various file formats.
"""

import os
import re

import numpy as np
import pandas as pd


def export_txt(x: np.ndarray, y: np.ndarray, path: str = "./file.txt",
               output_dir: str = "results") -> None:
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
    output_dir : str, optional
        Output directory for exported files. Default is "results".

    Returns
    -------
    None

    """
    line = "Baseline corrected chromatogram of: "
    adjusted_data = np.array([x, y]).T
    basename = os.path.splitext(os.path.basename(path))[0]
    header = line + basename + "\n\n\n\n\n\n"
    mobile_phase = os.path.basename(os.path.dirname(path))
    outdir = (
        os.path.join(output_dir, mobile_phase)
        if mobile_phase else output_dir
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, basename + "_bl.txt")
    np.savetxt(outpath, adjusted_data, delimiter='\t', header=header)
    return None

def export_csv(x: np.ndarray, y: np.ndarray, path: str = "./file.txt",
               output_dir: str = "results") -> None:
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
    output_dir : str, optional
        Output directory for exported files. Default is "results".

    Returns
    -------
    None

    """
    header = ["time","potential"]
    basename = os.path.splitext(os.path.basename(path))[0]
    outdata = np.array([x, y]).T
    df = pd.DataFrame(outdata)
    mobile_phase = os.path.basename(os.path.dirname(path))
    outdir = (
        os.path.join(output_dir, mobile_phase)
        if mobile_phase else output_dir
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, basename + ".csv")
    df.to_csv(outpath, index=False, header=header)
    return None

def export_dist(mol: str, g_fit: np.ndarray, sn_fit: np.ndarray,
                path: str, output_dir: str = "results") -> None:
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
    output_dir : str, optional
        Output directory for exported files. Default is "results".

    Returns
    -------
    None

    """
    solv_pattern = r"(^.+)__LPYE"   # not general...
    basename = os.path.basename(path)
    outname = os.path.splitext(basename)[0]
    m = re.match(solv_pattern, basename)
    solvent = m.group(1) if m else "unknown"
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
            "distribution": "Skew-Normal",
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
    mobile_phase = os.path.basename(os.path.dirname(path))
    outdir = (
        os.path.join(output_dir, mobile_phase)
        if mobile_phase else output_dir
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname + "_" + mol + ".csv")
    df.to_csv(outpath, index=False, header=header)
    return None
