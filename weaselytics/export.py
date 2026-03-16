#!/usr/bin/python
# coding: utf-8
"""
Functions to export data to various file formats.
"""

import os
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
