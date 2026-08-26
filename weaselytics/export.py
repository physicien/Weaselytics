# coding: utf-8
"""
Functions to export data to various file formats.

Three writers, each producing a different quantity: `export_txt` the
corrected signal, `export_csv` the raw signal for plotting outside the
package, and `export_dist` the parameters of the fitted peak models.
All three place their output under a subdirectory named after the
parent directory of the source file.

References
----------
.. [1] Navarro-Huerta, J. A. et al. Assisted baseline subtraction in
   complex chromatograms using the BEADS algorithm. J. Chromatogr. A
   1507, 1-10 (2017). doi:10.1016/j.chroma.2017.05.057
.. [2] Liland, K. H. et al. Customized baseline correction. Chemom.
   Intell. Lab. Syst. 109(1), 51-56 (2011).
   doi:10.1016/j.chemolab.2011.07.005
.. [3] Azzalini, A. A class of distributions which includes the normal
   ones. Scand. J. Statist. 12(2), 171-178 (1985).
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

    Notes
    -----
    Writes ``<stem>_bl.txt`` under a subdirectory named after the
    parent directory of `path`, tab separated, behind a header of the
    file stem and six blank lines.

    What `y` holds is the caller's choice. The CLI passes
    ``params["signal"]``, the denoised peak component BEADS returns,
    where Navarro-Huerta et al. and Liland et al. subtract the
    baseline and keep the noise, so the exported signal is not
    ``y - baseline``.

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

    Notes
    -----
    Writes ``<stem>.csv`` under a subdirectory named after the parent
    directory of `path`, with columns ``time`` and ``potential``.

    This is the raw signal, exported so it can be plotted from the csv
    by a script outside the package. `export_txt` beside it writes the
    corrected signal.

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
                path: str, output_dir: str = "results",
                p7_fit: np.ndarray | None = None) -> None:
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
            Scales the area of a normalised density, not the height.
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
    p7_fit : ndarray with shape (5,), optional
        Parameters of the modified Pearson VII fit: amplitude, centre,
        sigma, shape ``m`` and asymmetry ``E``. Default None, which
        writes the original two-row, seven-column file unchanged; when
        given, a third row is written and the columns ``m`` and ``E``
        are appended.

        Note ``m`` is **censored at 1000**, the upper bound of
        `peakfitting.PEARSON7_M_BOUNDS`: the Gaussian is only reached in
        the limit, so a genuinely Gaussian peak drives ``m`` to the rail
        and the value there means "Gaussian" rather than a fitted
        number. ``m`` and ``E`` are left empty on the Gaussian and
        Skew-Normal rows, not applicable being different from zero.

    Returns
    -------
    None

    Notes
    -----
    Writes ``<stem>_<mol>.csv`` under a subdirectory named after the
    parent directory of `path`, one row per distribution.

    **The ``A``, ``x0`` and ``sigma`` columns do not mean the same
    thing on every row.** On the Gaussian and Pearson VII rows they are
    the peak height, the apex and a width. On the Skew-Normal row they
    are Azzalini's amplitude, location and scale, none of which is a
    moment: the mean sits at ``x0 + sigma * b * d`` and the standard
    deviation is ``sigma * sqrt(1 - (b d)**2)``, with
    ``b = sqrt(2/pi)`` and ``d = alpha / sqrt(1 + alpha**2)``. See
    `peakfitting.skew_norm`.

    The solvent is recovered from the file name by matching
    ``(^.+)__LPYE``, and any name not carrying that literal yields
    "unknown". See the README TO DO.

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
    header = ["mol","solvent","distribution","A","x0","sigma","alpha"]
    if p7_fit is not None:
        # The modified Pearson VII carries two shape parameters where
        # the other two distributions carry one, so `m` and `E` are
        # appended as their own columns rather than folded into
        # `alpha`. They are left empty on the Gaussian and Skew-Normal
        # rows: not applicable, which is not the same as zero.
        for row in (data_gauss, data_skew_norm):
            row["m"] = np.nan
            row["E"] = np.nan
        mol_list.append({
            "mol": mol,
            "solvent": solvent,
            "distribution": "Pearson VII",
            "A": p7_fit[0],
            "x0": p7_fit[1],
            "sigma": abs(p7_fit[2]),
            "alpha": np.nan,
            "m": p7_fit[3],
            "E": p7_fit[4],
            })
        header = header + ["m", "E"]
    df = pd.DataFrame(mol_list)
    mobile_phase = os.path.basename(os.path.dirname(path))
    outdir = (
        os.path.join(output_dir, mobile_phase)
        if mobile_phase else output_dir
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname + "_" + mol + ".csv")
    df.to_csv(outpath, index=False, header=header)
    return None
