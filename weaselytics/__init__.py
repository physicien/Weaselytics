"""
====================================================================
weaselytics - A library to extract and analyse chromatographic data.
====================================================================
"""

from weaselytics.parsers import ParsedData
from weaselytics.baseline import auto_beads
from weaselytics.peakfitting import fit_peak, gauss, skew_norm
from weaselytics.utils import smooth_SG
from weaselytics.export import export_txt, export_csv, export_dist
from weaselytics.plot import plot, r2_plots

__all__ = [
    "ParsedData",
    "auto_beads",
    "fit_peak",
    "gauss",
    "skew_norm",
    "smooth_SG",
    "export_txt",
    "export_csv",
    "export_dist",
    "plot",
    "r2_plots",
]
