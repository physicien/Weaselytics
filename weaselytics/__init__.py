"""
====================================================================
weaselytics - A library to extract and analyse chromatographic data.
====================================================================
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("weaselytics")
except PackageNotFoundError:
    __version__ = "0.1.0"

from weaselytics.baseline import auto_beads
from weaselytics.export import export_csv, export_dist, export_txt
from weaselytics.parsers import ParsedData
from weaselytics.peakfitting import fit_peak, gauss, skew_norm
from weaselytics.plot import plot, r2_plots
from weaselytics.utils import smooth_SG


def main() -> None:
    from weaselytics.weaselytics import main as _main
    _main()


__all__ = [
    "ParsedData",
    "auto_beads",
    "fit_peak",
    "gauss",
    "main",
    "skew_norm",
    "smooth_SG",
    "export_txt",
    "export_csv",
    "export_dist",
    "plot",
    "r2_plots",
]
