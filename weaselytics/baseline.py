#!/usr/bin/python
# coding: utf-8
"""
Functions to perform the baseline correction.
"""
import numpy as np
from peakfitting import peaks_params
from utils import r2_fct#, smooth_SG

def relevant_range(s):
    """
    Limits the signal to the relevant range in order to find the optimal
    cutoff frequency for the BEADS algorithm.

    Parameters
    ----------
    s : array-like
        The signal to limit.

    Returns
    -------
    _last_arg : int
        Index of the last relevant data point of in the signal ``s``.

    """
    _peaks, _widths = peaks_params(s)   #Smooth `s`?
    _arg_last_peak = _peaks.argmax()
    _last_peak = _peaks[_arg_last_peak]
    _buffer = int(3*np.ceil(_widths)[_arg_last_peak])
    _limmax = _last_peak + _buffer
    if len(s) > _limmax:
        _last_arg = _limmax
    else:
        _last_arg = len(s)
    return _last_arg

def log_transform(s,epsilon):
    """
    Log transformation used in the calculation of BEADS frequency cutoff. For
    further information, see [1].

    Parameters
    ----------
    s : array-like
        The data to transform.
    epsilon : float
        An arbitrary positive offset. The larger the offset, the less
        aggressive the pre-treatment. Default is 1. This value is appropriate
        regarding the magnitude of the signals being processed, which reach
        maxima around 500–10,000. Another reason for selecting ``epsilon = 1``
        is because if ``yi = min (y)``, then 
        ``log (yi − min(y) + 1) = log 1 = 0``.

    Returns
    -------
    log_s : numpy.ndarray
        The log transformed data.
    
    References
    ----------
    [1] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex
        chromatograms using the BEADS algorithm. Journal of Chromatography A,
        2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057.

    """
    log_s = np.log10(s-np.min(s)+epsilon)
    return log_s


def r2_beads(f_cut, s, bl_fitter, asym=1.0, fp=True, hw=None):
    """
    Minimal baseline correction with the BEADS algorithm. Used to compute
    the autocorrelation plot.

    Parameters
    ----------

    Returns
    -------

    """
    _bl, _p = bl_fitter.beads(
            s,
            freq_cutoff = f_cut,
            fit_parabola = fp,
            asymmetry = asym,
            smooth_half_window = hw
            )
#    _s_corr = s - _bl
    _s_corr = _p["signal"]
    _r2 = r2_fct(_s_corr)
    return _r2

