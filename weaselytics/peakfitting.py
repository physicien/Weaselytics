# coding: utf-8
"""
Functions to perform Peak fitting.
"""
import logging
from collections.abc import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from weaselytics.export import export_dist
from weaselytics.utils import peaks_params

logger = logging.getLogger(__name__)


def gauss(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Generate a Gaussian distribution based on `params`.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    params : array-like with shape (n,)
        `params` with the following fields defined:

        amp : float
            The maximum height of the distribution.
        x0 : float
            The center of the distribution.
        sigma : float
            The standard deviation of the distribution.

    Returns
    -------
    dist : numpy.ndarray
        The Gaussian distribution evaluated with x.

    Raises
    ------
    ValueError
        Raised if `sigma` is not greater than 0.

    """
    amp, x0, sigma = params

    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")

    dist = amp*np.exp(-0.5*((x-x0)**2)/sigma**2)
    return dist

def skew_norm(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Generate a Skew normal distribution based on `params`.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    params : array-like with shape (n,)
        `params` with the following fields defined:

        amp : float
            The maximum height of the distribution.
        loc :  float
            The location parameter of the distribution.
        scale : float
            The scale parameter of the distribution.
        alpha : float
            The shape parameter of the distribution.

    Returns
    -------
    dist : numpy.ndarray
        The Skew normal distribution evaluated with x.

    """
    amp, loc, scale, alpha = params

    z = alpha*(x-loc)/scale
    norm = np.sqrt(2*np.pi*scale**2)**-1* np.exp(
            -((x-loc)**2)/(2*scale**2)
            )
    cdf = 0.5*(1+erf(z/np.sqrt(2)))

    dist = amp*2*norm*cdf
    return dist

#: Bounds on the modified Pearson VII shape parameters, from Milani
#: et al. (2024) §3.1.1: "M was restricted to a range of 1-1000, and E
#: to -0.3 to +0.3, preventing implausible peak fits while favoring
#: mathematically ideal solutions." M -> 1 is a Lorentzian and M -> inf
#: a Gaussian, so the upper bound only excludes shapes already
#: indistinguishable from a Gaussian.
#:
#: **A reported m at the upper bound is a rail, not a measurement.**
#: Because the Gaussian is only reached in the limit, a peak that really
#: is Gaussian drives m upward until it stops at 1000; the value then
#: says "Gaussian, as far as this model can express it" and carries no
#: further information. Measured on 92 real analyte peaks this happens
#: on 9 of them (10%), concentrated in the later-eluting molecules; away
#: from the rail m has a median of 6.6 and spans 1.0 to 109. Anything
#: consuming the exported `m` column should treat 1000 as censored.
PEARSON7_M_BOUNDS = (1.0, 1000.0)
PEARSON7_E_BOUNDS = (-0.3, 0.3)


def pearson7(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    r"""
    Generate a modified Pearson VII distribution based on `params`.

    .. math::

       f(x) = A\left(1 + \frac{(x-x_0)^2}
                          {m\,(\sigma + E (x-x_0))^2}\right)^{-m}

    The shape interpolates between a Lorentzian (``m`` near 1) and a
    Gaussian (``m`` large), with ``E`` producing tailing (positive) or
    fronting (negative). Three independent selections favour it over the
    Gaussian and the exponentially-modified Gaussian for chromatographic
    peaks: Niezen et al. compared fifteen distributions by the Akaike
    information criterion and ranked it first (Table 1); Milani et al.
    measured a lower RMSE over 458 fitted peaks; and on 60 randomly
    chosen peaks of this project's own dataset it won on 29, against 20
    for the EMG and 11 for the Gaussian.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    params : array-like with shape (5,)
        `params` with the following fields defined:

        amp : float
            The maximum height of the distribution.
        x0 : float
            The center of the distribution.
        sigma : float
            The width parameter.
        m : float
            The shape parameter; see `PEARSON7_M_BOUNDS`.
        asym : float
            The asymmetry parameter; see `PEARSON7_E_BOUNDS`.

    Returns
    -------
    dist : numpy.ndarray
        The modified Pearson VII distribution evaluated with x.

    Raises
    ------
    ValueError
        Raised if `sigma` or `m` is not greater than 0.

    Notes
    -----
    For a non-zero asymmetry the denominator vanishes at
    ``x - x0 = -sigma / asym``, beyond which the expression rises again
    into a spurious second lobe. It is negligible in magnitude, but is
    clipped to zero so the profile is single-lobed by construction
    rather than by numerical accident.

    References
    ----------
    Niezen, L.E., et al. Critical comparison of background correction
    algorithms used in chromatography. Anal. Chim. Acta 1201 (2022)
    339605, Eq. (14).
    Milani, N.B.L., et al. Anal. Chim. Acta 1312 (2024) 342724, Eq. (2).

    """
    amp, x0, sigma, m, asym = params

    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")
    if m <= 0:
        raise ValueError("m must be greater than 0.")

    dx = x - x0
    denom = sigma + asym * dx
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = amp * (1. + dx**2 / (m * denom**2))**-m
    dist = np.nan_to_num(dist, nan=0., posinf=0., neginf=0.)
    if asym != 0.:
        dist = np.where(np.sign(denom) == np.sign(sigma), dist, 0.)
    return dist

def _lsq_eq(
    p: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray, y: np.ndarray,
    ) -> np.ndarray:
    """
    Compute the vector of residuals in order to solve the least-squares
    problem.

    Parameters
    ----------
    p : array-like with shape (n,)
        Set of independent variables defining the function.
    func : callable
        Function used to solve the least-squares problem.
    x : numpy.ndarray
        Range on the x-axis to fit `func`.
    y : numpy.ndarray
        Values on which to fit `func` for each point of the x-axis range.

    Returns
    -------
    callable
        A function to feed to the `scipy.optimize.least_squares` method.

    """
    return func(x,p) - y

def _lsq_fit(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray, y: np.ndarray,
    n_params: int,
    tau_offset: Callable[[float], float] | float,
    extra: tuple[tuple[float, float, float], ...] = (),
    ) -> np.ndarray:
    """Shared robust least-squares fitting for peak distributions.

    Finds the main peak, builds initial guess and bounds, then runs
    ``scipy.optimize.least_squares`` with a soft_l1 loss.

    Parameters
    ----------
    func : callable
        Model function ``f(x, params)``.
    x, y : numpy.ndarray
        Data to fit.
    n_params : int
        Number of parameters (3 for Gaussian, 4 for Skew-Normal, 5 for
        modified Pearson VII). Must equal ``3 + len(extra)``.
    tau_offset : callable or float
        How far from ``tau0`` to set the ``x0`` bounds.  If callable it
        receives ``sigma0`` (computed from the main peak width).
    extra : sequence of (float, float, float), optional
        Initial value and bounds ``(p0, lower, upper)`` for each
        parameter beyond the shared ``(amp, x0, sigma)``. Carrying the
        bounds with the parameter keeps a model's admissible range next
        to the model instead of encoded in a parameter count.

    Returns
    -------
    s : numpy.ndarray
        Optimised parameters.
    """
    if n_params != 3 + len(extra):
        raise ValueError("n_params must equal 3 + len(extra)")
    peaks, widths = peaks_params(y)
    main_index = np.absolute(y[peaks]).argmax()
    peak = peaks[main_index]

    A0 = y[peak]
    tau0 = x[peak]
    sigma0 = x[peak + int(widths[main_index]/2)] - x[peak]
    p0 = [A0, tau0, sigma0] + [e[0] for e in extra]

    offset = tau_offset(sigma0) if callable(tau_offset) else tau_offset

    if A0 < 0:
        bA = [-np.inf, 0]
    else:
        bA = [0, np.inf]

    lower = [bA[0], tau0 - offset, 0] + [e[1] for e in extra]
    upper = [bA[1], tau0 + offset, np.inf] + [e[2] for e in extra]

    res_robust = least_squares(
        _lsq_eq, p0, loss="soft_l1", f_scale=0.1, args=(func, x, y),
        bounds=(lower, upper),
    )
    return res_robust.x


def _lsq_gauss_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a Gaussian distribution via robust least-squares.

    See `_lsq_fit` for details.
    """
    return _lsq_fit(gauss, x, y, n_params=3, tau_offset=0.1)


def _lsq_skew_norm_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a Skew-Normal distribution via robust least-squares.

    See `_lsq_fit` for details.
    """
    return _lsq_fit(skew_norm, x, y, n_params=4, tau_offset=lambda s: s,
                    extra=((0., -np.inf, np.inf),))


def _lsq_pearson7_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a modified Pearson VII distribution via robust least-squares.

    The shape bounds are `PEARSON7_M_BOUNDS` and `PEARSON7_E_BOUNDS`,
    both from Milani et al. (2024) §3.1.1. The starting values -- m = 10
    and a symmetric E = 0 -- are optimiser seeds only, chosen inside the
    published range rather than fitted to any dataset.

    See `_lsq_fit` for details.
    """
    return _lsq_fit(
        pearson7, x, y, n_params=5, tau_offset=lambda s: s,
        extra=((10., PEARSON7_M_BOUNDS[0], PEARSON7_M_BOUNDS[1]),
               (0., PEARSON7_E_BOUNDS[0], PEARSON7_E_BOUNDS[1])))

def fit_peak(
    s: np.ndarray, x: np.ndarray,
    x0: float | None = None,
    x1: float | None = None,
    mol: str | None = None,
    path: str | None = None,
    output_dir: str = "results",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit robustly the most prominent peak on `x` with both Gaussian and
    Skew-Normal distributions.

    Parameters
    ----------
    s : array-like, shape (N,)
        A signal with peaks.
    x : array-like, shape (N,)
        The x-values on which to fit a peak.
    x0 : float, optional
        Start of interval. The interval includes this value. If `x0` is set to
        `None` (default), then ``x0 = min(x)``.
    x1 : float, optional
        End of interval. The interval includes this value. If `x1` is set to
        `None` (default), then ``x1 = max(x)``.
    mol : str, optional
        Molecule identifier used to export and save the data of the peak. If
        `None` (default), will not export the data.
    path: str, optional
        Path of the data file. If `None` (default), will not export the data.

    Returns
    -------
    x_robust : array-like, shape (N,)
        The x-values of the fitted distributions.
    y_robust_g : array-like, shape (N,)
        The y-values of the Gaussian distribution.
    y_robust_sn : array-like, shape (N,)
        The y-values of the Skew-Normal distribution.
    y_robust_p7 : array-like, shape (N,)
        The y-values of the modified Pearson VII distribution. Of the
        three this is the shape best supported for chromatographic
        peaks; see `pearson7` for the evidence.

    """
    if x0 is not None:
        xmin = x0
    else:
        xmin = min(x)

    if x1 is not None:
        xmax = x1
    else:
        xmax = max(x)

    xdata = x[(x >= xmin) & (x <= xmax)]
    ydata = s[(x >= xmin) & (x <= xmax)]

    x_robust = np.arange(xdata.min() - 0.1, xdata.max() + 0.1, 0.001)

    # Gaussian curve fit
    p_lsq_g = _lsq_gauss_fit(xdata, ydata)
    y_robust_g = gauss(x_robust, p_lsq_g)
    A_g, x0_g, sigma_g = p_lsq_g
    sigma_g = abs(sigma_g)
    logger.info('The amplitude of the gaussian fit is %s', A_g)
    logger.info('The center of the gaussian fit is %s', x0_g)
    logger.info('The sigma of the gaussian fit is %s \n', sigma_g)

    # Skew-Normal curve fit
    p_lsq_sn = _lsq_skew_norm_fit(xdata, ydata)
    y_robust_sn = skew_norm(x_robust, p_lsq_sn)
    A_sn, x0_sn, sigma_sn, alpha_sn = p_lsq_sn
    sigma_sn = abs(sigma_sn)
    logger.info('The amplitude of the skew-normal fit is %s', A_sn)
    logger.info('The center of the skew-normal fit is %s', x0_sn)
    logger.info('The sigma of the skew-normal fit is %s', sigma_sn)
    logger.info('The skew parameter of the skew-normal fit is %s',
                alpha_sn)

    # Modified Pearson VII curve fit
    p_lsq_p7 = _lsq_pearson7_fit(xdata, ydata)
    y_robust_p7 = pearson7(x_robust, p_lsq_p7)
    A_p7, x0_p7, sigma_p7, m_p7, e_p7 = p_lsq_p7
    sigma_p7 = abs(sigma_p7)
    logger.info('The amplitude of the Pearson VII fit is %s', A_p7)
    logger.info('The center of the Pearson VII fit is %s', x0_p7)
    logger.info('The sigma of the Pearson VII fit is %s', sigma_p7)
    logger.info('The shape parameter m of the Pearson VII fit is %s', m_p7)
    logger.info('The asymmetry E of the Pearson VII fit is %s \n', e_p7)

    #if name is given - csv generation
    if mol and path:
        export_dist(mol, p_lsq_g, p_lsq_sn, path, output_dir=output_dir,
                    p7_fit=p_lsq_p7)

    return x_robust, y_robust_g, y_robust_sn, y_robust_p7

