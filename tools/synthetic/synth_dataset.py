#!/usr/bin/python3
"""
Generate synthetic chromatograms with known ground truth.

Each signal is a baseline plus peaks plus white noise, on a sample index
and in signal units, as the simulated chromatograms of Ning, Selesnick &
Duval (2014, §5.1). The true baseline is saved beside the signal, so a
baseline estimate can be scored against it.

The baseline is one of five types:

1. a polynomial plus a sinusoid, Ning et al.'s Type 1;
2. a stationary low-pass random process, Ning et al.'s Type 2;
3. to 5. the parabola, exponential decay and step that Donald Erb,
   author of pybaselines, wrote to test the cutoff selection of this
   project, legacy types that a set holds with `--legacy-baselines`
   only.

Every baseline is defined over the whole record: a longer record holds
as many waves, sampled more densely, while the peaks keep their width in
samples, unless the plate model sets it.

Between 1 and 30 analytes are drawn per signal (Komsta 2011), 0 to 30
with the plate model, in the shape `--peak-model` names, an
exponentially-modified Gaussian by default, a modified Pearson VII or a
Gaussian. Each draws its tailing factor, by default as one plus a
log-normal variate; `--tailing-factor` draws it uniformly in a range
instead. Their widths are drawn in samples, or follow the plate model,
sigma = tR / sqrt(N) (Snyder, Kirkland & Dolan 2009, Eqs. 2.9b and
2.10), with `--plate-model`. Each analyte stays where it was drawn or is
dropped. It must return to the noise before the 20 samples at each end
of the record, and keep a resolution of at least 1 from the others. A
share of the analytes, `--overlap-prob`, may overlap. The placement
stops at the first analyte that would take the share of the record they
cover above 33%. The name of a signal gives the number left. The plate
model brings the bipolar injection artifact, at the dead time, in the
first third of the record and ahead of every analyte.

Records are 500 to 40000 points long, drawn per signal unless
`--length` fixes them.

Every signal is written as a two-column text file, sample index and
signal, that the weaselytics pipeline reads unchanged. The split into
signal, baseline and noise goes to ``truth/<stem>__truth.npz``, one row
per signal to ``manifest.csv``, and a plot to ``plots/`` unless
``--no-plot`` is given. ``PROVENANCE.txt`` records the commit, the exact
call with every argument, the machine and the library versions. The
output directory must be new or empty.

Usage
-----
python tools/synthetic/synth_dataset.py OUTPUT_DIR [--seed 0]
    [--legacy-baselines] [--length N] [--n-peaks K] [--plate-model]
    [--overlap-prob P] [--peak-model emg|pearson7|gaussian]
    [--tailing-factor LO HI] [--pearson-m LO HI]
"""

import argparse
import datetime
import json
import os
import shlex
import socket
import subprocess
import sys

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402
from scipy.special import erfc, erfcx  # noqa: E402
from scipy.stats import exponnorm, skewnorm  # noqa: E402

from weaselytics.peakfitting import pearson7  # noqa: E402

# Analytes per signal, drawn uniformly unless `--n-peaks` fixes the
# count: 1 to 30, as in the synthetic data of Komsta (2011, p. 724,
# "Synthetic Data"). With the plate model the lower end is 0: a blank
# still holds the injection artifact.
N_PEAKS_RANGE = (1, 30)
# Standard deviation of an analyte, in samples, drawn uniformly per peak
# when the plate model is off. Chosen by eye to match the peaks of the
# 2000-sample chromatograms of Ning et al. (2014, Figs. 5 and 6).
FIXED_SIGMA_RANGE = (1.5, 8.)
# Column plate number of the plate model, drawn once per signal. The
# range was chosen around the N of about 10,000 that Snyder, Kirkland &
# Dolan (2009, §2.5, p. 57) call a good starting point in many cases.
PLATE_NUMBER_RANGE = (2000., 14000.)
# Levels of the white noise, by the name they take in the signals: the
# energy of the baseline over the energy of the noise, in dB; see
# `noise_for_snr`. Chosen. Ning et al. (2014, §5.1) vary an SNR from -5 to
# 25 dB without defining it; the ratio here has the form of the output
# SNR they do define, the energy of the generated baseline over that of
# the error of the estimated one.
NOISE_CASES = {'snr30': 30., 'snr25': 25., 'snr20': 20.}
# Height of an analyte, drawn log-uniformly in a range set by the baseline
# type, so that the tallest analytes stand 5 to 10 times as high as the
# baseline is large, its highest point less its lowest. The upper end,
# per type, is ten times the typical size of its baselines, and the lower
# end is it over `HEIGHT_SPAN`. Chosen by eye to match the contrast and
# the spread of heights in Ning et al. (2014, Figs. 5 and 6). No height is
# drawn below `MIN_SNR` noise standard deviations: MacDougall et al.
# (1980, Eq. (9)) place the limit of detection three standard deviations
# of the blank above it. Snyder, Kirkland & Dolan (2009, §11.2.5.1) call
# S/N = 3 a commonly accepted definition of the limit of detection. They
# measure the noise peak to peak (§4.2.3.1, Fig. 4.7), which is larger
# than the standard deviation, so their limit sits above this floor.
HEIGHT_MAX = {1: 40., 2: 90., 3: 50., 4: 80., 5: 120.}
HEIGHT_SPAN = 60.
MIN_SNR = 3.

_FWHM_PER_SIGMA = 2.3548

# Shapes of the analyte peaks, chosen with `--peak-model`; see
# `analyte_peak`.
PEAK_MODELS = ('emg', 'pearson7', 'gaussian')
# Tailing factor of an analyte, drawn per peak when no range is asked
# for: one plus a log-normal variate, with the median and the standard
# deviation of its logarithm given below. A Pearson VII analyte fronts
# instead, at one minus the variate, with the probability below; an EMG
# cannot. The values were chosen: half the tailing factors fall between
# about 1.08 and 1.22, and about one in seven is above 1.3. For
# reference, Snyder, Kirkland & Dolan (2009, §2.4.2) call tailing up to
# 1.2 minor, usually reducible to 1.5 or less, and give 2 as a common
# limit for routine separations.
TF_EXCESS_MEDIAN = 0.13
TF_EXCESS_LOG_SIGMA = 0.77
FRONTING_SHARE = 0.02
# Shape parameter of a Pearson VII analyte, drawn log-uniformly, so that
# heavy tails and the near-Gaussian limit are equally likely per decade.
# Chosen: a low m gives heavy, Lorentzian-like tails and a high m nearly
# Gaussian peaks. Niezen et al. (2022, Table 2) kept m from 3.5 to 51
# from their fits, and Milani et al. (2024, §3.1.3) bound it between 1
# and 1000.
PEARSON7_M_RANGE = (4., 1000.)
# The peak shapes of a dataset: the model, the range the tailing factor
# of each analyte is drawn in uniformly (None draws it as above), and
# the range of the Pearson VII shape parameter.
DEFAULT_SHAPES = {'model': 'emg', 'tailing_factor': None,
                  'pearson_m': PEARSON7_M_RANGE}
# Probability that an analyte may overlap the others, chosen. The peaks
# of Komsta (2011, p. 724, "Synthetic Data") have "a slight probability
# to be overlapped", without a value.
OVERLAP_PROB = 0.1
# The peak layout of a dataset: the analyte count (None draws it, see
# `N_PEAKS_RANGE`), the plate model, which brings the dead time and the
# injection artifact, and the probability that an analyte may overlap
# its neighbours.
DEFAULT_LAYOUT = {'n_peaks': None, 'plate_model': False,
                  'overlap_prob': OVERLAP_PROB}

def emg_peak(t: np.ndarray, tc: float, sigma: float, tau: float,
             height: float) -> np.ndarray:
    """
    Evaluate an exponentially-modified Gaussian peak.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    tc : float
        Center of the Gaussian component.
    sigma : float
        Standard deviation of the Gaussian component.
    tau : float
        Time constant of the exponential tail.
    height : float
        Peak maximum.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The peak profile, scaled so its maximum equals `height`.

    """
    shape = exponnorm.pdf(t, K=max(tau / sigma, 1e-3), loc=tc,
                          scale=sigma)
    peak_max = shape.max()
    if peak_max <= 0:
        return np.zeros(len(t))
    return height * shape / peak_max


def pearson7_width(sigma: float, m: float) -> float:
    """
    Width parameter of a symmetric Pearson VII with a Gaussian's FWHM.

    `peakfitting.pearson7` falls to half its height at
    ``w sqrt(m (2**(1/m) - 1))`` from the centre when its asymmetry is
    zero, so this `w` gives it the FWHM of a Gaussian of standard
    deviation `sigma`.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian, the peak's sigma.
    m : float
        Shape parameter of the Pearson VII, greater than 0.

    Returns
    -------
    w : float
        The width parameter.

    """
    return sigma * _FWHM_PER_SIGMA / (2. * np.sqrt(m * (2. ** (1. / m) - 1.)))


def pearson7_floor(m: float, e: float) -> float:
    """
    Height at which the modified Pearson VII levels off on its long side.

    With an asymmetry `e` other than zero, the term ``x / (w + e x)`` in
    `peakfitting.pearson7` tends to ``1 / e`` far from the centre, so the
    profile tends to ``(1 + 1 / (m e**2))**(-m)`` of its height instead
    of zero: after the peak when `e` > 0, before it when `e` < 0.

    Parameters
    ----------
    m, e : float
        Shape and asymmetry parameters.

    Returns
    -------
    floor : float
        The limit, as a fraction of the height; 0 when `e` is 0.

    """
    if e == 0.:
        return 0.
    return float((1. + 1. / (m * e * e)) ** (-m))


def pearson7_peak(t: np.ndarray, tc: float, sigma: float, m: float,
                  e: float, height: float) -> np.ndarray:
    """
    Evaluate a modified Pearson VII peak that tends to zero.

    The profile is `peakfitting.pearson7` with the width of
    `pearson7_width`, less its `pearson7_floor`, rescaled to keep its
    height and clipped at zero. The short side ends where the fitted
    function falls to the floor; the long side tends to zero as the
    inverse of the distance from the apex. The apex stays at `tc`.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    tc : float
        Apex.
    sigma : float
        Standard deviation of the Gaussian whose FWHM the symmetric
        profile has; see `pearson7_width`.
    m : float
        Shape parameter, greater than 0: near 1 a Lorentzian, large a
        Gaussian.
    e : float
        Asymmetry parameter: positive tails, negative fronts.
    height : float
        Peak maximum.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The peak profile.

    Notes
    -----
    Removing the floor changes the profile by at most the floor itself,
    as a fraction of the height. It is below 0.5% for ``m`` above 3.5
    and ``e`` below 0.28; at ``m = 1`` and ``e = 0.3`` it is 8%, and the
    shape departs visibly from the fitted function.
    The long side still falls slowly: at a low `m` and a large `e`, it
    stays above 0.1% of the height a hundred sigmas from the apex.

    """
    floor = pearson7_floor(m, e)
    y = pearson7(np.asarray(t, dtype=float),
                 [1., tc, pearson7_width(sigma, m), m, e])
    return height * np.clip((y - floor) / (1. - floor), 0., None)


def analyte_peak(t: np.ndarray, tc: float, sigma: float, shape: dict,
                 height: float) -> np.ndarray:
    """
    Evaluate one analyte peak in the model its `shape` names.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    tc : float
        Centre: of the Gaussian component for ``emg``, the apex
        otherwise.
    sigma : float
        The peak's standard deviation at its place. It is
        the Gaussian component's for ``emg``, and sets the FWHM of the
        symmetric profile for ``pearson7``.
    shape : dict
        ``model``, one of `PEAK_MODELS`, with ``tau_ratio`` for
        ``emg`` and ``m`` and ``e`` for ``pearson7``; see `draw_shape`.
    height : float
        Peak maximum.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The peak profile.

    """
    model = shape['model']
    if model == 'emg':
        return emg_peak(t, tc, sigma, shape['tau_ratio'] * sigma, height)
    if model == 'pearson7':
        return pearson7_peak(t, tc, sigma, shape['m'], shape['e'], height)
    if model == 'gaussian':
        return height * np.exp(-0.5 * ((np.asarray(t) - tc) / sigma) ** 2)
    raise ValueError(f'unknown peak model {model!r}')


def _crossings(t: np.ndarray, y: np.ndarray, frac: float
               ) -> tuple[float, float, float]:
    """
    Measure the apex of a sampled peak and where it crosses a level.

    The apex is refined by a parabola through the three samples around
    the maximum; each crossing is interpolated linearly between the two
    samples that bracket `frac` of the maximum.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Sample index.
    y : numpy.ndarray, shape (N,)
        Noise-free, single-peaked profile.
    frac : float
        The level, as a fraction of the maximum, between 0 and 1.

    Returns
    -------
    apex, left, right : float
        Apex time and the times where the profile crosses the level on
        either side, samples. A crossing beyond the record is reported
        at its end.

    """
    k = int(np.argmax(y))
    apex = t[k]
    if 0 < k < len(y) - 1:
        denom = y[k - 1] - 2. * y[k] + y[k + 1]
        if denom < 0:
            apex += 0.5 * (y[k - 1] - y[k + 1]) / denom * (t[1] - t[0])
    level = frac * y[k]
    below = np.nonzero(y[:k] < level)[0]
    if below.size:
        i = below[-1]
        left = t[i] + (level - y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i])
    else:
        left = t[0]
    below = np.nonzero(y[k:] < level)[0]
    if below.size:
        i = k + below[0]
        right = t[i - 1] + ((y[i - 1] - level) / (y[i - 1] - y[i])
                            * (t[i] - t[i - 1]))
    else:
        right = t[-1]
    return float(apex), float(left), float(right)


def tailing_factor(t: np.ndarray, y: np.ndarray) -> float:
    """
    Tailing factor of a sampled peak, ``(A + B) / 2A`` at 5% height.

    A and B are the distances from the apex to the leading and the
    trailing edge at 5% of the height (Snyder, Kirkland & Dolan 2009,
    §2.4.2, Fig. 2.16a): 1 for a symmetric peak, above 1 for tailing,
    below 1 for fronting.

    Parameters
    ----------
    t, y : numpy.ndarray, shape (N,)
        Time axis and noise-free, single-peaked profile.

    Returns
    -------
    tf : float
        The tailing factor. Biased when the profile does not fall to 5%
        on both sides within the record, and biased low on a peak with
        few samples per sigma, where the linear interpolation of the
        crossings cuts the curvature of the tail; `shape_factors` gives
        the value free of sampling.

    """
    apex, left, right = _crossings(t, y, 0.05)
    return (right - left) / (2. * (apex - left))


def asymmetry_factor(t: np.ndarray, y: np.ndarray) -> float:
    """
    Asymmetry factor of a sampled peak, ``B / A`` at 10% height.

    A and B as in `tailing_factor`, at 10% of the height (Snyder,
    Kirkland & Dolan 2009, §2.4.2, Fig. 2.16a).

    Parameters
    ----------
    t, y : numpy.ndarray, shape (N,)
        Time axis and noise-free, single-peaked profile.

    Returns
    -------
    a_s : float
        The asymmetry factor. Biased as `tailing_factor` is, at 10% of
        the height.

    """
    apex, left, right = _crossings(t, y, 0.10)
    return (right - apex) / (apex - left)


def _unit_profile(shape: dict):
    """
    Profile of a peak shape at unit sigma and its apex.

    Parameters
    ----------
    shape : dict
        As in `analyte_peak`.

    Returns
    -------
    f : callable
        The profile as a function of a scalar time, in sigmas from the
        centre of `analyte_peak`.
    apex : float
        Time of its maximum, found by bounded minimization for ``emg``.

    """
    model = shape['model']
    if model == 'pearson7':
        return (lambda x: pearson7_peak(np.atleast_1d(x), 0., 1., shape['m'],
                                        shape['e'], 1.)[0]), 0.
    if model == 'emg':
        k = max(shape['tau_ratio'], 1e-3)

        def f(x):
            """
            `scipy.stats.exponnorm` density at unit scale, evaluated
            directly: `scipy.stats.exponnorm.pdf` costs far more per
            scalar call. The scaled complementary error function keeps
            it finite far in the tail.
            """
            z = (1. / k - x) / np.sqrt(2.)
            if z > 0.:
                return float(np.exp(-0.5 * x * x) * erfcx(z) / (2. * k))
            return float(np.exp(0.5 / (k * k) - x / k) * erfc(z) / (2. * k))
        res = minimize_scalar(lambda x: -f(x), bounds=(-3., 3. + k),
                              method='bounded',
                              options={'xatol': 1e-10})
        return f, float(res.x)
    if model == 'gaussian':
        return (lambda x: float(np.exp(-0.5 * x * x))), 0.
    raise ValueError(f'unknown peak model {model!r}')


def _shape_edges(shape: dict, fracs: tuple[float, ...]
                 ) -> tuple[float, list[tuple[float, float]]]:
    """
    Apex of a peak shape at unit sigma and its crossings of levels.

    Parameters
    ----------
    shape : dict
        As in `analyte_peak`.
    fracs : tuple of float
        The levels, as fractions of the height.

    Returns
    -------
    apex : float
        Time of the maximum, in sigmas.
    edges : list of (float, float)
        For each level, the crossings before and after the apex, found by
        root finding.

    """
    f, apex = _unit_profile(shape)

    def edge(step, level):
        """The crossing of `level` on the side `step` points to."""
        near, far = apex, apex + step
        while f(far) >= level:
            near, far = far, far + 2. * (far - near)
        return brentq(lambda x: f(x) - level, min(near, far),
                      max(near, far), xtol=1e-12)

    top = f(apex)
    return apex, [(edge(-1., q * top), edge(1., q * top)) for q in fracs]


def shape_factors(shape: dict) -> tuple[float, float]:
    """
    Tailing and asymmetry factors of a peak shape, free of sampling.

    Both are ratios of distances, so they do not depend on the width or
    the height of the peak: they are solved here at unit sigma, each
    crossing found by root finding rather than read off a sampled
    profile as `tailing_factor` and `asymmetry_factor` do.

    Parameters
    ----------
    shape : dict
        As in `analyte_peak`.

    Returns
    -------
    tf, a_s : float
        The tailing factor at 5% and the asymmetry factor at 10% of the
        height, as defined in `tailing_factor` and `asymmetry_factor`.

    """
    apex, ((left, right), (left10, right10)) = _shape_edges(
        shape, (0.05, 0.10))
    return ((right - left) / (2. * (apex - left)),
            (right10 - apex) / (apex - left10))


def shape_tailing_factor(shape: dict) -> float:
    """
    Tailing factor of a peak shape, free of sampling.

    Parameters
    ----------
    shape : dict
        As in `analyte_peak`.

    Returns
    -------
    tf : float
        The tailing factor; see `shape_factors`.

    """
    apex, ((left, right),) = _shape_edges(shape, (0.05,))
    return (right - left) / (2. * (apex - left))


def shape_for_tf(model: str, tf: float, m: float | None = None) -> dict:
    """
    The shape of a model that has a given tailing factor.

    Solved by root finding on `shape_tailing_factor`: over the tail time
    constant for ``emg``, over the asymmetry at the given `m` for
    ``pearson7``.

    Parameters
    ----------
    model : str
        ``emg`` or ``pearson7``.
    tf : float
        Target tailing factor. At least 1 for ``emg``, which cannot
        front.
    m : float, optional
        Shape parameter of ``pearson7``. Required for that model.

    Returns
    -------
    shape : dict
        As in `analyte_peak`.

    Raises
    ------
    ValueError
        If `tf` is out of the model's reach: below 1 for ``emg``, or
        beyond what an asymmetry of magnitude 0.99 gives for
        ``pearson7`` at this `m`, below about 0.51 at any `m`.

    Notes
    -----
    A ``pearson7`` tailing factor above about 2.5 at ``m = 3.5`` or 2.0
    at ``m = 51``, or below about 0.63 at ``m = 3.5`` or 0.67 at
    ``m = 51``, needs an asymmetry of magnitude above 0.28, beyond the
    range Niezen et al. (2022, Table 2) kept from their fits of
    experimental peaks. Its floor, removed in `pearson7_peak`, grows
    with it.

    """
    if model == 'emg':
        if tf < 1.:
            raise ValueError('an EMG peak cannot front: tailing factor '
                             f'{tf} < 1')

        def gap(r):
            return shape_tailing_factor({'model': 'emg',
                                         'tau_ratio': r}) - tf
        lo, hi = 1e-3, 1.
        if gap(lo) >= 0.:
            return {'model': 'emg', 'tau_ratio': lo}
        while gap(hi) < 0.:
            lo, hi = hi, 2. * hi
            if hi > 1e3:
                raise ValueError(f'tailing factor {tf} is out of reach')
        return {'model': 'emg',
                'tau_ratio': brentq(gap, lo, hi, xtol=1e-10)}
    if model == 'pearson7':
        if m is None:
            raise ValueError('pearson7 needs its shape parameter m')
        if tf == 1.:
            return {'model': 'pearson7', 'm': m, 'e': 0.}

        def gap_e(e):
            return shape_tailing_factor({'model': 'pearson7', 'm': m,
                                         'e': e}) - tf
        far = 0.99 if tf > 1. else -0.99
        if np.sign(gap_e(far)) == np.sign(gap_e(0.)):
            raise ValueError(f'tailing factor {tf} is out of reach of a '
                             f'Pearson VII with m = {m}')
        return {'model': 'pearson7', 'm': m,
                'e': brentq(gap_e, min(0., far), max(0., far), xtol=1e-12)}
    raise ValueError(f'no tailing factor for peak model {model!r}')


def draw_shape(rng: np.random.Generator, shapes: dict) -> dict:
    """
    Draw the shape of one analyte.

    ``gaussian`` draws nothing. ``pearson7`` first draws its shape
    parameter log-uniformly in ``shapes['pearson_m']``. Both asymmetric
    models then draw a tailing factor, uniformly in
    ``shapes['tailing_factor']`` when it gives a range, else as one plus
    a log-normal variate (`TF_EXCESS_MEDIAN`, `TF_EXCESS_LOG_SIGMA`);
    a ``pearson7`` analyte fronts at one minus it with probability
    `FRONTING_SHARE`. The tailing factor is converted to the model's
    parameter with `shape_for_tf`. In the default draw a value out of
    the model's reach is drawn again, so the distribution is cut at that
    reach.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    shapes : dict
        As `DEFAULT_SHAPES`.

    Returns
    -------
    shape : dict
        As in `analyte_peak`.

    Raises
    ------
    ValueError
        If the model is unknown, or if ``shapes['tailing_factor']``
        draws a value out of the model's reach; see `shape_for_tf`.

    """
    model, tf_range = shapes['model'], shapes['tailing_factor']
    if model == 'gaussian':
        return {'model': 'gaussian'}
    if model not in ('emg', 'pearson7'):
        raise ValueError(f'unknown peak model {model!r}')
    m = None
    if model == 'pearson7':
        lo, hi = np.log10(shapes['pearson_m'])
        m = 10. ** rng.uniform(lo, hi)
    if tf_range is not None:
        return shape_for_tf(model, rng.uniform(*tf_range), m)
    while True:
        excess = rng.lognormal(np.log(TF_EXCESS_MEDIAN), TF_EXCESS_LOG_SIGMA)
        fronts = model == 'pearson7' and rng.random() < FRONTING_SHARE
        try:
            return shape_for_tf(model, 1. - excess if fronts
                                else 1. + excess, m)
        except ValueError:
            continue


def draw_height(rng: np.random.Generator, noise_sigma: float,
                height_max: float) -> float:
    """
    Draw the height of one analyte, or of the artifact spike.

    Log-uniform from `height_max` over `HEIGHT_SPAN`, or `MIN_SNR` times
    `noise_sigma` when that is higher, to `height_max`; see `HEIGHT_MAX`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    noise_sigma : float
        Standard deviation of the signal's noise.
    height_max : float
        Upper end of the range.

    Returns
    -------
    height : float
        The height.

    """
    lo = max(height_max / HEIGHT_SPAN, MIN_SNR * noise_sigma)
    return float(np.exp(rng.uniform(np.log(lo), np.log(height_max))))


###############################################################################
# Erb's test baselines: the analytic baselines of the `make_data` function
# Donald Erb, author of pybaselines, wrote to test the cutoff-frequency
# selection of this project, legacy baseline types 3 to 5. Posted in
# issue #4:
# https://github.com/physicien/Weaselytics/issues/4#issuecomment-4158610986
###############################################################################

def _g(x: np.ndarray, height: float, center: float,
       sigma: float) -> np.ndarray:
    """
    ``pybaselines.utils.gaussian`` in its own argument order.

    Evaluated with pybaselines' own expression,
    ``h * exp(-0.5 * (x - c)**2 / s**2)``, so `erb_baseline` uses the
    same expression as `make_data`.

    """
    return height * np.exp(-0.5 * ((x - center) ** 2) / sigma ** 2)


def erb_baseline(x: np.ndarray, kind: int,
                 exact_integral: bool = False) -> np.ndarray:
    """
    Evaluate one of Erb's four test baselines.

    =====  ==================================================
    kind   baseline
    =====  ==================================================
    0      parabola ending at zero on both ends
    1      exponentially decaying
    2      "very complicated": a logistic approximated by
           integrating a Gaussian, plus a Gaussian bump
    3      sinusoidal
    =====  ==================================================

    Kind 2 is the one the three plateau-count cases of `make_data` use.

    Parameters
    ----------
    x : array-like, shape (N,)
        Abscissa.
    kind : int
        One of 0, 1, 2, 3.
    exact_integral : bool, optional
        Only affects `kind` 2. False (default) reproduces `make_data`
        exactly, a bare ``cumsum`` of the Gaussian. True
        multiplies by the sample spacing, making it the actual integral.

        A bare ``cumsum`` is a Riemann sum missing its ``dx``, so the
        baseline's amplitude scales with the sampling density: at fixed
        span, doubling the point count doubles the drift. That is harmless
        for those three cases, which all use 1000 points, and wrong for
        a population that varies the record length, so `draw_baseline`
        sets True.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline, in the same arbitrary units as the peaks.

    """
    x = np.asarray(x, dtype=float)
    if kind == 0:
        return 2e-5 * (x - 500.) ** 2 - 5.
    if kind == 1:
        return 10. - 10. * np.exp(-x / 600.)
    if kind == 2:
        step = _g(x, 0.05, 400., 100.)
        if exact_integral:
            dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.
            step = step * dx
        return -np.cumsum(step) + _g(x, 3., 800., 100.) - 5.
    if kind == 3:
        return 10. + 1. * np.sin(x / 50.)
    raise ValueError(f"unknown erb baseline kind {kind!r}")


def _dip_scale(pos: np.ndarray, neg: np.ndarray, ratio: float) -> float:
    """
    Scale the dip so the artifact shows a given dip-to-spike ratio.

    The two lobes overlap, so the dip eats into the spike and a dip
    drawn at ``ratio`` times the spike height shows a larger ratio on
    the trace. The depth of ``pos - c * neg`` below zero grows with
    ``c`` and its height above zero shrinks, so exactly one ``c`` makes
    the depth ``ratio`` times the height.

    Parameters
    ----------
    pos : numpy.ndarray, shape (N,)
        The positive lobe.
    neg : numpy.ndarray, shape (N,)
        The negative lobe, of unit height.
    ratio : float
        Dip depth over spike height, as the trace shows them.

    Returns
    -------
    c : float
        The height to give the negative lobe.

    Raises
    ------
    ValueError
        If `neg` is zero everywhere, so no height gives the ratio.

    """
    if ratio <= 0:
        return 0.
    if neg.max() <= 0:
        raise ValueError("the dip lobe is zero everywhere on this grid.")

    def excess(c):
        """
        Dip depth minus `ratio` times spike height, with the dip lobe
        at height `c`.
        """
        y = pos - c * neg
        return -y.min() - ratio * y.max()

    hi = pos.max()
    while excess(hi) < 0:
        hi *= 2.
    return brentq(excess, 0., hi)


def draw_artifact(rng: np.random.Generator, sigma: float, height: float
                  ) -> dict:
    """
    Draw the bipolar injection artifact, relative to its dead time.

    A sharp positive spike at the dead time immediately followed by a
    wider negative undershoot. The undershoot is 0 to 2 times as deep as
    the spike is high, as the trace shows them, drawn as twice a
    Beta(1, 3) variate so small dips are the most common. This law was
    chosen. The negative lobe is genuine signal, which is why a
    symmetric BEADS cost (``asymmetry=1``) is required: an asymmetric
    cost would absorb it into the baseline.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    sigma : float
        Standard deviation of the spike, in samples: that of an analyte
        at its place, so the artifact keeps to the widths of the peaks.
    height : float
        Height of the spike.

    Returns
    -------
    params : dict
        ``pos_sigma``, ``pos_tau``, ``pos_height``, ``neg_offset``
        (centre of the dip after the dead time), ``neg_sigma`` and
        ``dip_ratio``, in samples and signal units; see
        `artifact_profile`.

    """
    tau = rng.uniform(0.3, 0.8) * sigma
    neg_sigma = rng.uniform(1.2, 2.5) * sigma
    ratio = 2. * rng.beta(1., 3.)
    return {'pos_sigma': sigma, 'pos_tau': tau, 'pos_height': height,
            'neg_offset': rng.uniform(1.0, 2.0) * sigma,
            'neg_sigma': neg_sigma, 'dip_ratio': ratio}


def artifact_profile(t: np.ndarray, t0: float, params: dict
                     ) -> tuple[np.ndarray, list[dict]]:
    """
    Evaluate the injection artifact at a dead time.

    The dip is scaled on the sampled trace, so the trace shows the drawn
    dip-to-spike ratio (`_dip_scale`).

    Parameters
    ----------
    t : array-like, shape (N,)
        Sample index.
    t0 : float
        Dead time, the centre of the positive spike.
    params : dict
        As `draw_artifact` returns it.

    Returns
    -------
    contribution : numpy.ndarray, shape (N,)
        The artifact, to add to the signal.
    entries : list of dict
        Per-lobe parameter records for the truth file: ``height`` is the
        lobe's own, ``trace_height`` the one the trace shows, where the
        dip lowers the spike. The negative lobe's entry also carries
        ``dip_ratio``.

    """
    p = params
    contrib = emg_peak(t, t0, p['pos_sigma'], p['pos_tau'], p['pos_height'])
    neg_tc = t0 + p['neg_offset']
    neg_shape = emg_peak(t, neg_tc, p['neg_sigma'], 0.5 * p['neg_sigma'], 1.)
    neg_h = _dip_scale(contrib, neg_shape, p['dip_ratio'])
    trace = contrib - neg_h * neg_shape
    entries = [{'tc': t0, 'sigma': p['pos_sigma'], 'tau': p['pos_tau'],
                'height': p['pos_height'],
                'trace_height': float(trace.max()),
                'fwhm_points': p['pos_sigma'] * _FWHM_PER_SIGMA,
                'artifact': True},
               {'tc': neg_tc, 'sigma': p['neg_sigma'],
                'tau': 0.5 * p['neg_sigma'], 'height': -neg_h,
                'trace_height': float(trace.min()),
                'fwhm_points': p['neg_sigma'] * _FWHM_PER_SIGMA,
                'dip_ratio': p['dip_ratio'], 'artifact': True}]
    return trace, entries


def artifact_extent(params: dict, noise_sigma: float) -> tuple[float, float]:
    """
    How far the artifact stands out of the noise around its dead time.

    Measured on a grid centred on the dead time, wide enough that both
    lobes are back far below any noise level at its ends: fifty times
    the dip's FWHM, where its exponential tail has decayed by more than
    e**-200.

    Parameters
    ----------
    params : dict
        As `draw_artifact` returns it.
    noise_sigma : float
        Noise standard deviation, the level the artifact must exceed.

    Returns
    -------
    lead, trail : float
        Samples before and after the dead time where the artifact still
        exceeds the noise, each plus one sample, so the extent holds at
        any position of the dead time between two samples.

    """
    half = int(np.ceil(50. * params['neg_sigma'] * _FWHM_PER_SIGMA))
    r = np.arange(-half, half + 1, dtype=float)
    contrib, _ = artifact_profile(r, 0., params)
    above = np.nonzero(np.abs(contrib) > noise_sigma)[0]
    if not above.size:
        return 1., 1.
    return -r[above[0]] + 1., r[above[-1]] + 1.


def peak_component(n: int, rng: np.random.Generator, noise_sigma: float,
                   shapes: dict = DEFAULT_SHAPES,
                   shape_rng: np.random.Generator | None = None,
                   height_max: float = HEIGHT_MAX[1],
                   layout: dict = DEFAULT_LAYOUT
                   ) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """
    Build the peak component: the analytes and the injection artifact.

    Everything a chromatogram contains except its baseline and its
    noise, on the sample index.

    The analyte count is drawn uniformly in `N_PEAKS_RANGE`, from 0 with
    the plate model, and ``layout['n_peaks']`` replaces it when given; the
    draw is taken either way. Another count draws other spots, widths and
    heights; the baseline and the noise keep their own draws.
    One spot per analyte, plus one, is drawn uniformly over the record.
    With the plate model, the dead time, the transit time of an
    unretained species, before which nothing elutes, is one of the spots
    in the first `DEAD_TIME_SHARE` of the record whose artifact overlaps
    neither the first nor the last `BORDER_MARGIN` samples, picked at
    random; the earlier spots are discarded. Without it, the last spot
    is discarded and the analytes may sit anywhere. The
    spots are drawn again when none can be the dead time.

    Each analyte stays where it was drawn or is dropped, in the order of
    the draw, which is random along the record. It is dropped when:

    - it is above the noise in the first or the last `BORDER_MARGIN`
      samples, or before the injection artifact is over;
    - its resolution from an analyte already kept is below
      `MIN_RESOLUTION` and it may not overlap;
    - keeping it would take the coverage above `MAX_COVERAGE`. The
      placement stops there, and every analyte after it is dropped too.

    Parameters
    ----------
    n : int
        Number of samples.
    rng : numpy.random.Generator
        Source of randomness.
    noise_sigma : float
        Standard deviation of the noise the signal will carry.
    shapes : dict, optional
        The peak shapes, as `DEFAULT_SHAPES`. Default `DEFAULT_SHAPES`.
    shape_rng : numpy.random.Generator, optional
        Source of randomness for the peak shapes alone. Default None,
        which draws them from `rng`.
    height_max : float, optional
        The tallest height; see `draw_height`. Default that of a
        Type 1 baseline.
    layout : dict, optional
        The peak layout, as `DEFAULT_LAYOUT`. Default `DEFAULT_LAYOUT`.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Sample index, as floats.
    signal : numpy.ndarray, shape (N,)
        Analytes plus the injection artifact.
    peaks : list of dict
        Per-peak parameters; entries with ``artifact=True`` are the
        two injection lobes, the others are analytes. Analytes also
        carry ``apex``, ``fwhm_true_points``, ``half_left``,
        ``half_right``, measured on the noise-free profile, their
        ``tailing_factor`` and ``asymmetry_factor``, from the shape,
        free of sampling (`shape_factors`), their ``shape`` and
        ``may_overlap``. For an EMG, ``tc`` and ``fwhm_points`` describe
        its Gaussian component, not the peak itself.
    info : dict
        ``n_requested`` (analytes drawn), ``dropped_early``
        (spots discarded before the dead time), ``dropped_cut``,
        ``dropped_overlap`` and ``dropped_coverage`` (analytes dropped
        for each reason above), ``n_plates`` (NaN without the plate
        model) and ``dead_time`` (NaN without a dead time).

    Raises
    ------
    ValueError
        If the margins leave no room in the record, or if none of
        `_MAX_REDRAWS` spot sets holds a dead time when one
        is needed, or leaves a feature in a signal without the plate model.

    Notes
    -----
    A signal always holds a feature: when every analyte of a signal
    without the plate model is dropped, they are drawn again.

    """
    plate = layout['plate_model']
    if n <= 2 * BORDER_MARGIN or (plate
                                  and DEAD_TIME_SHARE * n <= BORDER_MARGIN):
        raise ValueError(f'no dead time or analyte fits a {n}-sample '
                         f'record between its margins')
    t = np.arange(n, dtype=float)
    lo, hi = N_PEAKS_RANGE
    n_peaks = int(rng.integers(0 if plate else lo, hi + 1))
    if layout['n_peaks'] is not None:
        n_peaks = layout['n_peaks']
    for _ in range(_MAX_REDRAWS):
        # The extra spot is drawn either way.
        spots = rng.uniform(0., n - 1., n_peaks + 1)
        specs, n_plates = _draw_analytes(spots, layout, noise_sigma, rng,
                                         shapes, shape_rng, height_max)
        signal = np.zeros(n)
        artifact = []
        t0 = float('nan')
        t_front = -np.inf
        early = 0
        if plate:
            able = []
            for spec in specs:
                if not (BORDER_MARGIN <= spec['tc'] < DEAD_TIME_SHARE * n):
                    continue
                # As wide and as high as an analyte at its place.
                spec['artifact'] = draw_artifact(rng, spec['sigma'],
                                                 spec['height'])
                lead, trail = artifact_extent(spec['artifact'],
                                              noise_sigma)
                if (spec['tc'] - lead >= BORDER_MARGIN
                        and spec['tc'] + trail <= n - 1. - BORDER_MARGIN):
                    able.append(spec)
            if not able:
                continue
            start = able[int(rng.integers(len(able)))]
            t0 = start['tc']
            early = sum(q['tc'] < t0 for q in specs)
            specs = [q for q in specs if q['tc'] > t0]
            contrib, artifact = artifact_profile(t, t0, start['artifact'])
            signal += contrib
            above = np.nonzero(np.abs(contrib) > noise_sigma)[0]
            t_front = t[above[-1]] if above.size else t0
        else:
            specs = specs[:-1]
        kept, profiles, dropped = _keep_analytes(t, specs, t_front,
                                                 noise_sigma)
        if kept or plate:
            break
    else:
        raise ValueError(f'no dead time or analyte fits a {n}-sample '
                         f'record in {_MAX_REDRAWS} draws')
    for y in profiles:
        signal += y
    peaks = artifact + sorted(kept, key=lambda p: p['apex'])
    return t, signal, peaks, {'n_requested': n_peaks,
                              'dropped_early': early, **dropped,
                              'n_plates': n_plates, 'dead_time': t0}


def _keep_analytes(t: np.ndarray, specs: list[dict], t_front: float,
                   noise_sigma: float
                   ) -> tuple[list[dict], list[np.ndarray], dict]:
    """
    Keep the analytes that fit where they were drawn, drop the others.

    See `peak_component` for the rules.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Sample index.
    specs : list of dict
        As `_draw_analytes` returns them.
    t_front : float
        End of the injection artifact, or of the dead time, samples.
    noise_sigma : float
        Noise standard deviation, the level a peak must be within.

    Returns
    -------
    kept : list of dict
        Parameters of the kept analytes, in draw order.
    profiles : list of numpy.ndarray
        Their profiles.
    dropped : dict
        ``dropped_cut``, ``dropped_overlap`` and ``dropped_coverage``.

    """
    # A peak cut by an end of the record makes the endpoint mismatch that
    # BEADS' periodicity requirement forbids. No parabola correction
    # repairs it.
    quiet = t <= t_front
    quiet[:BORDER_MARGIN] = True
    quiet[-BORDER_MARGIN:] = True
    kept, profiles, widths = [], [], []
    dropped = dict.fromkeys(('dropped_cut', 'dropped_overlap',
                             'dropped_coverage'), 0)
    for i, spec in enumerate(specs):
        shape = spec['shape']
        y = analyte_peak(t, spec['tc'], spec['sigma'], shape,
                         spec['height'])
        # A profile narrower than the sampling can miss every sample.
        if y.max() <= noise_sigma or y[quiet].max() > noise_sigma:
            dropped['dropped_cut'] += 1
            continue
        width = _apex_and_half_width(t, y)
        if not spec['may_overlap'] and any(
                _resolution(width, o) < MIN_RESOLUTION for o in widths):
            dropped['dropped_overlap'] += 1
            continue
        if _coverage(t, widths + [width]) > MAX_COVERAGE:
            # Placement stops here: a narrower analyte drawn later could
            # still fit, which would favour narrow peaks.
            dropped['dropped_coverage'] = len(specs) - i
            break
        apex, left, right = width
        factors = shape_factors(shape)
        entry = {'tc': spec['tc'], 'sigma': spec['sigma']}
        if shape['model'] == 'emg':
            entry['tau'] = shape['tau_ratio'] * spec['sigma']
        entry.update({'height': spec['height'],
                      'fwhm_points': spec['sigma'] * _FWHM_PER_SIGMA,
                      'apex': apex,
                      'fwhm_true_points': right - left,
                      'half_left': left, 'half_right': right,
                      'shape': shape,
                      'tailing_factor': factors[0],
                      'asymmetry_factor': factors[1],
                      'may_overlap': spec['may_overlap'],
                      'artifact': False})
        kept.append(entry)
        profiles.append(y)
        widths.append(width)
    return kept, profiles, dropped


def _draw_analytes(centres: np.ndarray, layout: dict, noise_sigma: float,
                   rng: np.random.Generator,
                   shapes: dict = DEFAULT_SHAPES,
                   shape_rng: np.random.Generator | None = None,
                   height_max: float = HEIGHT_MAX[1]
                   ) -> tuple[list[dict], float]:
    """
    Draw one analyte set: widths, tails, heights and overlap rights.

    Every analyte takes the same draws from `rng` whatever the layout:
    a width variate, a height and an overlap variate, after one plate
    number per signal.

    Parameters
    ----------
    centres : numpy.ndarray, shape (K,)
        Drawn centres, in samples.
    layout : dict
        As `DEFAULT_LAYOUT`.
    noise_sigma : float
        Noise standard deviation, which sets the height floor; see
        `draw_height`.
    rng : numpy.random.Generator
        Source of randomness.
    shapes : dict, optional
        The peak shapes, as `DEFAULT_SHAPES`. Default `DEFAULT_SHAPES`.
    shape_rng : numpy.random.Generator, optional
        Source of randomness for the peak shapes alone. Default None,
        which draws them from `rng`.
    height_max : float, optional
        The tallest height; see `draw_height`. Default that of a
        Type 1 baseline.

    Returns
    -------
    specs : list of dict
        One per analyte, in the order of `centres`: ``tc``, ``sigma``,
        in samples, ``shape`` (see `draw_shape`), ``height`` (see
        `draw_height`) and ``may_overlap``.
    n_plates : float
        The plate number of the signal, NaN without the plate model.

    """
    n_plates = rng.uniform(*PLATE_NUMBER_RANGE)
    specs = []
    for tc in centres:
        width = rng.uniform()
        if layout['plate_model']:
            # One plate number per signal: W = 4 N^-0.5 tR with
            # W = 4 sigma, so sigma = tR / sqrt(N), tR counted from the
            # injection at the first sample (Snyder, Kirkland & Dolan
            # 2009, §2.3 p. 24, Eqs. 2.9b and 2.10, §2.4 p. 36). The first
            # peaks are sharp and the late ones broad. A mild per-peak
            # jitter keeps the compounds from being identical.
            sigma = (0.85 + 0.3 * width) * tc / np.sqrt(n_plates)
        else:
            lo, hi = FIXED_SIGMA_RANGE
            sigma = lo + (hi - lo) * width
        specs.append({'tc': float(tc), 'sigma': float(sigma),
                      'shape': draw_shape(
                          rng if shape_rng is None else shape_rng, shapes),
                      'height': draw_height(rng, noise_sigma, height_max),
                      'may_overlap': bool(rng.uniform()
                                          < layout['overlap_prob'])})
    return specs, float(n_plates if layout['plate_model'] else 'nan')


# Minimum chromatographic resolution between any two placed peaks,
# Rs = 1.18 (t2 - t1) / (w1 + w2) on the apex and the FWHM. At Rs = 1.0
# about 2% of each band lies under the other (Fornstedt, Forssén &
# Westerlund 2015, §1.2, Eqs. 1.17-1.18, p. 13; Gaussian zones of equal
# amount); 1.5 is baseline resolution (Snyder, Kirkland & Dolan 2009,
# §2.5, Eq. 2.23, p. 54), and 1.18 follows from their W1/2 = 0.588 W
# (§2.4, p. 35). BEADS assumes the peaks and their first two derivatives
# are sparse (Ning, Selesnick & Duval 2014, §1 and Fig. 1). A floor of
# 1.0 still allows small overlaps, to test the limits of that assumption.
# The formula assumes Gaussian peaks and can be far off for asymmetric
# ones, as Dyson (1998, p. 10) says of the same formula without the 1.18;
# these EMG peaks tail, so the 2% is approximate.
MIN_RESOLUTION = 1.0

# Largest share of the record the analytes may cover, with `_coverage`.
# Chosen: in 20000 simulated 2000-sample records of 30 Gaussian analytes
# of `FIXED_SIGMA_RANGE` placed at random, the largest coverage was 33%.
MAX_COVERAGE = 0.33

# Samples at each end of the record that no peak and no artifact may
# overlap, chosen.
BORDER_MARGIN = 20
# The dead time is picked in this first share of the record, chosen.
DEAD_TIME_SHARE = 1. / 3.

# Spot sets drawn before a signal is given up; only a record too short
# to hold its features reaches it.
_MAX_REDRAWS = 1000


def _apex_and_half_width(t: np.ndarray, y: np.ndarray
                         ) -> tuple[float, float, float]:
    """
    Measure the apex and the half-maximum crossings of a sampled peak.

    See `_crossings`, at half the maximum.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Sample index.
    y : numpy.ndarray, shape (N,)
        Noise-free, single-peaked profile.

    Returns
    -------
    apex, left, right : float
        Apex time and the times where the profile crosses half its
        maximum on either side, samples. A crossing beyond the record is
        reported at its end.

    """
    return _crossings(t, y, 0.5)


def _resolution(a: tuple[float, float, float],
                b: tuple[float, float, float]) -> float:
    """
    Chromatographic resolution of two peaks from their apex and FWHM.

    Parameters
    ----------
    a, b : tuple of float
        ``(apex, left, right)`` of each peak, as `_apex_and_half_width`
        returns them.

    Returns
    -------
    rs : float
        ``1.18 |apex_b - apex_a| / (fwhm_a + fwhm_b)``; see
        `MIN_RESOLUTION` for the source and its limits.

    """
    return 1.18 * abs(b[0] - a[0]) / ((a[2] - a[1]) + (b[2] - b[1]))


def _coverage(t: np.ndarray, widths: list[tuple[float, float, float]]
              ) -> float:
    """
    Share of the record inside the base widths of a set of peaks.

    The base of a peak extends from its apex, on each side, 4 / 2.3548
    times as far as its half-maximum crossing: four standard deviations
    in all for a Gaussian, the width W of which the half-height width is
    0.588 W (Snyder, Kirkland & Dolan 2009, §2.4, p. 35). The parts
    shared by several peaks count once.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Sample index.
    widths : list of (float, float, float)
        ``(apex, left, right)`` of each peak, as `_apex_and_half_width`
        returns them.

    Returns
    -------
    coverage : float
        Between 0 and 1.

    Notes
    -----
    The tail of a strongly tailing peak extends beyond this base.

    """
    ratio = 4. / _FWHM_PER_SIGMA
    covered, reach = 0., t[0]
    for left, right in sorted((a - ratio * (a - lo), a + ratio * (hi - a))
                              for a, lo, hi in widths):
        right = min(right, t[-1])
        covered += max(0., right - max(left, reach))
        reach = max(reach, right)
    return covered / float(t[-1] - t[0])


def _analyte_meta(t: np.ndarray, peaks: list[dict], info: dict) -> dict:
    """
    Summarize the placed analytes of one peak component.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Sample index.
    peaks : list of dict
        Peaks from `peak_component`.
    info : dict
        The ``info`` returned by `peak_component`.

    Returns
    -------
    meta : dict
        ``n_analytes``, ``rs_min`` (lowest resolution between adjacent
        analytes, NaN below two), ``coverage`` (see `_coverage`),
        everything in `info`, and ``tf_min`` and ``tf_max``, the range
        of the analytes' tailing factors (NaN without analytes).

    """
    analytes = sorted((p['apex'], p['half_left'], p['half_right'])
                      for p in peaks if not p.get('artifact', False))
    tfs = [p['tailing_factor'] for p in peaks
           if not p.get('artifact', False)]
    rs_min = min((_resolution(a, b) for a, b in zip(analytes,
                                                     analytes[1:])),
                 default=float('nan'))
    return {'n_analytes': len(analytes), 'rs_min': float(rs_min),
            'coverage': _coverage(t, analytes), **info,
            'tf_min': min(tfs, default=float('nan')),
            'tf_max': max(tfs, default=float('nan'))}


# Record length in samples. log10 of the length follows a skew-normal
# distribution, with the shape, location and scale below chosen, and
# draws outside the range are redrawn. Short records are the most
# common: half are under about 1500 samples and 2% are over 10000.
#
# Record length sets the fundamental frequency 1/n_used, which is exactly
# what `instability_boundary` keys on, so length must vary. A balanced
# ladder of lengths would over-weight the long records, which are the
# most expensive to sweep.
LOG10_N_SKEWNORM = (10.02, 2.818, 0.512)
N_RANGE = (500, 40000)

# The five baseline types, by number: Ning et al.'s Types 1 and 2, then
# Erb's parabola, exponential and step (`erb_baseline` kinds 0 to 2),
# which are legacy: a set holds them with `--legacy-baselines` only.
BASELINE_TYPES = (1, 2, 3, 4, 5)
LEGACY_TYPES = (3, 4, 5)

# Type 1, Ning et al. (2014, §5.1, p. 163): "the sum of an order-p
# polynomial and a sinusoidal signal of frequency f", p and f uniform in
# a range they do not give. The order is drawn in 1 to 5, to span the
# polynomials of the simulations they cite: degree up to 4 in Komsta
# (2011, p. 724, "Synthetic Data"), order 4 or 5 with N(0, 1)
# coefficients, offset to be positive, in Mazet et al. (2005, §3.1). The
# frequency, in cycles per record, the span of the polynomial and the
# amplitude of the sinusoid were chosen by eye to match Ning et al.'s
# Fig. 5, 2000-sample records.
TYPE1_ORDER_RANGE = (1, 5)
TYPE1_CYCLES_RANGE = (0.4, 1.6)
TYPE1_POLY_SPAN_RANGE = (1., 5.)
TYPE1_SINE_AMPLITUDE_RANGE = (0.5, 2.)
# Type 2, Ning et al. (2014, §5.1, p. 163): "a stationary random process
# with a power spectrum limited to a low-pass range of [0, fc]",
# obtained by low-pass filtering white Gaussian noise; fc is not given.
# The cutoff, in cycles per record (fc = 0.001 cycles per sample on 2000
# samples), and the standard deviation were chosen by eye to match their
# Fig. 6.
TYPE2_CYCLES = 2.
TYPE2_STD_RANGE = (1.5, 4.)
# Type 2 is filtered on a record this many times longer, and the middle
# kept: a low-pass in the Fourier domain is periodic over the filtered
# length, so a record filtered alone would end at the level it starts.
TYPE2_PADDING = 4


def draw_record_length(rng: np.random.Generator) -> int:
    """
    Draw a record length.

    log10 of the length is drawn from `LOG10_N_SKEWNORM` and redrawn
    until the length falls inside `N_RANGE`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    n : int
        Record length, samples.

    """
    lo, hi = N_RANGE
    while True:
        n = int(round(10 ** float(skewnorm.rvs(
            *LOG10_N_SKEWNORM, random_state=rng))))
        if lo <= n <= hi:
            return n


def type1_baseline(n: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, dict]:
    """
    Draw a Type 1 baseline: a polynomial plus a sinusoid.

    See `TYPE1_ORDER_RANGE` for the source and the values. The
    polynomial is evaluated on [-1, 1] and rescaled to its drawn span,
    the sinusoid has a random phase, and the sum is shifted to start at
    zero at its lowest point, as Mazet et al. (2005, §3.1) offset theirs
    into the positive part.

    Parameters
    ----------
    n : int
        Record length, samples.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline.
    info : dict
        ``order`` of the polynomial and ``cycles`` of the sinusoid over
        the record.

    Notes
    -----
    The frequency is set per record, so the baseline holds the same
    number of cycles at any length.

    """
    lo, hi = TYPE1_ORDER_RANGE
    order = int(rng.integers(lo, hi + 1))
    poly = np.polynomial.polynomial.polyval(np.linspace(-1., 1., n),
                                            rng.normal(0., 1., order + 1))
    poly = ((poly - poly.min()) / np.ptp(poly)
            * rng.uniform(*TYPE1_POLY_SPAN_RANGE))
    cycles = rng.uniform(*TYPE1_CYCLES_RANGE)
    amplitude = rng.uniform(*TYPE1_SINE_AMPLITUDE_RANGE)
    phase = rng.uniform(0., 2. * np.pi)
    baseline = poly + amplitude * np.sin(
        2. * np.pi * cycles * np.arange(n) / n + phase)
    return baseline - baseline.min(), {'order': order, 'cycles': cycles}


def type2_baseline(n: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, dict]:
    """
    Draw a Type 2 baseline: low-pass filtered white Gaussian noise.

    See `TYPE2_CYCLES` for the source and the values. White noise
    `TYPE2_PADDING` times longer than the record is cut in the Fourier
    domain above `TYPE2_CYCLES` cycles per record, the middle of the
    result is kept, and it is scaled to its drawn standard deviation.

    Parameters
    ----------
    n : int
        Record length, samples.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline. Its mean is not zero: waves slower than the record
        leave part of their level in it.
    info : dict
        ``cycles``, the cutoff over the record.

    Notes
    -----
    Keeping the middle of a longer filtered record breaks the
    periodicity of the Fourier-domain filter. The two ends of the kept
    record then differ, so its spectrum must be read through a tapering
    window: without one, the jump between the ends shows as power above
    the cutoff.

    """
    m = TYPE2_PADDING * n
    spectrum = np.fft.rfft(rng.normal(0., 1., m))
    spectrum[np.fft.rfftfreq(m) > TYPE2_CYCLES / n] = 0.
    start = (m - n) // 2
    baseline = np.fft.irfft(spectrum, m)[start:start + n]
    return (baseline / baseline.std() * rng.uniform(*TYPE2_STD_RANGE),
            {'cycles': TYPE2_CYCLES})


def draw_baseline(n: int, baseline_type: int, rng: np.random.Generator
                  ) -> tuple[np.ndarray, dict]:
    """
    Draw the baseline of one of the five types.

    Parameters
    ----------
    n : int
        Record length, samples.
    baseline_type : int
        One of `BASELINE_TYPES`: 1 and 2 are Ning et al.'s Types 1 and 2
        (`type1_baseline`, `type2_baseline`), 3 to 5 Erb's parabola,
        exponential and step (`erb_baseline` kinds 0 to 2).
    rng : numpy.random.Generator
        Source of randomness; Erb's baselines draw nothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline.
    info : dict
        What the type drew; empty for Erb's.

    """
    if baseline_type == 1:
        return type1_baseline(n, rng)
    if baseline_type == 2:
        return type2_baseline(n, rng)
    if baseline_type in (3, 4, 5):
        # Erb's coefficients are written for an abscissa of order 0-1000,
        # so the baseline is evaluated there and mapped onto the record,
        # and holds the same shape at any length. exact_integral keeps
        # the step's amplitude from scaling with the sample count.
        u = np.linspace(0., 1000., n)
        return erb_baseline(u, baseline_type - 3, exact_integral=True), {}
    raise ValueError(f'unknown baseline type {baseline_type!r}')


def noise_for_snr(baseline: np.ndarray, snr: float) -> float:
    """
    Standard deviation of white noise at a signal-to-noise ratio.

    Parameters
    ----------
    baseline : numpy.ndarray, shape (N,)
        The baseline, the signal of the ratio.
    snr : float
        Energy of the baseline over energy of the noise, in dB.

    Returns
    -------
    sigma : float
        Standard deviation of the noise.

    Notes
    -----
    The energy counts the level of the baseline, so a shift of the whole
    baseline changes the noise.

    """
    return float(np.sqrt(np.mean(baseline ** 2)) * 10. ** (-snr / 20.))


def make_signal(n: int, baseline_type: int, noise_sigma: float | None,
                rng: np.random.Generator,
                shapes: dict = DEFAULT_SHAPES,
                layout: dict = DEFAULT_LAYOUT,
                snr: float | None = None) -> dict:
    """
    Assemble one chromatogram: peaks on a baseline, plus white noise.

    The peak component comes from `peak_component` and the baseline
    from `draw_baseline`. White noise at one of `NOISE_CASES` is added.

    Parameters
    ----------
    n : int
        Number of samples.
    baseline_type : int
        One of `BASELINE_TYPES`.
    noise_sigma : float or None
        Standard deviation of the white noise; None with `snr`.
    rng : numpy.random.Generator
        Source of randomness.
    shapes : dict, optional
        The peak shapes, as `DEFAULT_SHAPES`. Default `DEFAULT_SHAPES`.
    layout : dict, optional
        The peak layout, as `DEFAULT_LAYOUT`. Default `DEFAULT_LAYOUT`.
    snr : float, optional
        Sets `noise_sigma` with `noise_for_snr`, in dB; see `NOISE_CASES`.
        Default None.

    Returns
    -------
    components : dict
        Keys ``x`` (the sample index), ``y``, ``signal``, ``baseline``,
        ``noise``, ``peaks`` and ``meta``.

        ``baseline`` alone is the truth a baseline estimator is scored
        against. The injection artifact is part of ``signal``: it is a
        sharp bipolar excursion that a low-pass baseline model is not
        meant to reproduce, and scoring on it would let one feature
        dominate the error. Entries of ``peaks`` with ``artifact=True``
        mark it.

        ``meta`` holds the layout, the baseline type and what it drew
        (``baseline_info``), ``n_requested`` (analytes drawn),
        ``n_analytes`` (those kept) and the counts dropped for each
        reason (`peak_component`), ``n_may_overlap`` (kept analytes
        allowed to overlap), ``n_plates`` and ``dead_time`` (NaN when
        unused), ``rs_min``, the lowest resolution between adjacent
        analytes (NaN below two), ``coverage`` (`_coverage`),
        ``sigma_min``, the narrowest analyte in
        samples, ``tf_min`` and ``tf_max``, the range of their tailing
        factors, and ``height_min_sigma``, the lowest analyte height over
        the nominal `noise_sigma` (NaN without analytes).

    """
    # The shapes, the baseline and the noise draw from streams of their
    # own, split from `rng`, so the rest of the signal does not depend on
    # how many draws they take: sets that differ only in the peak model
    # share their baselines and noise, and the places, widths and heights
    # their peaks are drawn with. The baseline type sets the height range
    # and, with `snr`, the noise level.
    shape_rng, baseline_rng, noise_rng = rng.spawn(3)
    baseline, baseline_info = draw_baseline(n, baseline_type, baseline_rng)
    if snr is not None:
        noise_sigma = noise_for_snr(baseline, snr)
    t, signal, peaks, info = peak_component(n, rng, noise_sigma, shapes,
                                            shape_rng,
                                            HEIGHT_MAX[baseline_type],
                                            layout)
    noise = noise_rng.normal(0., noise_sigma, n)
    y = signal + baseline + noise
    analytes = [p for p in peaks if not p['artifact']]
    return {'x': t, 'y': y, 'signal': signal, 'baseline': baseline,
            'noise': noise, 'peaks': peaks,
            'meta': {'n_points': n,
                     'baseline_type': baseline_type,
                     'baseline_info': baseline_info,
                     'noise_sigma': noise_sigma,
                     'peak_model': shapes['model'],
                     'plate_model': layout['plate_model'],
                     'overlap_prob': layout['overlap_prob'],
                     'baseline_range': float(np.ptp(baseline)),
                     'height_min_sigma': min(
                         (p['height'] / noise_sigma for p in analytes),
                         default=float('nan')),
                     'sigma_min': min((p['sigma'] for p in analytes),
                                      default=float('nan')),
                     'n_may_overlap': sum(p['may_overlap']
                                          for p in analytes),
                     **_analyte_meta(t, peaks, info)}}


def plot_signal(parts: dict, stem: str, out_path: str) -> None:
    """
    Render one synthetic signal against its known ground truth.

    Produced unless ``--no-plot`` is given, so the synthetic data can be
    reviewed by eye before it is trusted for scoring. Shows the raw
    measured signal, the true baseline, and the true baseline-corrected
    signal (peaks + noise), with the injection artifact marked.

    Parameters
    ----------
    parts : dict
        The component dictionary returned by `make_signal`.
    stem : str
        Signal identifier, used as the title.
    out_path : str
        Path of the PNG to write.

    """
    x = parts['x']
    y = parts['y']
    baseline = parts['baseline']
    corrected = y - baseline
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax0.plot(x, y, lw=0.6, c='0.35', label='measured $y$')
    ax0.plot(x, baseline, lw=1.6, c='tab:blue', ls='--',
             label='true baseline $b$')
    ax0.axhline(0., c='tab:red', lw=0.8, alpha=0.4)
    ax0.set_ylabel('signal')
    ax0.set_title(f'{stem}   (range {y.max() - y.min():.2f}, '
                  f'n={len(y)}, {parts["meta"]["peak_model"]} peaks)',
                  fontsize=10)
    ax0.legend(loc='upper right', fontsize=8)

    ax1.plot(x, corrected, lw=0.6, c='tab:green', label='$y - b$')
    ax1.axhline(0., c='tab:red', lw=0.8, alpha=0.4)
    ax1.set_xlabel('sample')
    ax1.set_ylabel('signal')
    ax1.legend(loc='upper right', fontsize=8)

    # Mark the injection artifact, from three sigmas before the spike to
    # three after the dip.
    arts = [p for p in parts['peaks'] if p['artifact']]
    if arts:
        lo = arts[0]['tc'] - 3. * arts[0]['sigma']
        hi = arts[1]['tc'] + 3. * arts[1]['sigma']
        for ax in (ax0, ax1):
            ax.axvspan(lo, hi, color='tab:orange', alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# Columns of `manifest.csv`, one row per signal; see `manifest_row`.
MANIFEST_COLUMNS = ('stem', 'baseline_type', 'noise_case', 'n_points',
                    'noise_sigma', 'replicate', 'peak_model',
                    'plate_model',
                    'overlap_prob', 'n_requested',
                    'n_analytes', 'dropped_early', 'dropped_cut',
                    'dropped_overlap', 'dropped_coverage',
                    'n_may_overlap', 'rs_min', 'coverage',
                    'n_plates', 'dead_time', 'sigma_min',
                    'tf_min', 'tf_max', 'height_min_sigma')


def peak_label(n_analytes: int, artifact: bool) -> str:
    """
    The peak field of a signal name: the analytes kept, then ``a``
    when the injection artifact is present.

    Parameters
    ----------
    n_analytes : int
        Analytes in the signal.
    artifact : bool
        Whether the signal carries the injection artifact.

    Returns
    -------
    label : str
        For example ``12`` or ``0a``.

    """
    return f'{n_analytes}{"a" if artifact else ""}'


def signal_stem(peaks: str, peak_model: str, baseline_type: int,
                noise_case: str, n: int, replicate: int) -> str:
    """
    Name one synthetic signal.

    Parameters
    ----------
    peaks : str
        The peak field, see `peak_label`.
    peak_model : str
        One of `PEAK_MODELS`.
    baseline_type : int
        One of `BASELINE_TYPES`.
    noise_case : str
        Key of `NOISE_CASES`.
    n : int
        Record length, samples.
    replicate : int
        Replicate number.

    Returns
    -------
    stem : str
        ``SYN__peaks__model__type<k>__noise__n__replicate``, which
        `synth_diag.stem_fields` reads. The length is padded to the five
        digits of the longest record and the replicate to four, so names
        sort in order.

    """
    return (f'SYN__{peaks}__{peak_model}__type{baseline_type}'
            f'__{noise_case}__{n:05d}__{replicate:04d}')


def manifest_row(stem: str, noise_case: str, replicate: int,
                 parts: dict) -> str:
    """
    Format the `manifest.csv` row of one signal.

    Parameters
    ----------
    stem : str
        Name of the signal.
    noise_case : str
        Key of `NOISE_CASES`.
    replicate : int
        Replicate number.
    parts : dict
        The signal as `make_signal` returns it.

    Returns
    -------
    row : str
        Comma-separated values in the order of `MANIFEST_COLUMNS`.

    """
    meta = parts['meta']
    values = {'stem': stem, 'noise_case': noise_case,
              'replicate': replicate, **meta}
    out = []
    for column in MANIFEST_COLUMNS:
        value = values[column]
        out.append(f'{value:.6g}' if isinstance(value, float)
                   else str(value))
    return ','.join(out)


def git_commit() -> str:
    """
    Commit of the checkout this script runs from.

    Returns
    -------
    commit : str
        The full hash of ``HEAD`` in the script's own checkout,
        followed by ``-dirty`` when tracked files carry uncommitted
        changes, or by ``-unknown`` when git cannot tell.

    Notes
    -----
    ``-dirty`` covers every tracked file of the repository, including
    files the generator never reads, and ignores untracked files.

    Fails when git is not installed or the script is not inside a git
    checkout, for example after being copied elsewhere. The commit then
    reads ``unknown``.

    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here,
                              capture_output=True, text=True, check=True)
        diff = subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=here,
                              capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    # `git diff --quiet` exits with 1 on changes; any other non-zero
    # code is an error, which says nothing about the tree.
    state = {0: '', 1: '-dirty'}.get(diff.returncode, '-unknown')
    return head.stdout.strip() + state


def _arg_text(value) -> str:
    """An argument value as one word: a range joined by a comma."""
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)


def write_provenance(path: str, args: argparse.Namespace,
                     argv: list[str], started: datetime.datetime,
                     commit: str) -> None:
    """
    Write ``PROVENANCE.txt``, the record of how a dataset was made.

    Parameters
    ----------
    path : str
        File to write.
    args : argparse.Namespace
        The parsed arguments, defaults included.
    argv : list of str
        The command line as called, ``sys.argv``.
    started : datetime.datetime
        Time the generation started, timezone-aware.
    commit : str
        The commit, read with `git_commit` when the generation started.

    Returns
    -------
    None

    Notes
    -----
    One line per item, as ``key value``:

    - ``commit``: see `git_commit`;
    - ``command``: the call, with the interpreter that ran it;
    - ``arguments``: every argument at its exact value, the output
      directory made absolute;
    - ``machine``: the host name, with the start and finish times;
    - ``python``: the versions of python, numpy, scipy and matplotlib.

    The commit reads ``unknown`` when git cannot be reached, and the
    record then no longer pins the code.

    """
    params = dict(vars(args))
    params['output'] = os.path.abspath(params['output'])
    params['plot'] = not params.pop('no_plot')
    finished = datetime.datetime.now().astimezone()
    lines = [
        f'commit {commit}',
        f'command {shlex.join([sys.executable] + list(argv))}',
        'arguments ' + ' '.join(f'{k}={_arg_text(v)}'
                                for k, v in params.items()),
        f'machine {socket.gethostname()}, '
        f'started {started.isoformat(timespec="seconds")}, '
        f'finished {finished.isoformat(timespec="seconds")}',
        f'python {sys.version.split()[0]} numpy {np.__version__} '
        f'scipy {scipy.__version__} matplotlib {matplotlib.__version__}',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> None:
    """
    CLI entry point of the synthetic dataset generator.

    """
    parser = argparse.ArgumentParser(
        prog='synth_dataset',
        description='synthetic chromatograms with known ground truth')
    parser.add_argument('output', help='output directory')
    parser.add_argument('--seed', type=int, default=0,
                        help='base random seed (default: 0)')
    parser.add_argument('--replicates', type=int, default=1,
                        help='replicates per case (default: 1)')
    parser.add_argument('--no-plot', action='store_true',
                        help='skip the per-signal PNG in plots/ '
                             '(rendered by default for review by eye)')
    parser.add_argument('--legacy-baselines', action='store_true',
                        help='add the legacy baseline types, Erb\'s '
                             'parabola, exponential and step')
    parser.add_argument('--length', type=int, default=None, metavar='N',
                        help='record length of every signal, samples '
                             '(default: drawn per signal, '
                             f'{N_RANGE[0]} to {N_RANGE[1]})')
    parser.add_argument('--n-peaks', type=int, default=None, metavar='K',
                        help='analytes drawn per signal, before any is '
                             'dropped (default: drawn '
                             f'uniformly, {N_PEAKS_RANGE[0]} to '
                             f'{N_PEAKS_RANGE[1]}, from 0 with --plate-model)')
    parser.add_argument('--plate-model', action='store_true',
                        help='widths from the plate model, sigma = tR / '
                             'sqrt(N), instead of drawn in samples, with '
                             'the injection artifact at the dead time')
    parser.add_argument('--overlap-prob', type=float, default=OVERLAP_PROB,
                        metavar='P',
                        help='probability that an analyte may overlap its '
                             'neighbours, skipping the resolution floor '
                             f'(default: {OVERLAP_PROB:g})')
    parser.add_argument('--peak-model', choices=PEAK_MODELS,
                        default='emg',
                        help='shape of the analyte peaks; gaussian turns '
                             'the asymmetry off (default: emg)')
    parser.add_argument('--tailing-factor', type=float, nargs=2,
                        metavar=('LO', 'HI'), default=None,
                        help='draw each analyte\'s tailing factor '
                             'uniformly in LO to HI; below 1 fronts, '
                             'pearson7 only (default: 1 plus a log-normal '
                             'variate, see TF_EXCESS_MEDIAN)')
    parser.add_argument('--pearson-m', type=float, nargs=2,
                        metavar=('LO', 'HI'), default=None,
                        help='range of the Pearson VII shape parameter, '
                             'drawn log-uniformly, pearson7 only '
                             f'(default: {PEARSON7_M_RANGE[0]:g} '
                             f'{PEARSON7_M_RANGE[1]:g})')
    args = parser.parse_args()
    if args.length is not None and args.length < N_RANGE[0]:
        parser.error(f'--length needs at least {N_RANGE[0]} samples')
    if args.n_peaks is not None:
        if args.n_peaks < 0:
            parser.error('--n-peaks cannot be negative')
        if args.n_peaks == 0 and not args.plate_model:
            parser.error('--n-peaks 0 needs --plate-model: a signal holds at '
                         'least one feature')
    if not 0. <= args.overlap_prob <= 1.:
        parser.error('--overlap-prob needs 0 <= P <= 1')
    tf = args.tailing_factor
    if tf is not None:
        if not 0. < tf[0] <= tf[1]:
            parser.error('--tailing-factor needs 0 < LO <= HI')
        if args.peak_model == 'gaussian':
            parser.error('a gaussian peak has a tailing factor of 1')
        if args.peak_model == 'emg' and tf[0] < 1.:
            parser.error('an emg peak cannot front: use pearson7 for a '
                         'tailing factor below 1')
    if args.pearson_m is not None:
        if args.peak_model != 'pearson7':
            parser.error('--pearson-m applies to pearson7 only')
        if not 0. < args.pearson_m[0] <= args.pearson_m[1]:
            parser.error('--pearson-m needs 0 < LO <= HI')
    shapes = {'model': args.peak_model,
              'tailing_factor': None if tf is None else tuple(tf),
              'pearson_m': (PEARSON7_M_RANGE if args.pearson_m is None
                            else tuple(args.pearson_m))}
    layout = {'n_peaks': args.n_peaks, 'plate_model': args.plate_model,
              'overlap_prob': args.overlap_prob}
    # Refuse a range the model cannot reach before anything is written,
    # rather than part-way through the set. The reach of a Pearson VII
    # depends on m, so both ends of its range are tried.
    if tf is not None:
        ms = shapes['pearson_m'] if args.peak_model == 'pearson7' else (None,)
        for m in ms:
            for value in tf:
                try:
                    shape_for_tf(args.peak_model, value, m)
                except ValueError as exc:
                    parser.error(str(exc))
    # One set per directory, so PROVENANCE.txt and manifest.csv describe
    # every file beside them.
    if os.path.isdir(args.output) and os.listdir(args.output):
        parser.error(f'output directory {args.output} is not empty')
    started = datetime.datetime.now().astimezone()
    commit = git_commit()

    sig_dir = os.path.join(args.output, 'signals')
    truth_dir = os.path.join(args.output, 'truth')
    plot_dir = os.path.join(args.output, 'plots')
    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)
    if not args.no_plot:
        os.makedirs(plot_dir, exist_ok=True)

    manifest = []
    jobs = [(bt, nk, ns) for bt in BASELINE_TYPES
            if args.legacy_baselines or bt not in LEGACY_TYPES
            for nk, ns in NOISE_CASES.items()]
    for k, (bt, nk, ns) in enumerate(jobs):
        for rep in range(args.replicates):
            rng = np.random.default_rng(args.seed + 7919 * (k + 1)
                                        + 104729 * rep)
            # Drawn even when fixed, so the draws after it are the same
            # variates with and without --length.
            n = draw_record_length(rng)
            if args.length is not None:
                n = args.length
            parts = make_signal(n, bt, None, rng, shapes=shapes,
                                layout=layout, snr=ns)
            stem = signal_stem(
                peak_label(parts['meta']['n_analytes'], args.plate_model),
                args.peak_model, bt, nk, n, rep)
            with open(os.path.join(sig_dir, f'{stem}.txt'), 'w') as f:
                for xi, yi in zip(parts['x'], parts['y']):
                    f.write(f'{int(xi)}\t{yi:.6e}\n')
            np.savez(
                os.path.join(truth_dir, f'{stem}__truth.npz'),
                x=parts['x'], y=parts['y'], signal=parts['signal'],
                baseline=parts['baseline'], noise=parts['noise'],
                peaks=json.dumps(parts['peaks']),
                meta=json.dumps(parts['meta']))
            if not args.no_plot:
                plot_signal(parts, stem,
                            os.path.join(plot_dir, f'{stem}.png'))
            manifest.append(manifest_row(stem, nk, rep, parts))
    with open(os.path.join(args.output, 'manifest.csv'), 'w') as f:
        f.write(','.join(MANIFEST_COLUMNS) + '\n')
        f.write('\n'.join(manifest) + '\n')
    write_provenance(os.path.join(args.output, 'PROVENANCE.txt'), args,
                     sys.argv, started, commit)
    print(f'{len(manifest)} signals -> {args.output}')


if __name__ == '__main__':
    main()
