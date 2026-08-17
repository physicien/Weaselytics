#!/usr/bin/python3
"""
Generate synthetic chromatograms with known ground truth.

Two families are produced, so the algorithm can be tested both on this
instrument and more generally:

* **native** — calibrated to the LPYE dataset. Exponentially-modified
  Gaussian peaks; a bipolar dead-time artifact (positive spike + a
  negative undershoot, the dominant feature of the real blanks, and the
  reason a symmetric BEADS cost is needed); a small baseline (real blank
  baseline structure is 0.33 mV median) built from a solvent-front
  decay, a gradient hump, a drift, a level step across the dead time and
  a mid-frequency wander; near-constant detector noise (~0.019 mV);
  120 points/min.

* **lit** — generic, not tied to this experiment, following the BEADS
  benchmark of Ning, Selesnick & Duval (2014), §5: superpositions of
  Gaussian peaks (including a dense, overlapping case) on a large
  Type-1 (polynomial + sinusoid) or Type-2 (low-pass-filtered noise)
  baseline, with the noise set by a target SNR spanning 8-25 dB.

Every signal is written in the same two-column text format as the real
data, so the whole weaselytics pipeline runs on it unchanged; the exact
baseline/signal/noise decomposition is stored in
``truth/<stem>__truth.npz``; and a per-signal plot is written to
``plots/`` (unless ``--no-plot``) so the synthetic data can be reviewed
against the real chromatograms before it is trusted for scoring.

Usage
-----
python tools/synth_dataset.py OUTPUT_DIR [--family both] [--seed 0]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import (  # noqa: E402
    binary_dilation,
    gaussian_filter1d,
    median_filter,
)
from scipy.stats import exponnorm  # noqa: E402

PEAK_CASES = {
    # (number of peaks, FWHM range in points at the run start);
    # 'isocratic' ignores the range and derives widths from a constant
    # plate number instead (width proportional to retention time)
    'blank': (0, None),
    'single_narrow': (1, (8, 14)),
    'multi_narrow': (6, (8, 14)),
    'multi_mixed': (6, (10, 45)),
    'multi_wide': (4, (40, 90)),
    'isocratic': (7, None),
}
BASELINE_CASES = ['exp', 'hump', 'exp_hump_drift']
# White detector noise. Measured on the 339 real signals (MAD of
# consecutive differences): the noise is nearly constant in absolute
# terms at ~0.019 mV regardless of amplitude or blank/non-blank, with a
# maximum of 0.047. These two cases bracket the measured median and the
# upper tail rather than the library-default 0.01/0.06 guesses.
NOISE_CASES = {'typ': 0.019, 'high': 0.038}
# Record lengths for the native family. Real n_points spans 473-39000
# (median 1180). The dead time is absolute at ~4.5 min and analytes
# elute after it, so a native record must span well beyond it: at
# 120 pts/min these are 7.5, 15 and 33 min, all comfortably past a
# 4.5 min dead time, and they bracket the real median and the long-run
# regime (Cyclohexane, Benzene).
N_CASES = [900, 1800, 4000]

# Points per minute. Real dt has median 0.00833 min, i.e. 120 pts/min
# (two samples per second), not the 60 the first version assumed.
PTS_PER_MIN = 120

_FWHM_PER_SIGMA = 2.3548

# Quantisation step of the LPYE detector, in mV. MEASURED, not taken from
# a datasheet: in all 339 reference signals every consecutive difference
# is an exact integer multiple of it to within 1e-9, a ~900-point record
# holds only 45-60 distinct values, and ~25% of consecutive samples are
# identical. See tools/synthetic_data.md §7.
ADC_STEP_MV = 0.008996

# Normal-consistency factor of the median absolute deviation, used by
# Niezen et al. (2022) Eqs. (12a)/(12b) to turn a MAD into a sigma.
MAD_TO_SIGMA = 1.4826

# Selection heuristic for the peak-free stretch (synthetic_data.md §3.1).
# NOT GROUNDED: these decide only which real data is admitted to the
# background pool, never a reported number, and the regions they produce
# are reviewed by eye before use. Reviewed and accepted 2026-07-27.
PEAK_FREE_SIGMA = 8.0        # excursion threshold, in sigma
PEAK_FREE_WINDOW_FRAC = 40   # median-filter width = len(signal) / this


def noise_sigma_mad(y: np.ndarray, on_derivative: bool = True) -> float:
    """
    Estimate the noise level of a trace by the median absolute deviation.

    Niezen et al. (2022) Eq. (12a) applies the MAD to the signal itself;
    their Eq. (12b) applies it to the first derivative, and they note
    that in the presence of a baseline and peaks the derivative gives a
    "more representative value", which is the default here. Both are
    scaled by `MAD_TO_SIGMA` for consistency with a normal distribution.

    Parameters
    ----------
    y : array-like, shape (N,)
        The trace.
    on_derivative : bool, optional
        If True (default), Eq. (12b); otherwise Eq. (12a).

    Returns
    -------
    sigma : float
        The noise estimate, in the units of `y`.

    Notes
    -----
    On quantised data this does not measure analogue noise: it returns a
    small integer multiple of the quantisation step, so it is an upper
    bound. See tools/synthetic_data.md §6.

    References
    ----------
    Niezen, Schoenmakers & Pirok (2022), Anal. Chim. Acta 1201, 339605,
    Eqs. (12a) and (12b).

    """
    v = np.diff(np.asarray(y, dtype=float)) if on_derivative else \
        np.asarray(y, dtype=float)
    return float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def peak_free_stretch(y: np.ndarray, k_sigma: float = PEAK_FREE_SIGMA,
                      window_frac: int = PEAK_FREE_WINDOW_FRAC
                      ) -> slice:
    """
    Locate the longest stretch of a trace showing no peak.

    Niezen et al. (2022) §4.1.1 require a background containing "only
    low-frequency drift and a small amount of initial noise", and obtain
    it by removing peaks from experimental blanks by curve fitting and
    subtraction. Here the same requirement is met by *selecting* a
    peak-free stretch instead, so the drift is the recorded signal
    itself and nothing is fitted or subtracted.

    A point is busy when the residual against a median filter exceeds
    `k_sigma` times the derivative-MAD noise estimate; busy points are
    dilated by half the filter width so a peak's shoulders are excluded
    with it, and the longest surviving run is returned.

    .. warning::
       `k_sigma` and `window_frac` are **not grounded**. They decide only
       which real data enters the background pool, never a reported
       number, and the regions they select are reviewed by eye. The
       criterion cannot certify the absence of peaks -- only the absence
       of excursions it can see, so a very broad, low feature could
       survive it and would then be scored as drift.

    Parameters
    ----------
    y : array-like, shape (N,)
        The trace to search.
    k_sigma : float, optional
        Excursion threshold in units of the noise estimate.
    window_frac : int, optional
        The median filter spans ``len(y) / window_frac`` points.

    Returns
    -------
    region : slice
        The longest peak-free run. Empty (``slice(0, 0)``) when the
        whole trace is busy or the noise estimate is degenerate.

    """
    y = np.asarray(y, dtype=float)
    sigma = noise_sigma_mad(y)
    if sigma <= 0 or len(y) < 8:
        return slice(0, 0)
    width = max(31, len(y) // window_frac) | 1
    resid = np.abs(y - median_filter(y, size=width))
    busy = binary_dilation(resid > k_sigma * sigma,
                           np.ones(max(width // 2, 1), dtype=bool))
    idx = np.flatnonzero(~busy)
    if idx.size == 0:
        return slice(0, 0)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    best = max(runs, key=len)
    return slice(int(best[0]), int(best[-1]) + 1)


def quantise(y: np.ndarray, step: float = ADC_STEP_MV) -> np.ndarray:
    """
    Round a trace onto the detector's quantisation lattice.

    The LPYE detector digitises at `ADC_STEP_MV`; synthetic signals are
    rounded likewise so the benchmark exercises the same code paths as
    real data. This is not cosmetic: `baseline._snr` divides by a MAD of
    consecutive differences, which on quantised data is pinned to the
    lattice and on continuous data is a true noise estimate, so an
    unquantised benchmark measures a different quantity under the same
    name. See tools/synthetic_data.md §7.

    Parameters
    ----------
    y : array-like, shape (N,)
        The trace, in mV.
    step : float, optional
        Quantisation step in mV. Default `ADC_STEP_MV`.

    Returns
    -------
    yq : numpy.ndarray, shape (N,)
        `y` rounded to the nearest multiple of `step`.

    """
    if step <= 0:
        raise ValueError("step must be greater than 0.")
    return np.round(np.asarray(y, dtype=float) / step) * step


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


def broadening_factor(tc: float, t0: float, t_end: float,
                      g: float) -> float:
    """
    Peak-width growth factor with retention time.

    At a constant plate number the peak standard deviation grows with
    the retention time (time since injection at ``t0``): plate theory
    gives sigma = tR / sqrt(N), so a late eluter is several times
    broader than an early one. Parametrised here as ``1 + g * r`` with
    ``r`` the retention fraction from the dead time to the run end, so
    the factor is 1 at ``t0`` and ``1 + g`` at ``t_end``.

    Parameters
    ----------
    tc : float
        Peak centre (retention time), minutes.
    t0 : float
        Dead time, minutes.
    t_end : float
        End of the run, minutes.
    g : float
        Growth strength; the last eluter is ``1 + g`` times broader
        than one at the dead time.

    Returns
    -------
    factor : float
        Multiplicative width factor, at least 1.

    """
    r = np.clip((tc - t0) / max(t_end - t0, 1e-9), 0., 1.)
    return 1. + g * r


def gauss_peak(t: np.ndarray, tc: float, sigma: float,
               height: float) -> np.ndarray:
    """
    Evaluate a plain Gaussian peak of maximum ``height``.

    Used by the ``lit`` family, whose peaks are superpositions of
    Gaussians per Ning, Selesnick & Duval (2014), §5.

    """
    return height * np.exp(-0.5 * ((t - tc) / sigma) ** 2)


# Parameter ranges of the modified Pearson VII. Each range is kept under
# its own provenance rather than blended, so any number taken from here
# can be attributed. See tools/synthetic_data.md §4.2.
#
# PUBLISHED: fitted by Niezen et al. (2022), Table 2, to gradient-RPLC of
# small uncharged molecules. The paper states the parameters "may change
# significantly in other modes of chromatography", so these are not
# transplantable on their own.
PEARSON7_KURTOSIS_NIEZEN = (3.5, 51.0)        # m
PEARSON7_ASYMMETRY_NIEZEN = (0.01, 0.28)      # A_s, tailing only
#
# MEASURED here on 60 randomly-chosen LPYE peaks (p10-p90), fitted by the
# procedure of synthetic_data.md §4.1. Two departures from the published
# range: peaks that are effectively Gaussian (m far above 51), and a
# substantial fraction that FRONT rather than tail (A_s < 0), which a
# positive-only range excludes outright.
PEARSON7_KURTOSIS_LPYE = (2.2, 2.1e4)
PEARSON7_ASYMMETRY_LPYE = (-0.191, 0.174)
#
# The generator samples the UNION, so the benchmark represents *a*
# chromatogram rather than one instrument's. The kurtosis upper bound is
# taken from the published range rather than the measured one: beyond
# m ~ 50 the profile is already indistinguishable from a Gaussian to
# within the quantisation step, so the measured 2.1e4 carries no extra
# shape -- it is the fit reporting "Gaussian", not a wider family.
PEARSON7_KURTOSIS = (min(PEARSON7_KURTOSIS_NIEZEN[0],
                         PEARSON7_KURTOSIS_LPYE[0]),
                     PEARSON7_KURTOSIS_NIEZEN[1])
PEARSON7_ASYMMETRY = (min(PEARSON7_ASYMMETRY_NIEZEN[0],
                          PEARSON7_ASYMMETRY_LPYE[0]),
                      max(PEARSON7_ASYMMETRY_NIEZEN[1],
                          PEARSON7_ASYMMETRY_LPYE[1]))


# --------------------------------------------------------------------
# The `pyb` family: idealised signals transcribed from the pybaselines
# documentation. See tools/synthetic_data.md §10.
#
# TRANSCRIBED, NOT IMPORTED. `pybaselines.utils.make_data` states that
# its output "may change without notice ... outside users are advised
# not to rely on the exact output", and this project has already been
# bitten once by an undocumented pybaselines change (the beads lam_0/1/2
# defaults). Calling it would make every score depend on the install
# date. The formulas below are copied verbatim from the documentation at
# the pinned commit and are covered by tests.
#
# Sources, both at pybaselines commit c36ce6128:
#   [A] docs/examples/misc/plot_beads_preprocessing.py, make_data()
#       -- three baselines forming a deliberate ladder of violation of
#          the BEADS periodicity requirement (Navarro-Huerta 2017
#          §3.3.1): ends at zero on both / one / neither end.
#   [B] docs/algorithms/algorithms_1d/misc.rst, create_data()
#       -- five datasets varying peak density, noise level and baseline
#          shape, one of which carries NEGATIVE peaks.

def _g(x: np.ndarray, height: float, center: float,
       sigma: float) -> np.ndarray:
    """
    ``pybaselines.utils.gaussian`` in its own argument order.

    Defined here so the transcribed formulas below read exactly as they
    do in the documentation, and evaluated with pybaselines' own
    expression, ``h * exp(-0.5 * (x - c)**2 / s**2)``, rather than with
    `gauss_peak`'s ``h * exp(-0.5 * ((x - c) / s)**2)``. The two are
    algebraically identical but differ in the last bits, and the point
    of this family is that the signals reproduce the published figures
    exactly; the tests assert bit-equality with pybaselines and
    approximate equality with `gauss_peak`.

    """
    return height * np.exp(-0.5 * ((x - center) ** 2) / sigma ** 2)


def pyb_peaks(x: np.ndarray, group: str) -> np.ndarray:
    """
    Evaluate one of the pybaselines peak groups.

    Parameters
    ----------
    x : array-like, shape (N,)
        The abscissa.
    group : {'A', 'B', 'C', 'preproc'}
        ``'A'``, ``'B'`` and ``'C'`` are ``signal``, ``signal_2`` and
        ``signal_3`` of source [B]; ``'preproc'`` is the eight-peak
        signal of source [A].

    Returns
    -------
    signal : numpy.ndarray, shape (N,)

    """
    if group == 'A':
        return (_g(x, 6, 180, 5) + _g(x, 8, 350, 10)
                + _g(x, 6, 550, 5) + _g(x, 9, 800, 10))
    if group == 'B':
        return (_g(x, 9, 100, 12) + _g(x, 15, 400, 8)
                + _g(x, 13, 700, 12) + _g(x, 9, 880, 8))
    if group == 'C':
        return (_g(x, 8, 150, 10) + _g(x, 20, 120, 12)
                + _g(x, 16, 300, 20) + _g(x, 12, 550, 5)
                + _g(x, 20, 750, 12) + _g(x, 18, 800, 18)
                + _g(x, 15, 830, 12))
    if group == 'preproc':
        return (_g(x, 9, 100, 12) + _g(x, 6, 180, 5) + _g(x, 8, 350, 11)
                + _g(x, 15, 400, 18) + _g(x, 6, 550, 6)
                + _g(x, 13, 700, 8) + _g(x, 9, 800, 9)
                + _g(x, 9, 880, 7))
    raise ValueError(f"unknown peak group {group!r}")


def pyb_baseline(x: np.ndarray, kind: str) -> np.ndarray:
    """
    Evaluate one of the pybaselines baselines.

    The three ``ends_*`` kinds are source [A] and are named for what
    they do to the BEADS periodicity requirement, which is the reason
    they exist: `ends_both` reaches zero at both ends, `ends_one` at
    one, `ends_neither` at neither.

    Parameters
    ----------
    x : array-like, shape (N,)
        The abscissa.
    kind : str
        One of ``'ends_both'``, ``'ends_one'``, ``'ends_neither'``,
        ``'linear'``, ``'exponential'``, ``'gaussian'``,
        ``'decreasing_bump'``, ``'linear_offset'``.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)

    """
    if kind == 'ends_both':          # [A] type 0, parabola
        return 2e-5 * (x - 500) ** 2 - 5
    if kind == 'ends_one':           # [A] type 1
        return 10 - 10 * np.exp(-x / 600)
    if kind == 'ends_neither':       # [A] type 2, integrated gaussian
        return (-np.cumsum(_g(x, 0.05, 400, 100))
                + _g(x, 3, 800, 100) - 5)
    if kind == 'linear':             # [B] linear_baseline
        return 3 + 0.01 * x
    if kind == 'exponential':        # [B] exponential_baseline
        return 5 + 15 * np.exp(-x / 400)
    if kind == 'gaussian':           # [B] gaussian_baseline
        return 5 + _g(x, 20, 500, 500)
    if kind == 'decreasing_bump':    # [B] baseline_4
        return 10 - 0.005 * x + _g(x, 5, 850, 200)
    if kind == 'linear_offset':      # [B] baseline_5
        return 3 + 0.01 * x + 20
    raise ValueError(f"unknown baseline kind {kind!r}")


#: The eight `pyb` cases. Each is
#: ``(n_points, x_range, peak_expression, baseline_kind, noise_scale,
#:   published_seed, source)``. The five from [B] reproduce y1..y5 of
#: that figure exactly, including its documented BEADS parameters; the
#: three from [A] reproduce its endpoint ladder. `neg` marks the case
#: carrying negative peaks -- the published analogue of the LPYE
#: dead-time undershoot, and the one for which the pybaselines
#: documentation itself uses ``asymmetry=1`` while using 6-8 elsewhere.
PYB_CASES = {
    #                       n     x_range      peaks          baseline
    'B1_sparse_hi_noise': (500, (1., 1000.), ('2A',),      'linear',
                           5.0, 1, 'B'),
    'B2_dense':           (500, (1., 1000.), ('A', 'B', 'C'), 'gaussian',
                           1.0, 1, 'B'),
    'B3_medium':          (500, (1., 1000.), ('A', 'B'),   'exponential',
                           1.0, 1, 'B'),
    'B4_lo_noise':        (500, (1., 1000.), ('A', 'B'),   'decreasing_bump',
                           0.5, 1, 'B'),
    'B5_negative_peaks':  (500, (1., 1000.), ('2A', '-B'), 'linear_offset',
                           1.0, 1, 'B'),
    'A0_ends_both':       (1000, (0., 1000.), ('preproc',), 'ends_both',
                           1.0, 0, 'A'),
    'A1_ends_one':        (1000, (0., 1000.), ('preproc',), 'ends_one',
                           1.0, 0, 'A'),
    'A2_ends_neither':    (1000, (0., 1000.), ('preproc',), 'ends_neither',
                           1.0, 0, 'A'),
}

#: Base noise standard deviation of both sources, scaled per case by the
#: `noise_scale` entry of `PYB_CASES`.
PYB_NOISE_STD = 0.2


# --------------------------------------------------------------------
# Randomised `pyb` generation.
#
# The eight fixed cases above are a vocabulary, not a benchmark: eight
# signals cannot separate a real effect from a coincidence. The ranges
# below turn that vocabulary into a population, and every one of them is
# the span of values actually used across the pybaselines documentation
# at the pinned commit -- collected from docs/examples/*/*.py,
# docs/algorithms/algorithms_1d/*.rst and utils.make_data. Where a range
# is WIDER than the published span, it says so and why.
#
# Peak parameters, from the published peak lists (heights 4-20, centres
# 100-880 on a span of 1000, sigmas 5-20, counts 4-15). The height upper
# bound is 40 because two of the published datasets use `signal * 2`.
PYB_N_PEAKS = (4, 15)
PYB_PEAK_HEIGHT = (4.0, 40.0)
PYB_PEAK_CENTER_FRAC = (0.10, 0.88)     # of the abscissa span
PYB_PEAK_SIGMA_FRAC = (0.005, 0.020)    # of the abscissa span

# Noise: the documentation uses std 0.05 and 0.2, scaled per dataset by
# 0.5, 1 and 5, giving 0.025 to 1.0.
PYB_NOISE_STD_RANGE = (0.025, 1.0)

# Record length: the documentation uses 500 and 1000 points. WIDENED to
# 300-4000 deliberately -- the real LPYE records span 473-39129 points,
# and record length changes the fundamental (1/n_used), which is what
# the instability trim keys on. A benchmark fixed at two lengths could
# not detect a constant that depends on it.
PYB_N_POINTS = (300, 4000)

#: Baseline component vocabulary. Each entry is a builder taking
#: ``(x, rng)`` and returning ``(values, description)``. Coefficient
#: ranges are the spans observed in the documentation; the sources for
#: each are named in tools/synthetic_data.md §9.4.
def _bc_linear(x, rng):
    a = rng.uniform(1., 30.)             # offsets 1, 3, 5, 10, 15, 30
    b = rng.uniform(-0.005, 0.01)        # slopes -0.005 .. +0.01
    return a + b * x, f'linear(a={a:.3g}, b={b:.3g})'


def _bc_exponential(x, rng):
    a = rng.uniform(5., 10.)
    c = rng.uniform(-15., 15.)           # both signs appear (10-10exp)
    tau = rng.uniform(150., 1200.)       # published 150 .. 1200
    span = x[-1] - x[0]
    return (a + c * np.exp(-(x - x[0]) / (tau / 1000. * span)),
            f'exponential(a={a:.3g}, c={c:.3g}, tau={tau:.3g})')


def _bc_gaussian_bump(x, rng):
    a = rng.uniform(5., 30.)
    h = rng.uniform(-6., 20.)            # gaussian(x, -6, ...) appears
    span = x[-1] - x[0]
    c = x[0] + rng.uniform(0.3, 0.9) * span
    s = rng.uniform(0.1, 0.5) * span     # published 100 .. 500 on 1000
    return (a + _g(x, h, c, s),
            f'gaussian_bump(a={a:.3g}, h={h:.3g}, c={c:.3g}, s={s:.3g})')


def _bc_sine(x, rng):
    a = rng.uniform(10., 70.)
    amp = rng.uniform(1., 5.)            # published 1 and 5
    # Period: only x/50 appears; WIDENED to 30-150 so the benchmark
    # spans baselines the cutoff selector must treat differently.
    period = rng.uniform(30., 150.)
    return (a + amp * np.sin((x - x[0]) / period),
            f'sine(a={a:.3g}, amp={amp:.3g}, period={period:.3g})')


def _bc_parabola(x, rng):
    span = x[-1] - x[0]
    k = rng.uniform(1e-5, 3e-5) * (1000. / span) ** 2
    x0 = x[0] + rng.uniform(0.35, 0.65) * span
    a = rng.uniform(-10., 5.)
    return k * (x - x0) ** 2 + a, f'parabola(k={k:.3g}, x0={x0:.3g})'


def _bc_logistic(x, rng):
    # The 'ends_neither' shape: a logistic approximated by integrating a
    # gaussian, which is the hardest published case for BEADS because it
    # is near zero at neither end.
    span = x[-1] - x[0]
    h = rng.uniform(0.02, 0.08)
    c = x[0] + rng.uniform(0.25, 0.6) * span
    s = rng.uniform(0.05, 0.2) * span
    a = rng.uniform(-10., 0.)
    return (-np.cumsum(_g(x, h, c, s)) + a,
            f'logistic(h={h:.3g}, c={c:.3g}, s={s:.3g})')


PYB_BASELINE_COMPONENTS = {
    'linear': _bc_linear,
    'exponential': _bc_exponential,
    'gaussian_bump': _bc_gaussian_bump,
    'sine': _bc_sine,
    'parabola': _bc_parabola,
    'logistic': _bc_logistic,
}


def pyb_random_signal(seed: int, n_points: int | None = None,
                      negative_fraction: float = 0.2,
                      max_components: int = 2) -> dict:
    """
    Generate one randomised ``pyb`` signal with exact ground truth.

    Built from the same vocabulary as `PYB_CASES` -- Gaussian peaks on a
    composed analytic baseline with white noise -- but with every
    parameter drawn from the ranges observed across the pybaselines
    documentation, so the family becomes a population rather than eight
    points. See tools/synthetic_data.md §9.4.

    The baseline is the sum of one or two components, as the
    documentation itself does (``10 - 0.005x + gaussian(x, 5, 850,
    200)``). Nothing forces an endpoint condition: where the baseline
    sits relative to zero at each end is *recorded* in the metadata
    rather than imposed, so the periodicity axis is measurable without
    being a design variable.

    Parameters
    ----------
    seed : int
        Seed for all draws, including the noise. The signal is a pure
        function of it.
    n_points : int, optional
        Record length. Default None draws from `PYB_N_POINTS`.
    negative_fraction : float, optional
        Probability that the signal carries a negative peak group, as
        the published ``B5`` case does. Default 0.2.
    max_components : int, optional
        Maximum number of summed baseline components. Default 2.

    Returns
    -------
    components : dict
        Keys ``x``, ``y``, ``signal``, ``baseline``, ``noise`` and
        ``meta``. ``meta`` records every drawn parameter, so a signal can
        be regenerated and any dependence on a parameter measured.

    """
    rng = np.random.default_rng(seed)
    n = int(n_points if n_points is not None
            else rng.integers(PYB_N_POINTS[0], PYB_N_POINTS[1] + 1))
    x = np.linspace(0., 1000., n)
    span = x[-1] - x[0]

    n_peaks = int(rng.integers(PYB_N_PEAKS[0], PYB_N_PEAKS[1] + 1))
    centers = x[0] + rng.uniform(*PYB_PEAK_CENTER_FRAC, n_peaks) * span
    heights = np.exp(rng.uniform(*np.log(PYB_PEAK_HEIGHT), n_peaks))
    sigmas = rng.uniform(*PYB_PEAK_SIGMA_FRAC, n_peaks) * span
    negative = rng.random() < negative_fraction
    signs = np.ones(n_peaks)
    if negative:
        # As in the published B5 case, a subset of the peaks is
        # subtracted rather than added.
        signs[rng.random(n_peaks) < 0.35] = -1.
        if np.all(signs > 0):
            signs[int(rng.integers(n_peaks))] = -1.
    signal = np.zeros(n)
    peaks = []
    for h, c, s, sg in zip(heights, centers, sigmas, signs):
        signal = signal + sg * _g(x, h, c, s)
        peaks.append({'height': float(sg * h), 'center': float(c),
                      'sigma': float(s)})

    kinds = list(PYB_BASELINE_COMPONENTS)
    n_comp = int(rng.integers(1, max_components + 1))
    chosen = list(rng.choice(kinds, size=n_comp, replace=False))
    baseline = np.zeros(n)
    descs = []
    for k in chosen:
        vals, desc = PYB_BASELINE_COMPONENTS[k](x, rng)
        baseline = baseline + vals
        descs.append(desc)

    noise_std = float(np.exp(rng.uniform(*np.log(PYB_NOISE_STD_RANGE))))
    noise = rng.normal(0., noise_std, n)

    rng_b = baseline.max() - baseline.min()
    ends = (abs(baseline[0] - baseline.min()) / max(rng_b, 1e-12),
            abs(baseline[-1] - baseline.min()) / max(rng_b, 1e-12))
    return {'x': x, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise,
            'meta': {'family': 'pyb_random', 'seed': seed, 'n_points': n,
                     'n_peaks': n_peaks, 'has_negative_peaks': bool(negative),
                     'baseline_kinds': [str(k) for k in chosen],
                     'baseline_desc': '; '.join(descs),
                     'noise_std': noise_std,
                     'baseline_range': float(rng_b),
                     'end_offsets': [float(e) for e in ends],
                     'peaks': peaks}}


def pyb_signal(case: str, seed: int | None = None) -> dict:
    """
    Assemble one ``pyb`` synthetic signal with its exact ground truth.

    Parameters
    ----------
    case : str
        A key of `PYB_CASES`.
    seed : int, optional
        Seed of the noise generator. Default None uses the seed
        published with the source, so the signal reproduces the
        documentation figure exactly; pass an integer for a replicate.

    Returns
    -------
    components : dict
        Keys ``x``, ``y``, ``signal``, ``baseline``, ``noise``, and
        ``meta`` describing the case and its provenance.

    """
    if case not in PYB_CASES:
        raise ValueError(f"unknown pyb case {case!r}")
    n, (x0, x1), groups, bkind, nscale, pub_seed, src = PYB_CASES[case]
    x = np.linspace(x0, x1, n)
    signal = np.zeros(n)
    for g in groups:
        if g.startswith('2'):
            signal = signal + 2. * pyb_peaks(x, g[1:])
        elif g.startswith('-'):
            signal = signal - pyb_peaks(x, g[1:])
        else:
            signal = signal + pyb_peaks(x, g)
    baseline = pyb_baseline(x, bkind)
    rng_seed = pub_seed if seed is None else seed
    noise = (np.random.default_rng(rng_seed).normal(0., PYB_NOISE_STD, n)
             * nscale)
    return {'x': x, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise,
            'meta': {'family': 'pyb', 'case': case, 'source': src,
                     'baseline_kind': bkind, 'noise_scale': nscale,
                     'seed': rng_seed, 'n_points': n,
                     'has_negative_peaks': any(g.startswith('-')
                                               for g in groups)}}


###############################################################################
# The `erb` family: the pybaselines author's own BEADS test signals.
#
# Source: `donnie/test_donnie.py`, a script written by Donald Erb, the
# pybaselines author, to exercise cutoff-frequency selection on this
# project's problem. It is correspondence, not a publication, so under
# the project's evidence rule it earns a test rather than a citation --
# but as a *generator* it defines the truth rather than processing it,
# and its provenance is recorded here.
#
# It is transcribed, not imported, for the same reason as the `pyb`
# family (§9.1) and one more: the script carries `np.log` for the
# transform, which was a MISTAKE, since corrected upstream. Production
# uses log10, matching Navarro-Huerta Eqs. (8)/(11). Do not carry that
# line across. Only `make_data` is transcribed here.
###############################################################################

# The seven Gaussian peaks of Erb's `make_data`, as (height, centre,
# sigma) in the units of `x`. His script carries an eighth, commented
# out: `gaussian(x, 15, 400, 18)`. It is kept, also disabled, because
# its parameters are the only evidence for the upper end of the height
# and width ranges the population draws from.
ERB_PEAKS = ((9., 100., 12.), (6., 180., 5.), (8., 350., 11.),
             (6., 550., 6.), (13., 700., 8.), (9., 800., 9.),
             (9., 880., 7.))
ERB_PEAK_DISABLED = (15., 400., 18.)

ERB_NOISE_STD = 0.05
ERB_SEED = 0

# The three x-ranges his script offers, with the plateau count each is
# annotated with in his own comments. MEASURED 2026-08-16 through the
# production path: one_plateau and three_plateaus reproduce (1 and 3
# detected flat regions); two_plateaus does NOT (1 detected). See
# tools/synthetic_data.md §10.2 for why.
#
#   name: (x_start, x_end, n_points, claimed_plateaus)
ERB_CASES = {
    'one_plateau': (0., 1000., 1000, 1),
    'two_plateaus': (0., 4000., 1000, 2),
    'three_plateaus': (-4000., 1000., 1000, 3),
}

# What the x-range knob actually varies is WHERE THE PEAKS SIT inside
# the record, not the abscissa: the peak centres are fixed numbers
# (100-880) while the span moves, so the peaks occupy 10-88% of the
# record in the first case, 2.5-22% in the second and 82-98% in the
# third. The population therefore varies that window directly, and the
# ranges below are exactly the values his three cases take -- nothing
# is widened.
ERB_PEAK_WINDOW_START = (0.025, 0.82)
ERB_PEAK_WINDOW_WIDTH = (0.16, 0.78)
# Height and sigma spans of his eight peaks, the disabled one included.
# Sigma is expressed as a fraction of the abscissa span, on which his
# 5-18 over a span of 1000 becomes 0.005-0.018.
ERB_PEAK_HEIGHT = (6., 15.)
ERB_PEAK_SIGMA_FRAC = (0.005, 0.018)
ERB_N_PEAKS = (7, 8)



def erb_baseline(x: np.ndarray, kind: int,
                 exact_integral: bool = False) -> np.ndarray:
    """
    Evaluate one of Erb's four test baselines.

    Transcribed from `make_data` in ``donnie/test_donnie.py``:

    =====  ==================================================
    kind   baseline
    =====  ==================================================
    0      parabola ending at zero on both ends
    1      exponentially decaying
    2      "very complicated": a logistic approximated by
           integrating a Gaussian, plus a Gaussian bump
    3      sinusoidal
    =====  ==================================================

    Kind 2 is the one his three plateau-count cases use, and the one
    behind the ``runs/DONNIE_2026-07-27`` signals.

    Parameters
    ----------
    x : array-like, shape (N,)
        Abscissa.
    kind : int
        One of 0, 1, 2, 3.
    exact_integral : bool, optional
        Only affects `kind` 2. False (default) reproduces his
        expression exactly, a bare ``cumsum`` of the Gaussian. True
        multiplies by the sample spacing, making it the actual integral.

        The distinction is not cosmetic. A bare ``cumsum`` is a Riemann
        sum missing its ``dx``, so the baseline's amplitude scales with
        the sampling density: at fixed span, doubling the point count
        doubles the drift. That is harmless for his three fixed cases,
        which all use 1000 points, and wrong for a population that
        varies the record length. Set True there.

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


def erb_signal(case: str, baseline_type: int = 2,
               seed: int | None = None) -> dict:
    """
    Assemble one of Erb's three fixed plateau-count cases.

    Reproduces ``make_data`` exactly: the seven Gaussian peaks at their
    published centres, one of the four baselines, and white noise at
    ``ERB_NOISE_STD`` from seed 0. The peaks do not move with the
    abscissa, which is what makes the three cases differ.

    .. warning::
       The plateau counts in `ERB_CASES` are **his annotations, not
       measurements**. Checked through the production path on
       2026-08-16, ``one_plateau`` and ``three_plateaus`` reproduce
       while ``two_plateaus`` yields one detected region, not two --
       there the peaks fall in the first quarter of the record, so
       ``baseline._relevant_regions`` truncates the sweep to 235 of
       1000 points and the fundamental moves with it. Treat the counts
       as labels to be verified, never as ground truth.

    Parameters
    ----------
    case : str
        A key of `ERB_CASES`.
    baseline_type : int, optional
        Which of the four baselines, see `erb_baseline`. Default 2, the
        one his plateau-count comments were written against.
    seed : int, optional
        Noise seed. Default None uses his published seed, so the signal
        reproduces his script exactly; pass an integer for a replicate
        with the same signal and baseline but fresh noise.

    Returns
    -------
    components : dict
        Keys ``x``, ``y``, ``signal``, ``baseline``, ``noise`` and
        ``meta``.

    """
    if case not in ERB_CASES:
        raise ValueError(f"unknown erb case {case!r}")
    x0, x1, n, claimed = ERB_CASES[case]
    x = np.linspace(x0, x1, n)
    signal = np.zeros(n)
    for height, centre, sigma in ERB_PEAKS:
        signal = signal + _g(x, height, centre, sigma)
    baseline = erb_baseline(x, baseline_type)
    rng_seed = ERB_SEED if seed is None else seed
    noise = np.random.default_rng(rng_seed).normal(0., ERB_NOISE_STD, n)
    span = x[-1] - x[0]
    lo = (ERB_PEAKS[0][1] - x[0]) / span
    hi = (ERB_PEAKS[-1][1] - x[0]) / span
    return {'x': x, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise,
            'meta': {'family': 'erb', 'case': case,
                     'source': 'donnie/test_donnie.py, make_data',
                     'baseline_type': baseline_type, 'seed': rng_seed,
                     'n_points': n, 'x_range': [float(x0), float(x1)],
                     'claimed_plateaus': claimed,
                     'peak_window': [float(lo), float(hi)],
                     'noise_std': ERB_NOISE_STD,
                     'peaks': [{'height': h, 'center': c, 'sigma': s}
                               for h, c, s in ERB_PEAKS]}}


def erb_random_signal(seed: int, n_points: int | None = None,
                      baseline_type: int | None = None,
) -> dict:
    """
    Generate one randomised ``erb`` signal with exact ground truth.

    Turns Erb's three fixed cases into a population. The variable his
    x-range knob actually moves is the **peak window** -- what fraction
    of the record the peaks occupy -- so that is what is drawn here,
    from the span his own three cases cover. The abscissa is held at a
    fixed 0-1000 instead, which removes the confound: in his cases the
    window and the record length move together, so a constant that
    depended on either could not be told apart.

    Every range is taken from values his script contains; none is
    widened, except the record length, which comes from `PYB_N_POINTS`
    and is justified there -- length sets the fundamental, ``1/n_used``,
    which is what `instability_boundary` keys on.

    Parameters
    ----------
    seed : int
        Seed for all draws, including the noise. The signal is a pure
        function of it.
    n_points : int, optional
        Record length. Default None draws from `PYB_N_POINTS`.
    baseline_type : int, optional
        Which of the four baselines. Default None draws uniformly.
    Returns
    -------
    components : dict
        Keys ``x``, ``y``, ``signal``, ``baseline``, ``noise`` and
        ``meta``. ``meta`` records every drawn parameter.

    """
    rng = np.random.default_rng(seed)
    n = int(n_points if n_points is not None
            else rng.integers(PYB_N_POINTS[0], PYB_N_POINTS[1] + 1))
    x = np.linspace(0., 1000., n)
    span = x[-1] - x[0]

    start = float(rng.uniform(*ERB_PEAK_WINDOW_START))
    width = float(rng.uniform(*ERB_PEAK_WINDOW_WIDTH))
    # Keep the window inside the record; his widest case ends at 0.98.
    width = min(width, 0.98 - start)
    n_peaks = int(rng.integers(ERB_N_PEAKS[0], ERB_N_PEAKS[1] + 1))

    centres = x[0] + (start + np.sort(rng.uniform(0., 1., n_peaks))
                      * width) * span
    heights = rng.uniform(*ERB_PEAK_HEIGHT, n_peaks)
    sigmas = rng.uniform(*ERB_PEAK_SIGMA_FRAC, n_peaks) * span

    signal = np.zeros(n)
    peaks = []
    for height, centre, sigma in zip(heights, centres, sigmas):
        signal = signal + _g(x, height, centre, sigma)
        peaks.append({'height': float(height), 'center': float(centre),
                      'sigma': float(sigma)})

    kind = (int(rng.integers(0, 4)) if baseline_type is None
            else int(baseline_type))
    # exact_integral so kind 2's amplitude does not scale with the
    # record length; see `erb_baseline`.
    baseline = erb_baseline(x, kind, exact_integral=True)
    noise = rng.normal(0., ERB_NOISE_STD, n)
    return {'x': x, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise,
            'meta': {'family': 'erb_random', 'seed': seed, 'n_points': n,
                     'baseline_type': kind, 'n_peaks': n_peaks,
                     'peak_window': [start, start + width],
                     'noise_std': ERB_NOISE_STD,
                     'baseline_range': float(baseline.max()
                                             - baseline.min()),
                     'peaks': peaks}}


def pearson7_peak(t: np.ndarray, tc: float, sigma: float, kurtosis: float,
                  asymmetry: float, height: float) -> np.ndarray:
    r"""
    Evaluate a modified Pearson VII peak.

    .. math::

       f(t) = A\left(1 + \frac{(t-\mu)^2}
                          {m\,(\sigma + A_s (t-\mu))^2}\right)^{-m}

    Two independent groups selected this shape over the usual
    alternatives for chromatographic peaks. Niezen et al. compared 15
    distributions by the Akaike information criterion on real peaks and
    found it best overall (Table 1: sum-AIC -7.20e3, against -6.91e3 for
    the exponentially-modified Gaussian). Milani et al. reached the same
    function from a different direction, calling it the Skewed
    Lorentz-Normal, and measured RMSE <= 0.0045 against <= 0.0048 for a
    Gaussian over 458 fitted peaks.

    The shape interpolates between a Gaussian (``kurtosis`` -> infinity)
    and a Lorentzian (``kurtosis`` -> 1), with ``asymmetry`` producing
    tailing (positive) or fronting (negative).

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    tc : float
        Peak centre, the retention time (``mu`` above).
    sigma : float
        Width parameter, in the units of `t`.
    kurtosis : float
        Shape parameter ``m``. See `PEARSON7_KURTOSIS` for the range
        fitted to real peaks.
    asymmetry : float
        Skew parameter ``A_s``. See `PEARSON7_ASYMMETRY`.
    height : float
        Peak maximum; ``f(tc) == height`` exactly.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The peak profile.

    Notes
    -----
    For ``asymmetry != 0`` the denominator vanishes at
    ``t - tc = -sigma / asymmetry``, and beyond that point the
    expression rises again into a spurious second lobe. It is tiny --
    measured at 9.3e-7 of `height` at the widest fitted asymmetry, some
    five orders below the detector's quantisation step -- but it is
    clipped to zero rather than left to a numerical accident, so the
    function is exactly single-lobed by construction.

    References
    ----------
    Niezen, Schoenmakers & Pirok (2022), Anal. Chim. Acta 1201, 339605,
    Eq. (14) and Table 2.
    Milani et al. (2024), Anal. Chim. Acta 1312, 342724, Eq. (2).

    """
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")
    if kurtosis <= 0:
        raise ValueError("kurtosis must be greater than 0.")
    dt = t - tc
    denom = sigma + asymmetry * dt
    with np.errstate(divide='ignore', invalid='ignore'):
        y = height * (1. + dt ** 2 / (kurtosis * denom ** 2)) ** -kurtosis
    y = np.nan_to_num(y, nan=0., posinf=0., neginf=0.)
    if asymmetry != 0.:
        # Keep only the lobe containing the centre: the sign of the
        # denominator at t == tc.
        y = np.where(np.sign(denom) == np.sign(sigma), y, 0.)
    return y


def dead_time_artifact(t: np.ndarray, t0: float, dt: float,
                       rng: np.random.Generator,
                       pos_height: tuple[float, float] = (1., 4.),
                       neg_abs: tuple[float, float] | None = None
                       ) -> tuple[np.ndarray, list[dict]]:
    """
    Bipolar injection artifact at the dead time.

    A sharp positive spike at ``t0`` immediately followed by a wider
    negative undershoot, the dominant feature of the real blanks. The
    negative lobe is genuine signal, which is why a symmetric BEADS
    cost (``asymmetry=1``) is required: an asymmetric cost would absorb
    it into the baseline.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    t0 : float
        Dead time (centre of the positive spike).
    dt : float
        Sample spacing, minutes.
    rng : numpy.random.Generator
        Source of randomness.
    pos_height : (float, float), optional
        Range of the positive-spike height. Default ``(1., 4.)``
        matches the LPYE data.
    neg_abs : (float, float), optional
        Absolute range of the negative-lobe depth. If given, the dip is
        drawn from it directly rather than as a multiple of the positive
        spike; used to cap the native dip at the measured real maximum
        (~1.2 mV). Default None keeps the 0.5-2x relative scaling.

    Returns
    -------
    contribution : numpy.ndarray, shape (N,)
        The artifact, to add to the signal.
    entries : list of dict
        Per-lobe parameter records for the truth file.

    """
    pos_fwhm = rng.uniform(6., 10.)
    pos_h = rng.uniform(*pos_height)
    pos_sigma = pos_fwhm * dt / _FWHM_PER_SIGMA
    pos_tau = rng.uniform(0.3, 0.8) * pos_sigma
    contrib = emg_peak(t, t0, pos_sigma, pos_tau, pos_h)
    entries = [{'tc': t0, 'sigma': pos_sigma, 'tau': pos_tau,
                'height': pos_h, 'fwhm_points': pos_fwhm, 'artifact': True}]

    neg_fwhm = pos_fwhm * rng.uniform(1.2, 2.5)
    neg_h = rng.uniform(*neg_abs) if neg_abs else pos_h * rng.uniform(0.5,
                                                                      2.0)
    neg_sigma = neg_fwhm * dt / _FWHM_PER_SIGMA
    neg_tc = t0 + rng.uniform(1.0, 2.0) * pos_sigma
    contrib = contrib - emg_peak(t, neg_tc, neg_sigma, neg_sigma * 0.5,
                                 neg_h)
    entries.append({'tc': neg_tc, 'sigma': neg_sigma,
                    'tau': neg_sigma * 0.5, 'height': -neg_h,
                    'fwhm_points': neg_fwhm, 'artifact': True})
    return contrib, entries


def make_baseline(t: np.ndarray, kind: str,
                  rng: np.random.Generator, t0: float | None = None,
                  is_blank: bool = False) -> np.ndarray:
    """
    Build the slowly varying baseline of a ``native`` signal.

    Calibrated to the LPYE instrument. The real blanks (which are
    baseline + noise + artifact) have a smoothed peak-to-peak baseline
    structure of only **0.33 mV median, 0.75 mV maximum** over the 67
    blanks: this detector's baseline is nearly flat, and the large
    excursions in the real chromatograms are peaks, not baseline. The
    component amplitudes below are scaled to that measurement, an order
    of magnitude smaller than a generic literature baseline (for which
    see ``lit_baseline``).

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    kind : str
        One of ``exp`` (solvent-front decay), ``hump`` (broad gradient
        hump) or ``exp_hump_drift`` (both plus a linear drift).
    rng : numpy.random.Generator
        Source of randomness for the component parameters.
    t0 : float, optional
        Dead time. If given, a small step in the baseline level is
        placed across it, as the real blanks show (the baseline settles
        at a different level after the injection).
    is_blank : bool, optional
        If True, the large gradient hump is omitted and the exp/drift
        amplitudes are reduced: the real blanks are nearly flat (smoothed
        baseline p2p 0.33 mV median, 0.75 max), whereas gradient runs
        with analytes may carry a larger solvent-programme baseline.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline.

    """
    span = t[-1] - t[0]
    b = np.full(len(t), rng.uniform(-0.5, 1.5))
    # Level step across the dead time: the real baselines settle at a
    # slightly different level after the injection (a smoothed
    # Heaviside, not a discontinuity, since the detector has a finite
    # response time).
    if t0 is not None:
        step_h = rng.uniform(-0.4, 0.4)
        b += step_h * 0.5 * (1. + np.tanh((t - t0) / (0.05 * span)))
    if kind in ('exp', 'exp_hump_drift'):
        tau = rng.uniform(0.10, 0.30) * span
        exp_amp = rng.uniform(0.1, 0.4) if is_blank else rng.uniform(0.3, 1.5)
        b += exp_amp * np.exp(-(t - t[0]) / tau)
    # The gradient hump is a solvent-programme feature: blanks do not
    # carry it (their baseline is flat plus drift plus the wander below).
    if kind in ('hump', 'exp_hump_drift') and not is_blank:
        center = t[0] + rng.uniform(0.35, 0.70) * span
        width = rng.uniform(0.15, 0.30) * span
        b += rng.uniform(0.3, 1.2) * np.exp(-0.5 * ((t - center) / width)**2)
    if kind == 'exp_hump_drift':
        drift = rng.uniform(-0.4, 0.4) if is_blank else rng.uniform(-1., 1.)
        b += drift * (t - t[0]) / span

    # Mid-frequency wander (pump, thermal and detector fluctuations),
    # which every real baseline carries and the slow components above
    # cannot represent. Its correlation length sits between the peak
    # widths and the run length, so it is what decides how flexible the
    # baseline has to be: on a signal with little analyte, capturing it
    # is the whole job.
    dt = t[1] - t[0]
    corr_len = rng.uniform(0.3, 0.8) / dt
    wander = gaussian_filter1d(rng.normal(size=len(t)), corr_len)
    peak_amp = np.abs(wander).max()
    if peak_amp > 0:
        # Absolute amplitude 0.1-0.35 mV, matching the measured blank
        # baseline structure (median p2p 0.33 mV).
        wander *= rng.uniform(0.1, 0.35) / peak_amp
        b = b + wander
    return b


def lit_baseline(t: np.ndarray, kind: str,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Build a generic literature baseline, not tied to the LPYE data.

    The two forms are the BEADS benchmark baselines of Ning, Selesnick
    & Duval (2014), §5:

    - ``poly_sine`` (their Type 1): a polynomial of random order plus a
      sinusoid of random frequency and phase;
    - ``lowpass`` (their Type 2): a white Gaussian process low-pass
      filtered to a small band, i.e. a random smooth curve.

    Amplitudes are on the order of the peak heights rather than the
    tenth-of-a-millivolt of the LPYE baseline, so these exercise the
    large, structured baselines the instrument does not produce.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    kind : str
        ``poly_sine`` or ``lowpass``.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline.

    """
    n = len(t)
    u = (t - t[0]) / (t[-1] - t[0])         # normalised abscissa in [0, 1]
    if kind == 'poly_sine':
        order = rng.integers(2, 6)
        coeffs = rng.uniform(-1., 1., order + 1)
        poly = np.polyval(coeffs, 2. * u - 1.)
        cycles = rng.uniform(0.5, 4.)
        sine = rng.uniform(0.3, 1.0) * np.sin(2. * np.pi * cycles * u
                                              + rng.uniform(0, 2 * np.pi))
        b = poly + sine
    else:                                    # lowpass
        corr = rng.uniform(0.08, 0.20) * n
        b = gaussian_filter1d(rng.normal(size=n), corr)
    # Normalise to a random peak-to-peak amplitude comparable to the
    # peaks (set by the caller's height scale, ~1-30).
    span = b.max() - b.min()
    if span > 0:
        b = (b - b.min()) / span * rng.uniform(4., 20.)
    return b


def native_peak_component(n: int, peak_case: str,
                          rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray, list[dict],
                                     float]:
    """
    Build the LPYE peak component: analytes, artefact and time axis.

    Everything a `native` chromatogram contains except its baseline and
    its noise. Factored out of `make_signal` verbatim -- the draws
    happen in the same order, so `make_signal` is unchanged -- so that
    the peak model Emmanuel reviewed on 2026-07-22 can be placed on a
    different baseline without being written a second time.

    The component is calibrated to the LPYE instrument: 120 points per
    minute, an absolute dead time near 4.5 min carrying the bipolar
    injection artefact, exponentially-modified Gaussian analytes
    eluting after it, and widths growing with retention time.

    Parameters
    ----------
    n : int
        Number of data points.
    peak_case : str
        Key of `PEAK_CASES`.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Time axis, minutes.
    signal : numpy.ndarray, shape (N,)
        Analytes plus the injection artefact.
    peaks : list of dict
        Per-peak parameters; entries with ``artifact=True`` are the
        injection lobes and the blank ghosts, not analytes.
    t0 : float
        The dead time used, minutes.

    """
    dt = 1. / PTS_PER_MIN               # minutes per sample
    t = np.arange(n) * dt
    # Dead time in ABSOLUTE minutes. Measured on the real blanks it is a
    # fixed 4.58 min regardless of run length (55-73% of the span), not
    # a fraction of it, so it is placed absolutely and only pulled in
    # for a record too short to contain it (none of the shipped native
    # lengths are).
    t0 = rng.uniform(4.3, 4.7)
    if t[-1] < t0 + 0.5:
        t0 = 0.85 * t[-1]

    n_peaks, fwhm_pts = PEAK_CASES[peak_case]
    signal = np.zeros(n)
    peaks = []

    # The dead-time injection artifact (bipolar; see dead_time_artifact).
    # On many real blanks the negative lobe is the largest excursion in
    # the whole trace, down to -1.2 mV on a 0.6 mV blank.
    # The negative dip is capped at the measured real maximum (~1.2 mV);
    # without the cap the relative scaling produced dips to -2.7 mV that
    # the real data never shows.
    contrib, entries = dead_time_artifact(t, t0, dt, rng,
                                          pos_height=(1., 4.),
                                          neg_abs=(0.3, 1.4))
    signal += contrib
    peaks += entries
    # For blanks, a few small ghost/carryover peaks along the run, so a
    # blank still carries the faint detectable features real blanks show.
    if n_peaks == 0:
        for _ in range(rng.integers(1, 4)):
            tc = rng.uniform(0.2, 0.85) * t[-1]
            fwhm_a = rng.uniform(8., 16.)
            height = rng.uniform(0.15, 0.8)
            sigma = fwhm_a * dt / _FWHM_PER_SIGMA
            tau = rng.uniform(0.3, 0.8) * sigma
            signal += emg_peak(t, tc, sigma, tau, height)
            peaks.append({'tc': tc, 'sigma': sigma, 'tau': tau,
                          'height': height, 'fwhm_points': fwhm_a,
                          'artifact': True})

    # Analytes elute after the dead time; the window is bounded below by
    # t0 and above by the run end. A record must be long enough to hold
    # it (all shipped native lengths are), but guard the degenerate case
    # so a short record never crashes the uniform draw.
    elute_lo = t0 + 0.02 * t[-1]
    elute_hi = 0.9 * t[-1]
    if elute_hi <= elute_lo:
        elute_lo, elute_hi = 0.9 * t[-1], 0.98 * t[-1]
    centers = np.sort(rng.uniform(elute_lo, elute_hi, n_peaks))
    if fwhm_pts is None and n_peaks >= 2:
        # isocratic case: guarantee the hard sharp-first/broad-last
        # contrast by pinning an early and a late eluter
        centers[0] = t0 * rng.uniform(1.05, 1.20)
        centers[-1] = rng.uniform(0.85, 0.92) * t[-1]
        centers = np.sort(centers)
    # Constant plate number of the isocratic case: sigma = tc/sqrt(Np),
    # so the first peaks are sharp and the late ones broad, with a
    # width ratio set by the retention ratio (the hard regime the real
    # dataset shows).
    n_plates = rng.uniform(3000., 12000.)
    broaden_g = rng.uniform(2., 4.)
    # One base width per signal, so the retention growth is the dominant
    # trend rather than being swamped by per-peak scatter; a mild
    # per-peak jitter keeps the compounds from being identical.
    base_fwhm = None if fwhm_pts is None else rng.uniform(*fwhm_pts) * dt
    for tc in centers:
        if fwhm_pts is None:
            # isocratic: the pure plate model, sigma proportional to the
            # retention time from injection.
            sigma = max(tc - t0, dt) / np.sqrt(n_plates)
        else:
            fwhm = (base_fwhm * broadening_factor(tc, t0, t[-1], broaden_g)
                    * rng.uniform(0.85, 1.15))
            sigma = fwhm / _FWHM_PER_SIGMA
        fwhm = sigma * _FWHM_PER_SIGMA
        tau = rng.uniform(0.3, 1.2) * sigma
        height = np.exp(rng.uniform(np.log(1.), np.log(30.)))
        signal += emg_peak(t, tc, sigma, tau, height)
        peaks.append({'tc': tc, 'sigma': sigma, 'tau': tau,
                      'height': height, 'fwhm_points': fwhm / dt,
                      'artifact': False})

    return t, signal, peaks, t0


def make_signal(n: int, peak_case: str, baseline_case: str,
                noise_sigma: float, rng: np.random.Generator
                ) -> tuple[np.ndarray, dict]:
    """
    Assemble one ``native`` synthetic chromatogram (LPYE instrument).

    Parameters
    ----------
    n : int
        Number of data points.
    peak_case : str
        Key of ``PEAK_CASES``.
    baseline_case : str
        Key of ``BASELINE_CASES``.
    noise_sigma : float
        Standard deviation of the white noise.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    components : dict
        Keys ``x``, ``y``, ``signal``, ``baseline``, ``noise`` and
        ``peaks`` (the per-peak parameters).

    """
    t, signal, peaks, t0 = native_peak_component(n, peak_case, rng)
    baseline = make_baseline(t, baseline_case, rng, t0=t0,
                             is_blank=(PEAK_CASES[peak_case][0] == 0))
    noise = rng.normal(0., noise_sigma, n)
    return {'x': t, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise, 'peaks': peaks}


# Record length for the `erb_native` family. MEASURED: the quantile
# function of the 339 real LPYE records, every 5% from the minimum to
# the maximum. Lengths are drawn from it by inverse-CDF sampling, so the
# generated distribution reproduces the real one by construction rather
# than approximating it with a ladder of fixed values.
#
# Why a distribution and not levels: record length sets the fundamental
# frequency 1/n_used, which is exactly what `instability_boundary` keys
# on, so length must vary -- but a balanced ladder misrepresents how
# often each length actually occurs. The real set is strongly skewed:
# the median record is 1176 points while the longest is 39129, and only
# 4.7% exceed 10000. A six-level geometric ladder puts 33% of the
# benchmark above 10000, over-weighting the most expensive signals by
# sevenfold and implying the instrument produces long runs routinely.
ERB_NATIVE_N_QUANTILES = (473, 732, 780, 819, 846, 853, 868, 893, 924,
                          994, 1176, 1259, 1400, 1546, 1707, 2062, 2857,
                          3495, 5399, 9251, 39129)


def draw_record_length(rng: np.random.Generator) -> int:
    """
    Draw a record length from the real LPYE length distribution.

    Inverse-CDF sampling on `ERB_NATIVE_N_QUANTILES`: a uniform draw is
    mapped through the measured quantile function, so the generated
    lengths follow the real distribution, span its full range, and take
    arbitrary values rather than a handful of fixed ones.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    n : int
        A record length in points, between the real minimum and maximum.

    """
    grid = np.linspace(0., 1., len(ERB_NATIVE_N_QUANTILES))
    u = float(rng.uniform(0., 1.))
    return int(round(float(np.interp(u, grid,
                                     ERB_NATIVE_N_QUANTILES))))


def erb_native_signal(n: int, peak_case: str, baseline_type: int,
                      noise_sigma: float, rng: np.random.Generator,
                      quantise_output: bool = True) -> dict:
    """
    Assemble a chromatogram: LPYE peak component on an Erb baseline.

    The hybrid of the two families that each hold one half of what the
    benchmark needs.

    From `native`, by way of `native_peak_component`: the peak model
    calibrated to the LPYE instrument and reviewed on 2026-07-22 --
    120 points per minute, the bipolar injection artefact at an
    absolute dead time near 4.5 min, exponentially-modified Gaussian
    analytes eluting after it, widths growing with retention time, and
    detector noise at the measured level.

    From `erb`, by way of `erb_baseline`: a baseline that is an exact
    analytic function rather than a sum of drawn components, written by
    the pybaselines author to exercise cutoff selection. That is what
    `native`'s own baseline cannot offer -- and Erb's three fixed
    x-ranges additionally dial in how many plateaus the r2 curve shows,
    which is the property issue #4 needs and which nothing else in the
    benchmark provides.

    **Scales.** The baseline is evaluated on a normalised 0-1000
    abscissa spanning the record, because `erb_baseline`'s coefficients
    are written for that range; its output is then read as mV. That is
    legitimate rather than a fudge: Erb's baselines span roughly 2-12 in
    his units, and the drift measured on the real records spans
    0.08-12.5 mV, so the two scales coincide to within the spread of the
    real data. It is recorded here rather than assumed silently.

    Parameters
    ----------
    n : int
        Number of data points.
    peak_case : str
        Key of `PEAK_CASES`.
    baseline_type : int
        Which of Erb's four baselines, see `erb_baseline`.
    noise_sigma : float
        Standard deviation of the white noise, mV. See `NOISE_CASES`.
    rng : numpy.random.Generator
        Source of randomness.
    quantise_output : bool, optional
        Round the assembled signal onto the detector lattice. Default
        True. This is not cosmetic: `baseline._snr` divides by a MAD of
        consecutive differences, which on quantised data is pinned to
        the lattice and on continuous data is a true noise estimate, so
        an unquantised benchmark exercises a different quantity under
        the same name. See tools/synthetic_data.md §7.

    Returns
    -------
    components : dict
        Keys ``x``, ``y``, ``signal``, ``baseline``, ``noise``,
        ``peaks`` and ``meta``.

        ``baseline`` alone is the truth a baseline estimator is scored
        against. The injection artefact is inside ``signal``, not inside
        ``baseline``: it is a sharp bipolar excursion that a low-pass
        baseline model is not meant to reproduce, and scoring on it
        would let one feature dominate the error. Entries of ``peaks``
        with ``artifact=True`` mark it.

    """
    t, signal, peaks, t0 = native_peak_component(n, peak_case, rng)
    # Erb's coefficients are written for an abscissa of order 0-1000, so
    # the baseline is evaluated there and mapped onto the record. Its
    # shape is therefore independent of the record's duration in
    # minutes, which is what makes the four types comparable across
    # lengths. exact_integral keeps type 2's amplitude from scaling with
    # the point count.
    u = np.linspace(0., 1000., n)
    baseline = erb_baseline(u, baseline_type, exact_integral=True)
    noise = rng.normal(0., noise_sigma, n)
    y = signal + baseline + noise
    if quantise_output:
        y = quantise(y)
    return {'x': t, 'y': y, 'signal': signal, 'baseline': baseline,
            'noise': noise, 'peaks': peaks,
            'meta': {'family': 'erb_native', 'n_points': n,
                     'peak_case': peak_case,
                     'baseline_type': baseline_type,
                     'noise_sigma': noise_sigma, 'dead_time': t0,
                     'quantised': quantise_output,
                     'points_per_minute': PTS_PER_MIN,
                     'minutes': float(t[-1] - t[0]),
                     'baseline_range': float(baseline.max()
                                             - baseline.min()),
                     'n_analytes': sum(not p.get('artifact', False)
                                       for p in peaks)}}


LIT_PEAK_CASES = {
    # (number of Gaussian peaks, FWHM range in points)
    'sparse': (6, (10, 40)),
    'medium': (14, (8, 30)),
    'dense': (28, (8, 20)),        # crowded: stresses the sparsity and
                                   # median-estimator assumptions
    'few_broad': (3, (60, 140)),
}
LIT_BASELINE_CASES = ['poly_sine', 'lowpass']
# Ning et al. used input SNR from -5 to 25 dB; these four span that
# range from clean to genuinely harsh.
LIT_SNR_DB = {'clean': 25., 'mid': 15., 'noisy': 8., 'harsh': 2.}
LIT_N_CASES = [1000, 2500, 6000]


def lit_signal(n: int, peak_case: str, baseline_case: str,
               snr_db: float, rng: np.random.Generator
               ) -> tuple[np.ndarray, dict]:
    """
    Assemble one ``lit`` synthetic chromatogram (generic, literature).

    Not tied to the LPYE instrument: Gaussian peaks of varying
    amplitude/width/position on a large structured baseline, with the
    noise set by a target signal-to-noise ratio rather than an absolute
    detector floor. Follows the BEADS benchmark of Ning, Selesnick &
    Duval (2014), §5, so the algorithm is tested on the conditions the
    method was originally validated against, not only on this
    experiment.

    Parameters
    ----------
    n : int
        Number of data points.
    peak_case : str
        Key of ``LIT_PEAK_CASES``.
    baseline_case : str
        Key of ``LIT_BASELINE_CASES``.
    snr_db : float
        Target signal-to-noise ratio in decibels; the noise standard
        deviation is set to ``rms(peaks) / 10**(snr_db / 20)``.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    components : dict
        Same keys as ``make_signal``.

    """
    dt = 1. / PTS_PER_MIN
    t = np.arange(n) * dt
    n_peaks, fwhm_pts = LIT_PEAK_CASES[peak_case]

    signal = np.zeros(n)
    peaks = []
    # Dead-time injection artifact. Unlike the LPYE instrument (fixed
    # ~4.5 min) the literature family varies it over 3.5-6 min, so the
    # detector step and the artifact are not always at the same place.
    t0 = rng.uniform(3.5, 6.0)
    if t[-1] > t0 + 0.5:
        contrib, entries = dead_time_artifact(t, t0, dt, rng,
                                              pos_height=(2., 8.))
        signal += contrib
        peaks += entries
    else:
        t0 = 0.

    # Analytes elute after the dead time.
    lo = min(t0 + 0.02 * t[-1], 0.9 * t[-1])
    centers = np.sort(rng.uniform(lo, 0.97 * t[-1], n_peaks))
    broaden_g = rng.uniform(2., 4.)
    base_fwhm = rng.uniform(*fwhm_pts) * dt
    for tc in centers:
        fwhm = (base_fwhm * broadening_factor(tc, t0, t[-1], broaden_g)
                * rng.uniform(0.85, 1.15))
        sigma = fwhm / _FWHM_PER_SIGMA
        height = np.exp(rng.uniform(np.log(1.), np.log(30.)))
        signal += gauss_peak(t, tc, sigma, height)
        peaks.append({'tc': tc, 'sigma': sigma, 'tau': 0.,
                      'height': height, 'fwhm_points': fwhm / dt,
                      'artifact': False})

    baseline = lit_baseline(t, baseline_case, rng)
    rms = np.sqrt(np.mean(signal ** 2))
    noise_sigma = max(rms, 1e-3) / 10. ** (snr_db / 20.)
    noise = rng.normal(0., noise_sigma, n)
    return {'x': t, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise, 'peaks': peaks}


def plot_signal(parts: dict, stem: str, out_path: str) -> None:
    """
    Render one synthetic signal against its known ground truth.

    Always produced so the synthetic data can be eyeballed against the
    real chromatograms before it is trusted for scoring. Shows the raw
    measured signal, the true baseline, and the true baseline-corrected
    signal (peaks + noise), with the dead-time region marked.

    Parameters
    ----------
    parts : dict
        The component dictionary returned by ``make_signal``.
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
    ax0.set_ylabel('Potential (mV)')
    ax0.set_title(f'{stem}   (range {y.max() - y.min():.2f} mV, '
                  f'n={len(y)})', fontsize=10)
    ax0.legend(loc='upper right', fontsize=8)

    ax1.plot(x, corrected, lw=0.6, c='tab:green', label='$y - b$')
    ax1.axhline(0., c='tab:red', lw=0.8, alpha=0.4)
    ax1.set_xlabel('Time (min.)')
    ax1.set_ylabel('mV')
    ax1.legend(loc='upper right', fontsize=8)

    # Mark the dead-time artifact region (first non-blank-ghost peak).
    arts = [p for p in parts['peaks'] if p.get('artifact')]
    if arts:
        t0 = min(p['tc'] for p in arts)
        for ax in (ax0, ax1):
            ax.axvspan(t0 - 0.3, t0 + 0.5, color='tab:orange', alpha=0.08)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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
    parser.add_argument('--family',
                        choices=('native', 'lit', 'erb_native', 'both',
                                 'all'),
                        default='both',
                        help="'native' = calibrated to the LPYE data; "
                             "'lit' = generic Ning 2014 benchmark "
                             "conditions; 'erb_native' = the LPYE peak "
                             "component on Erb's four analytic "
                             "baselines; 'both' (default) = native+lit; "
                             "'all' = every family")
    parser.add_argument('--replicates', type=int, default=1,
                        help='replicates per case (default: 1)')
    parser.add_argument('--no-plot', action='store_true',
                        help='skip the per-signal PNG in plots/ '
                             '(rendered by default so the synthetic data '
                             'can be reviewed against the real signals)')
    args = parser.parse_args()

    sig_dir = os.path.join(args.output, 'signals')
    truth_dir = os.path.join(args.output, 'truth')
    plot_dir = os.path.join(args.output, 'plots')
    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)
    if not args.no_plot:
        os.makedirs(plot_dir, exist_ok=True)

    # One flat list of (family, peak_case, baseline_case, noise_key,
    # noise_value_or_snr, n) jobs, so the two families share the writing,
    # plotting and manifest code below.
    jobs = []
    if args.family in ('native', 'both', 'all'):
        for pc in PEAK_CASES:
            for bc in BASELINE_CASES:
                for nk, ns in NOISE_CASES.items():
                    for n in N_CASES:
                        jobs.append(('native', pc, bc, nk, ns, n))
    if args.family in ('lit', 'both', 'all'):
        for pc in LIT_PEAK_CASES:
            for bc in LIT_BASELINE_CASES:
                for nk, snr in LIT_SNR_DB.items():
                    for n in LIT_N_CASES:
                        jobs.append(('lit', pc, bc, nk, snr, n))
    if args.family in ('erb_native', 'all'):
        for pc in PEAK_CASES:
            for bt in range(4):
                for nk, ns in NOISE_CASES.items():
                    # Length is drawn per signal, not crossed: see
                    # `draw_record_length`.
                    jobs.append(('erb_native', pc, f'erb{bt}', nk,
                                 ns, None))

    manifest = []
    for k, (family, pc, bc, nk, nv, n) in enumerate(jobs):
        for rep in range(args.replicates):
            rng = np.random.default_rng(args.seed + 7919 * (k + 1)
                                        + 104729 * rep)
            if family == 'native':
                parts = make_signal(n, pc, bc, nv, rng)
            elif family == 'erb_native':
                n = draw_record_length(rng)
                parts = erb_native_signal(n, pc, int(bc[-1]), nv, rng)
            else:
                parts = lit_signal(n, pc, bc, nv, rng)
            stem = f'SYN__{family}__{pc}__{bc}__{nk}__{n}__{rep}'
            with open(os.path.join(sig_dir, f'{stem}.txt'), 'w') as f:
                for xi, yi in zip(parts['x'], parts['y']):
                    f.write(f'{xi:.6f}\t{yi:.6e}\n')
            np.savez(
                os.path.join(truth_dir, f'{stem}__truth.npz'),
                x=parts['x'], y=parts['y'], signal=parts['signal'],
                baseline=parts['baseline'], noise=parts['noise'],
                peaks=json.dumps(parts['peaks']))
            if not args.no_plot:
                plot_signal(parts, stem,
                            os.path.join(plot_dir, f'{stem}.png'))
            manifest.append(
                f'{stem},{family},{pc},{bc},{nk},{n},{rep},'
                f'{len(parts["peaks"])}')
    with open(os.path.join(args.output, 'manifest.csv'), 'w') as f:
        f.write('stem,family,peak_case,baseline_case,noise_case,'
                'n_points,replicate,n_peaks\n')
        f.write('\n'.join(manifest) + '\n')
    print(f'{len(manifest)} signals -> {args.output}')


if __name__ == '__main__':
    main()
