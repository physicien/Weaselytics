#!/usr/bin/python3
"""
Generate synthetic chromatograms with known ground truth.

Each signal is a peak component plus a baseline plus white noise. The
true baseline is saved beside the signal, so a baseline estimate can be
scored against it.

In the peak component, a positive spike followed by a negative
undershoot marks the injection at the dead time. The analytes are
exponentially-modified Gaussian peaks whose widths follow the plate
model, sigma = tR / sqrt(N) (Snyder, Kirkland & Dolan 2009, Eqs. 2.9b
and 2.10). They elute after the injection artifact, return to the noise
before the record ends, and keep a resolution of at least 1 between
neighbours. Blanks hold only the artifact. The ranges of plate number,
dead time and noise were picked for this benchmark.

The baseline is one of the analytic baselines Donald Erb, author of
pybaselines, wrote to test the cutoff selection of this project: a
parabola, an exponential decay or a step. Records last 4.2 to 333 min,
500 to 40000 points at 120 pts/min, and the signal is rounded to a
detector's voltage step.

Every signal is written as a two-column text file the weaselytics
pipeline reads unchanged. The split into signal, baseline and noise
goes to ``truth/<stem>__truth.npz``, one row per signal to
``manifest.csv``, and a plot to ``plots/`` unless ``--no-plot`` is
given.

Usage
-----
python tools/synthetic/synth_dataset.py OUTPUT_DIR [--seed 0]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import exponnorm, skewnorm  # noqa: E402

PEAK_CASES = {
    # (fewest, most) analytes, the count drawn per signal. Every case
    # takes its widths from the plate model (`PLATE_NUMBER_RANGE`);
    # 'isocratic' also pins an early and a late eluter and keeps the
    # exact plate width.
    'blank': (0, 0),
    'single': (1, 1),
    'few': (3, 4),
    'many': (6, 8),
    'isocratic': (7, 7),
}
# Column plate number, drawn once per signal. The range was picked for
# this benchmark around the N of about 10,000 that Snyder, Kirkland &
# Dolan (2009, §2.5, p. 57) call a good starting point in many cases.
PLATE_NUMBER_RANGE = (2000., 14000.)
# Standard deviation of the white detector noise, in mV, at two levels
# picked for this benchmark. 'typ' is the reference level and 'high',
# two and a half times larger, gives noisier records.
NOISE_CASES = {'typ': 0.02, 'high': 0.05}
# Default points per minute, picked for this benchmark: two samples per
# second. `--rate` sets another.
PTS_PER_MIN = 120
# Width (FWHM) of the artifact spike, in min: 6 to 10 samples at the
# default rate, picked for this benchmark.
ARTIFACT_FWHM_MIN = (6. / PTS_PER_MIN, 10. / PTS_PER_MIN)

_FWHM_PER_SIGMA = 2.3548

# Quantization step of the detector, in mV, picked for this benchmark.
ADC_STEP_MV = 0.008996


def quantise(y: np.ndarray, step: float = ADC_STEP_MV) -> np.ndarray:
    """
    Round a trace onto the detector's quantization lattice.

    A digitized detector signal is an integer multiple of its
    quantization step, so the synthetic signals are rounded likewise.

    Parameters
    ----------
    y : array-like, shape (N,)
        The trace, in mV.
    step : float, optional
        Quantization step in mV. Default `ADC_STEP_MV`.

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


###############################################################################
# Erb's test baselines: the analytic baselines of the `make_data` function
# Donald Erb, author of pybaselines, wrote to test the cutoff-frequency
# selection of this project. The `erb` family places the plate-model
# peak component on them. Posted in issue #4:
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
        a population that varies the record length, so `erb_signal`
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


def dead_time_artifact(t: np.ndarray, t0: float, dt: float,
                       rng: np.random.Generator,
                       pos_height: tuple[float, float] = (1., 4.)
                       ) -> tuple[np.ndarray, list[dict]]:
    """
    Bipolar injection artifact at the dead time.

    A sharp positive spike at ``t0`` immediately followed by a wider
    negative undershoot. The undershoot is 0 to 2 times as deep as the
    spike is high, as the trace shows them, drawn as twice a Beta(1, 3)
    variate so small dips are the most common. This law was picked for
    this benchmark. The negative lobe is genuine signal, which is why a
    symmetric BEADS cost (``asymmetry=1``) is required: an asymmetric
    cost would absorb it into the baseline.

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
        Range of the positive-spike height, mV. Default ``(1., 4.)``,
        picked for this benchmark.

    Returns
    -------
    contribution : numpy.ndarray, shape (N,)
        The artifact, to add to the signal.
    entries : list of dict
        Per-lobe parameter records for the truth file. The negative
        lobe's entry also carries ``dip_ratio``, the dip-to-spike ratio
        the trace shows.

    """
    pos_fwhm = rng.uniform(*ARTIFACT_FWHM_MIN)
    pos_h = rng.uniform(*pos_height)
    pos_sigma = pos_fwhm / _FWHM_PER_SIGMA
    pos_tau = rng.uniform(0.3, 0.8) * pos_sigma
    contrib = emg_peak(t, t0, pos_sigma, pos_tau, pos_h)
    entries = [{'tc': t0, 'sigma': pos_sigma, 'tau': pos_tau,
                'height': pos_h, 'fwhm_points': pos_fwhm / dt,
                'artifact': True}]

    neg_fwhm = pos_fwhm * rng.uniform(1.2, 2.5)
    ratio = 2. * rng.beta(1., 3.)
    neg_sigma = neg_fwhm / _FWHM_PER_SIGMA
    neg_tc = t0 + rng.uniform(1.0, 2.0) * pos_sigma
    neg_shape = emg_peak(t, neg_tc, neg_sigma, neg_sigma * 0.5, 1.)
    neg_h = _dip_scale(contrib, neg_shape, ratio)
    contrib = contrib - neg_h * neg_shape
    entries.append({'tc': neg_tc, 'sigma': neg_sigma,
                    'tau': neg_sigma * 0.5, 'height': -neg_h,
                    'fwhm_points': neg_fwhm / dt, 'dip_ratio': ratio,
                    'artifact': True})
    return contrib, entries


def peak_component(n: int, peak_case: str, rng: np.random.Generator,
                   noise_sigma: float, rate: float = PTS_PER_MIN
                   ) -> tuple[np.ndarray, np.ndarray, list[dict],
                                     float, dict]:
    """
    Build the peak component, with the analytes, artifact and time axis.

    Everything an `erb` chromatogram contains except its baseline and
    its noise.

    The component is sampled at `rate` points per minute and has a dead
    time between 2.5 and 6.5 min carrying the bipolar injection
    artifact, exponentially-modified Gaussian analytes eluting after
    it, and widths growing with retention time.

    Parameters
    ----------
    n : int
        Number of data points.
    peak_case : str
        Key of `PEAK_CASES`.
    rng : numpy.random.Generator
        Source of randomness.
    noise_sigma : float
        Standard deviation of the noise the signal will carry. Every
        analyte is placed so that it is within it until the injection
        artifact has returned to it and again at the last sample, at a
        resolution of at least `MIN_RESOLUTION` from every other
        analyte (see `_place_peak`). When an analyte fits nowhere, the
        dead time and the analytes are drawn again, with the dead time
        earlier, until all fit; an analyte that still fits nowhere with
        the dead time at 2.5 min is dropped.
    rate : float, optional
        Points per minute. Default `PTS_PER_MIN`.

    Returns
    -------
    t : numpy.ndarray, shape (N,)
        Time axis, minutes.
    signal : numpy.ndarray, shape (N,)
        Analytes plus the injection artifact.
    peaks : list of dict
        Per-peak parameters; entries with ``artifact=True`` are the
        two injection lobes, the others are analytes. Analytes also
        carry ``apex``, ``fwhm_true_points``, ``half_left`` and
        ``half_right``, measured on the noise-free profile: ``tc`` and
        ``fwhm_points`` describe the Gaussian component of the EMG, not
        the peak itself.
    t0 : float
        The dead time used, minutes.
    info : dict
        ``n_requested`` (analytes the case drew), ``n_plates`` (NaN for
        a blank) and ``dead_time_redraws``.

    Raises
    ------
    ValueError
        If the record is too short to hold a dead time of 2.5 min.

    """
    dt = 1. / rate                      # minutes per sample
    t = np.arange(n) * dt
    # Dead time in absolute minutes, drawn between 2.5 and 6.5 min. The
    # range was picked for this benchmark. The dead time depends on the
    # column and the flow rate, so it changes from one setup to another.
    # The upper bound is lowered on a short record so the elution window
    # opened below, from 0.02 of the run after the dead time to 0.9 of
    # the run, is never empty; a record too short to hold the lower bound
    # is refused.
    t0_hi = min(6.5, 0.88 * t[-1])
    if t0_hi <= 2.5:
        raise ValueError(f'a {t[-1]:.2f} min record cannot hold a dead '
                         f'time of at least 2.5 min before its elution '
                         f'window')
    lo, hi = PEAK_CASES[peak_case]
    n_peaks = int(rng.integers(lo, hi + 1))

    # Packing many analytes into a short run is only possible with an
    # early dead time. When an analyte fits nowhere, the dead time is
    # drawn again below the one that failed, with a new set of analytes,
    # so it moves earlier until every analyte fits, or until it reaches
    # 2.5 min within a sample.
    redraws = -1
    while True:
        redraws += 1
        t0 = rng.uniform(2.5, t0_hi)
        signal, peaks, n_plates = _peak_component_at(
            t, t0, n_peaks, peak_case, noise_sigma, rng)
        n_placed = sum(not p['artifact'] for p in peaks)
        if n_placed == n_peaks or t0 - 2.5 < dt:
            break
        t0_hi = t0
    return t, signal, peaks, t0, {'n_requested': n_peaks,
                                  'n_plates': n_plates,
                                  'dead_time_redraws': redraws}


def _peak_component_at(t: np.ndarray, t0: float, n_peaks: int,
                       peak_case: str, noise_sigma: float,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, list[dict], float]:
    """
    Build the artifact and place the analytes for one dead time.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Time axis, minutes.
    t0 : float
        Dead time, minutes.
    n_peaks : int
        Number of analytes to place.
    peak_case : str
        Key of `PEAK_CASES`.
    noise_sigma : float
        Noise standard deviation; see `peak_component`.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    signal : numpy.ndarray, shape (N,)
        Analytes plus the injection artifact.
    peaks : list of dict
        The two artifact lobes, then the placed analytes by apex.
    n_plates : float
        Plate number drawn for the analytes, NaN without analytes.

    """
    dt = t[1] - t[0]
    signal = np.zeros(len(t))
    peaks = []

    # The dead-time injection artifact (bipolar; see dead_time_artifact).
    contrib, entries = dead_time_artifact(t, t0, dt, rng,
                                          pos_height=(1., 4.))
    signal += contrib
    peaks += entries
    # The dead time is the first thing to leave the column: it is the
    # transit time of an unretained species, so every other feature
    # starts only once the injection artifact is back in the noise. A
    # feature earlier than that would have travelled the column faster
    # than the mobile phase.
    above = np.nonzero(np.abs(contrib) > noise_sigma)[0]
    t_front = t[above[-1]] if above.size else t0
    # Peaks are first drawn in this window, bounded above by the run
    # end; `_place_peak` then moves any that does not fit.
    elute_lo = t0 + 0.02 * t[-1]
    elute_hi = 0.9 * t[-1]

    # A blank carries only the injection artifact.
    if not n_peaks:
        return signal, peaks, float('nan')
    specs, n_plates = _draw_analytes(t, t0, n_peaks,
                                     peak_case == 'isocratic', elute_lo,
                                     elute_hi, rng)

    # An analyte has eluted when its trailing edge is back in the noise.
    # A record that stops before that is impossible: the trace ends
    # part-way down a peak, which is the endpoint mismatch BEADS'
    # periodicity requirement forbids and which no parabola correction
    # repairs. Peaks are placed in elution order, each where it was
    # drawn when it neither overlaps the dead time, nor runs off the
    # end, nor crowds a peak already placed, else anywhere it does, and
    # dropped only when there is no such place.
    placed = []
    for spec in specs:
        found = _place_peak(t, spec['tc'], spec['sigma_at'],
                            spec['tau_ratio'], spec['height'], t_front,
                            noise_sigma, placed, rng)
        if found is None:
            continue
        tc, (apex, left, right) = found
        sigma = spec['sigma_at'](tc)
        tau = spec['tau_ratio'] * sigma
        signal += emg_peak(t, tc, sigma, tau, spec['height'])
        placed.append({'tc': tc, 'sigma': sigma, 'tau': tau,
                       'height': spec['height'],
                       'fwhm_points': sigma * _FWHM_PER_SIGMA / dt,
                       'apex': apex,
                       'fwhm_true_points': (right - left) / dt,
                       'half_left': left, 'half_right': right,
                       'artifact': spec['artifact']})
    peaks += sorted(placed, key=lambda p: p['apex'])
    return signal, peaks, n_plates


def _draw_analytes(t: np.ndarray, t0: float, n_peaks: int,
                   isocratic: bool, elute_lo: float, elute_hi: float,
                   rng: np.random.Generator) -> tuple[list[dict], float]:
    """
    Draw one analyte set: centres, width laws, tails and heights.

    A width is a function of where the peak sits, so each analyte is
    returned with the law that gives its sigma at any retention time:
    `_place_peak` may move it, and a moved peak takes the width of its
    new place.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis, minutes.
    t0 : float
        Dead time, minutes.
    n_peaks : int
        Number of analytes, at least 1.
    isocratic : bool
        Pin an early and a late eluter and keep the exact plate width,
        without the per-peak jitter.
    elute_lo, elute_hi : float
        Bounds of the window the centres are drawn in, minutes.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    specs : list of dict
        One per analyte, sorted by drawn centre: ``tc``, ``sigma_at``
        (retention time to sigma), ``tau_ratio`` (tail time constant
        over sigma), ``height`` and ``artifact=False``.
    n_plates : float
        The plate number drawn for the signal.

    """
    centers = np.sort(rng.uniform(elute_lo, elute_hi, n_peaks))
    if isocratic and n_peaks >= 2:
        # isocratic case: aim for a sharp first and a broad last peak by
        # pinning an early and a late eluter, both clipped into the
        # window so the sort keeps them in order. Placement may still
        # move them.
        centers[0] = np.clip(t0 * rng.uniform(1.05, 1.20),
                             elute_lo, elute_hi)
        centers[-1] = np.clip(rng.uniform(0.85, 0.92) * t[-1],
                              elute_lo, elute_hi)
        centers = np.sort(centers)
    # One plate number per signal: W = 4 N^-0.5 tR with W = 4 sigma, so
    # sigma = tR / sqrt(N), tR counted from injection (Snyder, Kirkland &
    # Dolan 2009, §2.3 p. 24, Eqs. 2.9b and 2.10, §2.4 p. 36). A width is
    # then a fixed share of its retention time, and since every peak
    # elutes inside the run, widths scale with the run length. The first
    # peaks are sharp and the late ones broad.
    n_plates = rng.uniform(*PLATE_NUMBER_RANGE)
    specs = []
    for tc in centers:
        # A mild per-peak jitter keeps the compounds from being identical;
        # the isocratic case keeps the exact plate width.
        jitter = 1. if isocratic else rng.uniform(0.85, 1.15)

        def sigma_at(x, jitter=jitter):
            """Sigma of this analyte at retention time `x`, minutes."""
            return jitter * x / np.sqrt(n_plates)
        specs.append({'tc': tc, 'sigma_at': sigma_at,
                      'tau_ratio': rng.uniform(0.3, 1.2),
                      'height': np.exp(rng.uniform(np.log(1.),
                                                   np.log(30.))),
                      'artifact': False})
    return specs, n_plates


# Minimum chromatographic resolution between any two placed peaks,
# Rs = 1.18 (t2 - t1) / (w1 + w2) on the apex and the FWHM. At Rs = 1.0
# about 2% of each band lies under the other (Fornstedt, Forssén &
# Westerlund 2015, §1.2, Eqs. 1.17-1.18, p. 13; Gaussian zones of equal
# amount); 1.5 is baseline resolution (Snyder, Kirkland & Dolan 2009,
# §2.5, Eq. 2.23, p. 54), and 1.18 follows from their W1/2 = 0.588 W
# (§2.4, p. 35). BEADS assumes the peaks and their first two derivatives
# are sparse (Ning, Selesnick & Duval 2014, §3). A floor of 1.0 still
# allows small overlaps, to test the limits of that assumption. The
# formula assumes Gaussian peaks and can be far off for asymmetric ones
# (Dyson 1998, p. 10); these EMG peaks tail, so the 2% is approximate.
MIN_RESOLUTION = 1.0

# Matrix entries evaluated at once while searching for a place, a memory
# bound only: it never changes where a peak lands.
_PLACEMENT_CHUNK = 2_000_000


def _apex_and_half_width(t: np.ndarray, y: np.ndarray
                         ) -> tuple[float, float, float]:
    """
    Measure the apex and the half-maximum crossings of a sampled peak.

    The apex is refined by a parabola through the three samples around
    the maximum; each crossing is interpolated linearly between the two
    samples that bracket half the maximum.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Time axis, minutes.
    y : numpy.ndarray, shape (N,)
        Noise-free, single-peaked profile.

    Returns
    -------
    apex, left, right : float
        Apex time and the times where the profile crosses half its
        maximum on either side, minutes. A crossing beyond the record is
        reported at its end.

    """
    k = int(np.argmax(y))
    apex = t[k]
    if 0 < k < len(y) - 1:
        denom = y[k - 1] - 2. * y[k] + y[k + 1]
        if denom < 0:
            apex += 0.5 * (y[k - 1] - y[k + 1]) / denom * (t[1] - t[0])
    half = 0.5 * y[k]
    below = np.nonzero(y[:k] < half)[0]
    if below.size:
        i = below[-1]
        left = t[i] + (half - y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i])
    else:
        left = t[0]
    below = np.nonzero(y[k:] < half)[0]
    if below.size:
        i = k + below[0]
        right = t[i - 1] + ((y[i - 1] - half) / (y[i - 1] - y[i])
                            * (t[i] - t[i - 1]))
    else:
        right = t[-1]
    return float(apex), float(left), float(right)


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


def _place_peak(t: np.ndarray, tc: float, sigma_at, tau_ratio: float,
                height: float, t_front: float, noise_sigma: float,
                placed: list[dict], rng: np.random.Generator
                ) -> tuple[float, tuple[float, float, float]] | None:
    """
    Place a peak at its drawn retention time, or anywhere it fits.

    A peak fits when its profile is within the noise at every sample up
    to `t_front`, where the injection artifact has returned to the
    noise, and at the last sample, and when its resolution from every
    peak in `placed` is at least `MIN_RESOLUTION`. The drawn `tc` is
    kept when it fits; otherwise the peak is drawn uniformly among the
    samples after `t_front` where it fits, each evaluated with the width
    `sigma_at` gives at that place.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Time axis, minutes.
    tc : float
        Drawn retention time, minutes.
    sigma_at : callable
        Maps a retention time to the peak's sigma, minutes.
    tau_ratio : float
        Tail time constant over sigma.
    height : float
        Peak maximum.
    t_front : float
        End of the injection artifact, minutes.
    noise_sigma : float
        Noise standard deviation, the level a peak must be within.
    placed : list of dict
        Peaks already placed, each with ``apex``, ``half_left`` and
        ``half_right``.
    rng : numpy.random.Generator
        Source of randomness for the new place.

    Returns
    -------
    found : tuple or None
        ``(tc, (apex, left, right))`` of the chosen place, or None when
        the peak fits nowhere.

    Notes
    -----
    Fails, by returning None, when the peak is wider than any room left
    between the end of the artifact, the end of the record and the
    peaks already placed. A peak that does not fit where it was drawn
    costs one profile per sample of the record, so its cost grows with
    the square of the record length.

    """
    others = [(p['apex'], p['half_left'], p['half_right']) for p in placed]
    front = t <= t_front

    def fitting(c):
        """
        The candidate centres `c` where the peak fits, each with its
        apex and half-maximum crossings.
        """
        sigma = np.array([sigma_at(x) for x in c])
        shape = exponnorm.pdf(t[None, :], K=max(tau_ratio, 1e-3),
                              loc=c[:, None], scale=sigma[:, None])
        peak_max = shape.max(axis=1)
        y = height * shape / np.where(peak_max > 0, peak_max, 1.)[:, None]
        ok = (peak_max > 0) & (y[:, -1] <= noise_sigma)
        if front.any():
            ok &= y[:, front].max(axis=1) <= noise_sigma
        fits = []
        for i in np.nonzero(ok)[0]:
            shape_i = _apex_and_half_width(t, y[i])
            if all(_resolution(shape_i, o) >= MIN_RESOLUTION
                   for o in others):
                fits.append((float(c[i]), shape_i))
        return fits

    drawn = fitting(np.array([tc]))
    if drawn:
        return drawn[0]
    later = t[t > t_front]
    chunk = max(1, _PLACEMENT_CHUNK // len(t))
    fits = []
    for start in range(0, len(later), chunk):
        fits += fitting(later[start:start + chunk])
    if not fits:
        return None
    return fits[int(rng.integers(len(fits)))]


def _analyte_meta(t: np.ndarray, peaks: list[dict], t0: float,
                  info: dict) -> dict:
    """
    Summarize the placed analytes of one peak component.

    Parameters
    ----------
    t : numpy.ndarray, shape (N,)
        Time axis, minutes.
    peaks : list of dict
        Peaks from `peak_component`.
    t0 : float
        Dead time, minutes.
    info : dict
        The ``info`` returned by `peak_component`.

    Returns
    -------
    meta : dict
        ``n_analytes``, ``rs_min`` (lowest resolution between adjacent
        analytes, NaN below two), ``coverage`` (share of the run inside
        their half-maximum widths), ``n_requested``, ``n_dropped``,
        ``n_plates``, ``dead_time`` and ``dead_time_redraws``.

    """
    analytes = sorted((p['apex'], p['half_left'], p['half_right'])
                      for p in peaks if not p.get('artifact', False))
    rs_min = min((_resolution(a, b) for a, b in zip(analytes,
                                                     analytes[1:])),
                 default=float('nan'))
    covered, reach = 0., -np.inf
    for _, left, right in sorted(analytes, key=lambda a: a[1]):
        covered += max(0., right - max(left, reach))
        reach = max(reach, right)
    return {'n_analytes': len(analytes), 'rs_min': float(rs_min),
            'coverage': covered / float(t[-1] - t[0]),
            'n_requested': info['n_requested'],
            'n_dropped': info['n_requested'] - len(analytes),
            'n_plates': info['n_plates'], 'dead_time': t0,
            'dead_time_redraws': info['dead_time_redraws']}


# Record length of the `erb` family. log10 of the length follows a
# skew-normal distribution, with the shape, location and scale below
# picked for this benchmark, and draws outside the range are redrawn.
# Short records are the most common. Half are under about 1500 points
# and 2% are over 10000. The length is read at the default rate and
# turned into a duration, so another `--rate` keeps the run as long and
# changes only how densely it is sampled.
#
# Record length sets the fundamental frequency 1/n_used, which is exactly
# what `instability_boundary` keys on, so length must vary. A balanced
# ladder of lengths would over-weight the long records, which are the
# most expensive to sweep.
ERB_LOG10_N_SKEWNORM = (10.02, 2.818, 0.512)
ERB_N_RANGE = (500, 40000)

# Erb's baseline 3 is a pure sinusoid, ``10 + sin(x/50)``, 2 mV peak to
# peak and about three cycles over the record. It is excluded from the
# benchmark because an HPLC baseline does not oscillate through several
# full cycles within one run.
EXCLUDED_ERB_BASELINES = ('sine',)

# Names of Erb's baselines in signal names, by `erb_baseline` kind.
ERB_BASELINE_NAMES = {'parabola': 0, 'exp': 1, 'step': 2, 'sine': 3}


def draw_record_duration(rng: np.random.Generator) -> float:
    """
    Draw a record duration of the ``erb`` family.

    log10 of the length is drawn from `ERB_LOG10_N_SKEWNORM` and
    redrawn until the length falls inside `ERB_N_RANGE`, then divided
    by `PTS_PER_MIN`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    duration : float
        Record duration, minutes.

    """
    lo, hi = ERB_N_RANGE
    while True:
        n = int(round(10 ** float(skewnorm.rvs(
            *ERB_LOG10_N_SKEWNORM, random_state=rng))))
        if lo <= n <= hi:
            return n / PTS_PER_MIN


def erb_signal(n: int, peak_case: str, baseline_type: int,
               noise_sigma: float, rng: np.random.Generator,
               quantise_output: bool = True,
               rate: float = PTS_PER_MIN) -> dict:
    """
    Assemble one ``erb`` chromatogram, the shared peak component on an
    Erb baseline.

    The peak component comes from `peak_component`. White noise
    at one of `NOISE_CASES` is added.

    The baseline, from `erb_baseline`, is an exact analytic function
    written by the pybaselines author to exercise cutoff selection.

    **Scales.** The baseline is evaluated on a normalized 0-1000
    abscissa spanning the record, because `erb_baseline`'s coefficients
    are written for that range. Its output is read as mV, so the
    parabola spans 5 mV, the exponential 8.1 mV and the step 12.1 mV.

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
        Round the assembled signal onto the detector lattice, see
        `quantise`. Default True.
    rate : float, optional
        Points per minute. Default `PTS_PER_MIN`.

    Returns
    -------
    components : dict
        Keys ``x``, ``y``, ``signal``, ``baseline``, ``noise``,
        ``peaks`` and ``meta``.

        ``baseline`` alone is the truth a baseline estimator is scored
        against. The injection artifact is part of ``signal``, and
        ``baseline`` holds the baseline alone: the artifact is a sharp
        bipolar excursion that a low-pass baseline model is not meant to
        reproduce, and scoring on it
        would let one feature dominate the error. Entries of ``peaks``
        with ``artifact=True`` mark it.

        ``meta['n_requested']`` is the analyte count the case drew,
        ``meta['n_dropped']`` the analytes that fit nowhere even with the
        dead time at 2.5 min, and ``n_analytes`` what remains.
        ``meta['n_plates']`` and ``meta['dead_time_redraws']`` record the
        plate number and how often the dead time was drawn again,
        earlier, to make room.
        ``meta['rs_min']`` is the lowest resolution between adjacent
        analytes (NaN below two) and ``meta['coverage']`` the share of
        the run inside their half-maximum widths. Both are measured after
        placement.

    """
    t, signal, peaks, t0, info = peak_component(n, peak_case, rng,
                                                noise_sigma, rate)
    # Erb's coefficients are written for an abscissa of order 0-1000, so
    # the baseline is evaluated there and mapped onto the record. Its
    # shape is therefore independent of the record's duration in
    # minutes, which is what makes the types comparable across
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
            'meta': {'family': 'erb', 'n_points': n,
                     'peak_case': peak_case,
                     'baseline_type': baseline_type,
                     'noise_sigma': noise_sigma,
                     'quantised': quantise_output,
                     'points_per_minute': rate,
                     'minutes': float(t[-1] - t[0]),
                     'baseline_range': float(baseline.max()
                                             - baseline.min()),
                     **_analyte_meta(t, peaks, t0, info)}}


def plot_signal(parts: dict, stem: str, out_path: str) -> None:
    """
    Render one synthetic signal against its known ground truth.

    Produced unless ``--no-plot`` is given, so the synthetic data can be
    reviewed by eye before it is trusted for scoring. Shows the raw
    measured signal, the true baseline, and the true baseline-corrected
    signal (peaks + noise), with the dead-time region marked.

    Parameters
    ----------
    parts : dict
        The component dictionary returned by ``erb_signal``.
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

    # Mark the dead-time artifact region.
    arts = [p for p in parts['peaks'] if p.get('artifact')]
    if arts:
        t0 = min(p['tc'] for p in arts)
        for ax in (ax0, ax1):
            ax.axvspan(t0 - 0.3, t0 + 0.5, color='tab:orange', alpha=0.08)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# Columns of `manifest.csv`, one row per signal; see `manifest_row`.
MANIFEST_COLUMNS = ('stem', 'family', 'peak_case', 'baseline_case',
                    'noise_case', 'n_points', 'replicate', 'n_analytes',
                    'n_dropped', 'rs_min', 'coverage', 'n_requested',
                    'n_plates', 'dead_time', 'dead_time_redraws',
                    'points_per_minute')


def signal_stem(family: str, peak_case: str, baseline_case: str,
                noise_case: str, replicate: int, n: int,
                rep_width: int) -> str:
    """
    Name one synthetic signal.

    Parameters
    ----------
    family, peak_case, baseline_case, noise_case : str
        The case keys of the signal.
    replicate : int
        Replicate number.
    n : int
        Record length, points.
    rep_width : int
        Digits of the widest replicate number, so names sort in order.

    Returns
    -------
    stem : str
        ``SYN__family__case__baseline__noise__replicate__n``.

    """
    return (f'SYN__{family}__{peak_case}__{baseline_case}__{noise_case}'
            f'__{replicate:0{rep_width}d}__{n}')


def manifest_row(stem: str, family: str, peak_case: str,
                 baseline_case: str, noise_case: str, n: int,
                 replicate: int, parts: dict) -> str:
    """
    Format the `manifest.csv` row of one signal.

    Parameters
    ----------
    stem : str
        Name of the signal.
    family, peak_case, baseline_case, noise_case : str
        The case keys of the signal.
    n : int
        Record length, points.
    replicate : int
        Replicate number.
    parts : dict
        The signal as `erb_signal` returns it.

    Returns
    -------
    row : str
        Comma-separated values in the order of `MANIFEST_COLUMNS`.

    """
    meta = parts.get('meta', {})
    n_analytes = meta.get('n_analytes',
                          sum(not p.get('artifact', False)
                              for p in parts['peaks']))
    return (f'{stem},{family},{peak_case},{baseline_case},{noise_case},'
            f'{n},{replicate},{n_analytes},{meta.get("n_dropped", "")},'
            f'{meta.get("rs_min", float("nan")):.4g},'
            f'{meta.get("coverage", float("nan")):.4g},'
            f'{meta.get("n_requested", "")},'
            f'{meta.get("n_plates", float("nan")):.0f},'
            f'{meta.get("dead_time", float("nan")):.3f},'
            f'{meta.get("dead_time_redraws", "")},'
            f'{meta.get("points_per_minute", PTS_PER_MIN):g}')


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
    parser.add_argument('--rate', type=float, default=PTS_PER_MIN,
                        help='points per minute (default: 120)')
    parser.add_argument('--no-plot', action='store_true',
                        help='skip the per-signal PNG in plots/ '
                             '(rendered by default for review by eye)')
    args = parser.parse_args()

    sig_dir = os.path.join(args.output, 'signals')
    truth_dir = os.path.join(args.output, 'truth')
    plot_dir = os.path.join(args.output, 'plots')
    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)
    if not args.no_plot:
        os.makedirs(plot_dir, exist_ok=True)

    jobs = []
    for pc in PEAK_CASES:
        for bname in ERB_BASELINE_NAMES:
            for nk, ns in NOISE_CASES.items():
                # Length is drawn per signal, not crossed: see
                # `draw_record_duration`.
                jobs.append((pc, bname, nk, ns))

    manifest = []
    # Zero-padded to the widest replicate number, so names sort in order.
    rep_width = len(str(args.replicates - 1))
    for k, (pc, bc, nk, ns) in enumerate(jobs):
        # Skipped inside the loop rather than dropped from `jobs`, so the
        # index that seeds every other signal is unchanged and the rest
        # of the set stays reproducible.
        if bc in EXCLUDED_ERB_BASELINES:
            continue
        for rep in range(args.replicates):
            rng = np.random.default_rng(args.seed + 7919 * (k + 1)
                                        + 104729 * rep)
            n = int(round(draw_record_duration(rng) * args.rate))
            parts = erb_signal(n, pc, ERB_BASELINE_NAMES[bc], ns, rng,
                               rate=args.rate)
            stem = signal_stem('erb', pc, bc, nk, rep, n, rep_width)
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
            manifest.append(manifest_row(stem, 'erb', pc, bc, nk, n, rep,
                                         parts))
    with open(os.path.join(args.output, 'manifest.csv'), 'w') as f:
        f.write(','.join(MANIFEST_COLUMNS) + '\n')
        f.write('\n'.join(manifest) + '\n')
    print(f'{len(manifest)} signals -> {args.output}')


if __name__ == '__main__':
    main()
