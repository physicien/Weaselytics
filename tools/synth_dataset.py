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
from scipy.ndimage import gaussian_filter1d  # noqa: E402
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

    baseline = make_baseline(t, baseline_case, rng, t0=t0,
                             is_blank=(n_peaks == 0))
    noise = rng.normal(0., noise_sigma, n)
    return {'x': t, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise, 'peaks': peaks}


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
    parser.add_argument('--family', choices=('native', 'lit', 'both'),
                        default='both',
                        help="'native' = calibrated to the LPYE data; "
                             "'lit' = generic Ning 2014 benchmark "
                             "conditions; 'both' (default)")
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
    if args.family in ('native', 'both'):
        for pc in PEAK_CASES:
            for bc in BASELINE_CASES:
                for nk, ns in NOISE_CASES.items():
                    for n in N_CASES:
                        jobs.append(('native', pc, bc, nk, ns, n))
    if args.family in ('lit', 'both'):
        for pc in LIT_PEAK_CASES:
            for bc in LIT_BASELINE_CASES:
                for nk, snr in LIT_SNR_DB.items():
                    for n in LIT_N_CASES:
                        jobs.append(('lit', pc, bc, nk, snr, n))

    manifest = []
    for k, (family, pc, bc, nk, nv, n) in enumerate(jobs):
        for rep in range(args.replicates):
            rng = np.random.default_rng(args.seed + 7919 * (k + 1)
                                        + 104729 * rep)
            if family == 'native':
                parts = make_signal(n, pc, bc, nv, rng)
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
