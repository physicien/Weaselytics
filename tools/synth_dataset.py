#!/usr/bin/python3
"""
Generate synthetic chromatograms with known ground truth.

Each signal is the sum of three physically motivated components:

* **peaks**: exponentially-modified Gaussians (the standard model of a
  chromatographic peak: Gaussian broadening + first-order tailing),
  with widths growing slowly along the run and log-distributed
  heights;
* **baseline**: a solvent-front exponential decay, a broad gradient
  hump, a slow linear drift and a mid-frequency wander whose
  correlation length lies between the peak widths and the run
  length;
* **noise**: white detector noise.

Blanks are baseline + noise only. Every signal is written in the same
two-column text format as the real data, so the whole weaselytics
pipeline runs on it unchanged, and the exact baseline/signal/noise
decomposition is stored next to it (``truth/<stem>__truth.npz``) for
objective scoring of any baseline-correction result.

Usage
-----
python tools/synth_dataset.py OUTPUT_DIR [--seed 0]
"""

import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import exponnorm

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
NOISE_CASES = {'low': 0.01, 'high': 0.06}
N_CASES = [800, 2500]

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


def make_baseline(t: np.ndarray, kind: str,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Build a slowly varying baseline.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time axis.
    kind : str
        One of ``exp`` (solvent-front decay), ``hump`` (broad gradient
        hump) or ``exp_hump_drift`` (both plus a linear drift).
    rng : numpy.random.Generator
        Source of randomness for the component parameters.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline.

    """
    span = t[-1] - t[0]
    b = np.full(len(t), rng.uniform(1., 4.))
    if kind in ('exp', 'exp_hump_drift'):
        tau = rng.uniform(0.10, 0.30) * span
        b += rng.uniform(3., 12.) * np.exp(-(t - t[0]) / tau)
    if kind in ('hump', 'exp_hump_drift'):
        center = t[0] + rng.uniform(0.35, 0.70) * span
        width = rng.uniform(0.15, 0.30) * span
        b += rng.uniform(2., 8.) * np.exp(-0.5 * ((t - center) / width)**2)
    if kind == 'exp_hump_drift':
        b += rng.uniform(-2., 2.) * (t - t[0]) / span

    # Mid-frequency wander (pump, thermal and detector fluctuations),
    # which every real baseline carries and the slow components above
    # cannot represent. Its correlation length sits between the peak
    # widths and the run length, so it is what decides how flexible the
    # baseline has to be: on a signal with little analyte, capturing it
    # is the whole job.
    dt = t[1] - t[0]
    # 0.3-0.8 min of correlation: only a few times the peak widths, and
    # an order of magnitude below the run length, so it produces the ten
    # or so undulations per run that the real blanks show. A slower
    # wander would be indistinguishable from the hump above.
    corr_len = rng.uniform(0.3, 0.8) / dt
    wander = gaussian_filter1d(rng.normal(size=len(t)), corr_len)
    peak_amp = np.abs(wander).max()
    if peak_amp > 0:
        wander /= peak_amp
        swing = max(b.max() - b.min(), 1.)
        b = b + rng.uniform(0.02, 0.08) * swing * wander
    return b


def make_signal(n: int, peak_case: str, baseline_case: str,
                noise_sigma: float, rng: np.random.Generator
                ) -> tuple[np.ndarray, dict]:
    """
    Assemble one synthetic chromatogram.

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
    dt = 1. / 60.                       # one point per second, in minutes
    t = np.arange(n) * dt
    # Dead time in ABSOLUTE minutes as in the real dataset (t0 ~ 4.5
    # min regardless of run length), clamped for very short records.
    # A much earlier dead time would make the injection artifact fail
    # the width/x relevance filter of _relevant_regions, which no real
    # signal does; and an absolute t0 lets long runs reach large
    # retention (hence width) ratios, as real experiments do.
    t0 = min(rng.uniform(4.0, 5.0), 0.35 * t[-1])

    n_peaks, fwhm_pts = PEAK_CASES[peak_case]
    signal = np.zeros(n)
    peaks = []

    # Injection artifacts present in every real chromatogram: a narrow
    # solvent-front disturbance at the dead time, and for blanks a few
    # small ghost/carryover peaks along the run. Without them a blank
    # has no detectable feature at all, which real blanks never show.
    artifacts = [(t0, rng.uniform(6., 10.), rng.uniform(1., 4.))]
    if n_peaks == 0:
        for _ in range(rng.integers(1, 4)):
            artifacts.append((rng.uniform(0.2, 0.85) * t[-1],
                              rng.uniform(8., 16.),
                              rng.uniform(0.15, 0.8)))
    for tc, fwhm_a, height in artifacts:
        sigma = fwhm_a * dt / _FWHM_PER_SIGMA
        tau = rng.uniform(0.3, 0.8) * sigma
        signal += emg_peak(t, tc, sigma, tau, height)
        peaks.append({'tc': tc, 'sigma': sigma, 'tau': tau,
                      'height': height, 'fwhm_points': fwhm_a,
                      'artifact': True})

    centers = np.sort(rng.uniform(t0 + 0.02 * t[-1], 0.9 * t[-1],
                                  n_peaks))
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
    for tc in centers:
        if fwhm_pts is None:
            sigma = tc / np.sqrt(n_plates)
        else:
            fwhm = rng.uniform(*fwhm_pts) * dt
            # widths grow mildly along the run
            fwhm *= 1. + 1.5 * (tc / t[-1])
            sigma = fwhm / _FWHM_PER_SIGMA
        fwhm = sigma * _FWHM_PER_SIGMA
        tau = rng.uniform(0.3, 1.2) * sigma
        height = np.exp(rng.uniform(np.log(1.), np.log(30.)))
        signal += emg_peak(t, tc, sigma, tau, height)
        peaks.append({'tc': tc, 'sigma': sigma, 'tau': tau,
                      'height': height, 'fwhm_points': fwhm / dt,
                      'artifact': False})

    baseline = make_baseline(t, baseline_case, rng)
    noise = rng.normal(0., noise_sigma, n)
    return {'x': t, 'y': signal + baseline + noise, 'signal': signal,
            'baseline': baseline, 'noise': noise, 'peaks': peaks}


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
    args = parser.parse_args()

    sig_dir = os.path.join(args.output, 'signals')
    truth_dir = os.path.join(args.output, 'truth')
    os.makedirs(sig_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)

    manifest = []
    k = 0
    for peak_case in PEAK_CASES:
        for baseline_case in BASELINE_CASES:
            for noise_case, noise_sigma in NOISE_CASES.items():
                for n in N_CASES:
                    for rep in range(args.replicates):
                        rng = np.random.default_rng(args.seed + 7919 * k)
                        parts = make_signal(n, peak_case, baseline_case,
                                            noise_sigma, rng)
                        stem = (f'SYN__{peak_case}__{baseline_case}__'
                                f'{noise_case}__{n}__{rep}')
                        with open(os.path.join(sig_dir, f'{stem}.txt'),
                                  'w') as f:
                            for xi, yi in zip(parts['x'], parts['y']):
                                f.write(f'{xi:.6f}\t{yi:.6e}\n')
                        np.savez(
                            os.path.join(truth_dir, f'{stem}__truth.npz'),
                            x=parts['x'], y=parts['y'],
                            signal=parts['signal'],
                            baseline=parts['baseline'],
                            noise=parts['noise'],
                            peaks=json.dumps(parts['peaks']))
                        manifest.append(
                            f'{stem},{peak_case},{baseline_case},'
                            f'{noise_case},{n},{rep},'
                            f'{len(parts["peaks"])}')
                        k += 1
    with open(os.path.join(args.output, 'manifest.csv'), 'w') as f:
        f.write('stem,peak_case,baseline_case,noise_case,n_points,'
                'replicate,n_peaks\n')
        f.write('\n'.join(manifest) + '\n')
    print(f'{k} signals -> {args.output}')


if __name__ == '__main__':
    main()
