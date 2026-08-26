# coding: utf-8
"""
Baseline correction, and the automatic choice of the BEADS cutoff.

Wraps the BEADS algorithm of Ning et al. (2014), on the whole signal
(``_beads``) or through the customized-baseline wrapper of Liland et al.
(2011) with per-region stiffness (``_custom_beads``). ``auto_beads`` is
the entry point; when no cutoff is given it sweeps the autocorrelation
of the baseline-corrected signal and hands the curve to
``weaselytics.segmentation`` to pick one.

The selection method is documented in ``tools/fcut/segmentation.md``.

References
----------
.. [1] Ning, X., Selesnick, I. W. and Duval, L. Chromatogram baseline
   estimation and denoising using sparsity (BEADS). Chemom. Intell.
   Lab. Syst. 139, 156-167 (2014). doi:10.1016/j.chemolab.2014.09.014
.. [2] Navarro-Huerta, J. A. et al. Assisted baseline subtraction in
   complex chromatograms using the BEADS algorithm. J. Chromatogr. A
   1507, 1-10 (2017). doi:10.1016/j.chroma.2017.05.057
.. [3] Liland, K. H. et al. Customized baseline correction. Chemom.
   Intell. Lab. Syst. 109(1), 51-56 (2011).
   doi:10.1016/j.chemolab.2011.07.005
.. [4] Bauer, F. and Kindermann, S. The quasi-optimality criterion for
   classical inverse problems. Inverse Probl. 24(3), 035002 (2008).
   doi:10.1088/0266-5611/24/3/035002
.. [5] MacDougall, D. et al. Guidelines for data acquisition and data
   quality evaluation in environmental chemistry. Anal. Chem. 52(14),
   2242-2249 (1980). doi:10.1021/ac50064a004
"""
import hashlib
import logging
import os
import time  #@EB temporary?
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pybaselines import Baseline
from scipy.ndimage import gaussian_filter1d

from weaselytics.plot import r2_plots
from weaselytics.segmentation import (
    classify_segments,
    detect_dips,
    dip_curve,
    dips_to_mask,
    pelt_linear,
    segment_features,
    select_center,
    trim_plateaus,
)
from weaselytics.utils import (
    _durbin_watson,
    end_window,
    merge_intervals,
    peaks_params,
)

logger = logging.getLogger(__name__)

# Quantity the autocorrelation of `_r2` is computed on. Part of the
# cache key: bump it whenever the definition changes, so that curves
# cached under the previous one are recomputed rather than reused.
_R2_CHANNEL = "y-baseline-dw"

# The cache also stores the baseline-sensitivity curve (see
# `_sensitivity_curve`). Bump this token whenever the sensitivity
# definition or the cache contents change, so old caches are recomputed
# rather than served silently. Bumped to "sens-1" on 2026-07-26 when the
# curve and its stored array were renamed from `stability`: the values
# are unchanged, but the npz key is not, so pre-rename caches must miss.
_SENSITIVITY_VERSION = "sens-1"


def _relevant_regions(
    s: np.ndarray, x: np.ndarray, tol: float = 6.,
    smooth_sigma: float = 3.
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """
    Locate the peaks and the useful extent of the signal.

    Finds the peaks in a lightly smoothed copy, discards those too wide
    relative to their position to be analyte rather than baseline
    structure, readmits very tall wide ones through a hard-coded
    exception, brackets every survivor but the narrowest with a window
    reaching 0.85 of that peak's FWHM on each side of its apex, and
    merges brackets that overlap. The 0.85 follows from the peak shape:
    a region has to cover the peak it brackets, for a Gaussian about 95%
    of the area lies within ``+-2 sigma``, so the region spans
    ``4 sigma``; since ``FWHM = 2.355 sigma`` that is 1.70 FWHM in total,
    and half of it, 0.85 FWHM, is what each side gets. The regions and
    their decimation factors control the per-region stiffness of
    ``_custom_beads``; `scut` truncates the signal past the last peak so
    the autocorrelation is not diluted by an empty tail.

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    tol : float, optional
        Largest peak width, per unit of `x`, still counted as analyte. A
        wider feature is treated as baseline structure and ignored,
        unless the exception in the body readmits it. Default is 6.
    smooth_sigma : float, optional
        Standard deviation, in points, of the Gaussian applied to the
        copy of `s` that peaks are detected on. It sets the scale below
        which structure is treated as noise rather than as a peak.
        Default is 3.

    Returns
    -------
    peak_regions : array-like, shape (M,2), or None
        Start and stop indices of each region holding a relevant peak,
        as ``data[start:stop]``. The final stop is clamped below
        ``len(s)``, so the last sample is never inside a region.
        `None` means no relevant peaks were found.
    sampling : array-like of shape (M,)
        Decimation factor for each region: one point in every
        `sampling` is kept when the baseline is fitted, which stiffens
        the fit there. ``[1]`` when there are no regions, which keeps
        every point and makes ``_custom_beads`` equivalent to
        ``_beads``.
    scut : int
        Exclusive bound of the useful extent: the sweep runs on
        ``s[:scut]``. Equals ``len(s)`` when the last peak leaves no
        room after it.

    Notes
    -----
    Negative peaks are gated on prominence, measured from the local
    level, so the detection does not depend on where the baseline sits.
    Fails on a signal carrying no genuine negative peak: the bar is
    then the noise floor, and the deepest noise dip is admitted. See
    `peaks_params`, where the gate lives.

    `smooth_sigma` is load-bearing and nothing fixes its value. It sets
    the scale below which structure counts as noise rather than as a
    peak. It is a standard deviation and not a window length, and
    scipy's default truncation carries the kernel four sigma either
    side of centre.

    What it costs is paid on the peaks it keeps. The smoothed apex is
    attenuated, and a peak of full width ``w`` is measured at
    ``sqrt(w**2 + (2.355 * smooth_sigma)**2)``, so a peak only a few
    points wide comes out substantially broader. Those widths are what
    the relevance filter and the decimation factors are computed from.

    **Reducing it towards a spike-removal width does not clean the
    signal, it stops cleaning it.** The detector then reads the noise
    floor as peaks, and since `scut` follows the last admitted feature,
    a noise bump near the tail extends the swept region.

    Widths arrive from `peaks_params` at half prominence, which is
    scipy's own default and is not overridden here, so every width in
    this chain is a half-prominence width: what `tol` filters on, what
    the 0.85 buffer is built from, and what sets the decimation.

    The acetonitrile exception is two numbers written for one sample
    and carries a ``# TODO`` in the code.

    """
    # NOTE: A weak smoothing helps to avoid peak detection in noisy region of
    #       the signal by:
    #           1) removing most of the spurious features in the raw signal
    #           2) sligntly enlarging features relevant for peaks detection
    #       `smooth_sigma` is a standard deviation, not a window: scipy
    #       truncates at 4 sigma, so the default spans 25 points.
    z = gaussian_filter1d(s, smooth_sigma)
    # A feature carrying taller peaks is structure, not a peak: the
    # peaks on it are what define the region.
    peaks, widths = peaks_params(z, width=3, rel_prom_p=0.01,
                                   adapt=True, drop_enclosing=True)

    # TODO: Find a way to make this part of the code more robust.
    width_per_x = widths/x[peaks]
    # In case of very tall and large peaks (see acetonitrile)
    exception = ((s[peaks] > 20) & (width_per_x < 11))
    # Signal splitting
    rel_peaks = peaks[((width_per_x < tol) | exception)]
    rel_widths = widths[((width_per_x < tol) | exception)]
    # No relevant peak at all (featureless signal, or every detected
    # peak rejected by the relevance filter): fall back to the
    # documented degraded mode (no peak regions, uniform sampling and
    # no truncation) instead of crashing on the empty array below.
    if len(rel_peaks) == 0:
        return None, np.array([1]), len(s)
    ratio_w = rel_widths/np.min(rel_widths)

    # Peak full width
    # NOTE: Assuming that `rel_widths` is the FWHM and that the peak is
    #       gaussian, `buffer` is equal to half of the full peak width.
    #       For a Gaussian: FWHM ≈ 2.355σ. To capture ~95 % of the peak
    #       area we need ±2σ from the center. That's 4σ total =
    #       4 / 2.355 ≈ 1.7 × FWHM, so each side gets 0.85 × FWHM.
    _FWHM_TO_HALF_PEAK = 0.85
    buffer = np.ceil(_FWHM_TO_HALF_PEAK * rel_widths).astype(int)
    left_lim = rel_peaks - buffer
    right_lim = rel_peaks + buffer
    full_widths = np.array([left_lim,right_lim]).T

    # Peak regions and sampling
    large_peaks = full_widths[ratio_w > 1]      # Ignore the narrowest peak
    peak_regions = merge_intervals(large_peaks)
    if len(peak_regions) == 0:
        peak_regions = None
        sampling = np.array([1])
    else:
        # Because values in regions must be less than len(data)
        if peak_regions[-1,-1] >= len(s):
            peak_regions[-1,-1] = len(s) - 1
        # The region width covers 2×buffer per peak (= 2 × 0.85 × FWHM),
        # so dividing by 2 converts it to buffer units. Combined with the
        # division by rel_widths (FWHM), an isolated peak yields sampling≈1
        # (keep all points). Without the /2, sampling would be ≈2, making
        # the baseline unnecessarily stiff for isolated peaks.
        sampling = np.ceil((peak_regions[:,1]-peak_regions[:,0])/2
                           /np.min(rel_widths)).astype(int)

    # Signal cutoff
    # TODO: Maybe it would be a good idea to cut the less relevant starting
    #       segment too.
    arg_last_peak = rel_peaks.argmax()
    pos_last_peak = rel_peaks[arg_last_peak]
    buffer_last_peak = int(np.ceil(2*rel_widths)[arg_last_peak])
    pos_max = pos_last_peak + buffer_last_peak
    if len(s) > pos_max:
        scut = pos_max
    else:
        scut = len(s)
    return peak_regions, sampling, scut

def _snr(s: np.ndarray) -> float:
    """
    Robust signal-to-noise ratio of a chromatogram.

    The tallest excursion above the median divided by the noise, with
    the noise estimated from the median absolute deviation of the
    consecutive differences, so peaks do not inflate it.

    ``auto_beads`` uses it for one decision: whether the optimum may lie
    past the collapse of the r2 curve, which is allowed on a weak signal
    and not on one carrying substantial analyte.

    Parameters
    ----------
    s : array-like, shape (N,)
        The measured chromatogram.

    Returns
    -------
    snr : float
        The signal-to-noise ratio; ``inf`` when the noise estimate is
        zero, which happens when over half the consecutive differences
        are identical.

    Notes
    -----
    The two constants in the noise estimate follow from the Gaussian.
    ``1.4826`` is the reciprocal of the 0.75 quantile of the standard
    normal, which converts a median absolute deviation into a standard
    deviation. The ``sqrt(2)`` removes the inflation from differencing:
    the difference of two independent samples of variance ``v`` has
    variance ``2v``.

    The numerator is the tallest excursion **above** the median, so a
    signal whose only features are negative scores near zero however
    strong they are.

    Fails as a ratio on quantisation-limited data. When the detector
    digitises in fixed steps, every consecutive difference is a multiple
    of that step, the denominator collapses onto a handful of values,
    and the statistic reduces to the tallest excursion in the units of
    the signal. It is then an absolute amplitude wearing a dimensionless
    name, and a threshold on it is instrument-specific.

    Whether the threshold separates anything has to be checked on the
    data at hand: a population whose values all sit far above it makes
    the gate a constant. See the README TO DO.

    """
    diffs = np.diff(s)
    noise = 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2)
    if noise <= 0:
        return np.inf
    return float((np.max(s) - np.median(s)) / noise)

def _log_transform(s: np.ndarray, epsilon: float = 1) -> np.ndarray:
    """
    Log transformation applied before the BEADS fit.

    ``log10(s - min(s) + epsilon)``, Navarro-Huerta et al. [1] Eq. (8).
    Their §3.3.2 introduces it to compress the dynamic range: a
    chromatogram with peaks of very different height makes BEADS return
    a baseline that ripples under the tall ones, and the transform
    removes those ripples.

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal to transform.
    epsilon : float, optional
        An arbitrary positive offset. The larger the offset, the less
        aggressive the pre-treatment. Default is 1 (see Notes).

    Returns
    -------
    log_s : numpy.ndarray
        The log transformed data.

    Notes
    -----
    Navarro-Huerta et al. [1] §3.3.2 give two reasons for
    ``epsilon = 1``: it suits signals whose maxima reach 500 to 10,000,
    and it sends the minimum of the signal to zero, since
    ``log10(1) = 0``. They apply it unchanged to every chromatogram
    they show, from the 60 A.U. of their Fig. 8 to the 2100 of their
    Fig. 6c.

    The offset behaves the same way at every scale. The local gain of
    the transform is ``1 / ((u + epsilon) ln 10)`` with
    ``u = s - min(s)``, so a signal of span ``S`` is compressed by
    ``(S + epsilon) / epsilon`` between its bottom and its top. Tall
    peaks are therefore compressed far more than small ones, and the
    ratio shrinks with the span, so a short signal is treated more
    gently than a tall one.

    ``epsilon`` has to be positive: at zero the minimum of the signal
    maps to negative infinity.

    Under the auto-scaled ``lam_d`` the penalty terms are
    scale-invariant while the data-fidelity term is not, so the base of
    the logarithm sets the balance between them.

    References
    ----------
    [1] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex
        chromatograms using the BEADS algorithm. Journal of Chromatography A,
        2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057.

    """
    log_s = np.log10(s - np.min(s) + epsilon)
    return log_s

def _beads(baseline_fitter: Baseline, s: np.ndarray,
           freq_cutoff: float = 0.005, asymmetry: float = 1.0,
           fit_parabola: bool = True, alpha: float = 1.0,
           parabola_len: int = 3, **kwargs
           ) -> tuple[np.ndarray, dict]:
    r"""
    Wrap ``pybaselines.Baseline.beads``.

    Parameters
    ----------
    baseline_fitter : pybaselines.Baseline
        The fitter, already built on the x-values of the signal.
    s : array-like, shape (N,)
        The signal to fit.
    freq_cutoff : float, optional
        Boundary between the baseline and the peaks plus noise, in
        cycles per sample. BEADS is a high-pass filter, so what falls
        below `freq_cutoff` becomes the baseline and what passes
        becomes the sparse signal; raising it lets the baseline follow
        faster structure. Default is 0.005, pybaselines' own, and in
        production this is the value `auto_beads` selects rather than
        one taken from here.
    asymmetry : float, optional
        Price of a negative excursion against a positive one in the
        BEADS cost. Default is 1.0, the one deliberate departure from
        pybaselines, which ships 6.0: these signals carry a genuine
        negative peak in the dead-time artefact, so the two directions
        are priced alike.
    fit_parabola : bool, optional
        Subtract a parabola through the endpoints before fitting, so
        the signal starts and ends near zero. Default is True,
        pybaselines' own.
    alpha : float, optional
        Scales all three sparsity penalties through
        ``lam_d = alpha / ||z^(d)||_1``. Default is 1.0, pybaselines'
        own; the form is Ning et al. (2014) §5.1.
    parabola_len : int, optional
        Window at each end used to reject an outlying endpoint before
        the parabola is built. Default is 3, pybaselines' own.
    **kwargs
        Accepted and discarded. Nothing here reaches ``beads``.

    Returns
    -------
    bl : numpy.ndarray, shape (N,)
        The fitted baseline.
    params : dict
        The parameter dictionary pybaselines returns. Its ``signal``
        key holds the denoised reconstruction.

    See Also
    --------
    auto_beads : selects `freq_cutoff` instead of taking it.

    """
    bl, params = baseline_fitter.beads(
            s,
            freq_cutoff=freq_cutoff,
            fit_parabola=fit_parabola,
            asymmetry=asymmetry,
            alpha=alpha,
            parabola_len=parabola_len
            )
    return bl, params

def _custom_beads(baseline_fitter: Baseline, s: np.ndarray,
                  regions: tuple | np.ndarray | None = None,
                  sampling: int | np.ndarray | None = None,
                  freq_cutoff: float = 0.005, asymmetry: float = 1.0,
                  fit_parabola: bool = True, alpha: float = 1.0,
                  parabola_len: int = 3, **kwargs
                  ) -> tuple[np.ndarray, dict]:
    """Customized BEADS with per-region stiffness control.

    Extends `_beads` by splitting the signal into regions, each fitted
    on a decimated subset of its points, which stiffens the baseline
    there. The decimated fit is interpolated back to full resolution by
    pybaselines' ``custom_bc``.

    Parameters
    ----------
    baseline_fitter : pybaselines.Baseline
        The fitter, already built on the x-values of the signal.
    s : array-like, shape (N,)
        The signal to fit.
    regions : array-like, shape (M,2), optional
        Start and stop indices for each region containing a relevant
        peak, as ``data[start:stop]``. If `None` (default), uses all
        points.
    sampling : int or array-like, optional
        Decimation step for each region in `regions`: one point in
        every `sampling` is kept when the baseline is fitted. Default
        1, which keeps every point.
    freq_cutoff, asymmetry, fit_parabola, alpha, parabola_len : optional
        The BEADS settings, passed through unchanged. See `_beads` for
        what each controls and where its default comes from.
    **kwargs
        Accepted and discarded. Nothing here reaches ``custom_bc``.

    Returns
    -------
    bl : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        The BEADS parameter dictionary, with ``signal`` rebuilt here;
        see Notes.

    Notes
    -----
    ``custom_bc`` returns only a baseline, so ``params["signal"]`` is
    rebuilt here as ``s - bl - noise``, with the noise interpolated from
    the reduced grid. Since points are dropped only inside `regions`,
    that noise estimate is exact outside them and a straight line
    through bin averages within them. ``params["signal"]`` is therefore
    a denoised signal whose region interiors keep their point-to-point
    noise, and it equals neither a uniformly denoised residual nor
    ``s - bl``.

    On raw signals the gap between the two is the BEADS noise, largest
    where there are no regions and so no binning at all. Choosing
    between them is a choice about whether to show the noise floor.

    The two paths give the identical baseline whenever `sampling` is all
    ones, which holds when every peak is within about 1.18 times the
    width of the narrowest. See ``_relevant_regions``.

    """
    if regions is None:
        regions = ((None, None),)
    if sampling is None:
        sampling = 1

    beads_kwargs = {'freq_cutoff': freq_cutoff,
                    'fit_parabola': fit_parabola,
                    'asymmetry': asymmetry,
                    'alpha': alpha,
                    'parabola_len': parabola_len
                    }

    bl, params = baseline_fitter.custom_bc(
            s,
            method="beads",
            regions=regions,
            sampling=sampling,
            lam=None,
            method_kwargs=beads_kwargs
            )

    noise_fit = (params['y_fit'] - params['baseline_fit']
                 - params['method_params']['signal'])
    params['noise'] = np.interp(baseline_fitter.x, params['x_fit'], noise_fit)
    params['signal'] = s - bl - params['noise']
    return bl, params

def _r2(
    algo: Callable[..., tuple[np.ndarray, dict]],
    baseline_fitter: Baseline, y: np.ndarray,
    p: float, param: str = "freq_cutoff", **kwargs
) -> float:
    """
    Autocorrelation of the baseline-corrected signal at one parameter.

    Fits a baseline with `algo` at ``param = p`` and returns
    ``r2 = ((2 - DW)/2)**2``, with `DW` the Durbin-Watson statistic of
    ``y - b``. Sweeping `p` over a grid gives the curve the cutoff is
    selected from.

    Parameters
    ----------
    algo : Callable
       The callable method corresponding to the input string.
    baseline_fitter : `Baseline` object
        Contains the x-values of the signal to baseline correct and all
        available baseline correction algorithms in pybaselines.
    y : array-like, shape (N,)
        The y-values of the signal.
    p : float
        Value of `param` at which r2 is evaluated.
    param : str, optional
        Label of the parameter to correlate with the value of r2. Default is
        "freq_cutoff".
    **kwargs
        Forwarded to `algo`, with `param` set to `p` among them.

    Returns
    -------
    r2 : float
        The autocorrelation of the baseline corrected signal for `param`=`p`.

    Notes
    -----
    The statistic runs on ``y - b``, the sparse chromatogram plus the
    noise, which is Navarro-Huerta et al. [1] Eq. (13), the quantity
    they monitor in §3.4. The subtraction happens in the log domain,
    since `y` arrives log-transformed; their Eq. (12) back-transforms
    the corrected signal to the original scale for the output. They
    read the cutoff off the log-domain plot by eye, which fixes their
    domain without testing it, so whether the statistic behaves
    differently on the back-transformed signal is unverified. The noise
    has to stay in. ``r2`` is a whiteness
    test, and the plateau structure exists because a good cutoff leaves
    the correlated peaks in the corrected signal while an excessive one
    leaves white noise behind, so the noise is the floor the drop is
    measured against.

    Taking the baseline as the only output used here also makes the
    quantity identical on the ``beads`` and ``custom_beads`` paths, so
    their curves can be compared.

    The statistic is a sum of squared differences between neighbouring
    points, so it is dominated by any small patch of un-subtracted noise
    once the sparse component is small. That is why the denoised
    ``params["signal"]`` is unusable here: on the ``custom_beads`` path
    its noise term is interpolated from the reduced grid, leaving raw
    point-to-point noise on the region interiors.

    References
    ----------
    [1] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in
        complex chromatograms using the BEADS algorithm. Journal of
        Chromatography A, 2017, 1507, 1-10, §3.4 Eq. (13).

    """
    r2, _, _ = _r2_and_baseline(algo, baseline_fitter, y, p, param=param,
                                **kwargs)
    return r2


def _r2_and_baseline(
    algo: Callable[..., tuple[np.ndarray, dict]],
    baseline_fitter: Baseline, y: np.ndarray,
    p: float, param: str = "freq_cutoff", **kwargs
) -> tuple[float, np.ndarray]:
    """
    Evaluate `_r2` and also return the fitted baseline.

    The baseline is computed anyway to form ``y - baseline``; returning
    it lets the sweep measure how much it moves between adjacent
    parameter values (see `_sensitivity_curve`) at no extra fit.

    Parameters
    ----------
    algo : Callable
        ``_beads`` or ``_custom_beads``.
    baseline_fitter : `Baseline` object
        Carries the x-values and the pybaselines algorithms.
    y : array-like, shape (N,)
        The y-values of the signal, already log-transformed and
        truncated.
    p : float
        Value of `param` at which to fit.
    param : str, optional
        Name of the swept parameter. Default is "freq_cutoff".
    **kwargs
        Passed through to `algo`.

    Returns
    -------
    r2 : float
        Autocorrelation of ``y - baseline``.
    dw : float
        The Durbin-Watson statistic of ``y - baseline``. Returned
        because ``r2`` is a square and loses the sign of the
        correlation: ``r = 1 - dw/2``, and ``dw > 2`` marks a residual
        that alternates sign point to point, which is the baseline
        tracking the noise rather than the fit improving.
    baseline : numpy.ndarray, shape (N,)
        The fitted baseline.

    Notes
    -----
    The autocorrelation is taken on ``y - baseline``, which is not
    centred on zero, so it carries the assumption `_durbin_watson`
    documents: an offset between the fit and the noise reads as
    correlated structure.

    """
    kwargs[param] = p
    baseline, _ = algo(baseline_fitter, y, **kwargs)
    resid = y - baseline
    dw = _durbin_watson(resid)
    return ((2.0 - dw) ** 2) / 4.0, dw, baseline


def _sensitivity_curve(steps: np.ndarray, param_range: np.ndarray,
                     signal_range: float) -> np.ndarray:
    """
    Baseline-sensitivity curve from the step-to-step baseline changes.

    ``steps[i]`` is the rms change of the baseline from ``param_range``
    ``[i-1]`` to ``[i]`` (``steps[0] = 0``). Dividing by the signal
    range and by the log-frequency spacing makes it scale-free: the rms
    baseline change per decade of cutoff frequency, relative to the
    signal. It runs large and erratic where the fit is unstable, which
    Navarro-Huerta et al. (2017) §3.1(iv) report at low frequencies, and
    settles where the fit becomes reliable.

    Parameters
    ----------
    steps : array-like, shape (M,)
        Rms baseline change between adjacent grid points, ``steps[0]``
        being 0.
    param_range : array-like, shape (M,)
        The swept parameter values, geometrically spaced.
    signal_range : float
        Peak-to-peak range of the signal, used to make `steps`
        dimensionless.

    Returns
    -------
    sensitivity : numpy.ndarray, shape (M,)
        Rms baseline change per decade, relative to the signal range.

    Notes
    -----
    This is the quasi-optimality functional of Bauer and Kindermann
    [1]_. Their Definition 1.1 Eq. (6) minimises
    ``||x_n - x_{n+1}||``, which is `steps`; their continuous Eq. (7)
    minimises ``||a dx/da||``, the change per unit log-parameter, which
    is what dividing by `dlog` produces. They note that discretising
    (7) on a uniform grid returns (6), and this grid is uniform in log,
    so the two agree up to the constant spacing. Dividing further by
    `signal_range` is a positive per-signal constant, so an argmin over
    this curve is an argmin of their criterion.

    **``sensitivity[0]`` is zero by construction**, the first grid
    point having no predecessor, so it is the global minimum of the
    curve whatever the signal does. An argmin over the whole support
    lands there and must be restricted to the range of interest.

    `signal_range` and the log spacing are both floored at 1e-12, so a
    constant signal gives a curve of zeros.

    References
    ----------
    .. [1] Bauer, F. and Kindermann, S. The quasi-optimality criterion
       for classical inverse problems. *Inverse Problems* **24**,
       035002 (2008), Definition 1.1.

    """
    S = np.zeros_like(steps, dtype=float)
    denom = max(float(signal_range), 1e-12)
    dlog = np.abs(np.diff(np.log10(param_range)))
    S[1:] = steps[1:] / denom / np.maximum(dlog, 1e-12)
    return S


def _r2_chunk(args: tuple) -> tuple[list[float], np.ndarray,
                                    np.ndarray, np.ndarray]:
    """
    Evaluate `_r2` on a chunk of parameter values (worker helper).

    Module-level function so that it can be pickled by the process pool
    used in ``_r2_array``.

    Parameters
    ----------
    args : tuple
        The payload ``(algo, baseline_fitter, signal, chunk, param,
        kwargs)`` where `chunk` is the sub-array of parameter values to
        evaluate; the other elements are as in ``_r2``.

    Returns
    -------
    r2_values : list of float
        The r2 value for each parameter in `chunk`, in order.
    dw_values : list of float
        The Durbin-Watson statistic for each parameter in `chunk`, which
        carries the sign `r2` squares away.
    steps : numpy.ndarray, shape (len(chunk),)
        The rms baseline change between consecutive parameters *within*
        the chunk. ``steps[0]`` is 0 (the seam to the previous chunk is
        stitched by the caller from the edge baselines below).
    first_baseline, last_baseline : numpy.ndarray, shape (N,)
        The baselines at the two ends of the chunk, so the caller can
        compute the step across the chunk seam.

    Notes
    -----
    `steps` is the only output that is not a per-point quantity: it
    depends on the neighbour, which is why the two edge baselines are
    returned and why the caller stitches each seam. Splitting the range
    and stitching it back gives the same `steps` array as sweeping it
    whole, so the worker count does not enter the swept curves and need
    not enter the cache key.

    A chunk of one parameter yields ``steps == [0]`` with identical
    first and last baselines, so its whole contribution comes from the
    seams the caller builds.

    """
    algo, baseline_fitter, signal, chunk, param, kwargs = args
    r2_values, dw_values = [], []
    prev = None
    steps = np.zeros(len(chunk))
    first = last = None
    for i, p in enumerate(chunk):
        r2, dw, b = _r2_and_baseline(algo, baseline_fitter, signal, p,
                                     param=param, **kwargs)
        r2_values.append(r2)
        dw_values.append(dw)
        if i == 0:
            first = b
        else:
            steps[i] = np.sqrt(np.mean((b - prev) ** 2))
        prev = b
    last = prev
    return r2_values, dw_values, steps, first, last

def _r2_array(
    algo: Callable[..., tuple[np.ndarray, dict]],
    baseline_fitter: Baseline,
    signal: np.ndarray, param_range: np.ndarray,
    param: str = "freq_cutoff", workers: int = 1,
    return_sensitivity: bool = False, return_dw: bool = False, **kwargs
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Sweep `r2` over a range of values of one algorithm parameter.

    Evaluates `_r2` at every point of `param_range`, holding everything
    else fixed. `param` names the keyword that is varied; in production
    it is the BEADS cutoff frequency and the resulting curve is what
    the cutoff is selected from.

    The M evaluations are independent and the sweep dominates the
    runtime of anything built on it, so they can be distributed over a
    pool of worker processes. The fits hold the GIL, so the pool is of
    processes. The parameter range is split into contiguous chunks,
    eight per worker, keeping the inter-process overhead small next to
    the fits.

    The baseline-sensitivity curve (`_sensitivity_curve`) is computed
    alongside r2 from the same fits, at no extra baseline fit; it is
    returned only when `return_sensitivity` is set.

    Parameters
    ----------
    algo : Callable
        The baseline algorithm to sweep, ``_beads`` or
        ``_custom_beads``.
    baseline_fitter : `Baseline` object
        Contains the x-values of the signal to baseline correct and all
        available baseline correction algorithms in pybaselines.
    signal : array-like, shape (N,)
        The y-values of the signal.
    param_range : array-like, shape (M,)
        Range of values taken by `param` and at which r2 is evaluated.
    param : str, optional
        Label of the parameter to correlate with the value of r2. Default is
        "freq_cutoff".
    workers : int, optional
        Number of worker processes used to parallelize the sweep.
        Default is 1, which keeps the evaluation serial in the current
        process.
    return_sensitivity : bool, optional
        If True, append the baseline-sensitivity curve to the result.
        Default False.
    return_dw : bool, optional
        If True, append the Durbin-Watson array to the result. Default
        False.
    **kwargs
        Forwarded to `algo` at every point of the sweep.

    Returns
    -------
    result : numpy.ndarray or tuple of numpy.ndarray
        The array of r2 alone when both flags are False, otherwise a
        tuple holding it followed by the Durbin-Watson array if
        `return_dw` and the sensitivity curve if `return_sensitivity`,
        in that order. All have shape (M,).

    Notes
    -----
    Chunks are contiguous slices, so every step inside a chunk is exact
    and only the seams are stitched, from the edge baselines each
    worker returns. The result is independent of `workers`.

    Each worker receives the signal and the fitter by pickling, so
    memory grows with the pool size while the work per point stays the
    same. The eight chunks per worker balance fits whose cost varies
    across the range; the number itself is a convention.

    `signal_range` is the peak-to-peak of `signal` and is floored at
    1e-12 inside `_sensitivity_curve`, so a constant signal yields a
    sensitivity curve of zeros.

    """
    signal_range = float(np.max(signal) - np.min(signal))
    vdw = np.empty(len(param_range))
    if workers is None or workers <= 1:
        vr2 = np.empty(len(param_range))
        steps = np.zeros(len(param_range))
        prev = None
        for i, p in enumerate(param_range):
            vr2[i], vdw[i], b = _r2_and_baseline(
                algo, baseline_fitter, signal, p, param=param, **kwargs)
            if prev is not None:
                steps[i] = np.sqrt(np.mean((b - prev) ** 2))
            prev = b
    else:
        # Several chunks per worker to balance the varying cost of the fits.
        # Chunks are contiguous slices, so the step within a chunk is exact
        # and only the chunk seams need stitching from the edge baselines.
        n_chunks = min(len(param_range), workers * 8)
        chunks = [c for c in np.array_split(param_range, n_chunks) if len(c)]
        payloads = [
            (algo, baseline_fitter, signal, chunk, param, kwargs)
            for chunk in chunks
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_r2_chunk, payloads))
        vr2 = np.array([r2 for r2s, _, _, _, _ in results for r2 in r2s])
        vdw = np.array([dw for _, dws, _, _, _ in results for dw in dws])
        steps = np.zeros(len(param_range))
        i = 0
        prev_last = None
        for r2s, _, chunk_steps, first, last in results:
            m = len(r2s)
            if prev_last is not None:            # seam to previous chunk
                steps[i] = np.sqrt(np.mean((first - prev_last) ** 2))
            steps[i + 1:i + m] = chunk_steps[1:]
            prev_last = last
            i += m
    out = (vr2,)
    if return_dw:
        out += (vdw,)
    if return_sensitivity:
        out += (_sensitivity_curve(steps, param_range, signal_range),)
    return out[0] if len(out) == 1 else out

def _r2_cache_key(algo: Callable[..., tuple[np.ndarray, dict]],
                  signal: np.ndarray, param_range: np.ndarray,
                  param: str, kwargs: dict) -> str:
    """
    Compute the cache key identifying an autocorrelation curve.

    The key is a hash of every input that determines the curve: the
    (log-transformed, truncated) signal values, the parameter range,
    the baseline algorithm, the correlated parameter and the keyword
    arguments passed to the algorithm. Array-valued keyword arguments
    (e.g. `regions` and `sampling` for ``_custom_beads``) are hashed
    from their raw bytes.

    Parameters
    ----------
    algo : Callable
        The callable method used for the baseline correction.
    signal : array-like, shape (N,)
        The y-values of the signal.
    param_range : array-like, shape (M,)
        Range of values taken by `param`.
    param : str
        Label of the parameter to correlate with the value of r2.
    kwargs : dict
        Additional keyword arguments passed to `algo`.

    Returns
    -------
    key : str
        The hexadecimal digest identifying the curve.

    Notes
    -----
    Two hand-bumped tokens are hashed alongside the inputs so that a
    change in what the curve *means* invalidates it: `_R2_CHANNEL` for
    the quantity `_r2` correlates, and `_SENSITIVITY_VERSION` for the
    sensitivity array stored beside it.

    **No library version enters the key.** The curve depends on the
    pybaselines build that fits the baselines and on the numpy and
    scipy beneath it, none of which is hashed, so an entry computed
    under one build is served unchanged under another.

    A key is a statement that two curves are the same, so it can only
    be as trustworthy as the inputs it names. Anything that changes the
    curve without changing a hashed input is invisible to it.

    """
    sha = hashlib.sha1()
    # Identifies the quantity `_r2` correlates, not just its inputs.
    # Changing the channel changes every curve while leaving the inputs
    # untouched, so without this token the cache would keep serving
    # curves computed on the previous definition. Bump it whenever the
    # definition of `y_corr` in `_r2` changes.
    sha.update(_R2_CHANNEL.encode())
    sha.update(_SENSITIVITY_VERSION.encode())
    # Both float arrays are hashed at reduced precision, NOT from their
    # float64 bytes, because neither is bit-reproducible across numpy
    # versions and platforms:
    #
    # - `np.geomspace`: between this machine and the cluster, 48 of the
    #   1000 grid values differed in their last bit (max relative
    #   difference 2.2e-16, one ulp);
    # - the signal, which reaches this function as
    #   `log10(s - min(s) + eps)`. The subtraction is exact, but
    #   `np.log10` is not identical across libm implementations.
    #
    # Either was enough to change the digest completely, so a cache
    # filled on one machine was never reused on the other and the
    # sweep, which is 99.9% of the cost of the selection, was silently paid
    # twice. Quantising only the grid (the first attempt) left the
    # signal free to break the key on its own.
    #
    # float32 has ~7 significant digits, far coarser than the ulp noise
    # and far finer than any distinction that matters here: adjacent
    # points of the production grid differ by ~1.1%, and two
    # chromatograms identical to 7 significant digits at every point
    # are the same measurement. The lengths are hashed explicitly so a
    # different `scut` can never collide. The residual risk is a value
    # sitting within 1 ulp of a float32 rounding boundary (~4e-9 per
    # value), and its consequence is a cache miss and a recomputation,
    # never a wrong curve.
    sha.update(str(len(signal)).encode())
    sha.update(np.ascontiguousarray(signal, dtype=np.float32).tobytes())
    sha.update(str(len(param_range)).encode())
    sha.update(np.ascontiguousarray(param_range, dtype=np.float32).tobytes())
    sha.update(algo.__name__.encode())
    sha.update(param.encode())
    for name in sorted(kwargs):
        value = kwargs[name]
        sha.update(name.encode())
        if isinstance(value, np.ndarray):
            sha.update(str(value.shape).encode())
            sha.update(np.ascontiguousarray(value).tobytes())
        else:
            sha.update(repr(value).encode())
    key = sha.hexdigest()[:12]
    return key

def _r2_array_cached(
    algo: Callable[..., tuple[np.ndarray, dict]],
    baseline_fitter: Baseline,
    signal: np.ndarray, param_range: np.ndarray,
    param: str = "freq_cutoff", cache_dir: str | None = None,
    path: str = "./file.txt", workers: int = 1,
    return_sensitivity: bool = False, return_dw: bool = False, **kwargs
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Compute the array of `r2` with an optional on-disk cache.

    The sweep is by far the most expensive step of anything built on
    the curve (seconds to minutes per signal), while the downstream
    plateau detection runs in milliseconds. Caching the curve to a
    ``.npz`` file therefore allows iterating on the plateau detection
    over a whole dataset in seconds instead of recomputing every BEADS
    sweep. The cache file name
    combines the stem of `path` with a hash of every input that
    determines the curve, so modified inputs miss.

    **A miss deletes every existing entry of the same stem before the
    new curve is written.** That keeps the directory at one ``.npz`` of
    roughly 16 kB per signal, and it means a run pointed at an existing
    cache destroys the curves it is replacing before the replacements
    are known good. Regeneration writes to a new directory.

    Parameters
    ----------
    algo : Callable
        The baseline algorithm to sweep, ``_beads`` or
        ``_custom_beads``.
    baseline_fitter : `Baseline` object
        Contains the x-values of the signal to baseline correct and all
        available baseline correction algorithms in pybaselines.
    signal : array-like, shape (N,)
        The y-values of the signal.
    param_range : array-like, shape (M,)
        Range of values taken by `param` and at which r2 is evaluated.
    param : str, optional
        Label of the parameter to correlate with the value of r2.
        Default is "freq_cutoff".
    cache_dir : str, optional
        Directory where the curves are cached. Default is None, which
        skips the cache and sweeps every call.
    path : str, optional
        Path of the data file, used to name the cache file. Default is
        "./file.txt". Two signals sharing a stem share a cache slot and
        evict one another.
    workers : int, optional
        Number of worker processes used to parallelize the sweep on a
        cache miss (see ``_r2_array``). Does not affect the cache key,
        since it changes who computes the curve, not the curve itself.
        Default is 1 (serial).
    return_sensitivity : bool, optional
        If True, append the baseline-sensitivity curve to the result.
        Default False.
    return_dw : bool, optional
        If True, append the Durbin-Watson array to the result. A cached
        entry written before ``dw_val`` existed cannot satisfy this and
        is recomputed. Default False.
    **kwargs
        Forwarded to `algo` through ``_r2_array``, and hashed into the
        cache key, so changing any of them misses.

    Returns
    -------
    result : numpy.ndarray or tuple of numpy.ndarray
        The array of r2 alone when both flags are False, otherwise a
        tuple holding it followed by the Durbin-Watson array if
        `return_dw` and the sensitivity curve if `return_sensitivity`,
        in that order. All have shape (M,).

    """
    def _out(vr2, vdw, stab):
        """Shape the result according to the two return flags."""
        out = (vr2,)
        if return_dw:
            out += (vdw,)
        if return_sensitivity:
            out += (stab,)
        return out[0] if len(out) == 1 else out

    if cache_dir is None:
        vr2, vdw, stab = _r2_array(algo, baseline_fitter, signal,
                                   param_range, param=param,
                                   workers=workers, return_dw=True,
                                   return_sensitivity=True, **kwargs)
        return _out(vr2, vdw, stab)

    key = _r2_cache_key(algo, signal, param_range, param, kwargs)
    stem = os.path.splitext(os.path.basename(path))[0]
    cache_file = os.path.join(cache_dir, f"{stem}__r2__{key}.npz")

    if os.path.isfile(cache_file):
        with np.load(cache_file) as data:
            vr2 = data["r2_val"]
            stab = data["sensitivity"]
            # `dw_val` was added after the first caches were written.
            # The channel token below makes those entries miss, so a
            # file without it can only come from a cache written by
            # hand; recompute rather than guess a sign.
            vdw = data["dw_val"] if "dw_val" in data.files else None
        if vdw is not None or not return_dw:
            logger.info(f"{'r2 cache:':<20}loaded {cache_file}")
            return _out(vr2, vdw, stab)

    vr2, vdw, stab = _r2_array(algo, baseline_fitter, signal, param_range,
                               param=param, workers=workers, return_dw=True,
                               return_sensitivity=True, **kwargs)
    os.makedirs(cache_dir, exist_ok=True)
    # Keep at most one cached curve per data file: a new write replaces
    # any entry of the same stem computed from other inputs, so stale
    # files cannot accumulate in the cache directory.
    prefix = f"{stem}__r2__"
    for name in os.listdir(cache_dir):
        if name.startswith(prefix) and name.endswith(".npz"):
            os.remove(os.path.join(cache_dir, name))
    np.savez(cache_file, fcut_range=param_range, r2_val=vr2, dw_val=vdw,
             sensitivity=stab)
    logger.info(f"{'r2 cache:':<20}saved {cache_file}")
    return _out(vr2, vdw, stab)

def _fcutoff(s: np.ndarray, x: np.ndarray, scut: int,
            num: int = 1000,
            method: str = "custom_beads", param: str = "freq_cutoff",
            cache_dir: str | None = None, path: str = "./file.txt",
            workers: int = 1, snr_threshold: float = 10.0, **kwargs
            ) -> tuple[float, dict]:
    """
    Find the optimal cutoff frequency.

    Sweeps the autocorrelation of the baseline-corrected signal over a
    geometric grid of cutoff frequencies, detects the plateaus of that
    curve, removes the ones that cannot hold the answer, and returns a
    point on what survives. The method, its parameters and what remains
    open are documented in ``tools/fcut/segmentation.md``.

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    scut : int
        Exclusive bound of the useful extent: the sweep runs on
        ``s[:scut]``. See `_relevant_regions`, which produces it.
    num : int, optional
        Number of grid points spanning the frequency range. The grid is
        geometric from 1e-5 up to but excluding the Nyquist limit of
        0.5, outside which pybaselines raises. Default is 1000.
    method : str, optional
        The method name passed to ``_beads`` or ``_custom_beads``.
        Default is "custom_beads".
    param : str, optional
        Label of the parameter to correlate with the value of r2. Default is
        "freq_cutoff".
    cache_dir : str, optional
        Directory where the autocorrelation curves are cached. Default is
        None, which disables caching.
    path : str, optional
        Path of the data file, used to name the cache file. Default is
        "./file.txt".
    workers : int, optional
        Number of worker processes used to parallelize the r2 sweep.
        Default is 1 (serial).
    snr_threshold : float, optional
        Signal-to-noise ratio above which the collapsed plateaus are
        excluded from the candidate regions (see `_snr` and
        `trim_candidates`). Default is 10, the limit of quantitation of
        MacDougall et al. [1]_ Table I, which puts quantitation above
        10 and detection between 3 and 10. Their sigma is a blank's
        standard deviation while `_snr` reads the signal itself, so the
        value is carried over.
    **kwargs
        Forwarded to the sweep and on to the fitting method, and hashed
        into the cache key.

    Returns
    -------
    fcut : float
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5.
    plot_data : dict
        Dictionary of internal variables needed to produce the r2 diagnostic
        plot. Empty dict if no plotting was requested.

    Raises
    ------
    ValueError
        If `method` is not one of the allowed methods, or if stage-2
        trimming removes every candidate, leaving no plateau to select
        from. Failing is deliberate: substituting a cutoff would bias
        every area derived from it.

    References
    ----------
    .. [1] MacDougall, D. et al. Guidelines for data acquisition and
       data quality evaluation in environmental chemistry.
       *Analytical Chemistry* **52**, 2242-2249 (1980), Table I.

    """
    tic = time.perf_counter()

    # Make sure that the method being passed is allowed
    allowed_methods = {"beads": _beads, "custom_beads": _custom_beads}
    if method not in allowed_methods:
        raise ValueError(f"method '{method}' is not implemented")

    algo = allowed_methods[method]

    baseline_fitter = Baseline(x_data=x[:scut])

    # log transform of the signal
    z = _log_transform(s[:scut])
    logger.info(f"{'Used points:':<20}{len(z):d}")

    fcut_range = np.geomspace(0.00001, 0.5, num=num, endpoint=False)

    # y-data
    r2_val, dw_val, sensitivity_val = _r2_array_cached(
        algo, baseline_fitter, z, fcut_range, param=param,
        cache_dir=cache_dir, path=path, workers=workers, return_dw=True,
        return_sensitivity=True, **kwargs)
    #####
    # Changepoint-based prototype (issue #4), for diagnostics only: the
    # detected plateaus/proto-plateaus and the stage-1 trimming are
    # overlaid on the r2 diagnostic plot. The selected fcut is unaffected.
    cp_segments = classify_segments(
        segment_features(fcut_range, r2_val, pelt_linear(r2_val)))
    # Full flat set and proto-plateaus; their union is the detected
    # plateau selection (strong and initial plateaus, plus the relative
    # flattenings the absolute flat test misses, detected as dips of the
    # rolling standard deviation).
    cp_flat = np.zeros(len(fcut_range), dtype=bool)
    for seg in cp_segments:
        if seg['flat']:
            cp_flat[seg['start']:seg['end']] = True
    cp_detected_dips = detect_dips(fcut_range, r2_val)
    cp_dips = dips_to_mask(fcut_range, cp_detected_dips)
    # Stage-1 trimming (single source: segmentation.trim_plateaus). The
    # sub-fundamental clip (#1) gives `cp_removed` (drawn red). The
    # SNR-gated collapse exclusion (#2) gives
    # `cp_snr_removed` (dark red) and IS applied to the selection,
    # unless applying it would leave nothing surviving.
    cp_trim = trim_plateaus(fcut_range, cp_segments, cp_detected_dips,
                            len(z), exclude_collapse=_snr(s) >= snr_threshold,
                            sensitivity=sensitivity_val)
    cp_surviving = cp_trim['surviving']
    cp_removed = cp_trim['removed']
    cp_snr_removed = cp_trim['snr_removed']
    cp_instab_removed = cp_trim['instab_removed']

    # Stage 3, PRELIMINARY: the cutoff is the centre of the surviving
    # plateau. This supersedes the legacy derivative route below, which
    # is kept only because the diagnostic overlay is built from its
    # intermediates. `None` means stage 2 left nothing, and the legacy
    # value is used as a fallback.
    fcut_center = select_center(fcut_range, cp_surviving)
    #####

    if fcut_center is None:
        raise ValueError(
            "no surviving plateau: stage-2 trimming removed every "
            "detected region, so there is no cutoff frequency to "
            "select. Pass an explicit cutoff with freq_cutoff=... to "
            "bypass the automatic selection."
        )
    fcut = fcut_center
    logger.info(f"{'fcut route:':<20}centre of the surviving plateau")

    toc = time.perf_counter()
    logger.info(f"Autocorrelation in {toc-tic:0.4f} seconds")
    # `select_center` returns a grid point, so the r2 there is already
    # in the swept curve: reading it saves a full baseline fit on every
    # run, and reports exactly the value the diagnostic draws rather
    # than a re-fit that can differ at low frequencies.
    fi_r2_val = float(r2_val[int(np.argmin(np.abs(fcut_range - fcut)))])
    logger.info(f"{'r2 value:':<20}{fi_r2_val:0.4f}")

    plot_data = {
        "fcut_range": fcut_range,
        "r2_val": r2_val,
        "sensitivity_val": sensitivity_val,
        "dip_curve": dip_curve(r2_val),
        "fcut": fcut,
        "fi_r2_val": fi_r2_val,
        "cp_flat": cp_flat,
        "cp_dips": cp_dips,
        "cp_surviving": cp_surviving,
        "cp_removed": cp_removed,
        "cp_snr_removed": cp_snr_removed,
        "cp_instab_removed": cp_instab_removed,
        # Number of points the sweep actually used; its reciprocal is
        # the signal's fundamental frequency, the slowest baseline the
        # data can constrain.
        "n_used": len(z),
    }
    return fcut, plot_data

###############################################################################
#BEADS baseline correction
def auto_beads(s: np.ndarray, x: np.ndarray,
               freq_cutoff: float | None = None, show_plot: bool = False,
               print_plot: bool = False, path: str = "./file.txt",
               output_dir: str = "results",
               method: str = "custom_beads", asymmetry: float = 1.0,
               fit_parabola: bool = True, alpha: float | None = None,
               parabola_len: int | None = 3,
               cache_dir: str | None = None,
               workers: int = 1, snr_threshold: float = 10.0
               ) -> tuple[np.ndarray, dict]:
    """
    Baseline-correct a chromatogram with BEADS.

    Automatic implementation of Baseline Estimation And Denoising with
    Sparsity [1], choosing the cutoff frequency when none is given.

    Decomposes the input data into baseline and pure, noise-free signal by
    modelling the baseline as a low pass filter and by considering the signal
    and its derivatives as sparse [1].

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    show_plot : bool, optional
        If True, the plot will be shown to the screen. Default is False.
    print_plot : bool, optional
        If True, the plot will be exported as an image. Default is False.
    path : str, optional
        Path of the data file.
    output_dir : str, optional
        Directory the plots are written to. Default is "results".
    method : str, optional
        Which fitting path to use, "beads" or "custom_beads". Default is
        "custom_beads", which decimates the points under each detected
        peak in proportion to that peak's width before fitting, making
        the baseline stiffer under the broad late peaks. "beads" fits
        the whole signal at one stiffness. Peak detection is needed for
        the regions, so it runs even when `freq_cutoff` is given.

        The two paths return the identical baseline whenever every peak
        is within about 1.18 times the width of the narrowest, since the
        decimation factor is then 1 and no point is dropped. They differ
        once the widths spread.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is None, which will calculate its
        value based on the autocorrelation plot of the log-transform of
        Navarro-Huerta et al. [2]_ §3.4.
    asymmetry : float, optional
        A number greater than 0 that determines the weighting of negative
        values compared to positive values in the cost function. For example,
        if is 6.0, it will give negative values six times more impact on the
        cost function that positive values. If set to 1 (default), the cost
        function is symmetric, and a value less than 1 will weigh positive
        values more.

        Default is 1, a symmetric cost, because a chromatogram carries
        a genuine negative peak at the dead time that an asymmetric cost
        would absorb into the baseline. BEADS and pybaselines default to
        6.0, which assumes positive-only peaks.
    fit_parabola : bool, optional
        If True (default), fit a parabola to the data and subtract it
        before the BEADS fit, as suggested in [2] §3.3.1. BEADS requires
        the endpoints to be close to 0, and the correction is a no-op on
        data that already satisfies that.
    alpha : float, optional
        Proportionality constant of the sparsity penalties. ``lam_0``,
        ``lam_1`` and ``lam_2`` are left to pybaselines, which follows
        Ning et al. [1]_ §5.1 in taking ``lam_d = alpha / ||D^d z||_1``,
        so `alpha` scales all three at once and a larger value gives a
        sparser peak component. Default is None, which selects 1.0.
        Only 1.0 is currently produced, so the value is fixed in
        practice.

        Ning et al. §5.1 set this constant "according to the noise
        variance" and tuned it by hand; 1.0 is a starting value, not an
        optimum. It is left alone because the cutoff frequency is the
        parameter to settle first: Navarro-Huerta et al. [2]_ §3.2 find
        it "has a major influence in the returned baseline" while "the
        other working parameters exhibit milder variations". Raising
        `alpha` is mostly useful at low signal-to-noise ratio, where it
        suppresses noise.
    parabola_len : int, optional
        Window at each end of the data used to reject an outlying
        endpoint before the parabola is fitted. Default is 3,
        pybaselines' own. `None` instead takes `end_window`, one percent
        of the signal length clamped to ``[3, 20]``, which is a different
        configuration and gives a different baseline.
    cache_dir : str, optional
        Directory where the autocorrelation curves used to select
        `freq_cutoff` are cached as ``.npz`` files. At most one cached curve
        is kept per data file (a new write replaces stale entries). Default
        is None, which disables caching. Only relevant when `freq_cutoff` is
        None.
    workers : int, optional
        Number of worker processes used to parallelize the autocorrelation
        sweep that selects `freq_cutoff`. Default is 1 (serial). Only
        relevant when `freq_cutoff` is None.
    snr_threshold : float, optional
        Signal-to-noise ratio (see `_snr`) above which the collapsed
        plateaus past the r2 drop are excluded from the candidate cutoff
        regions, because on a signal with analyte a cutoff there destroys
        peak area. Below it the shelves are kept, as a weak signal's optimum
        can lie past the collapse. Default is 10, the limit of
        quantitation of MacDougall et al. [3]_: below it a peak's area
        cannot be measured with acceptable precision, so there is little
        left for a cutoff past the collapse to destroy. Their ratio is
        built on the standard deviation of a blank, while `_snr` reads
        the tallest excursion of the signal itself against a robust
        noise estimate, so the value is carried over rather than
        derived. Only relevant when `freq_cutoff` is None.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    p : dict
        The BEADS parameter dictionary. Its ``signal`` key is the
        denoised component, which the two fitting paths build
        differently; see `_custom_beads`.

    Raises
    ------
    ValueError
        Raised if `asymmetry` is not greater than 0, if `method` is not
        one of the allowed methods, if `freq_cutoff` is not in (0, 0.5),
        or, when the cutoff is being selected, if stage-2 trimming leaves
        no plateau to select from.

    References
    ----------
    .. [1] Ning, X., et al. Chromatogram baseline estimation and denoising
        using sparsity (BEADS). Chemometrics and Intelligent Laboratory
        Systems, 2014, 139, 156-167.
    .. [2] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in
        complex chromatograms using the BEADS algorithm. Journal of
        Chromatography A, 2017, 1507, 1-10.
    .. [3] MacDougall, D., et al. Guidelines for data acquisition and
        data quality evaluation in environmental chemistry. Analytical
        Chemistry, 1980, 52(14), 2242-2249. Their Table I sets the
        region of quantitation at a signal-to-noise ratio above 10, and
        the region of detection between 3 and 10.

    """
    if asymmetry <= 0:
        raise ValueError('asymmetry must be greater than 0')

    # Make sure that the method being passed is allowed
    allowed_methods = {"beads": _beads, "custom_beads": _custom_beads}
    if method not in allowed_methods:
        raise ValueError(f"method '{method}' is not implemented")
    algo = allowed_methods[method]

    # Limits the range and splits the signal.
    # Only needed when freq_cutoff is auto-selected (needs scut) or when
    # using custom_beads (needs regions/sampling).
    if method == "custom_beads" or freq_cutoff is None:
        peak_regions, sampling, scut = _relevant_regions(s, x)
    else:
        peak_regions = None
        sampling = None
        scut = None

    # NOTE: The value of `alpha` doesn't need to change when looking for the
    #       best r**2 because of the log transform
    # NOTE: The default setting of `parabola_len=3` is suitable to determine
    #       `fcut` since the signal is log-transformed beforehand.
    #       regions and sampling are only relevant for custom_beads.
    method_kwargs = {
            "asymmetry": asymmetry,
            "fit_parabola": fit_parabola,
            "alpha": 1.0,
            "parabola_len": 3,
            }
    if method == "custom_beads":
        method_kwargs.update(regions=peak_regions, sampling=sampling)

    logger.info(f"{'Data points:':<20}{len(s):d}")

    # Cutoff frequency
    if freq_cutoff is None:
        fcut, plot_data = _fcutoff(
            s, x, scut, method=method, cache_dir=cache_dir, path=path,
            workers=workers, snr_threshold=snr_threshold, **method_kwargs)
    else:
        if ((freq_cutoff <= 0) or (freq_cutoff >= 0.5)):
            raise ValueError("cutoff frequency must be 0 < freq_cutoff < 0.5")
        fcut = freq_cutoff
        plot_data = {}
    # plot_data is empty when freq_cutoff is user-provided: there is no
    # autocorrelation sweep, hence no r2 diagnostic plot to draw.
    if (show_plot or print_plot) and plot_data:
        r2_plots(
            plot_data["fcut_range"], plot_data["r2_val"],
            plot_data["dip_curve"],
            plot_data["fcut"], plot_data["fi_r2_val"],
            cp_flat=plot_data["cp_flat"],
            cp_dips=plot_data["cp_dips"],
            cp_removed=plot_data["cp_removed"],
            cp_snr_removed=plot_data["cp_snr_removed"],
            cp_instab_removed=plot_data["cp_instab_removed"],
            sensitivity=plot_data["sensitivity_val"],
            n_used=plot_data["n_used"],
            show_plot=show_plot, print_plot=print_plot,
            path=path, output_dir=output_dir,
        )
    method_kwargs = {**method_kwargs, "freq_cutoff": fcut}

    # Change alpha for the final baseline correction
    # @EB TO CHANGE WHEN I KNOW HOW TO DO IT...
    if alpha is None:
        alpha=1.0
        method_kwargs["alpha"] = alpha

    # Change parabola_len for the final baseline correction
    if parabola_len is None:
        parabola_len=end_window(s)
        method_kwargs["parabola_len"] = parabola_len

    logger.info(f"{'Cutoff frequency:':<20}{fcut:0.4E}")
    logger.info(f"{'Asymmetry:':<20}{asymmetry:0.1f}")
    logger.info(f"{'Fit parabola:':<20}{str(fit_parabola):s}")
    logger.info(f"{'alpha:':<20}{alpha:0.2f}")
    logger.info(f"{'parabola_len:':<20}{parabola_len:d}")

    # Final baseline correction
    tic = time.perf_counter()                               #@TEMP

    baseline_fitter = Baseline(x_data=x)
    baseline, params = algo(baseline_fitter, s, **method_kwargs)

    toc = time.perf_counter()                               #@TEMP

    logger.info(f"Baseline correction in {toc-tic:0.4f} seconds") #@TEMP
    return baseline, params

