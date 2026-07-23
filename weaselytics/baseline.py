# coding: utf-8
"""
Functions to perform the baseline correction.
"""
import hashlib
import os
import time  #@EB temporary?
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pybaselines import Baseline
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import argrelmax, argrelmin  #, medfilt

from weaselytics.plot import r2_plots
from weaselytics.segmentation import (
    classify_segments,
    pelt_linear,
    refine_candidates,
    segment_features,
    trim_candidates,
)
from weaselytics.utils import (
    continuous_ranges,
    end_window,
    find_flat,
    find_plateaus,
    merge_intervals,
    peaks_params,
    r2_dw,
)

# Quantity the autocorrelation of `_r2` is computed on. Part of the
# cache key: bump it whenever the definition changes, so that curves
# cached under the previous one are recomputed rather than reused.
_R2_CHANNEL = "y-baseline"


def _relevant_regions(
    s: np.ndarray, x: np.ndarray, tol: float = 6.
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """
    Divide the signal into regions maximizing the contribution of the signal in
    the calculation of the autocorrelation plot. in order to find the optimal
    cutoff frequency for the BEADS algorithm.

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    tol : float, optional
        Threshold on the ratio of a peak’s width as a function of its location
        in `x`.

    Returns
    -------
    peak_regions : array-like, shape (M,2), or None
        The two dimensional array containing the start and stop indices for
        each region containing a relevant peak. Each region is defined as
        ``data[start:stop]``. `None` means no relevant peaks found.
    sampling : array-like of shape (M,)
        The sampling step size for each region defined in `peak_regions`.
    scut : int
        Index of the last data point in `s` (signal cutoff) relevant to the
        calculation of the autocorrelation.

    """
    # NOTE: A weak smoothing helps to avoid peak detection in noisy region of
    #       the signal by:
    #           1) removing most of the spurious features in the raw signal
    #           2) sligntly enlarging features relevant for peaks detection
    z = gaussian_filter1d(s,3)
    # Coarse detrend before measuring peak widths: half-prominence
    # widths are otherwise contaminated by slow baseline structure (a
    # narrow peak riding a broad hump measures the hump's width, not
    # its own, and gets rejected by the relevance filter below). The
    # rolling-median window only needs to separate the two scales: on
    # the reference dataset the widest relevant peak spans N/17.7 at
    # worst (median N/57) while baseline features span the record, so
    # N/4 clears every peak by a factor of 4 while following the
    # baseline. Measured over the 339 reference signals; note that
    # the detrend also shifts the peak regions or `scut` on 161 of
    # them, so artefacts derived from earlier runs are stale.
    window = max(31, len(z) // 4) | 1
    z = z - median_filter(z, size=window)
    peaks, widths = peaks_params(z, height_n=0.50, width=3, rel_prom_p=0.01,
                                   adapt=True)

    # TODO: Find a way to make this part of the code more robust.
    width_per_x = widths/x[peaks]
    # In case of very tall and large peaks (see acetonitrile)
    exception = ((s[peaks] > 20) & (width_per_x < 11))
    # Signal splitting
    rel_peaks = peaks[((width_per_x < tol) | exception)]
    rel_widths = widths[((width_per_x < tol) | exception)]
    # No relevant peak at all (featureless signal, or every detected
    # peak rejected by the relevance filter): fall back to the
    # documented degraded mode — no peak regions, uniform sampling and
    # no truncation — instead of crashing on the empty array below.
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

    Defined as the tallest excursion above the median divided by the
    noise, with the noise estimated from the median absolute deviation
    of the consecutive differences (so it is not inflated by the peaks
    themselves). This scalar carries the analyte-vs-baseline information
    the autocorrelation curve does not: a split near 25 separates the
    signals whose optimum sits at the shoulder from those whose optimum
    lies past the collapse, at 95% on the synthetic benchmark and 100%
    on the labeled real data.

    Parameters
    ----------
    s : array-like, shape (N,)
        The measured chromatogram.

    Returns
    -------
    snr : float
        The signal-to-noise ratio; ``inf`` if the noise estimate is
        zero, which happens only when the consecutive differences are
        constant (a flat or perfectly linear trace).

    """
    diffs = np.diff(s)
    noise = 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2)
    if noise <= 0:
        return np.inf
    return float((np.max(s) - np.median(s)) / noise)

def _log_transform(s: np.ndarray, epsilon: float = 1) -> np.ndarray:
    """
    Log transformation used in the calculation of the autocorrelation for the
    BEADS algorithm. For further information, see [1].

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
    The default value of ``epsilon = 1`` was originally suggested by
    Navarro-Huerta et al. [1] for two reasons:
        1) It was judged appropriate regarding the magnitude of the signals
        reaching maxima around 500-10000.
        2) If ``yi = min(y)``, then ``log(yi - min(y) + 1) = log 1 = 0``.

    However, it seems uncertain whether choosing ``epsilon = 1`` is optimal
    for a signal whose maximum value is below 500. My impression is that the
    500-10000 guideline reported by Navarro-Huerta et al. [1] was most likely
    derived purely from the particular signals they had at their disposal,
    rather than being grounded in a substantive theoretical argument.

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
    r"""Wrap ``pybaselines.Baseline.beads``.

    See `auto_beads` for parameter details.
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

    Extends `_beads` by splitting the signal into regions, each with its
    own sampling density. See `auto_beads` for common parameter details.

    Parameters
    ----------
    regions : array-like, shape (M,2), optional
        Start and stop indices for each region containing a relevant peak.
        ``data[start:stop]``. If `None` (default), uses all points.
    sampling : int or array-like, optional
        Sampling step size for each region in `regions`. Default 1.

    Returns
    -------
    bl : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        See `auto_beads` for details.

    Notes
    -----
    ``custom_bc`` returns only a baseline, so ``params["signal"]`` is
    rebuilt here as ``s - bl - noise``, with the noise interpolated from
    the reduced grid. ``custom_bc`` reduces points **only inside**
    ``regions`` and keeps everything else at full resolution, so that
    noise estimate is exact outside the regions and a straight line
    through bin averages inside them. ``params["signal"]`` is therefore
    the sparse component everywhere *plus* whatever point-to-point noise
    was never estimated on the region interiors -- it is not a uniformly
    denoised residual, and it is not ``s - bl``.

    How far it departs from ``s - bl`` is **not** governed by the
    binning, because the two conditions that would make binning matter
    are anti-correlated. Where binning is heaviest the analyte is large,
    so the residual noise is a negligible fraction of it; and where the
    departure is large -- a weak analyte -- ``_relevant_regions``
    returns few or no regions, so ``sampling`` is 1 and no binning
    happens at all. Measured on the raw signals at
    ``freq_cutoff=1.13e-2``, ``max|params["signal"] - (s - bl)|`` as a
    fraction of the signal range:

    ==================================  ==========  ========
    signal                              sampling    departure
    ==================================  ==========  ========
    2-Dichlorobenzene BLANK 1           [1]         15.0%
    2-Chlorotoluene BLANK 1             [2]          6.3%
    Toluene 60-100 #3                   [2,2,5,6,10] 0.79%
    2-Xylene 60-100 #1                  [2,2,...]    0.87%
    ==================================  ==========  ========

    On the two blanks the difference is simply the noise: the first has
    no regions at all, and ``params["signal"] - (s - bl)`` equals the
    BEADS noise exactly. So on raw signals this key is a *denoised*
    trace, and the choice between it and ``s - bl`` is a choice about
    showing the noise floor, not about contamination.

    This is separate from the reason `_r2` correlates ``y - baseline``:
    there the statistic is a sum of squared point-to-point differences
    on the log-transformed, truncated signal, where a small contaminated
    patch does dominate whenever the sparse component is small.

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
    Calculate the autocorrelation, based on the Durbin-Watson statistics, of
    the baseline corrected signal for a given value of a given parameter used
    for the substraction of the baseline.

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
        Value or `param` at which r2 is evaluated.
    param : str, optional
        Label of the parameter to correlate with the value of r2. Default is
        "freq_cutoff".
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    r2 : float
        The autocorrelation of the baseline corrected signal for `param`=`p`.

    Notes
    -----
    The statistic is computed on the **baseline-corrected signal**
    ``y - b``, i.e. the sparse chromatogram *plus* the noise, and not on
    the denoised ``params["signal"]``. This is the quantity monitored by
    Navarro-Huerta et al. [1] (Eq. 12), and the noise has to stay in:
    ``r2`` is a whiteness test, so the plateau structure exists only
    because a good cutoff leaves the correlated peaks in the corrected
    signal while an excessive one leaves white noise behind. Removing
    the noise removes the floor the drop is measured against.

    Keeping the baseline as the only algorithm output used here also
    makes the quantity identical on the ``beads`` and ``custom_beads``
    paths, so their curves are comparable. They were not before:
    ``_custom_beads`` rebuilds ``params["signal"]`` as
    ``y - b - noise`` with a noise term interpolated from the reduced
    grid, which is exact outside the peak regions but absent inside
    them, leaving raw point-to-point noise on the region interiors.

    References
    ----------
    [1] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in
        complex chromatograms using the BEADS algorithm. Journal of
        Chromatography A, 2017, 1507, 1-10, §3.3.2 and §3.4.

    """
    kwargs[param] = p
    baseline, _ = algo(baseline_fitter, y, **kwargs)
    y_corr = y - baseline
    r2 = r2_dw(y_corr)
    return r2

def _r2_chunk(args: tuple) -> list[float]:
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

    """
    algo, baseline_fitter, signal, chunk, param, kwargs = args
    r2_values = [
        _r2(algo, baseline_fitter, signal, p, param=param, **kwargs)
        for p in chunk
    ]
    return r2_values

def _r2_array(
    algo: Callable[..., tuple[np.ndarray, dict]],
    baseline_fitter: Baseline,
    signal: np.ndarray, param_range: np.ndarray,
    param: str = "freq_cutoff", workers: int = 1, **kwargs
) -> np.ndarray:
    """
    Calculate the array of `r2`, the Durbin-Watson autocorrelation of the
    baseline corrected signal, relative to a parameter on a specific range.

    The M evaluations are independent, and this sweep dominates the
    total runtime of the automatic cutoff-frequency selection (99.9% of
    the compute in the reference run), so they can be distributed over a
    pool of worker processes. Processes are required rather than threads
    because the iterative baseline fits hold the GIL. The parameter
    range is split into chunks of several evaluations each so that the
    inter-process overhead stays negligible compared to the fits.

    Parameters
    ----------
    algo : Callable
       The callable method corresponding to the input string.
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
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    vr2 : numpy.ndarray, shape (M,)
        The calculated array of r2.

    """
    if workers is None or workers <= 1:
        def _r2_wrapper(x):
            return _r2(algo, baseline_fitter, signal, x, param=param,
                       **kwargs)
        vr2_func = np.vectorize(_r2_wrapper)
        vr2 = vr2_func(param_range)
        return vr2

    # Several chunks per worker to balance the varying cost of the fits
    n_chunks = min(len(param_range), workers * 8)
    chunks = [c for c in np.array_split(param_range, n_chunks) if len(c)]
    payloads = [
        (algo, baseline_fitter, signal, chunk, param, kwargs)
        for chunk in chunks
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_r2_chunk, payloads))
    vr2 = np.array([r2 for chunk_r2 in results for r2 in chunk_r2])
    return vr2

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

    """
    sha = hashlib.sha1()
    # Identifies the quantity `_r2` correlates, not just its inputs.
    # Changing the channel changes every curve while leaving the inputs
    # untouched, so without this token the cache would keep serving
    # curves computed on the previous definition. Bump it whenever the
    # definition of `y_corr` in `_r2` changes.
    sha.update(_R2_CHANNEL.encode())
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
    # sweep -- 99.9% of the cost of the selection -- was silently paid
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
    path: str = "./file.txt", workers: int = 1, **kwargs
) -> np.ndarray:
    """
    Compute the array of `r2` with an optional on-disk cache.

    The autocorrelation sweep is by far the most expensive step of the
    automatic selection of the cutoff frequency (seconds to minutes per
    signal), while the downstream plateau detection runs in
    milliseconds. Caching the curve to a ``.npz`` file therefore allows
    iterating on the plateau detection over a whole dataset in seconds
    instead of recomputing every BEADS sweep. The cache file name
    combines the stem of `path` with a hash of every input that
    determines the curve, so a stale cache can never be returned for
    modified inputs. At most one cached curve is kept per data file:
    writing a new curve deletes the older entries of the same stem, so
    the cache directory stays bounded to one ``.npz`` (roughly 16 kB)
    per signal and cannot build up stale files.

    Parameters
    ----------
    algo : Callable
       The callable method corresponding to the input string.
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
        disables caching entirely.
    path : str, optional
        Path of the data file, used to name the cache file. Default is
        "./file.txt".
    workers : int, optional
        Number of worker processes used to parallelize the sweep on a
        cache miss (see ``_r2_array``). Does not affect the cache key,
        since it changes who computes the curve, not the curve itself.
        Default is 1 (serial).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    vr2 : numpy.ndarray, shape (M,)
        The calculated (or cached) array of r2.

    """
    if cache_dir is None:
        return _r2_array(algo, baseline_fitter, signal, param_range,
                         param=param, workers=workers, **kwargs)

    key = _r2_cache_key(algo, signal, param_range, param, kwargs)
    stem = os.path.splitext(os.path.basename(path))[0]
    cache_file = os.path.join(cache_dir, f"{stem}__r2__{key}.npz")

    if os.path.isfile(cache_file):
        with np.load(cache_file) as data:
            vr2 = data["r2_val"]
        print(f"{'r2 cache:':<20}loaded {cache_file}")
        return vr2

    vr2 = _r2_array(algo, baseline_fitter, signal, param_range,
                    param=param, workers=workers, **kwargs)
    os.makedirs(cache_dir, exist_ok=True)
    # Keep at most one cached curve per data file: a new write replaces
    # any entry of the same stem computed from other inputs, so stale
    # files cannot accumulate in the cache directory.
    prefix = f"{stem}__r2__"
    for name in os.listdir(cache_dir):
        if name.startswith(prefix) and name.endswith(".npz"):
            os.remove(os.path.join(cache_dir, name))
    np.savez(cache_file, fcut_range=param_range, r2_val=vr2)
    print(f"{'r2 cache:':<20}saved {cache_file}")
    return vr2

def _fcutoff(s: np.ndarray, x: np.ndarray, scut: int,
            smoothing_window: int = 15, slope_thresh: float = 5.0E-05,
            tol0: float = 1.0E-03, tol1_0: float = 1.0E-05,
            tol1_1: float = 5.0E-04, tol2: float = 2.0E-06,
            num: int = 1000,
            method: str = "beads", param: str = "freq_cutoff",
            cache_dir: str | None = None, path: str = "./file.txt",
            workers: int = 1, snr_threshold: float = 25.0, **kwargs
            ) -> tuple[float, int, dict]:
    """
    Find the optimal cutoff frequency.

    ###EXPERIMENTAL###
    Since this function is still under development and very unreliable, it is
    best to explain the general idea behind it rather than the details of the
    current implementation. This is done in fcut.md
    ##################

    Parameters
    ----------
    s : array-like, shape (N,)
        The y-values of the signal.
    x : array-like, shape (N,)
        The x-values of the signal.
    scut : int
        Index of the last data point in `s` (signal cutoff) relevant to the
        calculation of the autocorrelation.
    smoothing_window : int, optional
        Standard deviation for Gaussian kernel used to smooth the signal.
        Default is 15.
    slope_thresh : float, optional
        Threshold on the value of `smooth_d1` for the final shift of the
        frequency cutoff. Default is 5.0E-05.
    tol0 : float, optional
        Threshold used to find the first plateau on the autocorrelation plot.
        Default is 1.0E-03.
    tol1_0 : float, optional
        Tight threshold used to find plateaus on the first derivative of the
        smoothed autocorrelation plot. Default is 1.0E-05.
    tol1_1 : float, optional
        Loose threshold used to find plateaus on the first derivative of the
        smoothed autocorrelation plot. Default is 5.0E-04.
    tol2 : float, optional
        Threshold used to find plateaus on the second derivative of the
        smoothed autocorrelation plot. Default is 2.0E-06.
    num : int, optional
        Number of x-values spanning the frequency range to evaluate r2.
        Default is 1000.
    method : str
        The method name passed to ``_beads`` or ``_custom_beads``.
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
        `trim_candidates`). Default is 25.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    fcut : float
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5.
    case : int,
        The case rule from which `fcut` have been selected. Not necessarily
        useful in the current implementation, but it is advisable to keep it
        until proven otherwise.
    plot_data : dict
        Dictionary of internal variables needed to produce the r2 diagnostic
        plot. Empty dict if no plotting was requested.

    Raises
    ------
    ValueError
        Raised if `method` is not one of the allowed methods.

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
    print(f"{'Used points:':<20}{len(z):d}")

    fcut_range = np.geomspace(0.00001, 0.5, num=num, endpoint=False)

    # y-data
    r2_val = _r2_array_cached(algo, baseline_fitter, z, fcut_range,
                              param=param, cache_dir=cache_dir, path=path,
                              workers=workers, **kwargs)
    #####
    # Diagnostics only: these four feed the r2 overlay in `r2_plots` and
    # nothing downstream of them reaches `fcut`. `find_plateaus` shares
    # the absolute-tolerance fragility of the route below (utils.py, the
    # `tol0` level match on the initial plateau) and can raise on curves
    # the selection itself handles perfectly well, so a failure here
    # must disable the overlay, not abort the selection.
    try:
        test_plateaus, ends, test, test3 = find_plateaus(r2_val)
    except (IndexError, ValueError) as exc:
        print(f"WARNING: plateau overlay unavailable ({exc}).")
        test_plateaus = np.zeros(len(r2_val), dtype=bool)
        ends = np.zeros(len(r2_val), dtype=bool)
        test = np.zeros(len(r2_val))
        test3 = np.zeros(len(r2_val))

    # Changepoint-based prototype (issue #4), for diagnostics only: the
    # trimmed candidate plateau regions are overlaid on the r2
    # diagnostic plot for comparison. The selected fcut is unaffected.
    # Signal-to-noise ratio gates the collapse exclusion: on a signal
    # with analyte (high SNR) the optimum sits at the shoulder and the
    # collapsed low shelves are impossible, so they are trimmed; on a
    # blank (low SNR) they survive as legitimate candidates.
    snr = _snr(s)
    cp_segments = classify_segments(
        segment_features(fcut_range, r2_val, pelt_linear(r2_val)))
    cp_candidates = trim_candidates(fcut_range, cp_segments, len(z),
                                    exclude_collapse=snr >= snr_threshold)
    cp_refined = refine_candidates(fcut_range, cp_candidates)
    # Full flat set (before trimming), for the diagnostic overlay.
    cp_flat = np.zeros(len(fcut_range), dtype=bool)
    for seg in cp_segments:
        if seg['flat']:
            cp_flat[seg['start']:seg['end']] = True
    #####

    ##########################################################################
    # Smoothed data and derivatives
    smooth_d0 = gaussian_filter1d(r2_val,smoothing_window)
    #smooth_d0 = medfilt(r2_val, smoothing_window)
    smooth_d1 = np.gradient(smooth_d0)
    smooth_d2 = np.gradient(smooth_d1)
    min_d1 = argrelmin(smooth_d1)[0]
    max_d1 = argrelmax(smooth_d1)[0]
    d1_min = np.argmin(smooth_d1)
    #EB not general at all...
    # The threshold is absolute while the scale of the r2 curve is not:
    # on a signal whose baseline stays recoverable over the whole grid,
    # r2 never collapses (total drop of a few percent, steepest slope
    # below the threshold) and no point qualifies. Fall back on the
    # steepest descent, which is what the limit stands for.
    d1_drops = np.where(smooth_d1 < -1E-03)[0]
    lim_d1_drop = d1_drops[0] if len(d1_drops) > 0 else d1_min

    # Proto-plateaus from d1 and d2
    tight_d1_flats = find_flat(smooth_d1, tol1_0)
    loose_d1_flats = find_flat(smooth_d1, tol1_1)
    d2_flats = np.where(np.absolute(smooth_d2) < tol2)[0]

    # Find initial plateau
    tight_continuous = continuous_ranges(tight_d1_flats)
    starting_r2 = np.mean(smooth_d0[tight_continuous[0]])
    # Same class of mistuning as the secondary-plateau guard below:
    # `tol0` is an absolute level tolerance on a curve whose scale is
    # not fixed. When the curve never sits within `tol0` of the level
    # of its first tight-flat run before the steepest descent, there is
    # no initial plateau to end and the selection has no starting
    # point. Fail loudly rather than on an opaque IndexError.
    starting_candidates = np.where(
            np.absolute(starting_r2 - r2_val[:d1_min]) < tol0)[0]
    if len(starting_candidates) == 0:
        raise ValueError(
            "no initial plateau found: no point before the steepest "
            f"descent (index {d1_min:d}) lies within tol0={tol0:.1e} "
            f"of the level of the first flat run ({starting_r2:.4f}). "
            "This tolerance is absolute while the scale of the r2 "
            "curve is not; pass an explicit cutoff with "
            "freq_cutoff=... to bypass the automatic selection."
        )
    starting_end = starting_candidates[-1]
    starting_plateau = np.arange(starting_end+1)

    # Remove final plateau if it is tight
    last_r2 = num - 1
    if np.isin(last_r2, tight_continuous[-1]).any():
        last_r2 = tight_continuous[-1][0]

    # Plateaus
    plateaus = loose_d1_flats[(loose_d1_flats > starting_plateau[-1]) &
                              (loose_d1_flats < last_r2)]
    secondary_plateaus = np.intersect1d(plateaus, d2_flats)

    # No secondary plateau at all: every downstream branch indexes into
    # this array, so the legacy route has nothing to anchor on. Fail
    # loudly rather than crash on an opaque IndexError, and rather than
    # substitute a cutoff -- a wrong fcut silently biases every area
    # derived from it, which is worse than no answer.
    #
    # The cause is a mistuning, not a property of the signal: `tol1_1`
    # and `tol2` are absolute thresholds on the derivatives of a curve
    # whose scale is not fixed (see segmentation.md section 1). `tol2`
    # sits about 200x below the peak curvature of a typical r2 curve, so
    # only near-linear stretches qualify, and the intersection with the
    # d1-flat set can come out empty. Shorter signals are hit harder:
    # on the synthetic benchmark the median count of secondary-plateau
    # points is 111 for 800-point signals against 211 for 2500-point
    # ones, at identical peak curvature.
    if len(secondary_plateaus) == 0:
        raise ValueError(
            "no secondary plateau found: the d1-flat set (|d1| < "
            f"tol1_1={tol1_1:.1e}) and the d2-flat set (|d2| < "
            f"tol2={tol2:.1e}) do not overlap between the initial "
            f"plateau (index {starting_plateau[-1]:d}) and index "
            f"{last_r2:d}. These tolerances are absolute while the "
            "scale of the r2 curve is not; pass an explicit cutoff "
            "with freq_cutoff=... to bypass the automatic selection."
        )

    # Anchors
    sec_max_d1 = np.intersect1d(secondary_plateaus,max_d1)
    if len(sec_max_d1) == 0:
        p2_start = secondary_plateaus[0]
    else:
        # Make sure this is not on the tail of the initial plateau (p1) by
        # starting p2 at the first max of d1 on the secondary plateaus.
        p2_start = sec_max_d1[0]
    anchors = secondary_plateaus[((secondary_plateaus < lim_d1_drop) &
                                  (secondary_plateaus > p2_start))]

    # Differents cases
    if len(anchors) == 0:
        case = 1
        arg_l = continuous_ranges(secondary_plateaus)[0][-1]
        # Not needed if slope_arg is well chosen?
        slope_thresh = tol1_1*0.5    #@EB temporary?
    else:
        case = 2
        arg_l = anchors[np.argmin(np.absolute(smooth_d1[anchors]))]

    ##########################################################################
    # Shift relative to the chosen anchor
    slope_arg = np.where(np.absolute(smooth_d1) >= slope_thresh)[0]
    try:
        cutoff = slope_arg[slope_arg >= arg_l][0]
    except IndexError:
        print("WARNING: slope_arg < arg_l.")
        cutoff = arg_l

    fcut = fcut_range[cutoff]
    ##########################################################################

    print(f"Case {case:d}")
    toc = time.perf_counter()
    print(f"Autocorrelation in {toc-tic:0.4f} seconds")
    fi_r2_val = _r2(algo, baseline_fitter, z, fcut, param=param, **kwargs)
    print(f"{'r2 value:':<20}{fi_r2_val:0.4f}")

    plot_data = {
        "fcut_range": fcut_range,
        "r2_val": r2_val,
        "smooth_d0": smooth_d0,
        "test": test,
        "test3": test3,
        "min_d1": min_d1,
        "max_d1": max_d1,
        "ends": ends,
        "secondary_plateaus": secondary_plateaus,
        "test_plateaus": test_plateaus,
        "tol1_1": tol1_1,
        "tol2": tol2,
        "fcut": fcut,
        "fi_r2_val": fi_r2_val,
        "case": case,
        "cp_candidates": cp_candidates,
        "cp_refined": cp_refined,
        "cp_flat": cp_flat,
    }
    return fcut, case, plot_data

###############################################################################
#BEADS baseline correction
def auto_beads(s: np.ndarray, x: np.ndarray,
               freq_cutoff: float | None = None, show_plot: bool = False,
               print_plot: bool = False, path: str = "./file.txt",
               output_dir: str = "results",
               method: str = "beads", asymmetry: float = 1.0,
               fit_parabola: bool = True, alpha: float | None = None,
               parabola_len: int | None = 3,
               cache_dir: str | None = None,
               workers: int = 1, snr_threshold: float = 25.0
               ) -> tuple[np.ndarray, dict, int]:
    """
    Automatic implementation of the Baseline estimation and denoising with
    sparsity (BEADS) algorithm.

    Decomposes the input data into baseline and pure, noise-free signal by
    modeling the baseline as a low pass filter and by considering the signal
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
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is None, which will calculate its
        value based on the autocorrelation plot of the log-transform from
        Navarro-Huerta [2].
    asymmetry : float, optional
        A number greater than 0 that determines the weighting of negative
        values compared to positive values in the cost function. For example,
        if is 6.0, it will give negative values six times more impact on the
        cost function that positive values. If set to 1 (default), the cost
        function is symmetric, and a value less than 1 will weigh positive
        values more.

        The default of 1 departs from BEADS and from pybaselines, which use
        6.0 on the grounds that chromatographic peaks are positive, so that
        weighting negative values more pushes the baseline underneath them.
        Here the signals carry a genuine *negative* peak around the dead
        time; an asymmetric cost would absorb it into the baseline instead
        of returning it as signal, so the cost is kept symmetric.
    fit_parabola : bool, optional
        If True (default), will fit a parabola to the data and subtract it
        before performing the BEADS fit as suggested in [2]. This ensures the
        endpoints of the fit data are close to 0, which is required by BEADS.
        If the data is already close to 0 on both endpoints, set `fit_parabola`
        to False (but it does not change anything in reality).
    alpha : float, optional
        #@EB will change in pybaselines. If None (default), will automatically
        adjust the value (always to 1 for now).
    parabola_len : int, optional
        Size of the window used, at each ends of the data, to prevent issues
        in fitting a parabola before the baseline correction[2] when the first
        and/or last point is an outlier. If None, will be adjusted to the length
        of the data.
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
        peak area. Below it the shelves are kept, as a blank's optimum
        can lie past the collapse. Default is 25. Only relevant when
        `freq_cutoff` is None.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    p : dict
        A dictionary with the various parameters depending of the method used.
    case : int,
        The case rule from which `fcut` have been selected. Not necessarily
        useful in the current implementation, but it is advisable to keep it
        until proven otherwise.

    Raises
    ------
    ValueError
        Raised if `asymmetry` is not greater than 0, if `method` is not one of
        the allowed methods, or if `freq_cutoff` is not in (0, 0.5).

    References
    ----------
    .. [1] Ning, X., et al. Chromatogram baseline estimation and denoising
        using sparsity (BEADS). Chemometrics and Intelligent Laboratory
        Systems, 2014, 139, 156-167.
    .. [2] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in
        complex chromatograms using the BEADS algorithm. Journal of
        Chromatography A, 2017, 1507, 1-10.

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

    print(f"{'Data points:':<20}{len(s):d}")

    # Cutoff frequency
    if freq_cutoff is None:
        fcut, case, plot_data = _fcutoff(
            s, x, scut, method=method, cache_dir=cache_dir, path=path,
            workers=workers, snr_threshold=snr_threshold, **method_kwargs)
    else:
        if ((freq_cutoff <= 0) or (freq_cutoff >= 0.5)):
            raise ValueError("cutoff frequency must be 0 < freq_cutoff < 0.5")
        fcut = freq_cutoff
        case = 0
        plot_data = {}
    # plot_data is empty when freq_cutoff is user-provided: there is no
    # autocorrelation sweep, hence no r2 diagnostic plot to draw.
    if (show_plot or print_plot) and plot_data:
        r2_plots(
            plot_data["fcut_range"], plot_data["r2_val"],
            plot_data["smooth_d0"], plot_data["test"],
            plot_data["test3"], plot_data["min_d1"],
            plot_data["max_d1"], plot_data["ends"],
            plot_data["secondary_plateaus"], plot_data["test_plateaus"],
            plot_data["tol1_1"], plot_data["tol2"],
            plot_data["fcut"], plot_data["fi_r2_val"],
            case=plot_data["case"],
            cp_flat=plot_data["cp_flat"],
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

    print(f"{'Cutoff frequency:':<20}{fcut:0.4E}")
    print(f"{'Asymmetry:':<20}{asymmetry:0.1f}")
    print(f"{'Fit parabola:':<20}{str(fit_parabola):s}")
    print(f"{'alpha:':<20}{alpha:0.2f}")
    print(f"{'parabola_len:':<20}{parabola_len:d}")

    # Final baseline correction
    tic = time.perf_counter()                               #@TEMP

    baseline_fitter = Baseline(x_data=x)
    baseline, params = algo(baseline_fitter, s, **method_kwargs)

    toc = time.perf_counter()                               #@TEMP

    print(f"Baseline correction in {toc-tic:0.4f} seconds") #@TEMP
    return baseline, params, case

