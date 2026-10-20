# `Weaselytics`

Python package to extract and analyse chromatographic data.

A Python 3 library for (hassle-free) plotting of HPLC chromatograms from output
files with baseline correction, peak detection and retention time determination
through curve fitting.

## Quick start

```console
pip install weaselytics
weaselytics [OPTION] filename
```

## CLI usage

```console
weaselytics [-h] [-s] [-p] [-e] [-o] [-os OUTPUT_STATS] [-n] [-nb] [-sm]
            [-x0 STARTX] [-x1 ENDX] [-od OUTPUT_DIR] [-cd CACHE_DIR]
            [-w WORKERS] [-fc FREQ_CUTOFF] [-snr SNR_THRESHOLD]
            path
```

### Options

| Argument | Description |
|---|---|
| `path` | Input `.txt` data file (required) |
| `-s` | Show the matplotlib window(s) |
| `-p` | Print/export the matplotlib window(s) |
| `-e` | Export baseline-corrected data to `filename_bl.txt` |
| `-o` | Output data to `filename.csv` |
| `-os` | Output fitted stats for the given label to `filename_<label>.csv` |
| `-n` | Disable peak fitting |
| `-nb` | Disable baseline correction |
| `-sm` | Enable signal smoothing |
| `-x0` | Start fitting at `x0` min |
| `-x1` | End fitting at `x1` min |
| `-od` | Output directory for exported files (default: `results`) |
| `-cd` | Cache directory for the autocorrelation curves used to select `freq_cutoff`; reruns on an unchanged signal skip the expensive BEADS sweep. At most one cached curve is kept per data file (default: no caching) |
| `-w` | Number of worker processes for the autocorrelation sweep (default: 1, serial) |
| `-fc` | Bypass the automatic selection and use this cutoff frequency, `0 < freq_cutoff < 0.5` (default: automatic selection) |
| `-snr` | Threshold on `baseline._snr` above which the plateaus past the drop are excluded from the candidate cutoff regions (default: 10.0). **Note:** on quantisation-limited data this statistic is not a signal-to-noise ratio — see the TO DO below |

## Library usage

```python
import weaselytics as wl

data = wl.ParsedData("chromato.txt")
x, y = data.data

# Fixed cutoff frequency
baseline, params = wl.auto_beads(y, x, freq_cutoff=0.01)

# Automatic cutoff frequency, with the autocorrelation curve cached so
# that reruns on the same signal skip the expensive BEADS sweep
baseline, params = wl.auto_beads(y, x, freq_cutoff=None,
                                 cache_dir="r2_cache")
```

## Requirements

See `pyproject.toml` for the full dependency list.

## Install

Editable install (for development):

```console
pip install -e .
```

With test dependencies:

```console
pip install -e ".[test]"
```

## Tests

```console
pytest
```

## Contributor

Contributed by Emmanuel Bourret

## TO DO

- **The `_snr` gate never switches, and the exclusion it guards can delete
  correct answers.** `exclude_past_drop` is gated on `_snr >= snr_threshold`,
  but no signal processed so far falls below the threshold, so the gate is inert
  and the fallback in `segmentation.trim_plateaus` is unreachable. Peak area is
  already being absorbed before the drop level, so the exclusion removes a
  region where the loss has largely happened, and against known baselines it
  sometimes removes the flat region holding the optimal cutoff. Decide whether
  the gate has a discriminator worth keeping, or whether the exclusion itself is
  what should be reopened.

- **`utils.peaks_params` invents a negative peak when there is none, and it
  should be treated as a defect rather than a limitation.** The negative gate is
  `rel_prom_n` times the deepest negative prominence *present in the signal*, so
  on a signal carrying no genuine negative excursion the reference collapses to
  the noise floor and the deepest noise dip clears the bar by construction. Any
  caller that takes a returned negative peak to be real then acts on noise. It
  is reachable from production along two paths. `baseline._relevant_regions`
  calls `peaks_params` on every signal, and a spurious negative feature can
  enter the relevance filter and move the regions and `scut`.
  `peakfitting._lsq_fit` calls it as well, picks the main peak as the largest
  `|y|` among those returned, and opens the amplitude bounds downward when that
  peak is negative, so an invented negative feature can flip the sign of a
  fitted peak. The fix needs an absolute floor of some kind, so that "the
  deepest dip in this signal" is not automatically a peak; that floor needs
  grounding, so this is a halt as much as a repair.
- **Remove `utils.smooth_SG` and the `-sm/--dosmoothing` flag.** The wrapper is
  an artefact of the abandoned derivative approach, when smoothing the signal
  was thought to help, and nothing needs it now. Three further reasons, found
  2026-08-25: the single production call is `smooth_SG(ydata, 9, 0)`, and
  Savitzky-Golay with `polyorder=0` fits a constant per window, so it is a plain
  9-point moving average rather than the polynomial filter its name promises;
  neither 9 nor 0 is grounded; and the flag's help text reads `'do not smooth
  the signal'` while `action='store_true'` means passing it turns smoothing
  **on**, so `--help` states the opposite of the behaviour. It is not
  display-only: when the flag is set, the smoothed signal is what
  `auto_beads` fits, so it moves the baseline and every area derived from it.
  Removal touches `utils.smooth_SG`, `weaselytics.py:153` and its `y_sm` plot
  branch, the `-sm` argument, the `__init__.py` export, and two tests in
  `test_utils.py`.
- **`utils.merge_intervals` modifies its input in place.** The rows it keeps are
  the caller's own objects rather than copies, so extending one writes the new
  stop index back through: `[[0,5],[3,9],[20,25]]` returns as `[[0,9],[3,9]]`
  and leaves the caller holding `[[0,9],[3,9],[20,25]]`. This holds for a list
  of lists and for a two-dimensional array alike. It is currently harmless,
  since the one production caller, `baseline._relevant_regions`, passes an array
  it does not reuse, but it is a trap for the next caller. The repair is to
  append a copy of each row. It also returns shape `(0,)` rather than `(0,2)` on
  empty input, which the same caller happens to guard against with `len(...)`.
- **Dips and segments use opposite conventions for the same key name.** A
  segment dict from `segmentation.segment_features` carries an *exclusive*
  `end`, sliced as `[start:end]` at three call sites. A dip dict from
  `segmentation.detect_dips` carries an *inclusive* `end`, clamped to `n - 1`
  and sliced as `[start:end + 1]` in `dips_to_mask`. The two dictionaries look
  alike, travel through the same stages, and are indexed differently, which is
  the shape of an off-by-one that nobody finds. Unify on the exclusive
  convention, which is the one Python slicing expects and the one the majority
  of the code already uses.
- **`segmentation.dip_curve`'s `sigma` is tied to the grid density, and nothing
  else in the chain is.** It is passed to `gaussian_filter1d` as a standard
  deviation in *grid points*, so doubling `num` in the sweep halves the width it
  smooths over on the cutoff axis, and the proto-plateaus the detector finds
  move with a parameter that is supposed to control only sampling resolution.
  Every other constant in the classification is a ratio against the geometry of
  the curve and carries no such dependence, which is what
  `classify_segments` claims for the chain as a whole. Either express `sigma` in
  decades and convert using the grid's points-per-decade, so it means the same
  thing at any `num`, or establish that the detector is meant to work in sample
  space and say so. The value 8.0 itself is frozen and set by visual validation,
  so this is about the unit rather than the number. It is one constant declared
  in two signatures: `detect_dips(sigma=8.0)` threads its value straight through
  to `dip_curve(sigma=8.0)`, so the default is written twice and the two can
  drift apart silently. Whatever unit it ends up in, it should be declared once.
- **`baseline._beads` and `_custom_beads` accept `**kwargs` and forward none of
  it.** Both signatures end in `**kwargs`, and both call through to pybaselines
  with an explicit list of five arguments only, so anything else a caller
  supplies is swallowed without a warning: `lam_0`, `lam_1`, `lam_2`,
  `max_iter`, `tol`, `filter_type`, `cost_function` and `smooth_half_window` are
  all reachable in `pybaselines.Baseline.beads` and all unreachable through
  these wrappers. Accepting an argument and discarding it is worse than
  refusing it, because the caller believes the setting took effect. Either
  forward `**kwargs` or drop it from both signatures.
- **`export_dist` writes three columns whose meaning changes with the row.**
  `A`, `x0` and `sigma` are filled from the Gaussian fit and from the
  skew-normal fit into the same headers, but the two models do not define them
  the same way. `peakfitting.gauss` is un-normalized, so its `A` is the peak
  height, `x0` the apex and `sigma` the standard deviation.
  `peakfitting.skew_norm` is Azzalini's skew-normal (Scand. J. Statist. **12**,
  171-178, 1985) under a location-scale extension, so its `A` scales the area of
  a normalized density, its `loc` is the location parameter rather than the mean
  or the mode, and its `scale` is the scale parameter rather than the standard
  deviation. By his §2.3 Eq. (5) the mean is `loc + scale*b*d` and the standard
  deviation `scale*sqrt(1-(b*d)^2)`, with `b = sqrt(2/pi)` and
  `d = alpha/sqrt(1+alpha^2)`, both differing from the parameters whenever
  `alpha` is non-zero.
  A reader comparing rows across the `distribution` column is therefore
  comparing a height with an area and an apex with a location. Either convert
  the skew-normal parameters to apex, height and standard deviation on the way
  out, or give the columns names that do not promise a shared meaning. The
  modified Pearson VII row is not affected: its `A` is the peak height and its
  `x0` the apex at every asymmetry, so `gauss` and `pearson7` share a convention
  and only `skew_norm` departs from it.
- **The `-e` export writes the denoised peak component, not the
  baseline-corrected signal.** `weaselytics.py:168` sets the exported trace to
  `params["signal"]`, the sparse peak component BEADS returns, and `export_txt`
  writes it as `<stem>_bl.txt`. Navarro-Huerta et al. and Liland et al. subtract
  the baseline and keep the noise, which is `y - baseline` and a different file.
  The CLI table above describes `-e` as "baseline-corrected data", which matches
  the name rather than the content. Decide which quantity the flag should write,
  then make the name, the table and the docstring agree.
- **Ground the instability-exclusion thresholds.** `segmentation.instability_boundary`
  trims the stiff side up to where the baseline stops flailing, using
  `trigger=0.10` (is the fundamental inside a flailing region?) and
  `settled=0.05` (are the oscillations small enough?). Both are **adopted
  provisionally, not grounded**: they are amplitudes of the dimensionless
  sensitivity curve, so they read as statements about tolerable baseline
  movement rather than as instrument constants, but no reference fixes where
  that tolerance lies. `settled` is the sensitive one — it sets how far the
  exclusion reaches, while `trigger` only changes how many signals are
  affected. Ground it against baseline error on synthetic ground truth, where
  the true baseline is known.
- **The stiff-side instability trim is not reproducible across library
  versions, and that is a requirement its grounding must meet.** Measured
  2026-07-27 on one machine, same code, same pinned pybaselines, with **only
  numpy/scipy differing** (2.2.4/1.15.3 vs 2.5.0/1.18.0): the swept `r2` curve
  agrees to 1 ulp above fcut 0.1 but diverges by up to **5.6e-2 near fcut
  1e-4**, a one-ulp input difference amplified ~2e14 by the method's own
  low-frequency instability (Navarro-Huerta 2017 §3.1(iv)). That is the same
  magnitude as the Rorqual-vs-workstation difference, so **the cluster was
  never the variable — the library version is**, and no pin fixes it because
  every upgrade re-rolls it. Effect on the selected cutoff across that version
  pair, with the trim off and on:

  | | trim OFF | trim ON |
  |---|---|---|
  | identical fcut | 280/339 | 190/339 |
  | >= 0.1 decade apart | 1 | 6 |

  So the trim accounts for **5 of the 6 large shifts** and takes identical
  selections from 280 down to 190; every other stage — detection, the
  sub-fundamental clip, the frozen tail, the past-drop exclusion — is close to
  version-proof. `trigger`/`settled` must therefore be grounded against a
  **robustness** criterion as well as an accuracy one: a threshold that scores
  well but moves six signals more than 0.1 decade under a library upgrade is
  fitted, not grounded. Do not calibrate either constant to a precision finer
  than ~1e-3 near the fundamental. The `numpy`/`scipy` floors now match
  Rorqual, which narrows the gap but does not close it.
- **`baseline._snr` is not a signal-to-noise ratio on quantisation-limited
  data, and the name hides it.** The LPYE detector output is digitised at a
  step of q = 0.008996 mV: every consecutive difference in all 339 reference
  signals is an exact integer multiple of q, and ~25% of consecutive samples
  are identical. `_snr`'s denominator, `1.4826 * MAD(diff) / sqrt(2)`, is
  therefore pinned to that lattice — it takes **five distinct finite values
  across the whole dataset**, 86% of signals sharing one — so `_snr` reduces
  to the tallest excursion divided by a constant (corr of the logs = 0.979).
  `SNR >= 25` is in practice `excursion >= ~0.47 mV`, an absolute amplitude in
  mV wearing a dimensionless name, and therefore instrument-specific. It is
  load-bearing: it gates the past-drop exclusion, which since a3b7159 moves the
  selected cutoff. Two further consequences: `_snr` returns `inf` on 34 of 339
  signals — the docstring attributes this to "a flat or perfectly linear
  trace", but the real cause is that quantisation makes over half the
  consecutive differences exactly zero — and the synthetic benchmark adds
  Gaussian noise without quantising, so there `_snr` *is* a true ratio and the
  recorded "95% synthetic / 100% real" agreement compared two different
  quantities under one name. Decide whether the statistic keeps its role under
  an honest name and rationale, or is replaced.
- **Investigate whether the fixed parameters of the peak decimation can be
  computed at the start from the relative broadening of the peaks, under the
  isocratic elution hypothesis.** `baseline._relevant_regions` takes its unit of
  width from `ratio_w = rel_widths / min(rel_widths)`, so the reference is
  whichever detected peak happens to be narrowest, and `scut` adds a buffer of
  `2 * rel_widths` past the last peak with no stated derivation. Under isocratic
  elution the plate number is constant, so peak width grows in proportion to
  retention time and the relative broadening across a chromatogram is set by the
  separation rather than by one peak. Fitting that relation on the detected
  peaks at the start could yield the decimation factors and the buffer directly,
  replacing a data-dependent reference with a property of the run. The `0.85`
  and the `/2` in the same block are Gaussian geometry and are not in question.
- Generalize hardcoded `__LPYE__` pattern in `export_dist`
- `weaselytics/__init__.py` re-exports the `plot` **function**, which shadows
  the `weaselytics.plot` **submodule** of the same name. `import
  weaselytics.plot as p` therefore binds the function, not the module, and
  reaching the module needs `sys.modules["weaselytics.plot"]`. This silently
  breaks any script that introspects or patches module-level names (e.g.
  `_SENSITIVITY_COLOR`); it fails by doing nothing rather than by raising.
  Consider renaming the function or the module.
- Make `ParsedData` parser more general (support different delimiters, extra columns, headers)
- Clean up `#@EB`, `#@TEMP`, `#TODO` markers
- Add `examples/` directory with sample output images
- Add proper sample chromatogram data for demos
- Improve README with example CLI output images

## Dead code, kept on purpose

Functions with no caller in the package that are deliberately not removed. Each
carries the same statement at the top of its own docstring, so the two do not
drift apart.

- **`utils.rm_ends_outliers`.** The full prototype of the endpoint-outlier
  handling that preceded pybaselines
  [#70](https://github.com/derb12/pybaselines/issues/70). Its criterion is a fraction of
  the signal's full range, where pybaselines compares an endpoint against two
  standardized median absolute deviations of its own edge, so a peak anywhere in
  the signal raises this bar and can leave a bad endpoint in place. Kept because
  that behaviour may still be wanted. Its window-sizing half, `end_window`, is
  live: `auto_beads` passes it as `parabola_len` when a caller supplies `None`.
  Three tests in `test_utils.py` guard it.
