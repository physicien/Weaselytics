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
| `-snr` | Threshold on `baseline._snr` above which the plateaus past the drop are excluded from the candidate cutoff regions (default: 10.0). **Note:** on quantisation-limited data this statistic is not a signal-to-noise ratio (see the TO DO below) |

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

- Decide whether the `_snr` gate on the past-drop exclusion is worth keeping: no signal falls below its threshold.
- Rename or replace `baseline._snr`: on quantised data it is the tallest excursion over a fixed step, not a signal-to-noise ratio.
- Ground `trigger` and `settled` in `segmentation.instability_boundary`, including how far they move across numpy/scipy versions.
- Stop `utils.peaks_params` from returning a negative peak when the signal has none.
- Remove `utils.smooth_SG` and the `-sm/--dosmoothing` flag.
- Stop `utils.merge_intervals` from modifying its input.
- Use one convention for `end` in dips and segments (exclusive, as in segments).
- Express the `sigma` of `segmentation.dip_curve` in decades instead of grid points, and declare it once.
- Forward `**kwargs` in `baseline._beads` and `_custom_beads`, or drop it.
- Give the `A`, `x0` and `sigma` columns of `export_dist` the same meaning for every distribution.
- Decide what `-e` exports: the denoised peak component (now) or `y - baseline`.
- Investigate computing the decimation parameters of `baseline._relevant_regions` from the peak broadening, under isocratic elution.
- Rename the `plot` function or the `weaselytics.plot` module: the function shadows the module.
- Generalize hardcoded `__LPYE__` pattern in `export_dist`
- Make `ParsedData` parser more general (support different delimiters, extra columns, headers)
- Clean up `#@EB`, `#@TEMP`, `#TODO` markers
- Add `examples/` directory with sample output images
- Add proper sample chromatogram data for demos
- Improve README with example CLI output images

## Dead code, kept on purpose

- `utils.rm_ends_outliers` has no caller but is kept: its endpoint-outlier criterion may still be wanted.
