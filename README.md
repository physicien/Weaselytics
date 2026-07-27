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
| `-snr` | Threshold on `baseline._snr` above which the collapsed (past-drop) plateaus are excluded from the candidate cutoff regions (default: 25.0). **Note:** on quantisation-limited data this statistic is not a signal-to-noise ratio — see the TO DO below |

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

- **Re-read `segmentation.md` §4c and decide whether to delete it.** It records the
  removed `refine_candidates` bracket. That route was ill advised: its constants
  were fitted to a hand-labeled gallery that sampled only inside the very regions
  the bracket narrows and drew them on the figures the labeller saw, so the labels
  were anchored on the machinery they calibrated and could not test its boundaries.
  It also tied three shipped constants to one labeling session on one instrument.
  Both label sets have since been deleted, so none of the numbers in that section
  can be reproduced or checked. Decide whether it records anything worth keeping.
  The same caution applies to any other prose in the docs quoting measurements with
  no surviving dataset behind them.
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
  load-bearing: it gates the collapse exclusion, which since a3b7159 moves the
  selected cutoff. Two further consequences: `_snr` returns `inf` on 34 of 339
  signals — the docstring attributes this to "a flat or perfectly linear
  trace", but the real cause is that quantisation makes over half the
  consecutive differences exactly zero — and the synthetic benchmark adds
  Gaussian noise without quantising, so there `_snr` *is* a true ratio and the
  recorded "95% synthetic / 100% real" agreement compared two different
  quantities under one name. Decide whether the statistic keeps its role under
  an honest name and rationale, or is replaced.
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
