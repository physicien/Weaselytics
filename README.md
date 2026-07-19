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

## Library usage

```python
import weaselytics as wl

data = wl.ParsedData("chromato.txt")
x, y = data.data

# Fixed cutoff frequency
baseline, params, case = wl.auto_beads(y, x, freq_cutoff=0.01)

# Automatic cutoff frequency, with the autocorrelation curve cached so
# that reruns on the same signal skip the expensive BEADS sweep
baseline, params, case = wl.auto_beads(y, x, freq_cutoff=None,
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

- Generalize hardcoded `__LPYE__` pattern in `export_dist`
- Make `ParsedData` parser more general (support different delimiters, extra columns, headers)
- Clean up `#@EB`, `#@TEMP`, `#TODO` markers
- Add `examples/` directory with sample output images
- Add proper sample chromatogram data for demos
- Improve README with example CLI output images
