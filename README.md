# `Weaselytics`

[![CI](https://github.com/physicien/Weaselytics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/physicien/Weaselytics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/physicien/Weaselytics/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://github.com/physicien/Weaselytics/blob/main/LICENSE)

A Python 3 library to extract and analyse chromatographic data: (hassle-free) plotting of HPLC chromatograms from output files, with baseline correction, peak detection and retention time determination through curve fitting.

* For Python 3.11 or later
* Open source: BSD 3-Clause License
* Source code: https://github.com/physicien/Weaselytics

**Contents**

* [Quick start](#quick-start)
* [CLI usage](#cli-usage)
* [Library usage](#library-usage)
* [Install](#install)
* [Tests](#tests)
* [Contributing](#contributing)
* [License](#license)
* [Author](#author)

## Quick start

```console
pip install git+https://github.com/physicien/Weaselytics.git
weaselytics [OPTION] filename
```

## CLI usage

```console
weaselytics [-h] [-s] [-p] [-e] [-o] [-os OUTPUT_STATS] [-n] [-nb]
            [-x0 STARTX] [-x1 ENDX] [-od OUTPUT_DIR] [-cd CACHE_DIR]
            [-w WORKERS] [-fc FREQ_CUTOFF]
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
| `-x0` | Start fitting at `x0` min |
| `-x1` | End fitting at `x1` min |
| `-od` | Output directory for exported files (default: `results`) |
| `-cd` | Cache directory for the autocorrelation curves used to select `freq_cutoff`; reruns on an unchanged signal skip the expensive BEADS sweep. At most one cached curve is kept per data file (default: no caching) |
| `-w` | Number of worker processes for the autocorrelation sweep (default: 1, serial) |
| `-fc` | Bypass the automatic selection and use this cutoff frequency, `0 < freq_cutoff < 0.5` (default: automatic selection) |

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

## Install

From GitHub:

```console
pip install git+https://github.com/physicien/Weaselytics.git
```

Editable install (for development):

```console
git clone https://github.com/physicien/Weaselytics.git
cd Weaselytics
pip install -e .
```

With test dependencies:

```console
pip install -e ".[test]"
```

With numba, which makes BEADS use its banded solver:

```console
pip install -e ".[numba]"
```

### Dependencies

Weaselytics requires Python 3.11 or later and the following libraries, all installed automatically:

* [NumPy](https://numpy.org)
* [SciPy](https://scipy.org)
* [matplotlib](https://matplotlib.org)
* [seaborn](https://seaborn.pydata.org)
* [pandas](https://pandas.pydata.org)
* [pybaselines](https://github.com/derb12/pybaselines), pinned to a commit of its development version

The minimum versions are in `pyproject.toml`. [numba](https://numba.pydata.org) is optional.

## Tests

```console
pytest
```

## Contributing

Bug reports, feature requests and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Weaselytics is open source under the BSD 3-Clause License. See [LICENSE](LICENSE).

## Author

* Emmanuel Bourret ([@physicien](https://github.com/physicien))
