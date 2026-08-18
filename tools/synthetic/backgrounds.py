#!/usr/bin/python3
"""
Build the pool of real chromatographic backgrounds.

The ``hybrid`` family of the synthetic benchmark places known peaks and
known noise on a *real* drift, so that the background is experimental
while the truth stays exact (Niezen et al. 2022, §4.1, Fig. 1). This
module collects those drifts from three sources of deliberately
different provenance and writes them to a pool directory, one file per
background, with a manifest and a review plot each.

The pool is materialised rather than rebuilt on demand for two reasons:
scanning the 339 read-only LPYE records takes far longer than generating
a signal from them, and the pool is the artefact a human reviews by eye
before it is trusted for scoring.

**What is trimmed, and what is not.** The two kinds of source are not
treated alike, and the rule is provenance rather than a threshold:

* a trace **published as a background** is used at its full recorded
  length. Niezen's ``Baseline_*.csv`` are already the output of their
  peak-removal procedure (§4.1.1), and the MOCCA2 blanks are blanks;
  re-selecting a peak-free stretch inside them re-applies a selection
  their authors already performed, and measurably removes drift rather
  than peaks -- ``Baseline_3`` loses 16.0 min of structured drift for
  1.9 min of its flattest part.
* a trace that is an **ordinary run** carrying analytes has its longest
  peak-free stretch extracted, by
  `synth_dataset.peak_free_stretch`.

The reason the same criterion cannot serve both is its scale. It flags
an excursion above ``8 sigma`` with ``sigma`` the derivative-MAD, and
on LPYE that estimate is pinned by the detector's quantisation step, so
``8 sigma`` is 0.1-0.4 mV and sits well above any drift curvature. The
borrowed files are min-max normalised (Niezen) or very low-noise Waters
traces, with sigma from 3e-6 to 7e-4 in their own units, so the same
multiple falls *below* the curvature of their own drift.

**Units.** Backgrounds are stored as recorded, in whatever unit their
source uses, together with the peak-to-peak amplitude needed to rescale
them. Only LPYE is in mV. Assembly puts a borrowed shape onto the
synthetic instrument's mV scale by drawing a target peak-to-peak from
the measured LPYE range; see `synth_dataset` and
tools/synthetic/synthetic_data.md §3.

Usage
-----
python tools/synthetic/backgrounds.py POOL_DIR [--data DIR] [--external DIR]

References
----------
Niezen, Schoenmakers & Pirok (2022), Anal. Chim. Acta 1201, 339605,
§4.1.1.

"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synth_dataset as synth  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from weaselytics.parsers import ParsedData  # noqa: E402

# Default locations, as siblings of the repository. Both are read-only.
_HERE = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(os.path.dirname(_HERE), 'data')
DEFAULT_EXTERNAL = os.path.join(os.path.dirname(_HERE),
                                'backgrounds_external')

# Shortest LPYE peak-free stretch admitted to the pool, in minutes.
# NOT GROUNDED: like the `8 sigma` criterion it selects which real data
# enters the pool and never enters a reported number. This value is the
# one whose 130 extracted regions were reviewed by eye and accepted on
# 2026-07-27; changing it changes the pool that was reviewed.
LPYE_MIN_MINUTES = 5.0

# Wavelength taken from the two-dimensional DAD sources, in nm. A
# DECLARED CHOICE, not a grounded one: the sources record 200-400 nm and
# a background has to be one trace, so the channel nearest this value is
# taken and the wavelength actually used is recorded per background.
DAD_WAVELENGTH_NM = 254.0

# The external files, and what each one IS -- which is what decides
# whether it is trimmed. Transcribed from backgrounds_external/README.txt
# and tools/synthetic/synthetic_data.md §3.2, so the exclusions stay readable
# rather than being re-derived by a threshold.
#
#   (name, relative path, reader, kind, note)
#
# kind 'background' -> used whole;  kind 'run' -> peak-free stretch.
EXTERNAL_SOURCES = (
    ('NIEZEN__Baseline_1', 'niezen_FILTER/Baseline_1.csv', 'niezen',
     'background', 'Agilent 1260 DAD / Waters Acquity RI, 15.0 min'),
    ('NIEZEN__Baseline_3', 'niezen_FILTER/Baseline_3.csv', 'niezen',
     'background', 'Agilent 1260 DAD / Waters Acquity RI, 16.0 min'),
    ('MOCCA_AGILENT__blank', 'mocca2/benzaldehyde/blank.D/DAD1.CSV',
     'agilent_dad', 'background', 'Agilent ChemStation DAD, 3.99 min'),
    ('MOCCA_WATERS__blank1', 'mocca2/examples/blank1.arw', 'waters_arw',
     'background', 'Waters, 3.70 min'),
    ('MOCCA_WATERS__blank2', 'mocca2/examples/blank2.arw', 'waters_arw',
     'background', 'Waters, 3.75 min'),
    ('MOCCA_WATERS__blank3', 'mocca2/examples/blank3.arw', 'waters_arw',
     'background', 'Waters, 2.20 min'),
    ('MOCCA_DITERPENE', 'mocca2/diterpene_esters/data.mat',
     'mocca_mat', 'run', 'Agilent DAD, 16 calibration runs of 25 min'),
)

# Enumerated exclusions, kept so they are not silently re-admitted by a
# future change to a threshold. Each is a statement about what the file
# is, not about a number it scores.
EXTERNAL_EXCLUDED = (
    ('niezen_FILTER/Baseline_2.csv', 'LCxLC modulation slice (0.62 min),'
     ' not a 1D run'),
    ('niezen_FILTER/Baseline_4.csv', 'LCxLC modulation slice (0.50 min),'
     ' not a 1D run'),
    ('niezen_FILTER/Baseline_5.csv', 'LCxLC modulation slice (1.00 min),'
     ' not a 1D run'),
    ('mocca2/benzaldehyde/ba_05.D', 'a sample run, not a blank: carries'
     ' the benzaldehyde peak at 2.3 min'),
    ('mocca2/benzaldehyde/ba_1.D', 'a sample run, not a blank: carries'
     ' the benzaldehyde peak at 2.3 min'),
    ('mocca2/examples/chrom1-3.arw', 'sample runs; the blanks recorded'
     ' alongside them are used instead'),
)


def read_lpye(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one raw LPYE chromatogram.

    Parameters
    ----------
    path : str
        Path of the two-column Vernier-format file.

    Returns
    -------
    x : numpy.ndarray, shape (N,)
        Time, minutes.
    y : numpy.ndarray, shape (N,)
        Detector response, mV.

    """
    data = ParsedData(path).data
    return np.asarray(data[0], float), np.asarray(data[1], float)


def read_niezen(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one background of the Niezen et al. ``FILTER`` tool.

    Two comma-separated columns, time in minutes and a response that
    arrives min-max normalised to a range of exactly 1 (Niezen 2022,
    §4.1.3), so these carry shape only and not amplitude.

    Parameters
    ----------
    path : str
        Path of the CSV file.

    Returns
    -------
    x, y : numpy.ndarray, shape (N,)
        Time in minutes, and the normalised response.

    """
    a = np.loadtxt(path, delimiter=',')
    return a[:, 0].astype(float), a[:, 1].astype(float)


def read_agilent_dad(path: str, wavelength: float = DAD_WAVELENGTH_NM
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read one Agilent ChemStation exported DAD table.

    The file is UTF-16 with a byte-order mark, its first row listing the
    wavelengths in nm and each later row a time followed by one value per
    wavelength.

    Parameters
    ----------
    path : str
        Path of the ``DAD1.CSV`` file.
    wavelength : float, optional
        Target wavelength in nm; the nearest recorded channel is taken.

    Returns
    -------
    x, y : numpy.ndarray, shape (N,)
        Time in minutes, and the absorbance of the chosen channel.
    used : float
        The wavelength actually taken, nm.

    """
    with open(path, encoding='utf-16') as f:
        rows = [r for r in f.read().splitlines() if r.strip()]
    wl = np.array([float(v) for v in rows[0].split(',') if v.strip()])
    body = np.array([[float(v) for v in r.split(',')] for r in rows[1:]])
    j = int(np.argmin(np.abs(wl - wavelength)))
    return body[:, 0], body[:, j + 1], float(wl[j])


def read_waters_arw(path: str, wavelength: float = DAD_WAVELENGTH_NM
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read one Waters ``.arw`` two-dimensional export.

    The file uses classic Mac carriage returns as line endings, carries
    the wavelengths on its first row and a bare ``Time`` row before the
    data.

    Parameters
    ----------
    path : str
        Path of the ``.arw`` file.
    wavelength : float, optional
        Target wavelength in nm; the nearest recorded channel is taken.

    Returns
    -------
    x, y : numpy.ndarray, shape (N,)
        Time in minutes, and the absorbance of the chosen channel.
    used : float
        The wavelength actually taken, nm.

    """
    with open(path, encoding='latin-1') as f:
        text = f.read().replace('\r\n', '\n').replace('\r', '\n')
    lines = [ln for ln in text.split('\n') if ln.strip()]
    wl = np.array([float(v) for v in lines[0].split('\t')[1:]])
    body = np.array([[float(v) for v in ln.split('\t')]
                     for ln in lines[2:]])
    j = int(np.argmin(np.abs(wl - wavelength)))
    return body[:, 0], body[:, j + 1], float(wl[j])


def read_mocca_mat(path: str, wavelength: float = DAD_WAVELENGTH_NM
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read the MOCCA2 diterpene-ester calibration set.

    The MATLAB file holds ``Data_calibration`` as
    ``(time, wavelength, run)``; every calibration run is returned, since
    each is an independent chromatogram of the same 25 min method.

    Parameters
    ----------
    path : str
        Path of ``data.mat``.
    wavelength : float, optional
        Target wavelength in nm; the nearest recorded channel is taken.

    Returns
    -------
    x : numpy.ndarray, shape (N,)
        Time in minutes.
    y : numpy.ndarray, shape (N, R)
        One column per calibration run.
    used : float
        The wavelength actually taken, nm.

    """
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = mat['Data']
    x = np.asarray(data.Time, float).ravel()
    wl = np.asarray(data.Wavelength, float).ravel()
    cube = np.asarray(data.Data_calibration)
    j = int(np.argmin(np.abs(wl - wavelength)))
    return x, cube[:, j, :].astype(float), float(wl[j])


def admit(x: np.ndarray, y: np.ndarray, kind: str
          ) -> slice:
    """
    Choose the stretch of a trace that enters the pool.

    Provenance decides. A trace published as a background is taken
    whole; an ordinary run carrying analytes has its longest peak-free
    stretch extracted. See the module docstring for why one criterion
    cannot serve both.

    Parameters
    ----------
    x : array-like, shape (N,)
        Time axis. Unused for ``'background'``, present so both kinds
        share a signature.
    y : array-like, shape (N,)
        The trace.
    kind : {'background', 'run'}
        What the trace is.

    Returns
    -------
    region : slice
        The admitted stretch, empty when a run has none.

    """
    if kind == 'background':
        return slice(0, len(y))
    if kind == 'run':
        return synth.peak_free_stretch(y)
    raise ValueError(f"unknown kind {kind!r}")


def describe(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Summarise a background for the manifest.

    Parameters
    ----------
    x : array-like, shape (N,)
        Time, minutes.
    y : array-like, shape (N,)
        The drift, in the source's own units.

    Returns
    -------
    stats : dict
        ``n_points``, ``minutes``, ``dt_min``, ``p2p``, ``sigma`` and
        ``drift_over_sigma``. ``p2p`` is the plain range of the trace --
        no smoother, so no window length enters the number the assembler
        rescales by -- and ``sigma`` is the derivative-MAD estimate of
        Niezen Eq. (12b).

    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    p2p = float(y.max() - y.min())
    sigma = float(synth.noise_sigma_mad(y))
    return {'n_points': int(y.size),
            'minutes': float(x[-1] - x[0]),
            'dt_min': float(np.median(np.diff(x))) if y.size > 1 else 0.,
            'p2p': p2p,
            'sigma': sigma,
            'drift_over_sigma': float(p2p / sigma) if sigma > 0 else 0.}


def collect_lpye(data_dir: str = DEFAULT_DATA,
                 min_minutes: float = LPYE_MIN_MINUTES) -> list[dict]:
    """
    Extract every admissible background from the LPYE dataset.

    Every raw record is scanned, not only the blanks: the recorded LPYE
    blanks stop shortly after the injection peak (median 2.4 min beyond
    it) and are too short, whereas ordinary runs carry long peak-free
    stretches after the last analyte has eluted.

    Parameters
    ----------
    data_dir : str, optional
        Directory holding ``Molecule/stem.txt``. Read-only.
    min_minutes : float, optional
        Shortest stretch admitted. See `LPYE_MIN_MINUTES`.

    Returns
    -------
    entries : list of dict
        One per admitted background, with keys ``name``, ``x``, ``y``,
        ``x_full``, ``y_full``, ``slice`` and ``meta``.

    """
    entries = []
    for path in sorted(glob.glob(os.path.join(data_dir, '*', '*.txt'))):
        x, y = read_lpye(path)
        if y.size < 8:
            continue
        sl = admit(x, y, 'run')
        if sl.stop - sl.start < 8:
            continue
        if x[sl][-1] - x[sl][0] < min_minutes:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        entries.append({
            'name': f'LPYE__{stem}',
            'x': x[sl], 'y': y[sl], 'x_full': x, 'y_full': y,
            'slice': (int(sl.start), int(sl.stop)),
            'meta': {'source': 'LPYE', 'group': 'LPYE',
                     'kind': 'run', 'units': 'mV', 'quantised': True,
                     'file': os.path.relpath(path, data_dir),
                     'note': 'Cosmosil PYE semi-prep, 3.06 mL/min; '
                             'wavelength varies per run, see the header'},
        })
    return entries


def collect_external(external_dir: str = DEFAULT_EXTERNAL,
                     wavelength: float = DAD_WAVELENGTH_NM
                     ) -> list[dict]:
    """
    Read the published backgrounds enumerated in `EXTERNAL_SOURCES`.

    Admission here is by provenance, not by a threshold: the files that
    are usable and the reason each of the others is not are recorded in
    `EXTERNAL_SOURCES` and `EXTERNAL_EXCLUDED`.

    Parameters
    ----------
    external_dir : str, optional
        Directory holding ``niezen_FILTER/`` and ``mocca2/``.
    wavelength : float, optional
        Target wavelength for the two-dimensional sources, nm.

    Returns
    -------
    entries : list of dict
        As `collect_lpye`.

    """
    entries = []
    for name, rel, reader, kind, note in EXTERNAL_SOURCES:
        path = os.path.join(external_dir, rel)
        if not os.path.exists(path):
            continue
        used = None
        if reader == 'niezen':
            x, y = read_niezen(path)
            traces = [(name, y)]
            units, quantised = 'normalised (range 1)', False
        elif reader == 'agilent_dad':
            x, y, used = read_agilent_dad(path, wavelength)
            traces = [(name, y)]
            units, quantised = 'mAU', False
        elif reader == 'waters_arw':
            x, y, used = read_waters_arw(path, wavelength)
            traces = [(name, y)]
            units, quantised = 'AU', False
        elif reader == 'mocca_mat':
            x, cube, used = read_mocca_mat(path, wavelength)
            traces = [(f'{name}__calib{k:02d}', cube[:, k])
                      for k in range(cube.shape[1])]
            units, quantised = 'counts', False
        else:
            raise ValueError(f"unknown reader {reader!r}")

        for tname, ty in traces:
            sl = admit(x, ty, kind)
            if sl.stop - sl.start < 8:
                continue
            meta = {'source': name.split('__')[0], 'group': name,
                    'kind': kind, 'units': units, 'quantised': quantised,
                    'file': rel, 'note': note}
            if used is not None:
                meta['wavelength_nm'] = used
            entries.append({'name': tname, 'x': x[sl], 'y': ty[sl],
                            'x_full': x, 'y_full': ty,
                            'slice': (int(sl.start), int(sl.stop)),
                            'meta': meta})
    return entries


def plot_background(entry: dict, out_path: str) -> None:
    """
    Render one background for review.

    The full source trace is drawn in grey with the admitted stretch
    over it, so a trim can be judged against what it left out. Nothing
    in the pool is trusted before these are looked at.

    Parameters
    ----------
    entry : dict
        An entry as returned by `collect_lpye` or `collect_external`.
    out_path : str
        Destination PNG.

    """
    stats = describe(entry['x'], entry['y'])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(entry['x_full'], entry['y_full'], lw=.6, color='0.55',
            label='source record')
    ax.plot(entry['x'], entry['y'], lw=.9, color='tab:orange',
            label='admitted background')
    ax.set_xlabel('time (min)')
    ax.set_ylabel(entry['meta']['units'])
    ax.set_title(
        f"{entry['name']}   [{entry['meta']['kind']}]   "
        f"{stats['minutes']:.2f} min, {stats['n_points']} pts, "
        f"p2p {stats['p2p']:.4g}, drift/sigma {stats['drift_over_sigma']:.0f}",
        fontsize=9)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build_pool(pool_dir: str, data_dir: str = DEFAULT_DATA,
               external_dir: str = DEFAULT_EXTERNAL,
               wavelength: float = DAD_WAVELENGTH_NM,
               min_minutes: float = LPYE_MIN_MINUTES,
               plot: bool = True) -> list[dict]:
    """
    Collect every background and write the pool directory.

    Writes ``backgrounds/<name>.npz`` (arrays ``x`` and ``y`` plus a
    JSON ``meta``), ``manifest.csv``, and ``plots/<name>.png``.

    Parameters
    ----------
    pool_dir : str
        Destination directory.
    data_dir, external_dir : str, optional
        Sources. Both are read-only and are only read.
    wavelength : float, optional
        Target wavelength for the two-dimensional sources, nm.
    min_minutes : float, optional
        Shortest LPYE stretch admitted.
    plot : bool, optional
        Write the per-background review plots. Default True.

    Returns
    -------
    entries : list of dict
        The pool, in the order written.

    """
    entries = collect_lpye(data_dir, min_minutes) \
        + collect_external(external_dir, wavelength)
    bg_dir = os.path.join(pool_dir, 'backgrounds')
    os.makedirs(bg_dir, exist_ok=True)
    if plot:
        os.makedirs(os.path.join(pool_dir, 'plots'), exist_ok=True)

    rows = []
    for entry in entries:
        stats = describe(entry['x'], entry['y'])
        meta = dict(entry['meta'])
        meta.update(stats)
        meta['slice'] = list(entry['slice'])
        meta['name'] = entry['name']
        np.savez(os.path.join(bg_dir, f"{entry['name']}.npz"),
                 x=entry['x'], y=entry['y'], meta=json.dumps(meta))
        if plot:
            plot_background(entry, os.path.join(pool_dir, 'plots',
                                                f"{entry['name']}.png"))
        rows.append(
            f"{entry['name']},{meta['source']},{meta['group']},"
            f"{meta['kind']},{meta['units']},{stats['n_points']},"
            f"{stats['minutes']:.4f},{stats['dt_min']:.6f},"
            f"{stats['p2p']:.6g},{stats['sigma']:.6g},"
            f"{stats['drift_over_sigma']:.3f}")

    with open(os.path.join(pool_dir, 'manifest.csv'), 'w') as f:
        f.write('name,source,group,kind,units,n_points,minutes,dt_min,'
                'p2p,sigma,drift_over_sigma\n')
        f.write('\n'.join(rows) + '\n')
    return entries


def load_pool(pool_dir: str) -> list[dict]:
    """
    Read a pool directory written by `build_pool`.

    Parameters
    ----------
    pool_dir : str
        A directory holding ``backgrounds/*.npz``.

    Returns
    -------
    pool : list of dict
        One entry per background with keys ``x``, ``y`` and ``meta``,
        sorted by name so the pool order is reproducible.

    """
    bg_dir = os.path.join(pool_dir, 'backgrounds')
    pool = []
    for path in sorted(glob.glob(os.path.join(bg_dir, '*.npz'))):
        with np.load(path, allow_pickle=False) as z:
            pool.append({'x': z['x'], 'y': z['y'],
                         'meta': json.loads(str(z['meta']))})
    if not pool:
        raise FileNotFoundError(f"no backgrounds under {bg_dir}")
    return pool


def main() -> None:
    """
    CLI entry point of the background-pool builder.

    """
    parser = argparse.ArgumentParser(
        prog='backgrounds',
        description='build the pool of real chromatographic backgrounds')
    parser.add_argument('pool', help='output pool directory')
    parser.add_argument('--data', default=DEFAULT_DATA,
                        help='LPYE raw data directory (read-only)')
    parser.add_argument('--external', default=DEFAULT_EXTERNAL,
                        help='published-background directory (read-only)')
    parser.add_argument('--wavelength', type=float,
                        default=DAD_WAVELENGTH_NM,
                        help='wavelength taken from the 2D sources, nm '
                             f'(default: {DAD_WAVELENGTH_NM})')
    parser.add_argument('--min-minutes', type=float,
                        default=LPYE_MIN_MINUTES,
                        help='shortest LPYE stretch admitted '
                             f'(default: {LPYE_MIN_MINUTES})')
    parser.add_argument('--no-plot', action='store_true',
                        help='skip the per-background review PNG')
    args = parser.parse_args()

    entries = build_pool(args.pool, args.data, args.external,
                         args.wavelength, args.min_minutes,
                         plot=not args.no_plot)
    by_source: dict[str, int] = {}
    for e in entries:
        by_source[e['meta']['source']] = \
            by_source.get(e['meta']['source'], 0) + 1
    summary = ', '.join(f'{k} {v}' for k, v in sorted(by_source.items()))
    print(f'{len(entries)} backgrounds -> {args.pool}  ({summary})')


if __name__ == '__main__':
    main()
