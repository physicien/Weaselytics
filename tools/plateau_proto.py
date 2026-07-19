#!/usr/bin/python3
"""
Driver for the changepoint-based plateau detection prototype (issue #4).

Runs the segmentation + classification + fcut selection chain of
``weaselytics.segmentation`` on precomputed autocorrelation curves and
prints a per-segment diagnostic table. The input files are ``.npz``
archives containing the arrays ``fcut_range`` and ``r2_val`` (the
format written by the r2-curve cache), so the expensive BEADS sweep is
never recomputed here and the detection can be iterated on the whole
dataset in seconds.

Usage
-----
python tools/plateau_proto.py cache_dir/*.npz [--plot] [-od OUTPUT_DIR]
"""

import argparse
import os
import time

import numpy as np

from weaselytics.segmentation import select_fcut


def load_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an autocorrelation curve from a ``.npz`` cache file.

    Parameters
    ----------
    path : str
        Path of the ``.npz`` file. Must contain ``fcut_range`` and
        either ``r2_val`` or ``r2``.

    Returns
    -------
    fcut_range : numpy.ndarray, shape (N,)
        The cutoff frequencies.
    r2 : numpy.ndarray, shape (N,)
        The autocorrelation coefficients.

    Raises
    ------
    KeyError
        Raised if the archive does not contain the expected arrays.

    """
    data = np.load(path)
    fcut_range = data['fcut_range']
    key = 'r2_val' if 'r2_val' in data else 'r2'
    return fcut_range, data[key]


def plot_segments(fcut_range: np.ndarray, r2: np.ndarray,
                  segments: list[dict], chosen: int | None,
                  fcut: float | None, title: str, out_path: str) -> None:
    """
    Save a diagnostic plot of the segmentation.

    Parameters
    ----------
    fcut_range : numpy.ndarray, shape (N,)
        The cutoff frequencies.
    r2 : numpy.ndarray, shape (N,)
        The autocorrelation coefficients.
    segments : list of dict
        The classified segments returned by ``select_fcut``.
    chosen : int or None
        Index of the selected plateau in `segments`.
    fcut : float or None
        The selected cutoff frequency.
    title : str
        Title of the figure.
    out_path : str
        Path of the image file to write.

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogx(fcut_range, r2, '.', ms=2, color='0.3')
    for k, seg in enumerate(segments):
        if k == chosen:
            color = 'tab:purple'
        elif seg['flat']:
            color = 'tab:green'
        else:
            color = 'tab:red'
        ax.axvspan(fcut_range[seg['start']], fcut_range[seg['end'] - 1],
                   color=color, alpha=0.18)
    if fcut is not None:
        ax.axvline(fcut, color='tab:red', ls='--')
    ax.set_xlabel('fcut')
    ax.set_ylabel(r'$r^2_{y-b}$')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """
    CLI entry point of the prototype driver.

    """
    parser = argparse.ArgumentParser(
        prog='plateau_proto',
        description='changepoint-based plateau detection on cached r2 curves'
    )
    parser.add_argument('files', nargs='+',
                        help='.npz files with fcut_range and r2_val arrays')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='save a diagnostic plot per curve')
    parser.add_argument('-od', '--output-dir', default='proto_results',
                        help='output directory (default: proto_results)')
    parser.add_argument('--penalty', type=float, default=None,
                        help='changepoint penalty (default: 25*log(N))')
    parser.add_argument('--min-size', type=int, default=15,
                        help='minimal segment length (default: 15)')
    parser.add_argument('--rel-slope-max', type=float, default=0.2,
                        help='flatness threshold on the relative slope')
    parser.add_argument('--rel-noise-max', type=float, default=0.005,
                        help='flatness threshold on the relative noise')
    args = parser.parse_args()

    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)

    tic = time.perf_counter()
    for path in args.files:
        name = os.path.splitext(os.path.basename(path))[0]
        fcut_range, r2 = load_curve(path)
        fcut, segments, chosen = select_fcut(
            fcut_range, r2, penalty=args.penalty, min_size=args.min_size,
            rel_slope_max=args.rel_slope_max,
            rel_noise_max=args.rel_noise_max,
        )
        print(f'\n{name}')
        for k, seg in enumerate(segments):
            tag = 'CHOSEN' if k == chosen else ('flat' if seg['flat']
                                                else '')
            print(f"  seg {k:2d} [{seg['start']:4d},{seg['end']:4d}) "
                  f"mean={seg['mean']:.4f} "
                  f"rel_slope={seg['rel_slope']:8.3f} "
                  f"rel_noise={seg['rel_noise']:8.5f} {tag}")
        if fcut is None:
            print('  no plateau candidate found')
        else:
            print(f"  {'Cutoff frequency:':<20}{fcut:0.4E}")
        if args.plot:
            out_path = os.path.join(args.output_dir, f'{name}_proto.png')
            plot_segments(fcut_range, r2, segments, chosen, fcut, name,
                          out_path)
    toc = time.perf_counter()
    print(f'\nPlateau detection on {len(args.files)} curve(s) in '
          f'{toc - tic:0.4f} seconds')


if __name__ == '__main__':
    main()
