#!/usr/bin/python3
"""
Render a gallery of baselines across the candidate plateau regions.

For every signal of the dataset, the trimmed candidate regions of the
autocorrelation curve (``segmentation.trim_candidates``) are sampled at
a constant geometric ratio and one baseline-correction figure is written
per sampled cutoff frequency. The r2 curves are read from an existing
``r2_cache`` directory, so the expensive autocorrelation sweep is never
recomputed; only the (millisecond) final BEADS correction is run per
image.

The output mirrors the layout of the ``data`` directory::

    output_dir/
        Molecule/
            signal_stem/
                00_r2.png       r2 curve, candidate regions, sampled fcut
                k00_r0_fcut_1.023e-03.png
                ...
                index.csv       one row per image
        gallery_index.csv       one row per signal

Usage
-----
python tools/fcut_gallery.py CACHE_DIR DATA_DIR -od OUTPUT_DIR [-r 1.15]
                             [-w 8] [--pattern GLOB]
"""

import argparse
import csv
import fnmatch
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from pybaselines import Baseline  # noqa: E402

from weaselytics.baseline import _custom_beads, _relevant_regions  # noqa: E402
from weaselytics.parsers import ParsedData  # noqa: E402
from weaselytics.segmentation import (  # noqa: E402
    classify_segments,
    pelt_linear,
    segment_features,
    trim_candidates,
)


def load_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an autocorrelation curve from a ``.npz`` cache file.

    Parameters
    ----------
    path : str
        Path of the ``.npz`` file.

    Returns
    -------
    fcut_range : numpy.ndarray, shape (N,)
        The cutoff frequencies.
    r2 : numpy.ndarray, shape (N,)
        The autocorrelation coefficients.

    """
    data = np.load(path)
    key = "r2_val" if "r2_val" in data else "r2"
    return data["fcut_range"], data[key]


def candidate_regions(fcut_range: np.ndarray, r2: np.ndarray,
                      n_used: int) -> list[np.ndarray]:
    """
    Compute the trimmed candidate plateau regions of a curve.

    Parameters
    ----------
    fcut_range : numpy.ndarray, shape (N,)
        The cutoff frequencies.
    r2 : numpy.ndarray, shape (N,)
        The autocorrelation coefficients.
    n_used : int
        Number of signal points used for the autocorrelation sweep.

    Returns
    -------
    regions : list of numpy.ndarray
        Index arrays of the contiguous candidate regions, in increasing
        order of cutoff frequency.

    """
    segments = classify_segments(
        segment_features(fcut_range, r2, pelt_linear(r2)))
    mask = trim_candidates(fcut_range, segments, n_used)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return np.split(idx, breaks + 1)


def sample_regions(fcut_range: np.ndarray, regions: list[np.ndarray],
                   ratio: float) -> list[tuple[int, int]]:
    """
    Sample the candidate regions at a constant geometric ratio.

    The sampled cutoffs are snapped to the points of `fcut_range`, so
    every image corresponds to a grid point of the cached curve. The two
    ends of each region are always sampled.

    Parameters
    ----------
    fcut_range : numpy.ndarray, shape (N,)
        The (geometrically spaced) cutoff frequencies.
    regions : list of numpy.ndarray
        The candidate regions, as returned by `candidate_regions`.
    ratio : float
        Ratio between two consecutive sampled cutoff frequencies.

    Returns
    -------
    samples : list of (int, int)
        Pairs ``(region_id, grid_index)``, sorted by cutoff frequency.

    """
    samples = []
    for reg_id, reg in enumerate(regions):
        f_lo = fcut_range[reg[0]]
        f_hi = fcut_range[reg[-1]]
        if f_hi <= f_lo:
            samples.append((reg_id, int(reg[0])))
            continue
        n = int(round(np.log(f_hi / f_lo) / np.log(ratio))) + 1
        targets = np.geomspace(f_lo, f_hi, max(n, 2))
        # Snap to the nearest grid point inside the region.
        taken = []
        for t in targets:
            j = int(reg[np.argmin(np.abs(np.log(fcut_range[reg] / t)))])
            if j not in taken:
                taken.append(j)
        samples.extend((reg_id, j) for j in taken)
    return sorted(samples, key=lambda p: p[1])


def plot_baseline(x: np.ndarray, y: np.ndarray, bl: np.ndarray,
                  signal: np.ndarray, title: str, out_path: str,
                  ylim: tuple[float, float] | None = None,
                  dpi: int = 100) -> None:
    """
    Save the chromatogram, its baseline and the corrected signal.

    Parameters
    ----------
    x, y : numpy.ndarray, shape (N,)
        The raw chromatogram.
    bl : numpy.ndarray, shape (N,)
        The baseline.
    signal : numpy.ndarray, shape (N,)
        The baseline-corrected signal.
    title : str
        Title of the figure.
    out_path : str
        Path of the image file to write.
    ylim : (float, float), optional
        Limits of the y-axis. Shared by every image of a signal so that
        only the baseline moves when the images are browsed in order.
        Default is None, which autoscales.
    dpi : int, optional
        Resolution of the exported image. Default is 100.

    """
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    # Visual reference for the corrected signal: a baseline is good when
    # the adjusted data sits on this line between the peaks. Red because
    # the three traces already use grey, tan and blue.
    ax.axhline(0.0, c="tab:red", lw=1.0, alpha=0.35, zorder=0)
    ax.plot(x, y, marker=".", ls="", c=palette[7], label="raw data", ms=3)
    ax.plot(x, signal, ls="-", c=palette[5], lw=1.5, label="adjusted data")
    ax.plot(x, bl, ls="--", c=palette[0], lw=2.0, label="baseline")
    ax.set_xlabel("Time (min.)")
    ax.set_ylabel("Potential (mV)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_r2(fcut_range: np.ndarray, r2: np.ndarray,
            regions: list[np.ndarray], samples: list[tuple[int, int]],
            title: str, out_path: str, dpi: int = 100) -> None:
    """
    Save the r2 curve with the candidate regions and the sampled cutoffs.

    Parameters
    ----------
    fcut_range, r2 : numpy.ndarray, shape (N,)
        The autocorrelation curve.
    regions : list of numpy.ndarray
        The candidate regions.
    samples : list of (int, int)
        The sampled ``(region_id, grid_index)`` pairs.
    title : str
        Title of the figure.
    out_path : str
        Path of the image file to write.
    dpi : int, optional
        Resolution of the exported image. Default is 100.

    """
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    ax.semilogx(fcut_range, r2, marker=".", ls="", ms=3, label=r"$r^2$")
    for reg in regions:
        mask = np.zeros(len(fcut_range), dtype=bool)
        mask[reg] = True
        ax.fill_between(fcut_range, 0, 1, where=mask, color="none",
                        ec="tab:purple", alpha=0.3, hatch="\\\\",
                        hatch_linewidth=2,
                        transform=ax.get_xaxis_transform())
    for k, (_, j) in enumerate(samples):
        ax.axvline(fcut_range[j], color="tab:orange", lw=0.6, alpha=0.7)
        # Only every fifth sample is labelled: consecutive ticks are one
        # `ratio` apart and their labels would overlap.
        if k % 5 == 0:
            ax.annotate(f"{k:d}", xy=(fcut_range[j], 1.01),
                        xycoords=("data", "axes fraction"), ha="center",
                        fontsize=7, color="tab:orange")
    ax.set_xlabel("Cutoff frequency")
    ax.set_ylabel(r"$r^2_{y-b}$")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def process_signal(data_path: str, cache_path: str, out_root: str,
                   ratio: float, dpi: int) -> dict:
    """
    Render the whole fcut gallery of one signal.

    Parameters
    ----------
    data_path : str
        Path of the raw ``.txt`` chromatogram.
    cache_path : str
        Path of the cached ``.npz`` autocorrelation curve.
    out_root : str
        Root directory of the gallery.
    ratio : float
        Ratio between two consecutive sampled cutoff frequencies.
    dpi : int
        Resolution of the exported images.

    Returns
    -------
    summary : dict
        One row of the gallery index. The ``error`` key is an empty
        string on success and the exception message otherwise.

    """
    stem = os.path.splitext(os.path.basename(data_path))[0]
    molecule = os.path.basename(os.path.dirname(data_path))
    summary = {"molecule": molecule, "stem": stem, "n_points": 0,
               "n_used": 0, "n_regions": 0, "regions": "", "n_images": 0,
               "error": ""}
    try:
        x, y = ParsedData(data_path).data
        peak_regions, sampling, scut = _relevant_regions(y, x)
        n_used = int(scut)

        fcut_range, r2 = load_curve(cache_path)
        regions = candidate_regions(fcut_range, r2, n_used)
        samples = sample_regions(fcut_range, regions, ratio)

        out_dir = os.path.join(out_root, molecule, stem)
        os.makedirs(out_dir, exist_ok=True)

        plot_r2(fcut_range, r2, regions, samples,
                f"{stem} — {len(samples)} sampled fcut "
                f"(ratio {ratio:g})", os.path.join(out_dir, "00_r2.png"),
                dpi=dpi)

        # First pass: every baseline correction, so that all the images of
        # the signal can share the same y-limits.
        baseline_fitter = Baseline(x_data=x)
        corrections = []
        for _, j in samples:
            bl, params = _custom_beads(
                baseline_fitter, y, regions=peak_regions, sampling=sampling,
                freq_cutoff=float(fcut_range[j]), asymmetry=1.0,
                fit_parabola=True, alpha=1.0, parabola_len=3)
            corrections.append((bl, params["signal"]))

        lo = min([y.min()] + [min(b.min(), s.min()) for b, s in corrections])
        hi = max([y.max()] + [max(b.max(), s.max()) for b, s in corrections])
        margin = 0.05 * (hi - lo) if hi > lo else 1.0
        ylim = (lo - margin, hi + margin)

        # Second pass: the figures.
        rows = []
        for k, (reg_id, j) in enumerate(samples):
            fcut = float(fcut_range[j])
            bl, signal = corrections[k]
            name = f"k{k:02d}_r{reg_id:d}_fcut_{fcut:.3e}.png"
            plot_baseline(
                x, y, bl, signal,
                f"{stem}\nk={k:d}  region {reg_id:d}  "
                f"fcut={fcut:.4e}  $r^2$={r2[j]:.4f}",
                os.path.join(out_dir, name), ylim=ylim, dpi=dpi)
            reg = regions[reg_id]
            span = np.log(fcut_range[reg[-1]] / fcut_range[reg[0]])
            pos = (np.log(fcut / fcut_range[reg[0]]) / span
                   if span > 0 else 0.0)
            rows.append({"k": k, "region": reg_id, "fcut": f"{fcut:.6e}",
                         "r2": f"{r2[j]:.6f}", "pos_in_region": f"{pos:.3f}",
                         "image": name})

        with open(os.path.join(out_dir, "index.csv"), "w",
                  newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["k", "region", "fcut", "r2",
                                "pos_in_region", "image"])
            writer.writeheader()
            writer.writerows(rows)

        summary.update(
            n_points=len(x), n_used=n_used, n_regions=len(regions),
            regions=";".join(f"{fcut_range[r[0]]:.4e}:{fcut_range[r[-1]]:.4e}"
                             for r in regions),
            n_images=len(rows))
    except Exception as exc:  # noqa: BLE001 - one bad signal must not
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def main() -> None:
    """
    CLI entry point of the gallery renderer.

    """
    parser = argparse.ArgumentParser(
        prog="fcut_gallery",
        description="render baselines across the candidate plateau regions")
    parser.add_argument("cache_dir", help="directory of the r2 .npz cache")
    parser.add_argument("data_dir", help="root of the raw data directory")
    parser.add_argument("-od", "--output-dir", default="fcut_gallery",
                        help="output directory (default: fcut_gallery)")
    parser.add_argument("-r", "--ratio", type=float, default=1.15,
                        help="geometric step between two sampled cutoff "
                             "frequencies (default: 1.15)")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="number of worker processes (default: 1)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="resolution of the images (default: 200)")
    parser.add_argument("--pattern", default="*",
                        help="only process the stems matching this glob "
                             "pattern (default: all)")
    args = parser.parse_args()

    data_files = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(os.path.join(args.data_dir, "*", "*.txt"))}
    jobs = []
    for cache_path in sorted(glob.glob(os.path.join(args.cache_dir,
                                                    "*.npz"))):
        name = os.path.splitext(os.path.basename(cache_path))[0]
        stem = name.rsplit("__r2__", 1)[0]
        if not fnmatch.fnmatch(stem, args.pattern):
            continue
        if stem not in data_files:
            print(f"no data file for {stem}, skipped")
            continue
        jobs.append((data_files[stem], cache_path))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(jobs)} signal(s), ratio {args.ratio:g}, "
          f"{args.workers} worker(s)")

    tic = time.perf_counter()
    summaries = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_signal, d, c, args.output_dir,
                                   args.ratio, args.dpi)
                       for d, c in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                summary = future.result()
                summaries.append(summary)
                print(f"[{done:3d}/{len(jobs)}] {summary['stem']} "
                      f"{summary['n_images']} images {summary['error']}")
    else:
        for done, (data_path, cache_path) in enumerate(jobs, 1):
            summary = process_signal(data_path, cache_path, args.output_dir,
                                     args.ratio, args.dpi)
            summaries.append(summary)
            print(f"[{done:3d}/{len(jobs)}] {summary['stem']} "
                  f"{summary['n_images']} images {summary['error']}")

    summaries.sort(key=lambda s: (s["molecule"], s["stem"]))
    with open(os.path.join(args.output_dir, "gallery_index.csv"), "w",
              newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["molecule", "stem", "n_points", "n_used",
                            "n_regions", "regions", "n_images", "error"])
        writer.writeheader()
        writer.writerows(summaries)

    toc = time.perf_counter()
    failed = [s for s in summaries if s["error"]]
    total = sum(s["n_images"] for s in summaries)
    print(f"\n{total} images for {len(summaries) - len(failed)} signal(s) "
          f"in {toc - tic:0.1f} seconds")
    if failed:
        print(f"{len(failed)} signal(s) failed:")
        for s in failed:
            print(f"  {s['stem']}: {s['error']}")


if __name__ == "__main__":
    main()
