#!/usr/bin/python3
"""
Render a gallery of baselines across the representable cutoff range.

For every signal of the dataset, the cutoff frequencies above the
sub-fundamental limit ``c1 / n_used`` are sampled at a constant
geometric ratio and one baseline-correction figure is written per
sampled cutoff frequency. The r2 curves are read from an existing
``r2_cache`` directory, so the expensive autocorrelation sweep is never
recomputed; only the (millisecond) final BEADS correction is run per
image.

The sampling is **not** restricted to the candidate plateaus of
``segmentation.trim_candidates``, and those regions are not drawn on
the summary figure. The 2026-07-20 gallery did both, so its labels were
confined to, and visually anchored on, the machinery they were then
used to calibrate; they could not test its boundaries. Pass ``--trim``
to restore that behaviour if you need to reproduce that gallery.

The output mirrors the layout of the ``data`` directory::

    output_dir/
        Molecule/
            signal_stem/
                00_r2.png       r2 curve, candidate regions, sampled fcut
                k00_r0_fcut_1.023e-03.png
                ...
                index_signal_stem.csv   one row per image, with an
                                        empty `label` column to fill in
                                        by hand (start / end / best)
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

from weaselytics.baseline import (  # noqa: E402
    _custom_beads,
    _relevant_regions,
    _snr,
)
from weaselytics.parsers import ParsedData  # noqa: E402
from weaselytics.segmentation import (  # noqa: E402
    classify_segments,
    detect_dips,
    dips_to_mask,
    pelt_linear,
    segment_features,
    trim_candidates,
    trim_plateaus,
)

#: Columns of the per-signal index. The last one is left empty on
#: purpose: it is filled in by hand while browsing the gallery, with
#: ``start`` / ``end`` on the images bounding the optimal region and
#: ``best`` on the best sampled cutoff frequency.
INDEX_FIELDS = ["k", "region", "fcut", "r2", "pos_in_region", "image",
                "label"]

#: Columns of the gallery-wide index, one row per signal.
GALLERY_FIELDS = ["molecule", "stem", "n_points", "n_used", "n_regions",
                  "regions", "n_images", "error"]


def load_curve(
    path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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
    sensitivity : numpy.ndarray, shape (N,), or None
        The baseline-sensitivity curve, or None for caches written before
        it was stored. Without it the instability exclusion cannot be
        applied and the gallery would show the pre-exclusion survivors.

    """
    data = np.load(path)
    key = "r2_val" if "r2_val" in data else "r2"
    sensitivity = data["sensitivity"] if "sensitivity" in data.files else None
    return data["fcut_range"], data[key], sensitivity


def existing_labels(path: str) -> dict[str, str]:
    """
    Read back the hand-filled labels of a previous run.

    Re-rendering a signal rewrites its index file, which would discard
    the manual annotations; they are carried over instead. The key is
    the cutoff frequency and not `k`, so the labels survive a re-run at
    a different sampling ratio (where `k` is renumbered but the grid
    points keep their value).

    Parameters
    ----------
    path : str
        Path of the index file of a previous run. A missing file is not
        an error and yields an empty mapping.

    Returns
    -------
    labels : dict
        Mapping of the ``fcut`` column to the non-empty labels.

    """
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as fh:
        return {row["fcut"]: row["label"] for row in csv.DictReader(fh)
                if row.get("label", "").strip()}


#: Sub-fundamental clip of ``segmentation.trim_candidates``. Mirrored
#: here so the untrimmed gallery keeps the one exclusion that is
#: physical rather than calibrated.
C1 = 0.5


def _mask_regions(mask: np.ndarray) -> list[np.ndarray]:
    """Split a boolean mask into its contiguous index runs."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return np.split(idx, breaks + 1)


def candidate_regions(fcut_range: np.ndarray, r2: np.ndarray,
                      n_used: int, trim: bool = False) -> list[np.ndarray]:
    """
    Compute the regions of a curve to sample for the gallery.

    Parameters
    ----------
    fcut_range : numpy.ndarray, shape (N,)
        The cutoff frequencies.
    r2 : numpy.ndarray, shape (N,)
        The autocorrelation coefficients.
    n_used : int
        Number of signal points used for the autocorrelation sweep.
    trim : bool, optional
        If True, restrict the regions to the candidate plateaus of
        ``segmentation.trim_candidates``. Default is False, which
        returns the whole representable grid as a single region.

    Returns
    -------
    regions : list of numpy.ndarray
        Index arrays of the contiguous regions, in increasing order of
        cutoff frequency.

    Notes
    -----
    The default is deliberately *untrimmed*. The 2026-07-20 gallery
    sampled only inside the candidate regions **and** drew them on the
    summary figure, so the resulting labels were both restricted to,
    and visually anchored on, the machinery they were later used to
    calibrate (the `refine_candidates` bracket, since removed). A label
    set collected that way cannot test the boundaries of
    `trim_candidates`.

    The sub-fundamental clip is kept even when untrimmed: below
    ``C1 / n_used`` a cutoff describes an oscillation slower than half
    a cycle over the record, which the data cannot constrain, so those
    cutoffs are not worth a human decision. Every other exclusion of
    `trim_candidates` (flatness, frozen tail, cliff bridging) is
    calibrated and is dropped here.

    """
    if not trim:
        idx = np.flatnonzero(fcut_range >= C1 / n_used)
        return [idx] if idx.size else []
    segments = classify_segments(
        segment_features(fcut_range, r2, pelt_linear(r2)))
    mask = trim_candidates(fcut_range, segments, n_used)
    return _mask_regions(mask)


def surviving_regions(fcut_range: np.ndarray, r2: np.ndarray, n_used: int,
                      s: np.ndarray, snr_threshold: float = 25.0,
                      c1: float = 1.0, sensitivity: np.ndarray | None = None
                      ) -> tuple[list[np.ndarray], dict]:
    """
    Regions surviving the stage-1 trimming, plus the overlay masks.

    Uses ``segmentation.trim_plateaus`` (the same trimming as the
    diagnostic overlay) so the gallery and the diagnostic cannot drift.
    The collapse exclusion needs the signal-to-noise ratio, hence ``s``.

    Returns
    -------
    regions : list of numpy.ndarray
        The contiguous index runs of the surviving mask.
    overlay : dict of numpy.ndarray
        ``cp_flat``, ``cp_dips``, ``cp_removed``, ``cp_snr_removed``
        and ``cp_instab_removed`` masks, for the r2 summary figure.
    """
    segments = classify_segments(
        segment_features(fcut_range, r2, pelt_linear(r2)))
    dips = detect_dips(fcut_range, r2)
    high_snr = _snr(s) >= snr_threshold
    masks = trim_plateaus(fcut_range, segments, dips, n_used,
                          exclude_collapse=high_snr, c1=c1,
                          sensitivity=sensitivity)
    cp_flat = np.zeros(len(fcut_range), dtype=bool)
    for seg in segments:
        if seg['flat']:
            cp_flat[seg['start']:seg['end']] = True
    overlay = {"cp_flat": cp_flat,
               "cp_dips": dips_to_mask(fcut_range, dips),
               "cp_removed": masks['removed'],
               "cp_snr_removed": masks['snr_removed'],
               "cp_instab_removed": masks['instab_removed']}
    return _mask_regions(masks['surviving']), overlay


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
        Limits of the y-axis. Set from the raw data and shared by every
        image of a signal, so that only the baseline moves when the
        images are browsed in order. Default is None, which autoscales.
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
            title: str, out_path: str, dpi: int = 100,
            show_regions: bool = False, overlay: dict | None = None) -> None:
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
    show_regions : bool, optional
        Draw the plain candidate hatching (trimmed gallery). Default
        False.
    overlay : dict, optional
        The detection/trimming masks (``cp_flat``, ``cp_dips``,
        ``cp_removed``, ``cp_snr_removed``) of the surviving-trim
        gallery, drawn exactly as on the diagnostic r2 top panel so the
        summary shows what was detected and what the trimming removed.
        Default None.

    """
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    tr = ax.get_xaxis_transform()
    if overlay is not None:
        # Same layers as baseline.r2_plots: detected plateaus (purple)
        # and proto-plateaus (orange), with the trimmed (red) and the
        # SNR-collapse-trimmed (dark-red hatch) marked on top.
        ax.fill_between(fcut_range, 0, 1, where=overlay["cp_flat"],
                        color="none", ec="tab:purple", alpha=0.3,
                        hatch="\\\\", hatch_linewidth=2, label="CP flat",
                        transform=tr)
        ax.fill_between(fcut_range, 0, 1, where=overlay["cp_dips"],
                        color="tab:orange", alpha=0.25,
                        label="proto-plateau", transform=tr)
        ax.fill_between(fcut_range, 0, 1, where=overlay["cp_removed"],
                        color="red", alpha=0.15, label="trimmed",
                        transform=tr)
        if np.any(overlay.get("cp_instab_removed", np.zeros(1, bool))):
            # Stiff side removed because the baseline is still flailing
            # there; same colour as the sensitivity curve of the
            # diagnostic figure.
            ax.fill_between(fcut_range, 0, 1,
                            where=overlay["cp_instab_removed"],
                            color="darkslateblue", alpha=0.30,
                            label="unstable", transform=tr)
        if np.any(overlay["cp_snr_removed"]):
            ax.fill_between(fcut_range, 0, 1,
                            where=overlay["cp_snr_removed"], color="none",
                            ec="darkred", alpha=0.9, hatch="xxx",
                            hatch_linewidth=1.0, label="SNR-trimmed",
                            transform=tr)
    elif show_regions:
        # The candidate hatching is drawn only when the sampling was
        # actually restricted to those regions. On an untrimmed gallery
        # the whole grid is one region, and shading it would tell the
        # labeller where the machinery expects the answer.
        for reg in regions:
            mask = np.zeros(len(fcut_range), dtype=bool)
            mask[reg] = True
            ax.fill_between(fcut_range, 0, 1, where=mask, color="none",
                            ec="tab:purple", alpha=0.3, hatch="\\\\",
                            hatch_linewidth=2, transform=tr)
    ax.semilogx(fcut_range, r2, marker=".", ls="", ms=3, label=r"$r^2$")
    for k, (_, j) in enumerate(samples):
        ax.axvline(fcut_range[j], color="0.35", lw=0.6, alpha=0.7)
        # Only every fifth sample is labelled: consecutive ticks are one
        # `ratio` apart and their labels would overlap.
        if k % 5 == 0:
            ax.annotate(f"{k:d}", xy=(fcut_range[j], 1.01),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=8, color="0.35")
    ax.set_xlabel("Cutoff frequency")
    ax.set_ylabel(r"$r^2_{y-b}$")
    # Pad the title above the row of sample numbers printed at the top of
    # the axes, so the two do not collide.
    ax.set_title(title, fontsize=10, pad=22)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def process_signal(data_path: str, cache_path: str, out_root: str,
                   ratio: float, dpi: int, trim: bool = False,
                   survive: bool = False,
                   snr_threshold: float = 25.0, c1: float = 1.0) -> dict:
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

        fcut_range, r2, sensitivity = load_curve(cache_path)
        if survive:
            regions, overlay = surviving_regions(fcut_range, r2, n_used, y,
                                                 snr_threshold, c1,
                                                 sensitivity=sensitivity)
        else:
            regions = candidate_regions(fcut_range, r2, n_used, trim=trim)
            overlay = None
        samples = sample_regions(fcut_range, regions, ratio)

        out_dir = os.path.join(out_root, molecule, stem)
        os.makedirs(out_dir, exist_ok=True)
        index_path = os.path.join(out_dir, f"index_{stem}.csv")
        labels = existing_labels(index_path)

        plot_r2(fcut_range, r2, regions, samples,
                f"{stem} — {len(samples)} sampled fcut "
                f"(ratio {ratio:g})", os.path.join(out_dir, "00_r2.png"),
                dpi=dpi, show_regions=trim, overlay=overlay)

        # Y-limits shared by every image of the signal, so that only the
        # baseline moves when they are browsed in order. They are set by
        # the RAW data alone: a diverging baseline at an unsuitable cutoff
        # frequency would otherwise inflate the scale of the whole series
        # and squash the chromatogram into an unreadable band. Such a
        # baseline now simply runs out of the frame, which is the
        # information wanted anyway.
        lo, hi = float(y.min()), float(y.max())
        margin = 0.05 * (hi - lo) if hi > lo else 1.0
        ylim = (lo - margin, hi + margin)

        baseline_fitter = Baseline(x_data=x)
        rows = []
        for k, (reg_id, j) in enumerate(samples):
            fcut = float(fcut_range[j])
            bl, params = _custom_beads(
                baseline_fitter, y, regions=peak_regions, sampling=sampling,
                freq_cutoff=fcut, asymmetry=1.0, fit_parabola=True,
                alpha=1.0, parabola_len=3)
            signal = params["signal"]
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
                         "image": name,
                         "label": labels.get(f"{fcut:.6e}", "")})

        with open(index_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
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
    parser.add_argument("--trim", action="store_true",
                        help="restrict the sampling to the candidate "
                             "plateaus of trim_candidates and draw them "
                             "on 00_r2.png. Off by default: it censors "
                             "and visually anchors the labels, which is "
                             "how the 2026-07-20 gallery came to be "
                             "unusable for the calibration it was "
                             "collected for")
    parser.add_argument("--survive", action="store_true",
                        help="restrict the sampling to the regions "
                             "surviving the stage-1 trimming (sub-"
                             "fundamental clip, frozen tail, the "
                             "SNR-gated collapse exclusion and the "
                             "stiff-side instability exclusion), via "
                             "segmentation.trim_plateaus, and draw them "
                             "on 00_r2.png")
    parser.add_argument("--snr-threshold", type=float, default=25.0,
                        help="SNR above which the collapse exclusion is "
                             "applied under --survive (default: 25)")
    parser.add_argument("--c1", type=float, default=1.0,
                        help="sub-fundamental clip factor for --survive: "
                             "cutoffs below c1/n_used are trimmed on the "
                             "stiff (low-frequency) side (default: 1.0)")
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
    mode = ("surviving-trim" if args.survive
            else "trimmed" if args.trim else "untrimmed")
    print(f"{len(jobs)} signal(s), ratio {args.ratio:g}, "
          f"{mode}, {args.workers} worker(s)")

    tic = time.perf_counter()
    summaries = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_signal, d, c, args.output_dir,
                                   args.ratio, args.dpi, args.trim,
                                   args.survive, args.snr_threshold,
                                   args.c1)
                       for d, c in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                summary = future.result()
                summaries.append(summary)
                print(f"[{done:3d}/{len(jobs)}] {summary['stem']} "
                      f"{summary['n_images']} images {summary['error']}")
    else:
        for done, (data_path, cache_path) in enumerate(jobs, 1):
            summary = process_signal(data_path, cache_path, args.output_dir,
                                     args.ratio, args.dpi, args.trim,
                                     args.survive, args.snr_threshold,
                                     args.c1)
            summaries.append(summary)
            print(f"[{done:3d}/{len(jobs)}] {summary['stem']} "
                  f"{summary['n_images']} images {summary['error']}")

    # Merge into the index of a previous run: a --pattern run must
    # update the rows of the signals it rendered, not truncate the file
    # to them.
    gallery_path = os.path.join(args.output_dir, "gallery_index.csv")
    merged = {}
    if os.path.isfile(gallery_path):
        with open(gallery_path, newline="") as fh:
            merged = {(r["molecule"], r["stem"]): r
                      for r in csv.DictReader(fh)}
    merged.update({(s["molecule"], s["stem"]): s for s in summaries})
    with open(gallery_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GALLERY_FIELDS)
        writer.writeheader()
        writer.writerows(v for _, v in sorted(merged.items()))

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
