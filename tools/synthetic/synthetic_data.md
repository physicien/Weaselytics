# Synthetic chromatograms with known ground truth

This document specifies the synthetic benchmark: what each component of a
generated signal is, **where it comes from**, and what may be cited for it.

Generator: [`tools/synthetic/synth_dataset.py`](./synth_dataset.py).
Tests: [`weaselytics/tests/test_synth_dataset.py`](../weaselytics/tests/test_synth_dataset.py).

---

## 1. Why a synthetic benchmark exists

The constants that decide the BEADS cutoff frequency, `drop_level` and
`instability_boundary`'s `trigger` and `settled`, are adopted, not grounded.
Grounding them requires knowing the true baseline, which no real chromatogram
provides. Niezen et al. state the dilemma directly: real data has realistic
peak overlap, noise and drift but no ground truth, while fully simulated data
has ground truth but "typically represent idealistic situations and lack
signal distortions and properties pertaining to experimental data"
(Niezen 2022, §1).

Niezen et al. resolve this with **hybrid data**: an experimental background
carrying real drift, with peaks and noise added at known parameters (Niezen
2022, §4.1, Fig. 1). A hybrid family was prototyped here and removed before
use. Its background-pool loader (`tools/synthetic/backgrounds.py`), its
peak-free-stretch selection and its modified Pearson VII peaks are in commit
`efc7aa7`, with their design in that commit's
`tools/synthetic/synthetic_data.md` §2 to §8.

## 2. The LPYE dataset

The `native` and `erb_native` families are chosen to match an experimental
dataset that is not distributed with the package: 339 chromatograms of
fullerenes and fullertubes, whose separation chemistry is described in
Bourret, Stevenson & Côté (2024). Its run conditions:

- a PYE (pyrenylethyl) column, 150 mm long and 10 mm I.D. (header code
  `SAS-150-10-30`);
- isocratic elution at 3.06 mL/min, the mobile phase varying between runs;
- UV-visible detection at one of ten wavelengths between 300 and 568 nm, most
  often 410 nm (173 runs);
- the detector output in mV, sampled at 60 (79 runs), 120 (226) or 180 (34)
  points per minute, from 473 to 39,129 points per run;
- 67 blanks among the 339 runs.

## 3. The injection / dead-time artifact

The LPYE runs carry a bipolar excursion at the dead time: a positive spike
followed by a negative undershoot, which on many blanks is the largest
feature of the trace. The negative lobe is the documented reason the
production BEADS call uses a symmetric cost (`asymmetry = 1`).

It is generated as a **separate, known component**, not as part of `b_true`.
Two reasons:

1. Niezen §4.1.1 requires a background of low-frequency drift only, and notes
   that negative peaks are the critical case because "such peaks are
   generally treated as background drift" by correction algorithms.
2. Scoring a low-pass baseline model on its ability to reproduce a sharp
   bipolar spike measures the wrong thing, and would let that one feature
   dominate the error metric.

Keeping it in the signal but out of the truth means `asymmetry = 1` is still
exercised while the baseline is scored only on the drift.

**Its position is a declared variable, not a constant.** The dead time of the
LPYE setup is ~4.5 min and near-invariant, but that is a property of one
column and flow rate. Pinning it would build an instrument constant into the
benchmark. The position is drawn from a wide range and recorded per signal, so
that any dependence of a constant on it is measurable.

## 4. Noise

White Gaussian noise is **added** at a known level, so the noise component is
known exactly rather than inferred (Niezen 2022, §4.1.1, step ii).

## 5. Quantization

The LPYE dataset is digitized at

```
q = 0.008996 mV
```

In all its records every consecutive difference is an
exact integer multiple of `q` to within 1e-9, a ~900-point record holds only
45–60 distinct values, and ~25% of consecutive samples are identical.

Assembled signals are **rounded to multiples of `q`**. This matters beyond
realism: `baseline._snr` divides by a MAD of consecutive differences, which on
quantized data is pinned to the lattice and takes only five distinct values
across the dataset. On continuous synthetic data the same function
returns a true signal-to-noise ratio. Without quantization the benchmark would
exercise a different quantity under the same name. That has happened once
already; see the README TO DO.

## 6. Scoring

Two metrics, because Niezen measured that they disagree about which method
wins (§4.3.2 against §4.3.3):

1. **RMSE against the known background**, Niezen Eq. (15), computed on the
   drift component alone:
   `RMSE = sqrt( Σ (b_i − z_i)² / n )`
2. **Relative error in peak area**, integrated over the known peak regions.

Niezen found that the combination giving the lowest background RMSE was *not*
the one giving the smallest peak-area errors. Since the deliverable of this
project is peak areas, both are reported; optimizing on RMSE alone would risk
grounding a constant against the wrong quantity.

## 7. The `pyb` family: idealized signals from the pybaselines docs

### 7.1 Why `make_data` is not called

The docstring of `pybaselines.utils.make_data` states the output "may change
without notice to meet the needs of the examples in the pybaselines
documentation, so outside users are advised not to rely on the exact output".
An undocumented pybaselines change has already moved this project's results
once: the `beads` `lam_0/1/2` defaults went from `1.0` to auto-scaled with no
changelog entry, shifting r² from 0.885 to 0.950 at a fixed cutoff. A
benchmark that imported `make_data` would likewise depend on the install date.

The signals are therefore generated here from the examples of the
documentation as it stood at commit `c36ce6128`, and covered by tests,
including one asserting **bit-for-bit** equality with
`pybaselines.utils.gaussian`. Note the code evaluates `h·exp(-0.5(x-c)²/σ²)`
rather than `h·exp(-0.5((x-c)/σ)²)`: algebraically identical, different in the
last bits, and the first is what the source computes.

### 7.2 The two sources and what each contributes

**[A] `docs/examples/misc/plot_beads_preprocessing.py`**: three baselines that
violate the BEADS periodicity requirement by degrees, which is what that
example exists to show. Its own comments name them: ends at zero on *both*
ends, on *one*, on *neither*. Navarro-Huerta 2017 §3.3.1 addresses that
failure with the parabola pre-treatment, which production enables with
`fit_parabola=True`. No other axis of the benchmark exercises it.

**[B] `docs/algorithms/algorithms_1d/misc.rst`, `create_data()`**: five
datasets varying peak density (4 / 15 / 8 / 8 / 8 peaks), noise level
(×5, ×1, ×0.5 of σ=0.2) and baseline shape, one of which carries **negative
peaks** (`signal*2 - signal_2`).

That last case matters beyond variety. The documentation fits its five
datasets with these BEADS parameters:

| dataset | peaks | alpha | asymmetry | freq_cutoff |
|---|---|---|---|---|
| y1 | positive | 500 | 6 | 0.01 |
| y2 | positive, dense | 0.01 | 8 | 0.08 |
| y3 | positive | 80 | 8 | 0.01 |
| y4 | positive | 0.2 | 6 | 0.04 |
| **y5** | **negative present** | 100 | **1** | 0.01 |

`asymmetry = 1` is used on exactly the dataset with negative peaks, and 6–8 on
the others. Production uses `asymmetry = 1` because of the bipolar dead-time
undershoot (§3), and the library author uses the same value for the same
reason.

Second, `freq_cutoff` spans **0.01 to 0.08** across the five datasets of one
generator, an 8× range. That is Navarro-Huerta §3.1(iv)'s "the optimal cutoff
is sample-specific" made concrete, and a sanity bound on the span any selector
must cover.

**Caveat:** those `fit_params` are chosen for illustration, not optimized, so
they are *not* ground truth for the optimum. The **baselines** are exact,
which is what the benchmark needs.

### 7.3 The eight cases

`PYB_CASES` in `synth_dataset.py`; `pyb_signal(case, seed=None)` returns
`x`, `y`, `signal`, `baseline`, `noise` and a `meta` dict. With `seed=None`
each case reproduces its published figure exactly, using the seed published
with it (1 for [B], 0 for [A]); an explicit seed produces a replicate with
the same signal and baseline but fresh noise.

### 7.4 From eight cases to a population

Eight signals cannot separate a real effect from a coincidence: three of Erb's
signals already produced a monotone trade-off that no single threshold
resolves. `pyb_random_signal(seed)` therefore turns the fixed vocabulary into
a population: Gaussian peaks on a composed analytic baseline with white noise,
every parameter drawn from a declared range, and the signal a pure function of
the seed.

**Every range is the span of values actually used across the pybaselines
documentation** at the pinned commit, collected from `docs/examples/*/*.py`,
`docs/algorithms/algorithms_1d/*.rst` and `utils.make_data`. Where a range is
*wider* than the published span, it is marked and justified:

| parameter | range | provenance |
|---|---|---|
| peak count | 4 – 15 | published datasets use 4, 7, 8, 15 |
| peak height | 4 – 40 | published 4–20; ×2 because two datasets use `signal * 2` |
| peak centre | 0.10 – 0.88 of span | published centres 100–880 on a span of 1000 |
| peak σ | 0.005 – 0.020 of span | published σ 5–20 on a span of 1000 |
| noise σ | 0.025 – 1.0 | published 0.05 and 0.2, scaled ×0.5, ×1, ×5 |
| record length | 300 – 4000 | **WIDENED** from the published 500/1000 |
| sine period | 30 – 150 | **WIDENED** from the single published `x/50` |

The record-length widening is deliberate and load-bearing: length sets the
fundamental, `1/n_used`, which is exactly what `instability_boundary` keys
on. A benchmark fixed at two lengths could not detect a constant that depends
on it, and the LPYE records span 473–39,129 points.

**Baseline components.** One or two are summed, as the documentation itself
does (`10 - 0.005x + gaussian(x, 5, 850, 200)`): `linear`, `exponential`,
`gaussian_bump`, `sine`, `parabola`, `logistic`. Coefficient ranges come from
the collected published values, signs included, so
`exponential` can rise or decay and `gaussian_bump` can be a bump or a dip
(`gaussian(x, -6, 700, 500)` appears in the classification example).

**Negative peaks** occur with probability 0.2, mirroring the published `B5`
case; when they occur a random subset of peaks is subtracted. This is the
purely-synthetic analogue of the LPYE dead-time undershoot, and the reason
`asymmetry = 1` exists in production (§7.2).

**The endpoint condition is recorded, not forced.** How close the baseline
sits to its own minimum at each end is measured and stored in
`meta['end_offsets']` rather than imposed. The periodicity axis is thus
measurable across the population without being a design variable: the fixed
`A0/A1/A2` cases cover it deliberately, and the random population covers it
incidentally.

**Metadata.** Every drawn parameter is stored in `meta`, including the
per-peak list, so any dependence of a constant on any generator parameter can
be measured after the fact.

## 8. What this benchmark does not establish

- **Cross-machine reproducibility bounds what can be concluded.** The swept
  `r²` curve is ill-conditioned at low cutoff: a one-ulp difference in the
  frequency grid reaches 5.6e-2 of `r²` near `fcut` 1e-4, and 6 of the 339
  LPYE records select a cutoff more than 0.1 decade apart across a numpy/scipy
  version change. No constant should be grounded to a precision finer than
  ~1e-3 near the fundamental; below that, the benchmark measures the
  machine. See the README TO DO.

## 9. What BEADS assumes, and where this benchmark violates it

Measured on `SYNTH_ERB_2026-08-18` (432 signals, 2026-08-20), scored by
`target_rmse`, the RMS departure in mV between the baseline fitted at
the true optimum and the known baseline. "Bad" below means >= 0.10 mV,
Emmanuel's provisional line drawn by eye on the merged figures.

### 9.1 The assumptions, as the authors state them

Ning, Selesnick & Duval (2014):

1. **Peaks.** `x` and its first *M* derivatives are sparse (§3 p.158,
   §3.2). With M = 2 that is the peaks themselves plus their first and
   second derivatives.
2. **Baseline.** Low-pass only (§1, §3.1, Eq. 16-17). No parametric
   form, and **no stated rule relating the cutoff to peak width**.
3. **Noise.** Stationary white Gaussian, which is what justifies the
   quadratic fidelity term (Eq. 7-8, §3.2).
4. **Positivity.** A preference encoded in the asymmetry parameter `r`
   (§3.4), not a constraint; `r = 1` is the symmetric penalty.

Their validation: Gaussian peaks, two baseline types (polynomial plus
sinusoid; low-pass-filtered white noise), 500 realizations, SNR -5 to
25 dB (§5.1-5.2). **No peak density or resolution is reported**, and
the peak-generation procedure is cited to another reference rather
than specified.

Navarro-Huerta et al. (2017) list five limitations of conventional
BEADS (§3.1 i-v): the periodicity requirement, ripples under peaks of
very disparate size, sporadic negative peaks, `fc`/lambda instability at
low frequency, and per-chromatogram retuning. **Peak overlap is not
among them.** Their "coeluting peaks (sometimes highly overlapped)" is
motivation in the introduction; §3.6's "peak in a cluster" concerns
quantification error after subtraction, not baseline correctness.

**Both papers are silent on overlap, resolution and peak density.** The
split below is therefore a finding outside what either author
addressed.

### 9.2 Failure by peak case and baseline

Percentage at or above 0.10 mV, 72 signals per peak case, 144 per
baseline:

| peak case | <0.05 | 0.05-0.10 | 0.10-0.20 | >=0.20 | bad |
|---|---|---|---|---|---|
| blank | 69 | 3 | 0 | 0 | **0.0%** |
| single_narrow | 60 | 10 | 2 | 0 | 2.8% |
| isocratic | 49 | 14 | 9 | 0 | 12.5% |
| multi_narrow | 15 | 25 | 17 | 15 | 44.4% |
| multi_mixed | 19 | 20 | 14 | 19 | 45.8% |
| multi_wide | 21 | 4 | 12 | 35 | **65.3%** |
| **all** | 233 | 76 | 54 | 69 | **28.5%** |

By baseline: `erb0` 18.1%, `erb1` 18.8%, **`erb2` 48.6%**. By noise,
31.0% high against 25.9% typical, so noise barely matters.

The cross-tab, bad over 24 per cell:

```
                  erb0    erb1    erb2
blank             0/24    0/24    0/24
single_narrow     0/24    0/24    2/24
isocratic         2/24    0/24    7/24
multi_mixed       7/24    7/24   19/24
multi_narrow     12/24    2/24   18/24
multi_wide        5/24   18/24   24/24
```

`blank` is clean on every baseline: whatever `erb2` does to the others,
it does not defeat BEADS without analyte peaks. Failure needs both a
hard baseline and peaks.

### 9.3 Crowding, not the baseline alone

Chromatographic resolution between adjacent analytes, `Rs = 1.18 (t2 -
t1) / (w1 + w2)` on FWHM, computed from the truth parameters over the
288 signals with two or more analytes:

| | n | Rs_min median | record covered by peaks |
|---|---|---|---|
| passing (<0.10) | 167 | 0.34 | 6.9% |
| failing (>=0.10) | 121 | 0.13 | 19.9% |

    failure rate, any adjacent pair below Rs = 1.5 :  46.0%  (252 signals)
    failure rate, all pairs resolved               :  13.9%  ( 36 signals)

Within `multi_narrow x erb0` alone, where baseline and peak case are the
same, the 12 passing have Rs_min 0.71 and 8.1% coverage, the 12 failing have
0.12 and 14.5%. A sixfold difference in resolution inside one cell.

**It does not explain everything.** `multi_wide x erb0` inverts:
passing signals sit at Rs 0.06 and failing ones at 1.03.
`multi_mixed x erb0/erb1` show no separation at all (0.11 vs 0.10,
0.13 vs 0.09). So crowding accounts for `multi_narrow` and
`multi_mixed x erb2`, and does not account for `multi_wide x erb0`.

### 9.4 The generator makes mostly unresolved chromatograms

**252 of 288** multi-peak signals have at least one adjacent pair below
Rs = 1.5, the conventional baseline-resolution mark. Peaks covering a
fifth of the record and overlapping each other are not a sparse signal
in the sense of assumption 1, so a large part of the `multi_*` failure
population is the benchmark asking BEADS for something its model does
not describe.

### 9.5 The analytic baselines are unlike the LPYE ones

The baselines production fits to the 339 LPYE chromatograms:

| | p10 | median | p90 | max |
|---|---|---|---|---|
| LPYE baseline range (mV) | 0.113 | **0.279** | 1.604 | 7.799 |
| LPYE signal range (mV) | 1.700 | 5.092 | 14.165 | 65.500 |

Against `erb0` 5.00 mV, `erb1` 8.11 mV, `erb2` 12.13 mV. **The
benchmark's smallest baseline is eighteen times the median LPYE one.**
Only 5 of the 339 baselines exceed 5 mV of range, and only 56 exceed
1 mV.

Plotted on a shared axis with their signals, the LPYE baselines read as
flat with a slight tilt, or a single shallow bend; the few with appreciable
range are monotone (`Cyclohexane C60C70C90 6`, 7.8 mV over 650 min,
zero direction changes). The only visibly structured ones are the
runs where the baseline is following analyte, which is a
selection fault rather than a drift characteristic.

Do **not** read the direction-change counts from a plot that puts the
baseline on its own y-axis: a 0.05 mV wander then looks like a 7.8 mV
drift, and the statistic is meaningless at that amplitude.

## References

- Ning, X., Selesnick, I.W., Duval, L. (2014). Chromatogram baseline
  estimation and denoising using sparsity (BEADS). *Chemom. Intell. Lab.
  Syst.* **139**, 156–167. doi:10.1016/j.chemolab.2014.09.014
- Navarro-Huerta, J.A. et al. (2017). Assisted baseline subtraction in
  complex chromatograms using the BEADS algorithm. *J. Chromatogr. A*
  **1507**, 1–10. doi:10.1016/j.chroma.2017.05.057
- Bourret, E., Stevenson, S., Côté, M. (2024). Anisotropic Contributions in
  the Chromatographic Elution Behavior of Fullerenes and Fullertubes.
  *J. Phys. Chem. C* **128**, 13283–13298.
- Niezen, L.E., Schoenmakers, P.J., Pirok, B.W.J. (2022). Critical comparison
  of background correction algorithms used in chromatography. *Anal. Chim.
  Acta* **1201**, 339605. doi:10.1016/j.aca.2022.339605;
  tool: Zenodo doi:10.5281/zenodo.6969547 (CC BY 4.0)
