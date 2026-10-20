# Synthetic chromatograms with known ground truth

This document specifies the synthetic benchmark: what each component of a
generated signal is, **where it comes from**, and what may be cited for it.
It is written so that a number produced by scoring against this benchmark can
be traced back to a published source or to a stated measurement.

The rule followed throughout is the one the project applies to production
code: *a choice that changes a reported value needs a citation or an explicit
statement that it is unresolved*. Where a value is adopted rather than
grounded, it says so.

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

## 2. The injection / dead-time artefact

Real LPYE runs carry a bipolar excursion at the dead time: a positive spike
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
column and flow rate. Pinning it would build an instrument constant into a
benchmark, which is precisely the objection that has been raised against
fitted heuristics elsewhere in this project. The position is drawn from a
wide range and recorded per signal, so that any dependence of a constant on
it is measurable.

## 3. Noise

White Gaussian noise is **added** at a known level, so the noise component is
known exactly rather than inferred (Niezen 2022, §4.1.1, step ii).

## 4. Quantisation

The LPYE detector output is digitised at

```
q = 0.008996 mV
```

Measured, not taken from a datasheet: in all 339 reference signals every
consecutive difference is an exact integer multiple of `q` to within 1e-9, a
~900-point record holds only 45–60 distinct values, and ~25% of consecutive
samples are identical.

Assembled signals are **rounded to multiples of `q`**. This matters beyond
realism: `baseline._snr` divides by a MAD of consecutive differences, which on
quantised data is pinned to the lattice and takes only five distinct values
across the whole reference set. On continuous synthetic data the same function
returns a true signal-to-noise ratio. Without quantisation the benchmark would
exercise a different quantity under the same name — which has already happened
once, and is recorded as a TO DO in the README.

## 5. Scoring

Two metrics, because Niezen measured that they disagree about which method
wins (§4.3.2 against §4.3.3):

1. **RMSE against the known background**, Niezen Eq. (15), computed on the
   drift component alone:
   `RMSE = sqrt( Σ (b_i − z_i)² / n )`
2. **Relative error in peak area**, integrated over the known peak regions.

Niezen found that the combination giving the lowest background RMSE was *not*
the one giving the smallest peak-area errors. Since the deliverable of this
project is peak areas, both are reported; optimising on RMSE alone would risk
grounding a constant against the wrong quantity.

## 6. The `pyb` family — idealised signals from the pybaselines docs

### 6.1 Why `make_data` is not called

`pybaselines.utils.make_data` is **not called**. Its docstring states the
output "may change without notice to meet the needs of the examples in the
pybaselines documentation, so outside users are advised not to rely on the
exact output". This project has already been bitten once by an undocumented
pybaselines change — the `beads` `lam_0/1/2` defaults moved from `1.0` to
auto-scaled with no changelog entry, shifting r² from 0.885 to 0.950 at a
fixed cutoff — so a benchmark that imported it would silently depend on the
install date.

The signals are therefore generated here from the examples of the
documentation as it stood at commit `c36ce6128`, and covered by tests,
including one asserting **bit-for-bit** equality with
`pybaselines.utils.gaussian`. Note the code evaluates `h·exp(-0.5(x-c)²/σ²)` rather than
`h·exp(-0.5((x-c)/σ)²)`: algebraically identical, different in the last bits,
and the first is what the source computes.

### 6.2 The two sources and what each contributes

**[A] `docs/examples/misc/plot_beads_preprocessing.py`** — three baselines
that form a deliberate ladder of violation of the BEADS periodicity
requirement, which is the whole reason that example exists. Its own comments
name them: ends at zero on *both* ends, on *one*, on *neither*. That is the
failure mode Navarro-Huerta 2017 §3.3.1 addresses with the parabola
pre-treatment, which production enables via `fit_parabola=True`, so this axis
tests a preprocessing step nothing else in the benchmark exercises.

**[B] `docs/algorithms/algorithms_1d/misc.rst`, `create_data()`** — five
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

`asymmetry = 1` is used on exactly the dataset with negative peaks, and 6–8
on the others. Production uses `asymmetry = 1` because of the bipolar
dead-time undershoot (§2); until now that rested on local observation, and
this is the library author's demonstrated usage of the same parameter for
the same reason. It is evidence of practice, not a citable value.

Second, `freq_cutoff` spans **0.01 to 0.08** across five datasets from one
generator — an 8× range, which is Navarro-Huerta §3.1(iv)'s "the optimal
cutoff is sample-specific" shown concretely, and a sanity bound on the span
any selector must cover.

**Caveat:** those `fit_params` are chosen for illustration, not optimised, so
they are *not* ground truth for the optimum. The **baselines** are exact,
which is what the benchmark needs.

### 6.3 The eight cases

`PYB_CASES` in `synth_dataset.py`; `pyb_signal(case, seed=None)` returns
`x`, `y`, `signal`, `baseline`, `noise` and a `meta` dict. With `seed=None`
each case reproduces its published figure exactly, using the seed published
with it (1 for [B], 0 for [A]); an explicit seed produces a replicate with
the same signal and baseline but fresh noise.

### 6.4 From eight cases to a population

Eight signals cannot separate a real effect from a coincidence — the
donnie exercise showed three signals already producing a monotone
trade-off that no single threshold resolves. `pyb_random_signal(seed)`
therefore turns the fixed vocabulary into a population: Gaussian peaks on a
composed analytic baseline with white noise, every parameter drawn from a
declared range, and the signal a pure function of the seed.

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
on it, and the real LPYE records span 473–39,129 points.

**Baseline components.** One or two are summed, as the documentation itself
does (`10 - 0.005x + gaussian(x, 5, 850, 200)`): `linear`, `exponential`,
`gaussian_bump`, `sine`, `parabola`, `logistic`. Coefficient ranges come from
the collected published values — including the *signs* that appear there, so
`exponential` can rise or decay and `gaussian_bump` can be a bump or a dip
(`gaussian(x, -6, 700, 500)` appears in the classification example).

**Negative peaks** occur with probability 0.2, mirroring the published `B5`
case; when they occur a random subset of peaks is subtracted. This is the
purely-synthetic analogue of the LPYE dead-time undershoot, and the reason
`asymmetry = 1` exists in production (§6.2).

**The endpoint condition is recorded, not forced.** How close the baseline
sits to its own minimum at each end is measured and stored in
`meta['end_offsets']` rather than imposed. The periodicity axis is thus
measurable across the population without being a design variable — the fixed
`A0/A1/A2` cases cover it deliberately, and the random population covers it
incidentally, which is the more honest test.

**Metadata.** Every drawn parameter is stored in `meta`, including the
per-peak list, so any dependence of a constant on any generator parameter can
be measured after the fact rather than guessed.

## 7. What this benchmark does not establish

- **Cross-machine reproducibility bounds what can be concluded.** The swept
  `r²` curve is ill-conditioned at low cutoff: a one-ulp difference in the
  frequency grid reaches 5.6e-2 of `r²` near `fcut` 1e-4, and 6 of 339
  signals select a cutoff more than 0.1 decade apart across a numpy/scipy
  version change. No constant should be grounded to a precision finer than
  ~1e-3 near the fundamental — below that the benchmark is measuring the
  machine. See the README TO DO.

## 8. What BEADS assumes, and where this benchmark violates it

Measured on `SYNTH_ERB_2026-08-18` (432 signals, 2026-08-20), scored by
`target_rmse`, the RMS departure in mV between the baseline fitted at
the true optimum and the known baseline. "Bad" below means >= 0.10 mV,
Emmanuel's provisional line drawn by eye on the merged figures.

### 8.1 The assumptions, as the authors state them

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
addressed, not a documented violation.

### 8.2 Failure by peak case and baseline

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
31.0% high against 25.9% typical -- noise barely matters.

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

### 8.3 Crowding, not the baseline alone

Chromatographic resolution between adjacent analytes, `Rs = 1.18 (t2 -
t1) / (w1 + w2)` on FWHM, computed from the truth parameters over the
288 signals with two or more analytes:

| | n | Rs_min median | record covered by peaks |
|---|---|---|---|
| passing (<0.10) | 167 | 0.34 | 6.9% |
| failing (>=0.10) | 121 | 0.13 | 19.9% |

    failure rate, any adjacent pair below Rs = 1.5 :  46.0%  (252 signals)
    failure rate, all pairs resolved               :  13.9%  ( 36 signals)

Within `multi_narrow x erb0` alone -- same baseline, same peak case --
the 12 passing have Rs_min 0.71 and 8.1% coverage, the 12 failing have
0.12 and 14.5%. A sixfold difference in resolution inside one cell.

**It does not explain everything.** `multi_wide x erb0` inverts:
passing signals sit at Rs 0.06 and failing ones at 1.03.
`multi_mixed x erb0/erb1` show no separation at all (0.11 vs 0.10,
0.13 vs 0.09). So crowding accounts for `multi_narrow` and
`multi_mixed x erb2`, and does not account for `multi_wide x erb0`.

### 8.4 The generator makes mostly unresolved chromatograms

**252 of 288** multi-peak signals have at least one adjacent pair below
Rs = 1.5, the conventional baseline-resolution mark. Peaks covering a
fifth of the record and overlapping each other are not a sparse signal
in the sense of assumption 1, so a large part of the `multi_*` failure
population is the benchmark asking BEADS for something its model does
not describe.

### 8.5 The analytic baselines are unlike the real ones

From `runs/REAL_BASELINES_2026-08-20`, the baselines production fits to
the 339 real chromatograms:

| | p10 | median | p90 | max |
|---|---|---|---|---|
| real baseline range (mV) | 0.113 | **0.279** | 1.604 | 7.799 |
| real signal range (mV) | 1.700 | 5.092 | 14.165 | 65.500 |

Against `erb0` 5.00 mV, `erb1` 8.11 mV, `erb2` 12.13 mV. **The
benchmark's smallest baseline is eighteen times the median real one.**
Only 5 of 339 real baselines exceed 5 mV of range, and only 56 exceed
1 mV.

Plotted on a shared axis with their signals, the real baselines read as
flat with a slight tilt, or a single shallow bend; the few with real
range are monotone (`Cyclohexane C60C70C90 6`, 7.8 mV over 650 min,
zero direction changes). The only visibly structured ones are the
gradient runs where the baseline is following analyte, which is a
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
- Niezen, L.E., Schoenmakers, P.J., Pirok, B.W.J. (2022). Critical comparison
  of background correction algorithms used in chromatography. *Anal. Chim.
  Acta* **1201**, 339605. doi:10.1016/j.aca.2022.339605 —
  tool: Zenodo doi:10.5281/zenodo.6969547 (CC BY 4.0)
