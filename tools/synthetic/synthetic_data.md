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

The constants that decide the BEADS cutoff frequency — `collapse_level`, and
`instability_boundary`'s `trigger` and `settled` — are adopted, not grounded.
Grounding them requires knowing the true baseline, which no real chromatogram
provides. Niezen et al. state the dilemma directly: real data has realistic
peak overlap, noise and drift but no ground truth, while fully simulated data
has ground truth but "typically represent idealistic situations and lack
signal distortions and properties pertaining to experimental data"
(Niezen 2022, §1).

The resolution used here is theirs: **hybrid data** — an experimental
background carrying real drift, with peaks and noise added at known
parameters, so the background and every peak area are known exactly
(Niezen 2022, §4.1, Fig. 1).

## 2. Architecture

A generated signal is the sum of four components, each stored separately in
the truth file:

```
y  =  drift  +  peaks  +  artefact  +  noise      then quantised
      ^^^^^
      this, and only this, is b_true
```

| component | what it is | source |
|---|---|---|
| `drift` | a real, peak-free chromatographic background | measured, §3 |
| `peaks` | modified Pearson VII profiles at known areas | Niezen Eq. (14), §4 |
| `artefact` | the bipolar injection/dead-time excursion | measured, §5 |
| `noise` | white Gaussian at a known level | Niezen §4.1.1, §6 |

Storing them separately is what makes `b_true` unambiguous. It is the drift
alone: the algorithm is not scored on reproducing the injection artefact,
which is a sharp bipolar feature that a low-pass baseline model is not
designed to represent (see §5).

## 3. Drift — the experimental background

### 3.1 Method

Niezen et al. build their backgrounds from a library of experimental blanks,
requiring "an experimental background that contains only low-frequency drift
and a small amount of initial noise", and manually removing any peaks by
curve fitting and subtraction (Niezen 2022, §4.1.1). Milani et al. use the
same device from the other direction, overlaying "the signal obtained from a
blank measurement … onto the peak signal, ensuring alignment of the baseline
and noise characteristics with the original dataset" (Milani 2024, §3.2).

Here the same requirement is met by **extracting the longest peak-free
stretch** of a real chromatogram rather than by removing peaks from it.
Nothing is fitted or subtracted, so the drift is the recorded signal itself.

The stretch is located as the longest run with no local excursion above
`8 σ`, where the residual is taken against a median filter of width
`max(31, N/40)` and `σ` is the derivative-MAD of §6. **The factor 8 and the
filter width are not grounded**; they are a selection heuristic, and the
regions they produce are reviewed by eye before use. They never enter a
reported number — they only decide which real data is admitted to the pool.

### 3.2 Sources

| source | count | spans | instrument | licence / provenance |
|---|---|---|---|---|
| LPYE dataset | ~130 | 5 – 554 min | Cosmosil PYE (semi-prep), 3.06 mL/min, 8 detection wavelengths | local, `data/` |
| Niezen `FILTER` | 2 of 5 usable | 15.0, 16.0 min | Agilent 1260 DAD; Waters Acquity RI | Zenodo `10.5281/zenodo.6969547`, CC BY 4.0 |
| MOCCA2 | ~5 usable | 4.0 – 7.6 min | Agilent ChemStation DAD; Waters | `github.com/bayer-group/MOCCA`, MIT |

Notes on each:

- **LPYE** is this project's shorthand for a semi-preparative **Nacalai
  Tesque Cosmosil PYE** column, a pyrenylethyl stationary phase run in
  reversed phase. The separation chemistry is described in Bourret,
  Stevenson & Côté, *J. Phys. Chem. C* **128**, 13283–13298 (2024). The
  analytes are fullerenes and fullertubes; the directory names in `data/`
  are the **mobile phase**, not the analyte.

  The run conditions below are read from the header line of each raw file
  rather than from the paper, because they vary per run and the paper
  describes a different configuration. Across all 339 files:

  | field | value |
  |---|---|
  | flow rate | 3.06 mL/min — **constant on all 339** |
  | column code in header | `SAS-150-10-30` — constant on all 339 |
  | stationary-phase token | `LGPYE` (321) / `LPYE` (18) — the same column, recorded two ways over time |
  | detection wavelength | **varies**: 410 nm (173), 568 (60), 420 (60), 450 (13), 478 (10), 340 (10), 300 (9), 377 (2) |

  The header code decodes as the initials of the collaborator who recorded
  the runs (`SAS`) followed by the column geometry: **150 mm length,
  10 mm I.D.**; the trailing `30` is not established. Note this is a
  *different column* from the 250 mm one described in the paper below —
  which is the concrete reason to read run parameters from the headers
  rather than from the publication.

  The varying wavelength is worth noting as a property of the pool rather
  than a nuisance: absorbance scale and therefore the drift-to-noise ratio
  differ between them, so the LPYE backgrounds already span eight detection
  conditions on one column.

  **Do not quote instrument parameters for this dataset from the paper.**
  Its column dimensions do not match the `SAS-150-10-30` code carried in the
  headers, and 478 nm — the value of the first file one happens to open — is
  10 of 339. Read the headers.

  LPYE carries the length and the bulk of the pool, and is the only source
  that is **quantised** (§7). Backgrounds are the peak-free stretches of
  ordinary runs, not dedicated blanks: the recorded blanks stop shortly
  after the injection peak (median 2.4 min beyond it) and are too short.
- **`FILTER`** ships five backgrounds; three span ≤ 1 min — they are LC×LC
  modulation slices, not 1D runs — so only two are usable at native
  duration. They arrive min-max normalised to a range of exactly 1.0
  (Niezen 2022, §4.1.3), so they carry shape only, not amplitude.
- **MOCCA2** contributes real traces from two further vendors. Reaction
  screening runs are short, so it adds diversity rather than length. Its
  DAD data is 2D; a single wavelength is taken, which is a declared
  transformation. Note the *deprecated* `HaasCP/mocca` repository contains
  no data at all, and the Zenodo benchmark attached to Haas 2023 is CADET
  simulation output with **no baseline and no noise** (SI §S7) — it cannot
  serve as a background.

### 3.3 Record length

Records take the **native duration** of their background; drift is never
stretched in time. Rescaling a drift rescales its frequency content, and
`fcut` is a frequency parameter, so a stretch factor would silently become a
variable of the thing being measured. This is a deliberate restriction, and
it costs coverage: backgrounds up to 145 min cover 96% of the LPYE run-length
distribution, and the two Cyclohexane backgrounds (526 and 555 min peak-free)
cover the rest. Rescaling remains available as a declared, recorded variable
if the pool proves too small.

## 4. Peaks — modified Pearson VII

### 4.1 The model

```
f(t) = A · ( 1 + (t − t_R)² / ( m · (σ + A_s (t − t_R))² ) )^(−m)
```

This is Niezen 2022 Eq. (14). Milani 2024 Eq. (2) is the same function under
a different name, "Skewed Lorentz-Normal", with `h/πγ ↔ A`, `γ ↔ σ`,
`M ↔ m`, `E ↔ A_s`; their Fig. 3E labels the fit "modified Pearson VII".
The shape interpolates between Lorentzian (`m → 1`) and Gaussian
(`m → ∞`), with `A_s` producing tailing (positive) or fronting (negative).

**Why this shape.** Three independent selections agree:

| evidence | result |
|---|---|
| Niezen 2022, Table 1 — AIC over **15 distributions** on real peaks | Pearson VII best; ΣAIC −7.20e3 vs −6.91e3 for EMG |
| Milani 2024, §3.1.2 — RMSE over **458 fitted peaks** | ≤ 0.0045 vs ≤ 0.0048 for a Gaussian |
| **this project**, 60 random LPYE peaks, AIC per Niezen Eq. (13) | ΣAIC −69,359 vs −68,002 (EMG), −55,209 (Gaussian); best on 29/60 |

The third row is a measurement made here, not a citation. Method: an isolated
peak is windowed at ±3 half-widths, linearly detrended from the first to the
last point of the window (Niezen 2022, §4.1.2), and fitted by soft-L1 least
squares; models are compared by `AIC = n·ln(SSE/n) + 2K`, Niezen Eq. (13).

### 4.2 Parameter ranges

Niezen Table 2 gives the ranges fitted to their data. Measured on LPYE peaks
they do not transfer, in two specific ways:

| parameter | Niezen Table 2 | LPYE (60 peaks, this project) |
|---|---|---|
| kurtosis `m` | 3.5 – 51 | median 11.4, p10–p90 2.2 – 2.1e4 |
| asymmetry `A_s` | 0.01 – 0.28 (tailing only) | median +0.072, p10–p90 −0.191 – +0.174 |

Two departures: LPYE contains peaks that are effectively Gaussian
(`m` ≫ 51), and a substantial fraction that **front** (`A_s < 0`), which a
positive-only range excludes. Niezen anticipate this — their ranges come
from gradient-RPLC of small uncharged molecules, and they state the
parameters "may change significantly in other modes of chromatography"
(§4.1.3).

The generator therefore samples the **union** of the published and the
measured ranges, so that the benchmark represents *a* chromatogram rather
than one instrument's. Both provenances are recorded in
`PEARSON7_KURTOSIS` / `PEARSON7_ASYMMETRY` and in this section.

### 4.3 Numerical note

For `A_s ≠ 0` the denominator vanishes at `t − t_R = −σ/A_s`, beyond which
the expression rises into a spurious second lobe. Measured at 9.3e-7 of peak
height at the widest fitted asymmetry — five orders below the quantisation
step — it is nonetheless clipped to zero, so the profile is single-lobed by
construction rather than by numerical accident. Covered by
`test_single_lobed_past_the_singularity`.

## 5. The injection / dead-time artefact

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

## 6. Noise

White Gaussian noise is **added** at a known level, so the noise component is
known exactly rather than inferred (Niezen 2022, §4.1.1, step ii).

To choose the level, the residual noise already present in a background must
be estimated. Niezen use the median absolute deviation with the
normal-consistency factor `k = 1.4826`, and note that in the presence of
drift and peaks a more representative estimate comes from the **first
derivative** of the signal, their Eq. (12b):

```
σ = k · median | dx_i − median(dx) |
```

**Caveat specific to this instrument.** On LPYE data this estimator does not
measure analogue noise: the detector output is quantised (§7), and the
statistic returns a small integer multiple of the quantisation step on nearly
every signal. It is therefore an upper bound on the true analogue noise, not
a measurement of it. The added noise level is a declared variable of the
benchmark regardless, so this affects the interpretation of the *background's*
residual noise, not the known noise the generator adds.

## 7. Quantisation

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

Backgrounds from `FILTER` and MOCCA2 are **not** quantised at source; when
used, the assembled signal is quantised as a whole, so the property belongs to
the synthetic instrument rather than to the borrowed drift.

## 8. Scoring

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

## 9. The `pyb` family — idealised signals from the pybaselines docs

Sections 1–8 describe the **hybrid** family. It is not the only one, and it
should not be: hybrid data buys realism at the cost of a background nobody
else can reproduce. Purely synthetic signals are worth keeping alongside it
because they are exactly reproducible, comparable with the literature, and
free of any instrument.

There are three families, with deliberately different provenance:

| family | peaks | baseline | source |
|---|---|---|---|
| `lit` | Gaussians | Type-1 poly+sinusoid, Type-2 low-pass | Ning 2014 §5 |
| **`pyb`** | Gaussians | eight cases, below | pybaselines docs |
| `hybrid` | modified Pearson VII | real peak-free drift | Niezen 2022, Milani 2024 |

### 9.1 Transcribed, not imported

`pybaselines.utils.make_data` is **not called**. Its docstring states the
output "may change without notice to meet the needs of the examples in the
pybaselines documentation, so outside users are advised not to rely on the
exact output". This project has already been bitten once by an undocumented
pybaselines change — the `beads` `lam_0/1/2` defaults moved from `1.0` to
auto-scaled with no changelog entry, shifting r² from 0.885 to 0.950 at a
fixed cutoff — so a benchmark that imported it would silently depend on the
install date.

The formulas are therefore copied from the documentation at the pinned
commit `c36ce6128` and covered by tests, including one asserting
**bit-for-bit** equality with `pybaselines.utils.gaussian`. Note the
transcription evaluates `h·exp(-0.5(x-c)²/σ²)` rather than
`h·exp(-0.5((x-c)/σ)²)`: algebraically identical, different in the last bits,
and the first is what the source computes.

### 9.2 The two sources and what each contributes

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
dead-time undershoot (§5); until now that rested on local observation, and
this is the library author's demonstrated usage of the same parameter for
the same reason. It is evidence of practice, not a citable value.

Second, `freq_cutoff` spans **0.01 to 0.08** across five datasets from one
generator — an 8× range, which is Navarro-Huerta §3.1(iv)'s "the optimal
cutoff is sample-specific" shown concretely, and a sanity bound on the span
any selector must cover.

**Caveat:** those `fit_params` are chosen for illustration, not optimised, so
they are *not* ground truth for the optimum. The **baselines** are exact,
which is what the benchmark needs.

### 9.3 The eight cases

`PYB_CASES` in `synth_dataset.py`; `pyb_signal(case, seed=None)` returns
`x`, `y`, `signal`, `baseline`, `noise` and a `meta` dict. With `seed=None`
each case reproduces its published figure exactly, using the seed published
with it (1 for [B], 0 for [A]); an explicit seed produces a replicate with
the same signal and baseline but fresh noise.

### 9.4 From eight cases to a population

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
`asymmetry = 1` exists in production (§9.2).

**The endpoint condition is recorded, not forced.** How close the baseline
sits to its own minimum at each end is measured and stored in
`meta['end_offsets']` rather than imposed. The periodicity axis is thus
measurable across the population without being a design variable — the fixed
`A0/A1/A2` cases cover it deliberately, and the random population covers it
incidentally, which is the more honest test.

**Metadata.** Every drawn parameter is stored in `meta`, including the
per-peak list, so any dependence of a constant on any generator parameter can
be measured after the fact rather than guessed.

## 10. What this benchmark does not establish

- **Long records are LPYE-only.** No published source ships peak-free
  chromatographic background beyond ~16 min. Above that the drift is
  necessarily from one instrument.
- **The background-selection heuristic is not grounded** (§3.1). It is
  reviewed by eye, and it decides only which real data enters the pool.
- **The `8 σ` peak-free criterion cannot certify absence of peaks**, only
  absence of excursions it can see. A very broad, low feature could survive
  it and would then be scored as drift.
- **Cross-machine reproducibility bounds what can be concluded.** The swept
  `r²` curve is ill-conditioned at low cutoff: a one-ulp difference in the
  frequency grid reaches 5.6e-2 of `r²` near `fcut` 1e-4, and 6 of 339
  signals select a cutoff more than 0.1 decade apart across a numpy/scipy
  version change. No constant should be grounded to a precision finer than
  ~1e-3 near the fundamental — below that the benchmark is measuring the
  machine. See the README TO DO.

## 11. What BEADS assumes, and where this benchmark violates it

Measured on `SYNTH_ERB_2026-08-18` (432 signals, 2026-08-20), scored by
`target_rmse`, the RMS departure in mV between the baseline fitted at
the true optimum and the known baseline. "Bad" below means >= 0.10 mV,
Emmanuel's provisional line drawn by eye on the merged figures.

### 11.1 The assumptions, as the authors state them

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

### 11.2 Failure by peak case and baseline

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

### 11.3 Crowding, not the baseline alone

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

### 11.4 The generator makes mostly unresolved chromatograms

**252 of 288** multi-peak signals have at least one adjacent pair below
Rs = 1.5, the conventional baseline-resolution mark. Peaks covering a
fifth of the record and overlapping each other are not a sparse signal
in the sense of assumption 1, so a large part of the `multi_*` failure
population is the benchmark asking BEADS for something its model does
not describe.

### 11.5 The analytic baselines are unlike the real ones

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
- Milani, N.B.L. et al. (2024). Generating realistic data through modeling
  and parametric probability for the numerical evaluation of data processing
  algorithms in two-dimensional chromatography. *Anal. Chim. Acta* **1312**,
  342724. doi:10.1016/j.aca.2024.342724
- Haas, C.P. et al. (2023). Open-Source Chromatographic Data Analysis for
  Reaction Optimization and Screening. *ACS Cent. Sci.* **9**, 307–317.
  doi:10.1021/acscentsci.2c01042 — data: `github.com/bayer-group/MOCCA` (MIT)
- Bourret, E., Stevenson, S., Côté, M. (2024). Anisotropic Contributions in
  the Chromatographic Elution Behavior of Fullerenes and Fullertubes.
  *J. Phys. Chem. C* **128**, 13283–13298. — the PYE separation chemistry
  behind the local dataset. Cite it for the chemistry, **not** for this
  dataset's run parameters: those are in the raw-file headers and differ.
