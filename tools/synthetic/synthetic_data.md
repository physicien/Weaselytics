# Synthetic chromatograms with known ground truth

Generator: [`tools/synthetic/synth_dataset.py`](./synth_dataset.py).
Tests: [`weaselytics/tests/test_synth_dataset.py`](../weaselytics/tests/test_synth_dataset.py).

## 1. Why a synthetic benchmark exists

Judging how well a method selects the BEADS cutoff frequency requires knowing the true baseline, which no real chromatogram provides. Niezen et al. \[[1](#niezen)\] state the dilemma: real data has realistic peak overlap, noise and drift but no ground truth, while fully simulated data has ground truth but "typically represent idealistic situations and lack signal distortions and properties pertaining to experimental data".

Niezen et al. \[[1](#niezen)\] resolve this with **hybrid data** made of an experimental background carrying real drift, with peaks and noise added at known parameters. A hybrid family was prototyped here. Its background-pool loader (`tools/synthetic/backgrounds.py`), its peak-free-stretch selection and its modified Pearson VII peaks are in commit `efc7aa7`.

## 2. What a signal is

A signal is a peak component, a baseline and white noise, summed and rounded to a detector's voltage step (§5). The truth file `truth/<stem>__truth.npz` stores each part separately, and the baseline alone is the truth a baseline estimate is scored against.

Three families share this format:

- `erb` puts the peak component of §3 on the analytic baselines of §4;
- `native`, deprecated, puts the same peak component on a sum of drawn components;
- `lit`, deprecated, is the BEADS benchmark of Ning et al. \[[2](#ning)\].

The ranges below were picked for this benchmark.

## 3. The peak component

**Analytes.** Exponentially-modified Gaussian peaks whose widths follow the plate model, σ = tR / √N, with tR counted from injection (Snyder et al. \[[3](#snyder)\]). The plate number N is drawn once per signal between 2,000 and 14,000, around the N of about 10,000 that they call a good starting point in many cases. A width is then a fixed share of its retention time, so the first peaks are sharp and the late ones broad. The cases, named by analyte count, are `blank` (none), `single`, `few` (3 or 4), `many` (6 to 8) and `isocratic` (7, with an early and a late eluter pinned).

**Placement.** Every analyte stays within the noise until the injection artifact has returned to it, is back within the noise at the last sample, and keeps a resolution Rs ≥ 1 from every other analyte, with `Rs = 1.18 (t2 − t1) / (w1 + w2)` on the apex and the FWHM. When an analyte fits nowhere, the dead time and the analytes are drawn again with an earlier dead time; an analyte that still fits nowhere with the dead time at 2.5 min is dropped. The manifest records the analytes kept and dropped, the resolution, the coverage, N and the dead time.

**The injection artifact.** A positive spike followed by a negative undershoot marks the injection at the dead time, drawn between 2.5 and 6.5 min. The dip, as the trace shows it, is 0 to 2 times the spike, drawn as twice a Beta(1, 3) variate. The production BEADS call uses a symmetric cost (`asymmetry = 1`) because of this negative lobe. A blank holds only this artifact.

The artifact is part of the signal, and `b_true` holds the baseline alone. Niezen et al. \[[1](#niezen)\] require a background of low-frequency drift only, and note that negative peaks are the critical case because "such peaks are generally treated as background drift" by correction algorithms. A sharp bipolar spike would also dominate a baseline error metric. So `asymmetry = 1` is exercised while the baseline is scored on the drift alone.

## 4. Baselines and record length

Donald Erb, author of pybaselines, wrote four analytic baselines to test the cutoff selection of this project. The three used are a `parabola` ending at zero at both ends, an exponential decay (`exp`), and a `step`, a logistic built by integrating a Gaussian, plus a Gaussian bump. The fourth, a sinusoid, is excluded because an HPLC baseline does not oscillate through several full cycles within one run. Each is evaluated on a 0–1000 abscissa spanning the record, the range its coefficients are written for, and read as mV. The parabola spans 5 mV, the exponential 8.1 mV and the step 12.1 mV.

Record length sets the fundamental frequency, 1/n_used, which `instability_boundary` keys on, so it must vary. log10 of the length is drawn from a skew-normal and redrawn until the length falls between 500 and 40,000 points. Lengths are drawn as durations in minutes, at 120 points per minute by default (4.2 to 333 min); `--rate` changes only how densely a run is sampled.

## 5. Noise and quantization

White Gaussian noise is **added** at a known level, 0.02 or 0.05 mV, so the noise component is known exactly (Niezen et al. \[[1](#niezen)\]). The assembled signal is then rounded to multiples of a voltage step, q = 0.008996 mV, as a digitized detector signal is. `baseline._snr` divides by a MAD of consecutive differences, which on quantized data is pinned to the lattice and on continuous data is a true noise estimate. Without quantization the benchmark would exercise a different quantity under the same name; see the README TO DO.

## 6. Scoring

Niezen et al. \[[1](#niezen)\] found that the combination with the lowest background RMSE was not the one with the smallest peak-area errors, so both are reported:

1. RMSE against the known baseline, `RMSE = sqrt( Σ (b_i − z_i)² / n )`;
2. relative error in peak area, integrated over the known peak regions.

The project's deliverable is peak areas.

## 7. Limits

Cross-machine reproducibility bounds what can be concluded. The swept `r²` curve is ill-conditioned at low cutoff. A one-ulp difference in the frequency grid reaches 5.6e-2 of `r²` near `fcut` 1e-4, and across a numpy/scipy version change some selected cutoffs move by more than 0.1 decade. No result should rest on a precision finer than ~1e-3 near the fundamental; below that, the benchmark measures the machine. See the README TO DO.

## 8. What BEADS assumes

Ning et al. \[[2](#ning)\] make four assumptions.

1. **Peaks.** `x` and its first *M* derivatives are sparse. With M = 2 that is the peaks themselves plus their first and second derivatives.
2. **Baseline.** Low-pass only. No parametric form, and **no stated rule relating the cutoff to peak width**.
3. **Noise.** Stationary white Gaussian, which is what justifies the quadratic fidelity term.
4. **Positivity.** A preference encoded in the asymmetry parameter `r`, not a constraint; `r = 1` is the symmetric penalty.

They validated on Gaussian peaks, two baseline types (polynomial plus sinusoid; low-pass-filtered white noise), 500 realizations and an SNR from -5 to 25 dB. **No peak density or resolution is reported**, and the peak-generation procedure is cited to another reference rather than specified.

Navarro-Huerta et al. \[[4](#nh)\] list five limitations of conventional BEADS, namely the periodicity requirement, ripples under peaks of very disparate size, sporadic negative peaks, `fc`/lambda instability at low frequency, and per-chromatogram retuning. **Peak overlap is not among them.**

The generator keeps neighbouring analytes at Rs ≥ 1 (§3), which still allows small overlaps, to test the limits of the sparsity assumption.

## References

<a name="niezen"></a>
1. Niezen, L.E.; Schoenmakers, P.J.; Pirok, B.W.J. Critical comparison of background correction algorithms used in chromatography. Analytica Chimica Acta, 2022, 1201, 339605. https://doi.org/10.1016/j.aca.2022.339605
<a name="ning"></a>
2. Ning, X.; Selesnick, I.W.; Duval, L. Chromatogram baseline estimation and denoising using sparsity (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167. https://doi.org/10.1016/j.chemolab.2014.09.014
<a name="snyder"></a>
3. Snyder, L.R.; Kirkland, J.J.; Dolan, J.W. Introduction to Modern Liquid Chromatography, 3rd ed. John Wiley & Sons, 2009.
<a name="nh"></a>
4. Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex chromatograms using the BEADS algorithm. Journal of Chromatography A, 2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057
