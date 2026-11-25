# Synthetic chromatograms with known ground truth

Generator: [`tools/synthetic/synth_dataset.py`](./synth_dataset.py).
Tests: [`weaselytics/tests/test_synth_dataset.py`](../weaselytics/tests/test_synth_dataset.py).

## 1. Why a synthetic benchmark exists

Judging how well a method selects the BEADS cutoff frequency requires knowing the true baseline, which no real chromatogram provides. Niezen et al. \[[1](#niezen)\] state the dilemma: real data has realistic peak overlap, noise and drift but no ground truth, while fully simulated data has ground truth but "typically represent idealistic situations and lack signal distortions and properties pertaining to experimental data".

Niezen et al. \[[1](#niezen)\] resolve this with **hybrid data** made of an experimental background carrying real drift, with peaks and noise added at known parameters. A hybrid family was prototyped here. Its background-pool loader (`tools/synthetic/backgrounds.py`), its peak-free-stretch selection and its modified Pearson VII peaks are in commit `efc7aa7`.

The baselines here are drawn at random instead, following the simulated chromatograms of Ning et al. \[[2](#ning)\].

## 2. What a signal is

A signal is a baseline, peaks and white noise, summed. It is sampled on an index, 0 to *N* − 1, and has no physical units, like those of Ning et al. \[[2](#ning)\]. The length *N* is drawn for each signal between 500 and 40,000 samples; `--length` fixes it.

The truth file `truth/<stem>__truth.npz` stores each part separately, and the baseline alone is the truth a baseline estimate is scored against. Beside the signals, a dataset holds `manifest.csv`, one row per signal, and `PROVENANCE.txt`, which records the commit, the exact call with every argument, the machine and the library versions.

A signal is named `SYN__peaks__model__type__noise__length__replicate`: the number of peaks left (with `a` when the plate model is on), the peak model, the baseline type, the noise level, the length on five digits and the replicate on four. The default set holds two baseline types and three noise levels, six signals per replicate.

## 3. Baselines

Every baseline is defined over the whole record, so a longer record holds as many waves, sampled more densely. By default the peaks keep their width in samples.

| Type | Baseline |
|---|---|
| 1 | a polynomial of random order plus a sinusoid of random frequency |
| 2 | white Gaussian noise through a low-pass filter |
| 3, 4, 5 | Erb's parabola, exponential and step, with `--legacy-baselines` |

Ning et al. \[[2](#ning)\] give the first two, without their ranges. The ranges used here were chosen by eye to match their Figs. 5 and 6. They are next to the constants in `synth_dataset.py`.

## 4. Peaks

Between 1 and 30 peaks are drawn per signal, as in the simulated chromatograms of Komsta \[[6](#komsta)\], or 0 and 30 with the plate model; `--n-peaks` fixes the number. Every value below is next to its constant in `synth_dataset.py`, with where it comes from or that it was chosen.

**Shape.** `--peak-model` chooses an exponentially-modified Gaussian (`emg`, the default), a modified Pearson VII (`pearson7`) or a Gaussian (`gaussian`). Each peak draws its tailing factor, so that most tail a little and a few strongly; `--tailing-factor LO HI` draws it in a range instead.

**Width.** By default the width is drawn in samples, at every record length. With `--plate-model` it follows σ = tR / √N, with tR counted from the first sample (Snyder et al. \[[3](#snyder)\]), so the first peaks are sharp and the late ones broad.

**Height.** Heights are drawn evenly on a log scale, in a range set by the baseline type, so that the tallest peaks typically stand 5 to 10 times as high as the baseline is large. No peak is below three standard deviations of the noise of its signal (MacDougall et al. \[[5](#macdougall)\]).

**Placement.** Each peak stays where it was drawn or is dropped. It is dropped when it rises above the noise in the first or the last 20 samples, when its resolution from a peak already kept is below 1, unless it is one of the share `--overlap-prob` allows to overlap (10% by default), or when it would take the peaks above 33% of the record. The placement stops at the first peak that does. The name of a signal gives the number of peaks left.

**The plate model brings a dead time.** One of the spots drawn in the first third of the record becomes the dead time. The earlier spots are discarded and nothing elutes before it. A positive spike followed by a negative undershoot marks it there, with the width and the height a peak would have at that place. The production BEADS call uses a symmetric cost (`asymmetry = 1`) because of this negative lobe. The artifact is part of the signal, and `baseline` holds the baseline alone: Niezen et al. \[[1](#niezen)\] note that negative peaks "are generally treated as background drift" by correction algorithms, and a sharp spike would dominate a baseline error metric.

## 5. Noise

White Gaussian noise is added, so the noise component is known exactly. Its level is the energy of the baseline over the energy of the noise: 30, 25 or 20 dB (`snr30`, `snr25`, `snr20`). The ratio has the form of the output SNR of Ning et al. \[[2](#ning)\], who do not define the input SNR they vary from −5 to 25 dB. The noise follows the size of the baseline, and a shift of the whole baseline changes it.

## 6. Scoring

Niezen et al. \[[1](#niezen)\] found that the combination with the lowest background RMSE was not the one with the smallest peak-area errors, so both are reported:

1. RMSE against the known baseline, `RMSE = sqrt( Σ (b_i − z_i)² / n )`;
2. relative error in peak area, integrated over the known peak regions.

The project's deliverable is peak areas.

## 7. Limits

Cross-machine reproducibility bounds what can be concluded. The swept `r²` curve is ill-conditioned at low cutoff. A one-ulp difference in the frequency grid reaches 5.6e-2 of `r²` near `fcut` 1e-4, and across a numpy/scipy version change some selected cutoffs move by more than 0.1 decade. No result should rest on a precision finer than ~1e-3 near the fundamental; below that, the benchmark measures the machine. See GitHub issue #20.

The peak detection of the package uses widths in samples per minute and heights in mV (GitHub issue #23), which the signals here do not have. Its result on them depends on the units the signals are read in, and some of them crash the fit.

## 8. What BEADS assumes

Ning et al. \[[2](#ning)\] make four assumptions.

1. **Peaks.** `x` and its first *M* derivatives are sparse. With M = 2 that is the peaks themselves plus their first and second derivatives.
2. **Baseline.** Low-pass only. No parametric form, and **no stated rule relating the cutoff to peak width**.
3. **Noise.** Stationary white Gaussian, which is what justifies the quadratic fidelity term.
4. **Positivity.** A preference encoded in the asymmetry parameter `r`, not a constraint; `r = 1` is the symmetric penalty.

They validated on Gaussian peaks, two baseline types (polynomial plus sinusoid; low-pass-filtered white noise), 500 realizations and an SNR from -5 to 25 dB. No peak density, width, height or resolution is reported, and the peak generation is cited to another reference.

Navarro-Huerta et al. \[[4](#nh)\] list five limitations of conventional BEADS, namely the periodicity requirement, ripples under peaks of very disparate size, sporadic negative peaks, `fc`/lambda instability at low frequency, and per-chromatogram retuning. **Peak overlap is not among them.**

The generator keeps most peaks at a resolution of at least 1 from their neighbours (§4), which still allows small overlaps, to test the limits of the sparsity assumption.

## References

<a name="niezen"></a>
1. Niezen, L.E.; Schoenmakers, P.J.; Pirok, B.W.J. Critical comparison of background correction algorithms used in chromatography. Analytica Chimica Acta, 2022, 1201, 339605. https://doi.org/10.1016/j.aca.2022.339605
<a name="ning"></a>
2. Ning, X.; Selesnick, I.W.; Duval, L. Chromatogram baseline estimation and denoising using sparsity (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167. https://doi.org/10.1016/j.chemolab.2014.09.014
<a name="snyder"></a>
3. Snyder, L.R.; Kirkland, J.J.; Dolan, J.W. Introduction to Modern Liquid Chromatography, 3rd ed. John Wiley & Sons, 2009.
<a name="nh"></a>
4. Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex chromatograms using the BEADS algorithm. Journal of Chromatography A, 2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057
<a name="macdougall"></a>
5. MacDougall, D., et al. Guidelines for data acquisition and data quality evaluation in environmental chemistry. Analytical Chemistry, 1980, 52, 2242-2249. https://doi.org/10.1021/ac50064a004
<a name="komsta"></a>
6. Komsta, Ł. Comparison of several methods of chromatographic baseline removal with a new approach based on quantile regression. Chromatographia, 2011, 73, 721-731. https://doi.org/10.1007/s10337-011-1962-1
