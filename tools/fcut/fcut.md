# How to find fcut?
In an autocorrelation plot of the baseline-corrected signal, a plateau represents a frequency range over which the baseline remains relatively unchanged, i.e., few or no “features” of the raw signal (in the broad sense) are designated as belonging to the baseline. According to Navarro-Huerta et al. \[[1](#navarro-huerta_assisted_2017)\], the optimal value of `fcut` lies on such a plateau. However, at this point it is not clear how to robustly detect the presence of a plateau, let alone how to identify the correct plateau. In general, the strategy we will use is:

> **What exactly is plotted, and two departures from the reference.** See [§0b](#0b-what-the-autocorrelation-measures) below. The reference paper defines *two* different plots, ours corresponds to the second, and our implementation departs from it on the quantity being autocorrelated and on the region it selects from. Both departures are recorded there; neither has been shown to be wrong, and one has been shown not to be worth fixing on the evidence available.
1. Identify plateaus in the autocorrelation plot;
2. Exclude plateaus that do not meet certain criteria, with the goal of identifying the plateau that includes the optimal value of `fcut`;
3. Find the optimal value of `fcut` on that plateau.

> ## ⚠ Status of this document (2026-07-26)
>
> **The three-step strategy above is current.** The three sections that
> implement it — [§1](#1-plateaus-identification), [§2](#2-plateau-exclusions)
> and [§3](#3-choosing-the-right-fcut-on-a-given-plateau) — are **HISTORICAL**.
> They describe the derivative-based route that was **removed on 2026-07-26**
> (commit `a3b7159`), together with its four absolute tolerances, the `Case`
> machinery, `select_fcut`, `refine_candidates` and `find_plateaus`. They are
> kept deliberately, as a record of what was tried and why it failed; do not
> read them as a description of the code.
>
> Sections [§0a](#0a-the-low-frequency-end-of-the-curve-is-not-reproducible),
> [§0b](#0b-what-the-autocorrelation-measures) and
> [§0](#0-preprocessing) remain current.
>
> **Where the live pipeline is documented:** [segmentation.md](./segmentation.md),
> and the docstrings of `segmentation.classify_segments` (detect),
> `segmentation.trim_plateaus` (exclude — four exclusions: sub-fundamental
> clip, frozen tail, SNR-gated collapse, stiff-side instability) and
> `segmentation.select_center` (select — the geometric centre of the surviving
> region, preliminary).
>
> **A standing caution for everything below.** Several passages quote
> measurements made against a set of 51 hand labels that was **deleted on
> 2026-07-26** as untrustworthy — 11 of the 51 sat below 0.90 of the r²
> shoulder and 5 below 0.80, where the optimum is expected near 0.97. Those
> numbers cannot be reproduced or checked against any surviving dataset. They
> are marked individually where they appear.

### 0b. What the autocorrelation measures

The quantity plotted is $`r^2 = \big((2 - DW)/2\big)^2`$, where $`DW`$ is the Durbin-Watson statistic — that is, approximately $`\rho_1^2`$, the squared lag-1 autocorrelation. It is therefore **not** a goodness of fit: it measures *how much smooth, structured content remains* in the channel it is computed on. $`r^2 \to 1`$ means what remains is strongly correlated point to point; $`r^2 \to 0`$ means it is white noise, i.e. nothing structured survives.

This has a consequence worth stating plainly, because it bounds what any plateau-based method can achieve. The drop in $`r^2`$ marks the cutoff at which the baseline starts absorbing the autocorrelated content of the measurement — but the statistic **cannot tell whether that content is analyte peaks or baseline structure**. For a strong analyte, crossing the drop destroys peak area and must be avoided; for a weak analyte on a large baseline (a blank), the structured content *is* the baseline, absorbing it is precisely the job, and a lower shelf beyond the drop can be the right answer. The information needed to choose is the ratio of analyte structure to baseline structure, which is *orthogonal to the curve*. Empirically this is exactly what was observed: on the labeled dataset, rules based on the shape of the $`r^2`$ curve selected the correct `fcut` at 23–25%, against 25.9% for a random pick inside the candidate regions. *(Those two figures come from the 51 hand labels deleted on 2026-07-26 and cannot be reproduced; the conclusion they support was independently reached from the structure of the problem — the deciding information is the analyte-to-baseline ratio, which is orthogonal to the curve.)* **The curve brackets the answer; it does not select it.**

### 0a. The low-frequency end of the curve is not reproducible

A second, independent bound. The same signal swept on two machines — identical code, identical pinned pybaselines, identical `scut` and regions, and the same selected `fcut` to five digits — gives $`r^2`$ curves that agree at high cutoff and diverge at low cutoff:

| $`f_{cut}`$ | \|local − cluster\| |
|---|---|
| $`5.1\times10^{-5}`$ | $`3.4\times10^{-3}`$ |
| $`1.5\times10^{-4}`$ | $`9.3\times10^{-5}`$ |
| $`4.4\times10^{-4}`$ | $`5.6\times10^{-6}`$ |
| $`1.3\times10^{-3}`$ | $`4.5\times10^{-7}`$ |
| $`\ge 3.8\times10^{-3}`$ | $`< 2.5\times10^{-8}`$ |
| $`\ge 9.9\times10^{-2}`$ | $`\sim 10^{-16}`$ |

The only inputs that differ are last-bit: `np.geomspace` and `np.log10` are not identical across numpy and libm versions. What turns one ulp into $`3.4\times10^{-3}`$ is the method itself — Navarro-Huerta et al. \[[1](#navarro-huerta_assisted_2017)\] §3.1(iv) state that the baseline is *"particularly susceptible to the selected cutoff frequency at low frequencies, which results in an unstable adjustment process"*.

So **below $`f_{cut} \approx 10^{-3}`$ the curve is only reproducible to about $`10^{-3}`$**, and above it to $`10^{-8}`$ or better. Any feature detected in that range — a plateau, a step, an inflection — must be larger than that floor to be a property of the data rather than of the machine. It is also an independent argument for the sub-fundamental clip of `trim_candidates`: on a 5000-point record that clip sits at $`10^{-4}`$, squarely inside the unreliable zone.

Two departures from Navarro-Huerta et al. \[[1](#navarro-huerta_assisted_2017)\] are worth recording. The paper defines two different plots: one on the BEADS **noise** $`e`$, whose *minimum* marks the optimum, and one on the **baseline-corrected signal**, which gives a stepped plot. Only the second applies here, since the paper notes that the log transformation makes the returned noise unusable for the autocorrelation.

A point of vocabulary first, because it is easy to get backwards. The paper's log transform is **base 10** — Eq. (8) is $`z = \log(y - \min(y) + \varepsilon)`$ and its inverse Eq. (11) is $`b_{corr,y} = 10^{b_z} + \min(y) - \varepsilon`$, and a $`10^{\,\cdot}`$ inverse fixes the base. `_log_transform` uses `np.log10` and so **matches the paper**. The upstream pybaselines example `plot_beads_param_selection.py` uses the *natural* log (`np.log` / `np.exp`, internally consistent but a different base); do not "align" with it. The base is load-bearing rather than cosmetic: under the pinned auto-scaled `lam_d` the penalty terms are scale-invariant but the data-fidelity term is not, so switching base rescales the fidelity/penalty balance by $`\ln(10)^2 \approx 5.3`$. Measured at a fixed cutoff on two reference signals, the resulting baselines differ by 7–9% of the signal range, and on a blank $`r^2`$ moves from 0.026 to 0.929.

1. **The quantity — resolved, and no longer a departure.** `_r2` now computes the statistic on the baseline-corrected signal $`z - b`$, which is $`c + e`$: the quantity Navarro-Huerta monitor. Only the fitted baseline is taken from the algorithm, so the definition is identical on the `beads` and `custom_beads` paths and their curves are finally comparable.

   The noise has to stay in. $`r^2`$ is a whiteness test: the stepped structure exists because a good cutoff leaves the correlated peaks in the corrected signal while an excessive one leaves white noise behind, so the noise *is* the floor the drop is measured against. Monitoring $`c`$ alone removes that floor and measures how correlated the sparse component is with itself, which has no reason to be stepped.

   **What it used to compute, and why it failed.** `_r2` previously used `params["signal"]`. On the plain `beads` path that is the algorithm's own sparse solve, and the BEADS identity $`v = y - s - H(y-s)`$ makes it exactly $`z - b - \hat e`$ (verified to 1.4e-17). On the `custom_beads` path `_custom_beads` rebuilds the same expression on the full grid — but $`\hat e`$ is interpolated from the reduced grid. `custom_bc` only reduces points **inside** `regions`; everything outside is kept at full resolution and enters `x_fit` unchanged. So $`\hat e`$ is *exact outside the regions* and, inside them, a straight line through bin averages. The reconstruction is therefore $`c`$ everywhere **plus raw point-to-point noise on the region interiors** — not a uniformly partially-denoised residual.

   Because Durbin-Watson is $`\sum(\Delta e)^2/\sum e^2`$, that small contaminated patch dominates the statistic whenever $`c`$ is small. On `2-Chlorotoluene__LPYE__BLANK__1` the single region spans 22 of 580 points (3.8%), where the reconstruction has std $`1.0\times10^{-2}`$ against $`3.0\times10^{-5}`$ outside — and $`r^2`$ reads **0.026** against **0.898** for the algorithm's own $`c`$ on the reduced grid. Across eight nominally identical blanks the old channel spanned **0.026–0.896**; the new one spans **0.816–0.849**, with `beads` now within 0.03–0.05 of `custom_beads` on the same signals.

   The failure scaled as (region coverage × noise) / analyte amplitude, so it was worst on blanks and invisible on strong peaks. It also made $`r^2`$ depend on the region layout, which is chosen for baseline stiffness, coupling the measurement channel to a stiffness decision.

   One qualification, because it is easy to over-read the mechanism. The contamination does **not** generally sit under the broad peaks, even though `sampling` is highest there: where binning is heaviest the analyte is large, so the un-subtracted noise is a negligible fraction of it. Measured on the *raw* signals, `params["signal"]` departs from $`y - b`$ by 0.79% and 0.87% of the signal range on two peaked chromatograms, against 6.3% and 15.0% on two blanks — and the worst of those blanks has **no peak regions at all**, so no binning occurs and the difference is exactly the BEADS noise. The two conditions that would make binning bite are anti-correlated: a weak analyte is precisely when `_relevant_regions` returns no regions. What breaks the statistic on the log-transformed truncated signal is that Durbin-Watson is a sum of squared *point-to-point* differences, so a small contaminated patch dominates it once the sparse component is small — not that the contamination is large in absolute terms. See the Notes of `_custom_beads`.

   Superseded by this change: the earlier claim that switching to the paper's quantity "did not locate the optimum better". That rested on three signals and an uncommitted script, and the alternative it tested was the log-scale residual rather than Eq. (12).

   **Still open — the scale.** Eq. (12) is $`(c+e)_{corr,y} = y - 10^{b_z} - \min(y) + \varepsilon`$, i.e. the corrected signal back-transformed to the **original** scale, because linearity is not preserved on the way back; the upstream pybaselines example agrees, computing `autocorrelation(y - fit)` after back-transforming. We monitor $`z - b`$ on the log scale. Note the paper's stated reason for using $`c+e`$ concerns original-scale separability, so it does not by itself settle the log-scale case either way. This remains to be measured against synthetic ground truth.
2. **The region.** The paper places the optimum near the centre of the *last* step at high frequencies, recommending in practice a point between the beginning and the centre of that last horizontal region. **This is no longer a departure.** `segmentation.select_center` takes the **last** surviving region and its centre, citing NH17 §3.4 — the remaining difference is that the paper biases slightly below that centre and we do not yet (see §3). The earlier implementation did the opposite: `last_r2` and the frozen-tail exclusion discarded the final region, and a first-two-regions rule in `refine_candidates` narrowed what remained to the first two. Both `last_r2` and `refine_candidates` were removed on 2026-07-26; `refine_candidates` was never in the production path, and its label-calibrated constants became unauditable when the labels were deleted. The frozen-tail exclusion survives in `trim_plateaus`, but it removes a *degenerate* tail rather than the last step as such. The older note that the paper's landmark missed the true optimum by roughly a factor of three on the clearest case was measured on the pre-`41a7580` r² channel and has not been rechecked since.

   It is not unopposed, and the counter-evidence should be cited with its conditions. The upstream `plot_beads_param_selection.py` selects "the middle of the last plateau" and reports that it coincides with the cutoff giving the lowest MSE against the *known* baseline — a ground-truth-validated hit for the paper's landmark. But it runs on `pybaselines.utils.make_data()`, a single synthetic trace with a simple baseline, monitored on the original scale after back-transforming, with `lam_d` fixed by hand and no region splitting. Those are not the conditions here, and the disagreement is a reason to re-test the landmark on our own ground truth rather than to assume either side.

### 0. Preprocessing
Before searching for plateaus, the signal must be restricted to a region of interest. For instance, consider a raw signal with 11000 data points, where the clean signal is confined to the first 1000 points. If we include all points from the raw signal in the analysis, the autocorrelation will be biased because the contribution from the key components in the region of interest is diluted. Consequently, the autocorrelation plot often becomes noisier, hard to interpret and longer to compute.

<a name="fig_scut"></a>
![Effect on the autocorrelation plot of changing scut.](./images/2-Xylene__LPYE__CS2__15_scut.png)

In the current implementation, this occurs at the start of `auto_beads` via the `_relevant_regions` function. In short, if we assume that a raw signal consists of a baseline, noise, and a sparse signal component with a reasonable signal-to-noise ratio, then we can approximately locate the positions and widths of the signal peaks directly in the raw signal. This is (part of) what `_relevant_regions` is designed to do. For reference, this function is also used to identify the regions of interest and the sampling strategy for `custom_bc`. As of today, the signal is only truncated on the right (via `scut`), but not on the left. In the future, it may be advantageous to clip the signal on the left side as well.

### 1. Plateaus identification
> **HISTORICAL — removed 2026-07-26 (`a3b7159`).** The derivative-tolerance
> method described below is no longer in the code. Detection is now
> `segmentation.classify_segments` (changepoint segmentation, `CP flat`), with
> `segmentation.detect_dips` supplying proto-plateaus as a **fallback** that
> contributes only when the flat channel leaves nothing surviving. See
> [segmentation.md](./segmentation.md). The paragraph below is kept as the
> record of what was tried; its closing sentence — that a more robust method
> was required — is what motivated the replacement.

In my view, plateau detection is the most challenging aspect of this problem. Up to now, the most effective approach I have found is to impose a small threshold around 0 on both the first (see the yellow \[loose tolerance\] and dashed orange regions \[tight tolerance\] on the [autocorrelation plot](#fig_scut)) and second derivatives (see the blue region on the [autocorrelation plot](#fig_scut)), and to mark as plateaus those points that fall within these tolerances. I experimented with several alternative techniques based on identifying features in the autocorrelation curve (for instance, inflection points or the extrema of its first derivative), but these approaches were too sensitive to the instabilities and discontinuities that can occur in the autocorrelation plot. A more robust method is therefore required in order to correctly identify all the plateaus.

### 2. Plateau exclusions
> **HISTORICAL — removed 2026-07-26 (`a3b7159`).** `last_r2`, named below, no
> longer exists anywhere in the code; nor does the derivative-threshold
> detection of `p_ini`. Exclusion is now `segmentation.trim_plateaus`, whose
> four exclusions are the sub-fundamental clip (`c1=1.0`, below `1/n_used` —
> the physical successor to the `p_ini` rule below), the frozen tail (the
> successor to `last_r2`), the SNR-gated collapse, and the stiff-side
> instability trim. Kept as the record of the reasoning that led there.

Regarding the regions to discard, let us first assume that there is always an “initial” plateau, `p_ini`, on the far left of the autocorrelation curve, i.e. where the `fcut` values are smallest. On this plateau, any choice of `fcut` yields a baseline that is overly rigid. In addition, when `fit_parabola` is set to True, the baseline begins to deform, much like a stiff beam under compression. Consequently, `p_ini` (in red on the [autocorrelation plot](#fig_scut)) is systematically excluded from the set of candidate plateaus. At present, this plateau is detected using a threshold on the drop of the `r2` value in reference to the average value of the first (from the left) tight flat region of the first derivative and by selecting the last point (before the minimum of the first derivative) that still satisfies this threshold. This approach allows us to better approximate the true end of the first plateau when it is very noisy. However, a more robust method is required.

If the last point (right side) of the autocorrelation is part of a continuous region where the first derivative is tightly flat, this last region is also not considered further through the use of `last_r2`.

At this point in time, there is no reliable way to find the "anchoring" plateau. In fact, this part of the code in more of less an artifact of an old implementation based on the use of extrema of the first derivative of autocorrelation.

### 3. Choosing the right `fcut` on a given plateau
Finally, once we assume that the correct plateau has been located, selecting `fcut` should be fairly straightforward, taking the plateau’s center and/or boundaries as references.

> **HISTORICAL, and partly unverifiable — 2026-07-26.** Selection is now
> `segmentation.select_center`: the geometric centre of the surviving region,
> snapped to a grid point so its r² is read from the swept curve rather than
> refitted. That is preliminary — NH17 §3.4 recommends *slightly below* the
> centre, and no offset has been derived. Of the four bullets below, the first
> and the last rest wholly or partly on the **deleted** 51 hand labels and
> cannot be checked; the second and third also cite synthetic measurements
> from a benchmark that predates the current `_r2` channel (`41a7580`) and
> peak-width geometry (`6a1a380`), so their numbers are not comparable with
> anything measurable today. The *conclusions* — not the edges, and not a
> power law — were reached by more than one route and are still held.

This last step is **the open half of the problem**, and it is harder than it looks. What was established at the time, from the (since deleted) hand-labeled dataset and from synthetic signals with a known baseline:

- **Where the answer sits.** In log-relative position inside the first wide candidate region (0 = left edge, 1 = right edge), the hand-labeled ranges have their centre at 0.55 and their upper edge at 0.71, and the objectively optimal cutoff of the synthetic signals has a median of 0.65 with an interquartile range of 0.55–0.77. The two agree, which is the strongest signal available.
- **Not the edges.** Taking the right edge of the region — the natural reading of “anchor near the drop” — costs a median of 3.5× the optimal baseline error, roughly 19× the noise level, and lands within 1.25× of the optimum for only 8 of 71 synthetic signals. The interior matters.
- **Not a power law either.** `fcut` does scale with the peak width, strongly (R² 0.86 when nothing else varies), but the fitted exponent is −0.735 under full control, −0.88 on the synthetic benchmark and −0.50 on the real dataset. It is a property of the signal population, not a transferable law, so it is deliberately **not** used in production.
- **Blanks are the hard case, for a reason.** With no analyte peak there is no width to set the scale, and the structured content of the signal is the baseline itself, so the optimum can legitimately lie past the collapse. Which side of the collapse is correct is predicted by the analyte-to-baseline ratio at roughly 88–90% accuracy — enough to *gate* the decision, not enough to make it.

## References

<a name="navarro-huerta_assisted_2017"></a>
1. Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex chromatograms using the BEADS algorithm. Journal of Chromatography A, 2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057.
