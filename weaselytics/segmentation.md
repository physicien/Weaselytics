# Changepoint-based plateau detection

This page documents the method implemented in `segmentation.py` (issue [#4](https://github.com/physicien/Weaselytics/issues/4)). It is the prototype alternative to the plateau detection described in [fcut.md](./fcut.md): the goal is unchanged (locate the plateaus of the autocorrelation plot $`r^2(f_{cut})`$ and pick the optimal `fcut` on the right one, following Navarro-Huerta et al. \[[1](#navarro-huerta)\]), but the detection strategy is reframed as a *changepoint* (segmentation) problem instead of a pointwise classification problem.

## 1. Why pointwise thresholding is fragile

The previous approaches (`_fcutoff` derivatives, `find_plateaus` rolling statistics) all share the same structure: compute a local statistic at every point of the curve (first/second derivative, rolling standard deviation, rolling MAD), then classify each point as "flat" or "not flat" by comparing that statistic to a threshold. Three structural problems follow:

1. **The plateaus are not flat.** On real chromatographic data, the plateau containing the optimal `fcut` typically *drifts* slowly (e.g. $`r^2`$ decreasing from 0.9955 to 0.9805 over more than a decade of `fcut`). A slow drift eventually crosses any fixed local threshold, so pointwise methods inevitably fragment a single physical plateau into many short segments — which then requires repair heuristics (`_long_segments`, merging rules).
2. **Absolute thresholds are not transferable.** Tolerances such as `tol0 = 1e-3` or `diff_std_mad < 5e-5` implicitly depend on the sampling density `num`, the rolling window, the signal length and the noise level. A threshold tuned on one dataset silently mistunes on the next.
3. **The criteria interact opaquely.** Each repair heuristic (dip test, local Sauvola threshold, triangle threshold, end trimming, length filter) is coupled to the others, and the combined behavior becomes difficult to predict — the main complaint recorded in issue #4.

The changepoint framing removes all three at the root: segments are contiguous by construction (no fragmentation), the two classification criteria are expressed relative to the geometry of the curve itself (no absolute scales), and the pipeline factorizes into two decoupled stages (segment, then classify).

## 2. The model: penalized piecewise-linear segmentation

The curve $`y_1, \dots, y_N`$ (the $`r^2`$ values on the geometric `fcut` grid) is modeled as a sequence of $`K`$ contiguous segments. On each segment, $`y`$ follows a straight line with its own Gaussian noise level:

$$y_t = a_k + b_k\, t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma_k^2), \qquad t \in (\tau_{k-1}, \tau_k].$$

Maximizing the Gaussian likelihood with per-segment variance gives the segment cost implemented in `_linear_costs`:

$$\mathcal{C}(y_{i..j}) = m \,\log\!\left(\frac{\mathrm{SSE}_{ij}}{m}\right), \qquad m = j - i,$$

where $`\mathrm{SSE}_{ij}`$ is the sum of squared residuals of the least-squares line fitted on $`y_{i..j}`$. Two properties of this cost matter here:

- **Slope changes are changepoints.** The transitions between plateaus and drops of the autocorrelation plot are detected as breaks in $`b_k`$, and a slowly drifting plateau is fitted by a *single* segment instead of a staircase of short ones. (A piecewise-*constant* model, by contrast, over-segments the drifting regions — this was verified on the real data and is the reason for the linear mean.)
- **Variance changes are changepoints.** Because each segment carries its own $`\sigma_k^2`$, a boundary between a quiet plateau and a region of BEADS instabilities (the low-frequency `p_ini` region, the chaotic high-frequency tail) is detected *even when the mean does not move*. This is exactly the information that the rolling standard deviation of `find_plateaus` was designed to capture, but here it is a by-product of the model instead of a separate machinery: the dip test, the Sauvola local threshold and the triangle threshold all become unnecessary.

The number of segments is not fixed a priori. The total objective is penalized by a constant $`\beta`$ per changepoint,

$$\min_{K,\ \tau_1 < \dots < \tau_{K-1}} \; \sum_{k=1}^{K} \mathcal{C}\!\left(y_{\tau_{k-1}..\tau_k}\right) + \beta K,$$

following the classical penalized model-selection approach of Yao \[[5](#yao)\] based on Schwarz's criterion \[[4](#schwarz)\]. The default $`\beta = 25 \ln N`$ is a BIC-like value; larger $`\beta`$ gives fewer, coarser segments.

## 3. Exact optimization

The objective is minimized *exactly* (global optimum over all $`2^{N-1}`$ partitions) by dynamic programming — the "optimal partitioning" recursion of Jackson et al. \[[6](#jackson)\]:

$$F(j) = \min_{i \,\le\, j - m_{\min}} \; \Big[ F(i) + \mathcal{C}(y_{i..j}) + \beta \Big], \qquad F(0) = -\beta,$$

with backtracking to recover the breakpoints. All segment costs are evaluated in $`O(1)`$ from cumulative sums, so the total cost is $`O(N^2)`$ — a few tens of milliseconds for the typical $`N = 1000`$ of an autocorrelation plot. The PELT pruning of Killick et al. \[[7](#killick)\] would reduce this to expected $`O(N)`$, and the `ruptures` library \[[8](#truong)\] provides equivalent off-the-shelf implementations (`Pelt` with a linear cost); the pure-NumPy implementation in `pelt_linear` was preferred to avoid a dependency. A minimal segment length `min_size` (default 15 points) excludes degenerate fits.

## 4. Scale-free classification of the segments

`segment_features` describes each segment by its mean level, its fitted slope $`b_k`$ and its residual standard deviation $`s_k`$, then normalizes the latter two by the geometry of the curve itself:

- **Relative slope.** Let $`\Delta = \max(r^2) - \min(r^2)`$ be the total drop of the curve and $`d`$ the number of `fcut` decades spanned by the grid. The natural slope scale is "the whole drop spread over one decade", i.e. $`\Delta / (N/d)`$ per grid point, and $`\mathrm{rel\_slope}_k = |b_k| \,/\, \big(\Delta\, d / N\big)`$.
- **Relative noise.** $`\mathrm{rel\_noise}_k = s_k / \Delta`$, the residual noise as a fraction of the total drop.

`classify_segments` marks a segment as a plateau candidate ("flat") when $`\mathrm{rel\_noise} < 0.006`$ and its relative slope satisfies a **two-tier (tight/loose) criterion**, the dimensionless analogue of the tight and loose derivative tolerances of `_fcutoff`:

- *tight*: $`\mathrm{rel\_slope} < 0.2`$ — strictly flat on the scale of the whole curve;
- *loose*: $`\mathrm{rel\_slope} < 0.6`$ **and** the segment is bracketed by at least one cliff ($`\mathrm{rel\_slope} > 1`$) on each side.

The loose tier exists for staircase-shaped curves (blank injections, multi-step programs such as Chlorobenzene 60-70): there the total drop of $`r^2`$ is split across several cliffs, so every intermediate shelf drifts at a substantial fraction of the global slope scale and a purely global criterion rejects it — even though the shelf is obviously flat *compared to the cliffs surrounding it*. On the 339-signal benchmark, the two-tier criterion places every accepted `fcut` of the reference implementation inside a flat segment (339/339, versus 336/339 for the tight tier alone) without increasing the total fraction of the curve classified as flat. All thresholds are dimensionless and independent of `num`, of the signal length and of the units of the statistic.

## 4b. Trimming the flat set into candidate regions

`trim_candidates` reduces the flat segments to the regions where the optimal `fcut` can actually lie, using only *a-priori* exclusions (deliberately less aggressive than the trimming stack of `find_plateaus` — no level rules, no end heuristics, no length filters):

- **Sub-fundamental clip.** The slowest oscillation representable on a record of $`N`$ points has frequency $`1/N`$, so every cutoff below it requests the identical, maximally rigid baseline — which is why the initial plateau of the autocorrelation plot always ends at $`\approx 1/N`$. Grid points below $`c_1/N`$ (default $`c_1 = 0.5`$; the dataset contains accepted values down to $`0.69/N`$) are removed: they only duplicate the solution at the fundamental. This alone removes roughly 40% of the geometric grid.
- **Frozen exclusion.** In the saturated far tail the baseline no longer responds to `fcut` and the residual noise of the segments collapses to $`\mathrm{rel\_noise} \lesssim 2 \times 10^{-7}`$, an order of magnitude below the quietest genuine plateau observed ($`6.7 \times 10^{-7}`$). Flat segments at or below `noise_floor` (default $`4 \times 10^{-7}`$) are removed.
- **Bridging.** A non-flat segment lying between candidate regions is absorbed when it is not a cliff ($`\mathrm{rel\_slope} < 1`$), so short drifting connectors do not split one plateau into several displayed pieces, while genuine staircase steps still separate regions.

On the 339-signal reference dataset (corrected parser), the trimmed mask retains the accepted `fcut` of **every** signal while halving the covered grid area (median 89% → ~50%) and collapsing the display to 2–3 contiguous regions per signal.

## 4c. Narrowing the candidates with the labeled bracket

`trim_candidates` uses only a-priori exclusions, and the result is a valid but *loose* bracket: on the labeled sample the candidate regions span a median of 2.33 decades against 0.62 decades for the hand-labeled acceptable range, a ratio of 3.6. `refine_candidates` narrows it further, using the only evidence available about where the answer actually lies — the hand-labeled gallery of the 339-signal dataset (`FCUT_GALLERY_2026-07-20`). All its rules are expressed in the same scale-free vocabulary as the classification above (region widths in decades, positions as log-relative fractions of a region), and every constant sits at a *"labels never go there"* extreme with a margin:

- **Sliver exclusion.** Regions narrower than `min_width` decades (default 0.5) are dropped. The narrowest region a label ever touches spans 0.67 decades, and the widest sliver below that is also 0.67 — the two populations are cleanly separated, and any threshold between roughly 0.4 and 1.1 gives the same partition. When every region is a sliver, the widest is kept as a fallback.
- **Ordinal cut.** Only the first two remaining regions are kept: across all 348 labeled ranges, no label ever touches a later one.
- **Left cut.** The first `left_cut` fraction of the first region is removed (default 0.12) — the over-rigid edge, where 99% of labels start beyond 0.12 and 95% beyond 0.18.
- **Right cut.** The second region is truncated past `right_cut` of its width (default 0.55): labels enter it only from the left and never beyond 0.41.

On the labeled sample this keeps the labeled range (recall $`\ge 0.99`$) for **336/339** signals while shrinking the median candidate span from 2.33 to **1.83 decades**. Under leave-one-solvent-family-out recalibration the constants are identical in 13 of the 14 folds and every held-out family keeps a median recall of 1.000; the exception is 2-Chlorotoluene, whose blanks penetrate deepest into the second region and pull `right_cut` down to 0.19. `tools/fcut_bracket_calib.py` reproduces all of these numbers and checks that the shipped implementation agrees with them.

Two caveats belong with these constants. First, the labels are **censored by the candidate set**: the gallery only rendered images at `fcut` values inside the candidate regions, so the user could not label outside them. The bracket can therefore answer *"which part of the candidates is dead weight?"* but cannot validate the outer boundaries of `trim_candidates` itself. Second, they are calibrated on one instrument and one labeling session, so they carry the same epistemic status as the tolerances they replace — reasonable, physically motivated, validated on the data at hand, and recalibratable with one command.

## 5. Selecting the plateau and the value of `fcut`

`select_fcut` applies the following rule, in the spirit of steps 2–3 of [fcut.md](./fcut.md):

1. locate the **first steep descending segment** ($`b_k < 0`$ and $`\mathrm{rel\_slope} > 0.2`$) — the onset of the main drop, where the baseline starts absorbing signal features;
2. among the flat segments *before* it, choose the **last** one;
3. return the **right edge** of that plateau as `fcut` (consistent with the `slope_thresh` shift of `_fcutoff`, which also anchors near the drop);
4. if no flat segment precedes a steep drop, fall back to flat segments whose mean lies above half of the total drop; if none, return no answer rather than a guess.

This "last plateau before the drop" is the discrete analogue of locating the **corner of an L-curve**, the classical criterion for choosing a regularization parameter in inverse problems (Hansen & O'Leary \[[2](#hansen-oleary)\]): the autocorrelation plot plays the role of the L-curve, the plateau is its flat branch, and the main drop is its vertical branch. The adaptive-pruning corner-finding algorithm of Hansen, Jensen & Rodriguez \[[3](#hansen-pruning)\] is built on the same idea of reducing the curve to a small set of candidate segments first — which is what the segmentation stage provides here.

Note that the residual per-plateau ambiguity (several plausible plateaus, e.g. the staircase-shaped curves of blank injections) is *not* resolved by this rule; it is a modeling decision that requires information beyond the geometry of the curve. The segmentation reduces the problem to a handful of quantified candidates on which such a decision can be made explicitly.

> **`select_fcut` is superseded and kept only for `tools/plateau_proto.py`.** Its step 3 returns the *right edge* of the chosen plateau, and that convention has since been measured against synthetic ground truth: it costs a median of **3.5× the optimal baseline error**, about 19× the noise level, and lands within 1.25× of the optimum for only 8 of 71 signals. The true optimum sits in the **interior**, at a log-relative position of roughly 0.65–0.75, because the error valley is steep on the flexible side and the right edge sits on the shoulder of the collapse. Nothing in production calls this function — `_fcutoff` still selects `fcut` by the legacy derivative route — but the rule should not be read as a recommendation.

## 6. Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `penalty` | $`25 \ln N`$ | Cost of adding a segment; fewer/coarser segments when larger |
| `min_size` | 15 | Minimal segment length (points) |
| `rel_slope_max` | 0.2 | Tight flatness threshold on the relative slope |
| `rel_slope_loose` | 0.6 | Loose slope threshold for cliff-bracketed shelves |
| `cliff_min` | 1.0 | Minimum relative slope of a bracketing cliff |
| `rel_noise_max` | 0.006 | Flatness threshold on the relative residual noise |
| `c1` | 0.5 | Sub-fundamental clip: grid points below `c1/N` are removed |
| `noise_floor` | $`4 \times 10^{-7}`$ | Relative noise at or below which a flat segment is frozen |
| `min_width` | 0.5 | Minimum width of a real plateau region, in decades |
| `left_cut` | 0.12 | Log-relative fraction cut from the first kept region |
| `right_cut` | 0.55 | Log-relative fraction of the second region kept |

The first two belong to `trim_candidates`, the last three to `refine_candidates`. Only the last three are calibrated on labeled data; all of them are dimensionless.

## 7. Validation against synthetic ground truth

Because the labels are hand-drawn and censored by the candidate set, `tools/synth_dataset.py` generates chromatograms whose baseline is *known*: exponentially-modified Gaussian peaks (including an isocratic case with widths proportional to retention time), a baseline built from a solvent-front decay, a gradient hump, a drift and a mid-frequency wander, plus white noise. `tools/synth_diag.py` then runs the production sweep and this whole chain on each signal and computes the **true error curve** $`E(f_{cut})`$, the RMSE of the fitted baseline against the known one, whose minimum is an objective optimum free of any labeling.

On a 72-signal benchmark the bracket contains a cutoff within a fraction of a percent of the optimal error for the large majority of signals, and the residual failures are concentrated where theory predicts: signals whose analyte content is negligible compared with the baseline, for which the optimum can lie past the collapse of the autocorrelation curve. Two properties of the error curve are worth recording, since they justify choices made above:

- **The valley is asymmetric.** It is flat and forgiving on the rigid (low-`fcut`) side and steep on the flexible side, because leaving baseline structure in the signal channel is a visible, correctable bias whereas destroying peak area is not. Erring left is therefore the safe direction — which is what the `left_cut` of 0.12, small compared with the 0.65–0.75 position of the true optimum, deliberately preserves.
- **Peak width sets the scale, but not with a transferable exponent.** Regressing the optimal cutoff on the peak width gives $`-0.735`$ under full control (`tools/synth_width_sweep.py`), $`-0.88`$ on the full benchmark and $`-0.50`$ on the real dataset. The scaling is real and strong, but its exponent is a property of the signal population, so no power law of the form $`f_{cut} = b\,w^{a}`$ is used in production: it would not fail loudly on a different instrument, it would mispredict quietly.
| `level_frac` | 0.5 | Fallback level criterion (fraction of the total drop) |

## 7. Practical usage

- Every automatic run overlays the trimmed candidate regions (§4b) on the diagnostic figure of `r2_plots` as a light solid purple fill, on top of the regions of the current method. The production `fcut` is **not** affected by the prototype at this stage. (The earlier `CP chosen` fill, dash-dotted line and `Proto fcut:` log line were removed: the selection rule of §5 proved unreliable on the reference dataset and is pending redesign; §5 is kept as documentation of the prototype in `select_fcut` and `tools/plateau_proto.py`.)
- The autocorrelation curves can be cached with `auto_beads(..., cache_dir=...)` (CLI: `-cd`), and `tools/plateau_proto.py` re-runs the full segmentation + classification + selection chain on the cached curves in milliseconds per signal, which is the intended loop for tuning the parameters against a labeled benchmark.

## 8. Correspondence with the previous pipeline

| Previous ingredient | Replaced by |
|---|---|
| Rolling std / rolling MAD (`_rolling_std`, `_rolling_mad`) | Per-segment residual variance $`\sigma_k^2`$ in the cost |
| Dip test + Sauvola + triangle thresholding | Variance changepoints of the same cost |
| `_long_segments` length filter | `min_size` and the penalty $`\beta`$ |
| `_flat_ends` end trimming (`tol0`, `tol1`, `tol_rdiff`) | Selection rule of §5 (steep-drop anchor, level fallback) |
| Seven absolute tolerances | Two dimensionless thresholds (§4) + one penalty (§2) |

## References

1. <a name="navarro-huerta"></a>Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex chromatograms using the BEADS algorithm. Journal of Chromatography A, 2017, 1507, 1-10. https://doi.org/10.1016/j.chroma.2017.05.057

2. <a name="hansen-oleary"></a>Hansen, P.C.; O'Leary, D.P. The use of the L-curve in the regularization of discrete ill-posed problems. SIAM Journal on Scientific Computing, 1993, 14(6), 1487-1503. https://doi.org/10.1137/0914086

3. <a name="hansen-pruning"></a>Hansen, P.C.; Jensen, T.K.; Rodriguez, G. An adaptive pruning algorithm for the discrete L-curve criterion. Journal of Computational and Applied Mathematics, 2007, 198(2), 483-492. https://doi.org/10.1016/j.cam.2005.09.026

4. <a name="schwarz"></a>Schwarz, G. Estimating the dimension of a model. The Annals of Statistics, 1978, 6(2), 461-464. https://doi.org/10.1214/aos/1176344136

5. <a name="yao"></a>Yao, Y.-C. Estimating the number of change-points via Schwarz' criterion. Statistics & Probability Letters, 1988, 6(3), 181-189. https://doi.org/10.1016/0167-7152(88)90118-6

6. <a name="jackson"></a>Jackson, B.; Scargle, J.D.; et al. An algorithm for optimal partitioning of data on an interval. IEEE Signal Processing Letters, 2005, 12(2), 105-108. https://doi.org/10.1109/LSP.2001.838216

7. <a name="killick"></a>Killick, R.; Fearnhead, P.; Eckley, I.A. Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 2012, 107(500), 1590-1598. https://doi.org/10.1080/01621459.2012.737745

8. <a name="truong"></a>Truong, C.; Oudre, L.; Vayatis, N. Selective review of offline change point detection methods. Signal Processing, 2020, 167, 107299. https://doi.org/10.1016/j.sigpro.2019.107299

9. <a name="ning"></a>Ning, X.; Selesnick, I.W.; Duval, L. Chromatogram baseline estimation and denoising using sparsity (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167. https://doi.org/10.1016/j.chemolab.2014.09.014

10. <a name="durbin-watson"></a>Durbin, J.; Watson, G.S. Testing for serial correlation in least squares regression: I. Biometrika, 1950, 37(3-4), 409-428. https://doi.org/10.1093/biomet/37.3-4.409
