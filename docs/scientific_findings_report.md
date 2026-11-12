# How accurately does the event centroid measure vibration?

## Executive scientific assessment

The baseline reproduces. Its strongest result is localization of successfully detected stationary components. At ten peaks per recording, 237 of the 239 matches permitted by the cap were recovered. At that setting 624 of 861 reference lines went unmatched. Strong localization and low pooled recall are compatible.

The largest qualification concerns displacement amplitude and phase. On the x axis, the median interpolated ON-centroid/reference RMS ratio was 1.79. Even after broadband reconstruction the median ratio was 1.72 and the median displacement NRMSE was 1.04. The combined-polarity representation reduced the median reconstructed displacement NRMSE to 0.65 while degrading stationary frequency recovery. The baseline retains ON centroids throughout.

The gradient advantage persists after removing 5% at each end. Lower x-acceleration RMSE occurred in 29/30 cases both on the full record and in the central 90%. The composed gradient attenuates 2-kHz acceleration by 12.49% relative to Fourier differentiation. This provides a mechanism for suppressing high-frequency reconstruction error while also attenuating true motion. Median gradient acceleration NRMSE was 1.081, with 17/30 cases above unity.

The 30 prescribed cases support paired, within-benchmark descriptions. They supply little independent leverage for separating contrast from displacement or motion family. They do not establish real-camera accuracy or sensitivity to controlled nuisance factors. All reference-assisted diagnostics described below are evaluation only. No correction estimated from a reference was fed back into inference.

## A. How closely does the ON centroid follow the marker?

**Question and method.** R1.8, R1.13 and R2.1 concern the geometry of the measurement. For each active axis we compared (i) observed bin centroids on their actual mask, (ii) the complete, linearly interpolated centroid and (iii) the frozen broadband displacement with the baseline reference interpolated to the bin centers. Position centroids use the known rendering origin (120,90) px. The reference displacement uses the baseline's mean-centering convention. RMSE is the square root of mean squared paired error and NRMSE divides by reference RMS on the same sample mask. The amplitude columns report RMS ratio and difference in px. The observed mean error therefore combines geometric bias and finite-record centering. The broadband filter removes DC by construction.

**Results.** Across the 30 x axes, median observed/interpolated/reconstructed correlations were 0.808/0.808/0.847 and median NRMSEs were 1.203/1.203/1.042. Median signed interpolated mean error was −0.194 px. Median reference-power-weighted magnitude-squared coherence was 0.943 (Hann Welch segments, 512 samples, half overlap, 50–2000 Hz). High coherence coexists with substantial amplitude mismatch.

**Frequency dependence.** At each stationary reference frequency, the diagnostic divides the complex Hann-weighted Fourier projection of the demeaned estimate by the corresponding reference projection. Positive phase denotes an estimate leading the reference under the exp(−i2πft) convention. We flag nearest-neighbour separations below two actual FFT bins and denominator magnitude ≤10⁻⁸ as unreliable. This spacing screen reduces interference without guaranteeing separation of unequal-strength lines. Of 861 x-axis stationary reference projections, 752 passed. In ascending 500-Hz bands their pooled interpolated median gains were 1.46, 1.71, 1.80 and 1.82. Circular mean phases were +17.7°, +17.1°, +23.0° and +26.9°. The release includes phase concentration and experiment counts. These are pooled line diagnostics, with dense experiments receiving more weight.

**Chirps.** For each original ridge-window center we projected the signal and reference onto the known linear or logarithmic sweep phase within a 50-ms Hann window. This compares local complex amplitudes along the prescribed sweep. Interpolated ON median ratios across the same four frequency bands were 1.00, 1.60, 1.60 and 1.61. Window counts were 14, 15, 14 and 8. The overlapping windows of four recordings provide descriptive evidence only.

**Interpretation and placement.** Centroid motion contains frequency information but can substantially amplify displacement and introduce a frequency-dependent phase lead. Event activity samples moving contrast boundaries. The experiment cannot uniquely attribute the phase to a single delay, boundary process or emulator effect. State the amplitude limitation alongside the first kinematic result. Put individual-axis results and transfer spectra in the supplement. [Accuracy](../results/centroid_accuracy.csv) · [Strata](../results/centroid_accuracy_strata.csv) · [Transfer diagnostics](../results/transfer_summary.csv) · [Amplitude and phase](../figures/centroid/amplitude_phase.pdf)

## B. Which geometric quantity does each polarity measure?

**Method.** R1.8 asks why ON events are primary. Within the same ROI and 50-µs bin, C+ and C− are the arithmetic means of coordinates from positive and negative events. The combined centroid weights them by event counts, (N+C+ + N−C−)/(N+ + N−). The difference is C+ − C− in bins where both exist. The original binner interpolates each constructed representation on its own validity mask and extends the nearest observed value at an unobserved end. All four pass through the same baseline reconstruction and spectral settings.

| Representation | Median reconstructed x-displacement NRMSE | Macro precision / recall | Median Chirp ridge RMSE (Hz) |
|---|---:|---:|---:|
| ON | 1.042 | 0.918 / 0.551 | 22.1 |
| OFF | 1.637 | 0.892 / 0.539 | 27.5 |
| Combined | 0.646 | 0.735 / 0.410 | 30.3 |
| Difference | 1.897 | 0.915 / 0.553 | 23.8 |

**Interpretation and placement.** Combining polarities improves median position reconstruction while losing spectral completeness. Taking a difference cancels common translation of the two event clouds. It can encode edge separation or motion direction and still yield spectral peaks. Its median reconstructed RMS ratio is 2.54 and it has the largest median position error. Polarity-specific leading/trailing-edge sampling is a plausible interpretation. The benchmark has no event-by-event edge correspondence labels. Retain the original ON baseline, acknowledge the task-dependent trade-off in Methods/Discussion and give the comparison in the supplement. [Polarity metrics](../results/polarity_comparison.csv) · [Coverage and strata](../results/polarity_comparison_strata.csv) · [Spectral variants](../results/spectral_variant_summary.csv)

## C. What does applying a gradient twice do?

**Derivation.** For h=1/20,000 s and angular frequency ω=2πf, Fourier differentiation on the periodic DFT modes multiplies displacement by iω or −ω². Interior first differences are (u[n+1]−u[n−1])/(2h), with response i sin(ωh)/h. Applying that stencil twice gives (u[n+2]−2u[n]+u[n−2])/(4h²) and response −sin²(ωh)/h², valid at least two samples from either end. Relative magnitude is sin(ωh)/(ωh) for velocity and its square for acceleration. The first gradient uses the second-order one-sided stencil (−3,4,−1)/(2h) at the first sample and its reflected counterpart at the last. Applying it twice also affects the adjacent sample. A single shift-invariant frequency response describes only the interior.

**Results and interpretation.** At 2 kHz the respective relative gains are 0.93549 and 0.87514. Over 1–2 kHz the median ratio of gradient to Fourier x-acceleration spectral power is 0.828. For residuals against GT the ratio is 0.789. The velocity ratios are 0.918 and 0.892. Per-record power is summed from squared full-record DFT magnitudes on the same bins for both methods. Descriptive Spearman correlations between the fraction of positive-frequency reference power at 1–2 kHz and relative RMSE improvement were 0.504 for acceleration and 0.592 for velocity. Full-record transforms include edge discontinuities. These associations cannot quantify a causal fraction of the gain or identify noise separately from biased motion estimates.

**Impact.** R1.12 and R2.2 require a narrower interpretation. Fourier differentiation implements the ideal multiplier on the retained periodic modes. Gradient differentiation trades high-frequency fidelity for lower empirical RMSE on these reconstructions. Include the two gain ratios and one sentence on residual power in the paper. [Analytical gains](../results/differentiation_response.csv) · [Experiment-level power](../results/differentiation_empirical.csv) · [Operator responses](../figures/differentiation/operator_responses.pdf)

## D. Does the advantage survive boundary exclusion?

**Method.** Keep every original sample and reference fixed. Exclude ceil(pN) samples from each end for p=1%, 2.5%, 5%. Score the retained interior and the union of removed ends separately. The paired outcome is 100(RMSE_F−RMSE_T)/RMSE_F. Bootstrap intervals below resample 30 experiment IDs with replacement for 10,000 median draws, seed 2027. These conditional sensitivity intervals differ in purpose from the original 100,000-draw analysis.

| Excluded per end | Interior velocity reduction | Interior acceleration reduction | Acceleration wins |
|---|---:|---:|---:|
| 0% | 4.92% | 10.84% | 29/30 |
| 1% | 4.94% | 10.89% | 29/30 |
| 2.5% | 4.96% | 10.84% | 29/30 |
| 5% | 4.95% | 10.88% | 29/30 |

**Interpretation.** At 5%, the acceleration interval is 10.03–11.95% and the excluded ends alone have a median reduction of 10.75%. Both paths share the periodic displacement filter, so trimming the scoring region does not remove all possible wraparound effects. It establishes that the observed ordering persists away from the ends. Include the central-90% result beside the full-record result for R1.12 and R2.3. [All boundaries](../results/boundary_sensitivity.csv) · [Intervals](../results/boundary_summary.csv)

## E. How much recall does the reporting cap cost?

**Method.** Freeze periodogram, prominence, distance, interpolation and ranking. Extract the entire candidate list and truncate it at each K. The original no-candidate fallback was never needed. Use the original ±10-Hz one-to-one assignment independently for each case and K. The per-case ideal ceiling is min(K,M)/M for M references. Macro rates weight cases equally; micro rates use pooled counts. K equal to the reference count is an oracle diagnostic. No K change enters the displacement filter.

**Results.** At K=10 the ceilings are 0.557 macro and 0.278 micro, compared with 0.551 and 0.275 observed. At least 622 of the 624 missing components were inevitable under the cap across this allocation. This is a structural lower bound on misses, not a causal identification of which individual lines were missed. At K=5 all 130 estimates matched. At K=100, 758 of 1,118 reported peaks matched 861 reference lines, with micro precision 0.678 and recall 0.880. The reference-count rule yielded 750/802 matches and micro recall 0.871.

**Remaining loss.** In the unrestricted candidate audit 50 reference lines had no candidate within tolerance. Another 53 had a nearby candidate but remained unassigned under the original assignment. Close lines, prominence rejection, band-edge handling and biased localization can overlap; a missed line alone does not identify its cause. Explain the cap in the paper and preserve all family/K results for R1.10 and R2.5. [Sensitivity](../results/peak_count_sensitivity.csv) · [Reference-level candidate audit](../results/candidate_audit.csv) · [Precision and recall](../figures/frequency/cap_and_bias.pdf)

## F. Is sub-bin localization credible?

**Method.** For each of the 237 baseline matches, compare the refined frequency and its own original FFT bin against the same assigned reference. Signed error is estimate minus reference. Holding assignment fixed isolates peak refinement conditional on the baseline matches. Nearest-neighbour separation uses every reference, including misses. Grid-aligned means distance ≤10⁻⁶ Hz from a multiple of the actual Fs/N, with nominal 5-Hz distances also retained in the candidate audit.

**Results.** Pooled MAE fell from 1.293 Hz for the bin centers to 0.367 Hz for the interpolated peaks. Mean signed error was −0.032 Hz, median −0.001 Hz and SD 0.639 Hz. Median absolute error was 0.184 Hz, 95th percentile 1.259 Hz and maximum 3.351 Hz. Equal weighting of 26 within-case MAEs gives the reported 0.355 Hz. Of 861 references, 15 aligned to the actual grid and 17 to the nominal grid. None of the 237 matches aligned to the actual grid. Thus a matched aligned-group MAE is undefined. The nearest-neighbour bins (0,5], (5,10] and >50 Hz had 19, 15 and 75 matches with MAEs 0.855, 1.545 and 0.181 Hz. Error does not vary monotonically across the two closest strata.

**Impact.** Sub-bin localization is possible for isolated peaks. The study establishes neither sub-bin resolving power for adjacent components nor assurance that a line will be detected. Report the two MAE weightings and their denominators for R1.11 and R2.4. Grid alignment, component strength, band edges and family were not randomized independently. [Signed-error distributions](../results/frequency_error_summary.csv) · [Individual matches](../results/frequency_error_analysis.csv)

## G. Where are centroid samples missing?

**Method.** Count observed ON/OFF bins independently per active axis, record all contiguous runs, and assign bins to four successive record quarters. Compare original linear interpolation to a diagnostic previous-observation hold with first-observation fill for leading gaps. Recompute spectra, ridges, filtered displacement and derivatives with all other settings fixed.

**Results.** Across 119,982 bins counted once per recording, 202 ON and 139 OFF bins lacked observations. Median missing percentage was zero for both polarities. The maximum ON fraction was 1.60%; longest ON and OFF runs lasted 600 and 550 µs. With hold filling, aggregate matches stayed at 237 and macro MAE shifted from 0.354519 to 0.354864 Hz. Median Chirp RMSE stayed at 22.1 Hz. Median absolute change in x-gradient acceleration NRMSE was zero and the maximum was 0.0081.

**Impact.** R1.7 can be answered with observed coverage and this controlled sensitivity. Values inside missing bins have no observed event centroid for direct validation. Results do not extend to long dropouts or weakly textured real images. Use one sentence in Methods and provide run positions, family/displacement strata and per-case sensitivity in the repository. [Missing bins](../results/missing_bin_analysis.csv) · [Individual gaps](../results/missing_runs.csv) · [Ablation](../results/interpolation_kinematics.csv) · [Diagnostic figure](../figures/diagnostics/boundaries_missing.pdf)

## H. Does spectral smoothing affect peaks or ridges?

**Method.** The offline 11-point cubic SG smoother applies centered coefficients at offsets −5…5 with SciPy mode interp. Its interior response is the sum of h[k]exp(−i2πfk/Fs). The positive real response in the analyzed band gives zero phase and zero interior group delay. A causal replay of this symmetric kernel would introduce five samples (250 µs) of delay. At the first and last five samples the implementation evaluates a local polynomial fitted to the end window. Those boundary coefficients depend on position. The ablation disables only smoothing on the spectral path and retains the original centroid signal, peak procedure and ridge settings.

**Results.** The 2-kHz amplitude gain is 0.6942 (−3.17 dB). Removing smoothing gives 238 matches among 257 reports, macro MAE 0.328 Hz, precision 0.927 and recall 0.553. Median Chirp RMSE falls from 22.1 to 16.7 Hz while median within-25-Hz fraction remains 0.846. Different retained matches contribute to each MAE.

**Impact.** R1.9's proposed claim of immaterial filtering would be too broad. State its upper-band attenuation and observed Chirp sensitivity. Four Chirps do not establish an optimal filter. The original kinematic path uses the unsmoothed centroid. [Exact coefficients](../results/sg_coefficients.csv) · [Response](../results/savitzky_golay_analysis.csv) · [Per-case ablation](../results/spectral_variants.csv)

## I. Which experimental effects are identifiable?

**Method and results.** Enumerate family, displacement, contrast and axes, preserve the full component/phase lists and recorded motion seeds. Only 21 of 270 family × displacement × contrast × axis cells are occupied. A treatment-coded categorical additive design including the intercept has 16 columns and rank 12. Each contrast occurs at one displacement. The 21 x-only and 9 xy cases also vary by family and amplitude. X-gradient acceleration median NRMSE was 2.117, 0.977 and 0.438 at 2.5, 6 and 11 pixels respectively.

**Interpretation.** Within-benchmark severity differs across allocations. An adjusted additive model cannot uniquely estimate all the proposed effects. Four Chirps and sparse cells provide little basis for stable interactions or a hierarchical generalization. Per-case seeds generate distinct prescribed trajectories and phases; there is no documented sampling frame or randomized allocation of nuisance conditions. The experiment is the paired unit. Frequency lines, axes and overlapping windows are dependent within a recording. The original Holm family contains 35 finite comparisons; the overall x-displacement comparison is undefined under the recorded implementation. Both reported x derivative endpoints give W=1, rank-biserial 0.9957 and adjusted p=1.304×10⁻⁷. This expresses directional consistency of ranked differences, with magnitude reported separately as paired percentage reductions. State the conditional scope for R1.5/R1.14. [Design](../results/experiment_factor_allocation.csv) · [Cells](../results/factor_crosstab.csv) · [Statistical assessment](statistical_assessment.md)

## J. What does a typical experiment look like?

**Selection and results.** Select the case nearest its group's median on a prespecified relevant measure, breaking ties by archive order. Experiment 006 represents sparse matched-frequency MAE; 012 represents dense recall; 030 represents low-displacement x-gradient acceleration NRMSE; 023 represents Chirp ridge RMSE. Both time histories use the same fixed 80–90 ms display interval. Full-record paired NRMSE distributions include all 30 cases. Separately, the missing-bin panel deliberately selects the largest ON missing fraction to show the extent of dropouts.

**Interpretation and placement.** The absolute-error distribution resolves a central ambiguity in the relative improvements: in 17/30 cases gradient acceleration RMSE exceeds the reference RMS. A trajectory overlay supplies timing and amplitude context. Individual examples add no independent replicates. For R1.13/R1.17 and R2.6–R2.8, replace part of Figure 2 with the paired absolute-error distribution and a short time trace; place spectra and ridges in the supplement. [Time histories](../figures/kinematics/time_histories.pdf) · [All paired errors](../figures/kinematics/absolute_errors.pdf) · [Example spectra](../figures/frequency/representative_spectra.pdf) · [Chirp](../figures/chirp/ridge.pdf)

## Top scientific findings

1. ON centroids preserve frequency information while systematically distorting amplitude and phase.
2. The ten-peak cap alone limits pooled recall to 0.278; higher caps introduce substantial false positives.
3. The gradient advantage survives trimming and accompanies measurable high-frequency attenuation.
4. Absolute acceleration error exceeds reference RMS in over half the cases despite paired improvements.
5. Factor allocation prevents a causal displacement or contrast interpretation. SG sensitivity matters for the four Chirps.

## Recommended manuscript changes

Use the abstract and conclusions to delimit pixel-domain synthetic evidence. In the introduction, distinguish a known-motion benchmark from validation on instrumented structures. Specify ROI look-ahead, masks, SG attenuation and composed-gradient response in Methods. In Results, replace unqualified differentiation superiority with paired reductions, the interior result and absolute NRMSE. Add 237/239 attainable matches to Table 2. Replace a relative-only panel in Figure 2 and relabel the two processing paths in Figure 1. [Space estimates](../reviewer_mapping/six_page_prioritization.csv)

## Reviewer response readiness and remaining risks

The numerical concerns about centroid validity, polarities, differentiation, boundaries, cap sensitivity, missing bins and smoothing now have inspectable per-case evidence. The absence of matched grid-aligned lines, aliasing among design factors, unknown historical v2e build, full-record ROI and lack of physical validation remain substantive limitations. Neither high temporal timestamp precision nor statistically consistent paired improvement establishes real-camera accuracy. A controlled nuisance-factor generation study, causal ROI demonstration and calibrated physical validation each need separate work. [Full comment dispositions](../reviewer_mapping/reviewer_evidence_matrix.md) · [Methodological scope](methodological_limitations.md)

