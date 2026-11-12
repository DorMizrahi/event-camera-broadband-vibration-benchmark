# Final verification

**Outcome.** The portable release driver completed an independent rebuild from the 30 archived HDF5 files. All 32 result files (30 CSVs and two JSON files) and all nine PNGs matched the first evidence run byte for byte. Nine accompanying vector PDFs passed format checks. The figure contact sheet was visually inspected. PDF metadata can depend on export time.

| Check | Verified outcome |
|---|---|
| Baseline preservation | All 609 saved per-experiment files matched the original Phase 1 ZIP byte for byte. The pre-analysis summary SHA-256 matched the frozen value. |
| Input identity | 122/122 minimal-release manifest entries passed. All 30 recording IDs, 26 stationary cases, four Chirps and 39 active coordinate series were accounted for. |
| Direct event checks | ROI-filtered HDF5 ON counts matched all original per-bin counts. Sampled direct coordinate means matched frozen centroids on each active axis. |
| Polarity and missingness | Per-case reconstructed missing masks exactly matched count-based masks. Direct count-weighted combined centroids, polarity differences and RMS ratios passed. |
| Frequency and phase | All baseline stationary peak lists and Chirp ridges matched. Independently computed complex projections passed. Conditional MAEs, cap bounds and matching count identities passed. |
| Differentiation and statistics | Interior gradients on sampled sinusoids reproduced analytic gains. Independent vector norms reproduced the original x endpoints and each trim's time-domain RMSE. Wilcoxon and Holm checks passed. |
| SG and design | SG response agreed with direct filtering of sinusoids at four frequencies. Factor counts and design occupancy checks passed. The baseline and controlled ablations were repeated without changing the estimator. |
| Numerical completeness | Every result table populated; numerical columns had no infinity. Context-dependent missing fields describe distinct diagnostic populations, including undefined full x-displacement Wilcoxon p, Chirp versus stationary fields and unassigned reference peaks. No failed cases were discarded. |
| Release | Scientific Python source in the frozen estimator modules, standalone generator and historical broadband generator matched the supplied files. The portable repository passed a private-path and compiled-file scan. Relative scientific-report links and all matrix evidence paths resolved. |
| Reviewer coverage | 17 numbered Reviewer A comments and 14 stable Reviewer B requests. 24 complete, six partial, one outside the existing data. A complete disposition describes the evidence for the current phase; manuscript edits are subsequent work. |

**Limits.** No matched actual-grid-aligned line was available for conditional localization comparison. Contrast and displacement were aliased. Historical v2e and Python builds could not be recovered. The live Zenodo record could not be independently accessed during packaging; the local minimal release and checksums were verified. Primary full texts were unavailable for some suggested literature, as documented in reports/literature.md. Real-camera calibration, prospective causal ROI performance, between-frame continuous-time fidelity and independently controlled nuisance-factor effects remain unvalidated.

From the portable release, run `python scripts/rebuild.py DATASET reproduced` to repeat the end-to-end computation and independent checks. The top-level SHA256SUMS.txt records hashes of package contents except itself.
