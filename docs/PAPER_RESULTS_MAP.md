# Manuscript V6 result-to-file mapping

The results and supplementary figures below derive from the archived 30-experiment dataset and the previously verified reviewer-analysis package.

| Paper content | Supplied files | Interpretation |
| --- | --- | --- |
| Table 1: motion conditions, recording parameters | `environment/emulator_settings.csv`, `results/experiment_factor_allocation.csv` | Some factors co-vary; allocation was not randomized |
| Fig. 1: marker, ON/OFF events and workflow | `figures/manuscript/Figure1_FINAL.png`; `scripts/figure1_figure2_final_generation.py` | Representative case experiment_025 |
| Sec. 5.1: matched-frequency errors | `results/frequency_error_analysis.csv`, `results/frequency_error_summary.csv` | Matched-only MAE, not all reference components |
| Sec. 5.1: K sensitivity | `results/peak_count_sensitivity.csv`, `results/peak_count_experiments.csv` | 237 matches / 861 references at K = 10; ceiling 239 |
| Table 2: overall frequency, chirp and paired tests | `results/experiment_level_summary.csv`, `results/statistical_reassessment.csv`, `results/baseline_summary.json` | Experiment is unit for paired statistics |
| Sec. 5.2: Chirp performance and SG ablation | `results/spectral_variants.csv`, `results/spectral_variant_summary.csv`, `figures/chirp/ridge.png` | Four Chirp cases |
| Fig. 2 and Sec. 5.3: acceleration time histories/NRMSE | `figures/manuscript/Figure2_FINAL.png`, `figures/kinematics/time_histories.png`, `figures/kinematics/absolute_errors.png` | Pixel-domain acceleration; limited absolute accuracy |
| Sec. 5.3: differentiator transfer / boundary checks | `results/differentiation_response.csv`, `results/boundary_sensitivity.csv` | Centered gradients attenuate high frequencies |
| Sec. 5.3: ON/OFF ablations | `results/polarity_comparison.csv`, `results/polarity_comparison_strata.csv` | Performance trade-offs; no edge labels |
| Sec. 5.3: ON centroid amplitude/phase | `results/component_transfer.csv`, `results/transfer_summary.csv`, `figures/centroid/amplitude_phase.png` | 752 resolved components, pooled analysis |
| Interpolation and missing events | `results/missing_bin_analysis.csv`, `results/interpolation_kinematics.csv` | Missing-bin fractions are small in this dataset |

## Amplitude and phase details (Reviewer B1)

For interpolated ON centroids, stationary frequency strata have 182/192/190/188 valid components in 50–500/500–1000/1000–1500/1500–2000 Hz, respectively (sum: 752). Median gains are approximately 1.46/1.71/1.80/1.82, with circular phase offsets 17.7°/17.1°/23.0°/26.9°. These are component-level pooled summaries, not 752 independent experiment replicates.

## Data origin

All precomputed CSVs and accompanying figures were copied from the archived `VIBROENGINEERING2027_REVIEWER_ANALYSIS.zip`. The generator, estimator and analysis code are preserved. No new synthetic experiments or invented data were introduced when assembling the repository.
