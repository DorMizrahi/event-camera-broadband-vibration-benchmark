# Broadband Vibration Estimation from Event-Camera Streams: A Synthetic Benchmark

**Source code, benchmark configuration, evaluation results, and supplementary material** for the VIBROENGINEERING 2027 conference manuscript, *“Broadband Vibration Estimation from Event-Camera Streams without Learned Calibration: A Synthetic Benchmark.”*

**Status:** Public research reproducibility package. The author has confirmed release permission and access to the separate Zenodo dataset. The historical v2e environment remains unidentified; see the reproduction limitations below.

## What this project does

A deterministic, calibration-free pipeline extracts polarity-specific event centroids from asynchronous DVS streams; estimates stationary frequency components and Chirp ridges; reconstructs band-limited displacement; and computes pixel-domain velocity and acceleration through both Fourier and centered finite-difference paths. All inference uses event streams only. Archived ground truth is used **for evaluation and supplementary transfer diagnostics**, never for reference-dependent adjustment of the estimator.

**Scope:** 30 synthetic tests (26 stationary and four Chirp), one moving circular marker, uniform background, 50–2000 Hz. All kinematics are expressed in **px, px/s, and px/s²** rather than physical units. Robustness to real scenes, sensor noise, illumination changes and geometric calibration is **not established**.

## Headline results and interpretation

- **0.355 Hz matched-frequency macro MAE** over the 26 stationary experiments. This **excludes unmatched reference frequencies**; it is not a completeness score.
- Frequency macro Precision/Recall: **0.918/0.551**; micro Recall **0.275** with the ten-peak cap (**237 matches among 861 reference components**).
- Four Chirps: **22.1 Hz median ridge RMSE**.
- Numerical time differentiation reduced acceleration RMSE relative to Fourier differentiation in **29/30** experiments (median reduction **10.84%**), but **absolute reconstruction accuracy can remain poor**.
- The ON-centroid has amplitude/phase bias: 752 sufficiently separated stationary x-axis components were analyzed; see `results/transfer_summary.csv` and `figures/centroid/amplitude_phase.png`.

See `docs/PAPER_RESULTS_MAP.md` and the CSV tables for exact definitions and evaluation populations.

## Repository structure

| Directory | Purpose |
| --- | --- |
| `estimator/` | Original deterministic event-processing and evaluation source |
| `simulator/` | Standalone generator, presets, optional notebook, tests and v2e installation notes |
| `broadband_generator/` | Historical broadband generation source and configuration |
| `baseline/` | Baseline evaluation entry points |
| `scripts/` | Full rebuild, reviewer diagnostics and figure-generation scripts |
| `tests/` | Numerical checks against original archived event streams |
| `environment/` | Per-experiment recorded emulator arguments, historical limitations, dependency manifest |
| `results/` | Precomputed CSV/JSON results for 30 experiments, spectra, sensitivity, centroid transfer and other checks |
| `figures/` | Supplementary PNG/PDF analysis figures and figures used in the manuscript |
| `docs/` | Methodological limitations, release checklist, dataset instructions and verification report |

## Data: download separately from Zenodo

The original synthetic benchmark is described in **[Zenodo record 22539609](https://zenodo.org/records/22539609)**. The minimal dataset used for analysis has the following structure:

```text
DATASET/
  checksums.sha256
  dataset_summary.csv
  experiments/
    experiment_001/
      events.h5
      ground_truth.csv
      metadata.json
      video.mp4
    ...
    experiment_030/
      ...
```

**The 30 recorded HDF5 data streams, videos and reference trajectories are intentionally excluded from GitHub**. To run the code, download and extract the actual dataset package. Check its license and accessibility independently before public release; the historical verification was against a supplied local copy, not a live Zenodo download.

## Rebuild the results (Windows, Python 3.12)

From the project root, after placing the dataset in a separate folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements.txt
python scripts/rebuild.py "C:\path\to\DATASET" reproduced
```

The script performs baseline inference and evaluation, computes reviewer diagnostics, generates figures and executes numerical checks. **Choose an empty output directory.** `reproduced/` is ignored by Git. For Unix-like environments, create/activate a compatible virtual environment using `python3.12 -m venv .venv`; note that the historical dependency lock targets Windows and may require platform-specific adjustments. No claim of cross-platform byte-identical reproduction is made.

If a full rebuild is unavailable, the verified analysis tables are available in `results/` and the key plots in `figures/`. See `docs/GETTING_STARTED.md` for a beginner-friendly guide and `environment/reproducibility_notes.md` for explicit limitations.

## What **can** and **cannot** be reproduced

**Can:** Reprocess the archived event streams using the supplied estimator, compare against archived ground truth, regenerate frequency and kinematic tables, and repeat sensitivity analyses and figure generation (subject to environment availability).

**Cannot be guaranteed:** Byte-identical generation of the historical events. The original `v2e` build/commit and generation-time Python environment are unknown. The `v2e 1.5.1` reference mentioned in the manuscript is not confirmed to be the exact historical release. Running the standalone generator presets does not reproduce the archived event streams by itself.

## Citation and permissions

Please cite the manuscript if and when bibliographic publication details become available, and cite the benchmark dataset using its Zenodo metadata. This repository **does not currently contain a public software license**: rights/permissions for source and generated materials must be confirmed before switching the repository to Public. Third-party notices are under `simulator/THIRD_PARTY_NOTICES.md`.

## Release and manuscript link

The repository is public at https://github.com/DorMizrahi/event-camera-broadband-vibration-benchmark. See [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) for the release audit and remaining optional verification tasks. The final manuscript's Data Availability statement should link to this repository and the separate Zenodo dataset.
