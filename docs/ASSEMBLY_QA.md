# Local assembly quality checks

The repository was assembled from the previously verified reviewer-analysis archive, preserved standalone generator package, final manuscript figure assets and analysis results. No synthetic data were generated during assembly.

**Checks performed on the assembled source:**

- 124 files and 37 Python modules/files packaged; 37 Python files passed a `py_compile` syntax check.
- SHA-256 manifest verified against every included file (123 files excluding the manifest itself).
- `results/experiment_level_summary.csv`: 30 distinct experiments, 28 columns.
- Stationary interpolated ON-centroid transfer: four bin counts sum to 752 (182, 192, 190, 188); gains agree with the paper's rounded values.
- A single-case smoke test (`experiment_001`) ran with the included estimator and a data excerpt from the locally provided minimal dataset. 21 output files/directories were produced, including centroid, displacement/velocity/acceleration, frequency matching, and kinematic metrics.
- Numerical CSV arrays for `experiment_001` matched the earlier archived baseline within `numpy.allclose(rtol=1e-8, atol=1e-8)` under the present runtime. **Output files were not byte identical**, consistent with platform/version formatting differences.

**Not established by these assembly checks:** A fresh, clean Python 3.12 Windows environment end-to-end run of all 30 cases; browser access to the live Zenodo record; manuscript figure asset identity against a newly exported V6 PDF; legal permission to distribute all sources. These remain release checklist items.

Recorded archived full-dataset verification is in `ARCHIVED_VERIFICATION.md`, and distinct from this assembly's smoke test.
