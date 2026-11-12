# Public-release checklist

Before setting visibility to **Public**:

- [ ] Confirm institutional and author permissions to release project source code, generated images and supporting analysis files. No license is granted by this package.
- [ ] Check for secrets, personal paths, identifiable metadata and organization-confidential material. Source files are included unchanged.
- [ ] Confirm that Zenodo record 22539609 is publicly accessible, its download works, and the downloaded dataset matches the expected 30-case minimal release/checksums.
- [ ] Verify that the GitHub upload includes **real source files** rather than only the ZIP archive.
- [ ] Confirm reproduction on an appropriate Python 3.12 environment, or document any platform-specific limitations discovered.
- [ ] Confirm that `results/experiment_level_summary.csv` contains 30 rows and the frequency/centroid/kinematic totals match the manuscript.
- [ ] Confirm that `results/transfer_summary.csv` includes 752 reliable stationary x-axis components for ON-centroid phase/gain characterization.
- [ ] Verify that manuscript figures `figures/manuscript/Figure1_FINAL.*` and `Figure2_FINAL.*` match those in the V6 final PDF.
- [ ] Review the scope of any future software LICENSE with the institutional rights holder, and add it only if approved.
- [ ] Switch visibility to **Public** once release is permitted; open the URL in a private browser window to check public access.
- [ ] Replace `[GitHub repository URL]` in the manuscript with the actual repository URL and verify the final PDF remains within six pages.

**Known historical limit:** Original v2e build and Python environment remain unknown. The archived event streams are the reproducibility baseline.
