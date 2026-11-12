# Getting started (first time using GitHub)

## What you have

- A local source and results directory, ready to upload to a new GitHub repository.
- The 30-experiment dataset stays on Zenodo, rather than GitHub.
- GitHub's README appears automatically on the repository home page.

## 1. Before making the repository public

Review `RELEASE_CHECKLIST.md` for copyright / institutional permission, dataset accessibility, and identifiers. For the initial upload, **Private** is a safer choice while auditing all file contents. No license has been added by default.

## 2. Sign up and install GitHub Desktop (recommended for Windows)

1. Open https://github.com/signup, create an account, confirm your email and turn on two-factor authentication.
2. Install GitHub Desktop from https://desktop.github.com and sign in with the new account.
3. Extract the downloaded ZIP. Keep the *folder with README.md and source directories*, not the ZIP file itself.
4. In GitHub Desktop choose **File → Add local repository → Choose...**, and select the extracted `event-camera-broadband-vibration-benchmark` folder.
5. Since this folder does not yet contain `.git`, GitHub Desktop can offer **Create a repository here**. Accept this option to initialize the existing project *in place*. Do not create an extra nested folder and do not overwrite the included README, `.gitignore`, or `.gitattributes`.
6. Commit the source changes if GitHub Desktop asks for a commit. Suggested message: `Initial research reproducibility release`.
7. Select **Publish repository** and **Keep this code private**. Use Public only after the checks in `RELEASE_CHECKLIST.md` pass.

Official help: https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop

### Alternative: upload from github.com

Create a repository at https://github.com/new with **Private** visibility and do not initialize it with a README. Then choose **Add file → Upload files** and drag the *unzipped files/folders*. Browser uploads accept up to 100 files at a time and up to 25 MiB per file. This release contains more than 100 files: use multiple batches, or use GitHub Desktop instead. Do **not** upload only the ZIP archive.

## 4. Verify the uploaded repository

Check that README.md, estimator/, scripts/, environment/, results/, figures/, tests/ and docs/ are visible and that Python source files can be opened. Compare the uploaded tree against MANIFEST_SHA256.csv. Open figure thumbnails to inspect them. The repository URL must be accessible without login **after** switching it to Public.

## 5. Final manuscript change

Only after the release passes review, replace `[GitHub repository URL]` in the paper's Data Availability section with the exact public URL, and regenerate the six-page PDF without changing its contents otherwise.

## Running locally (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements.txt
python scripts/rebuild.py "C:\path\to\DATASET" reproduced
```

The path `DATASET` must contain `checksums.sha256`, `dataset_summary.csv`, and `experiments/experiment_001` through `experiment_030`.

**Care:** Installing Python dependencies does not install or recover the original `v2e` emulator. That emulator is not required to reproduce analyses from the archived HDF5 streams.
