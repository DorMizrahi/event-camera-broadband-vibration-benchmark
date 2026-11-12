# Verification Report

Verification date: 2026-08-26

## Scope

The release was checked as a standalone single-experiment generator. Reduced
frame counts were used for execution smoke tests; these tests use the same shared
engine and V2E command builder as the 8,000-frame defaults.

## Results

| Check | Result |
|---|---|
| Python compilation of all modules | PASS |
| Notebook JSON/nbformat structure | PASS |
| Six scenario trajectory tests | PASS |
| Peak-displacement normalization | PASS |
| Nyquist rejection | PASS |
| SI normalization requires physical scale | PASS |
| Missing-V2E behavior, 20 frames | PASS |
| No synthetic event fallback | PASS |
| Programmatic execution of all 11 notebook cells | PASS |
| Real local V2E conversion, 50 frames | PASS |
| V2E HDF5 readability and quality control | PASS |

The real V2E smoke experiment used a 50 Hz single sine sampled at 1,000 FPS.
It produced 18,083 events. The output contained one HDF5 dataset named `events`
with shape `(18083, 4)`, dtype `uint32`, columns `[t_us, x, y, p]`, and polarities
`0` and `1`. The per-experiment quality-control report passed all checks.

The missing-V2E smoke experiment produced the video, 20 PNG frames, labels,
ground truth, plots, metadata, status files, and summaries. It did not create
`events.h5`, and `fallback_used` was false.

## Default Validation

The defaults resolve to 40,000 FPS, 0.2 s, and 8,000 frames. Every bundled JSON
preset resolves to one of the six supported motion types and 8,000 frames. The
highest-frequency broadband defaults provide 20 samples per 2,000 Hz cycle.

## Conclusion

PASS. The Python script and notebook are operational wrappers around the same
simulation engine. V2E is the only event conversion path; absence or failure of
V2E leaves the experiment complete without an event file and records the reason.
