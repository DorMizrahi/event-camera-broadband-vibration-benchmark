# V2E Installation and Detection

V2E is an external scientific dependency and is not copied into this package.
Use the official repository and record the exact V2E revision used for published
experiments: <https://github.com/SensorsINI/v2e>.

## Recommended Installation

The official project recommends a dedicated Conda environment, PyTorch installed
for the available CPU/GPU platform, and an editable installation of the checkout.
A typical setup is:

```bash
conda create -n v2e python=3.10
conda activate v2e
# Install the appropriate PyTorch build for this machine first.
git clone https://github.com/SensorsINI/v2e
cd v2e
python -m pip install -e .
```

Then install this generator's core dependencies in the same environment:

```bash
python -m pip install -r requirements.txt
```

Run `v2e --help` and `python generate_single_experiment.py --validate-only` before
a long experiment.

## Separate V2E Environment

If the generator and V2E use different Python environments, configure either:

```python
"v2e": {
    "script_path": r"C:\path\to\v2e\v2e.py",
    "python_executable": r"C:\path\to\v2e-env\python.exe"
}
```

or the environment variables:

```powershell
$env:V2E_SCRIPT = "C:\path\to\v2e\v2e.py"
python generate_single_experiment.py
```

On Linux/macOS:

```bash
export V2E_SCRIPT=/path/to/v2e/v2e.py
python generate_single_experiment.py
```

The `v2e.command`/`V2E_COMMAND` option can instead point to a working `v2e`
console executable.

## Why SuperSloMo Is Disabled by Default

The default source video is already rendered at 40,000 FPS, providing 20 samples
per cycle at 2,000 Hz. The command therefore passes `--disable_slomo` and sends
the high-rate frames directly through V2E's DVS event emulator. Set this option
to false only when the source-sampling design intentionally requires V2E's frame
interpolation and its model checkpoint is installed.

## Missing or Failed V2E

Missing V2E is non-fatal for base data generation. The directory retains the
video, PNG frames, ground truth, labels, plots, parameters, and metadata. It does
not create a substitute `events.h5`.

For every run, inspect:

- `event_conversion_status.json`
- `v2e_output/v2e_stdout.txt` and `v2e_stderr.txt` when V2E started
- `run_v2e_later.ps1` or `run_v2e_later.sh`

Only `status: completed` and `used_v2e: true` indicate a validated event file.
