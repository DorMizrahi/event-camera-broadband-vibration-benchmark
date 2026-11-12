# Tests

From the package root, run:

```powershell
python -m unittest discover -s tests -v
```

The suite verifies all six motion scenarios, normalization, Nyquist validation,
the explicit no-V2E path, output completeness, and that the notebook calls the
shared engine instead of defining scientific code.

To execute every notebook cell with a small automated parameter override, run:

```powershell
python tests/execute_notebook_smoke.py
```
