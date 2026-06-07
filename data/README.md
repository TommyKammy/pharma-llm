# Data Directory

This directory documents local dataset handling. Most dataset files are intentionally not tracked by Git.

## Policy

- Do not commit model weights, internal documents, secrets, raw data, or large derived datasets.
- Do not use raw AI output directly as a training target.
- Training records must include provenance metadata and review state.
- Evaluation data must be kept separate from training data.

## Expected Local Layout

```text
data/
  raw/              # local only, not tracked
  internal/         # local only, not tracked
  prepared/         # reviewed and transformed local datasets, mostly not tracked
  argilla_exports/  # local import/export artifacts, not tracked
```

Small schema examples may be added later under tracked example directories if needed.

