# GitHub Workflow

Remote repository:

```text
https://github.com/TommyKammy/pharma-llm
```

## Local Remote Setup

```bash
git remote add origin git@github.com:TommyKammy/pharma-llm.git
git push -u origin main
```

Use HTTPS instead if this host is not configured for GitHub SSH:

```bash
git remote add origin https://github.com/TommyKammy/pharma-llm.git
git push -u origin main
```

## CI

GitHub Actions runs on pushes to `main` and on pull requests.

The CI job uses `uv` and checks:

- `uv sync --extra dev --locked`
- `uv run ruff check .`
- `uv run pytest`

## Artifact Policy

Do not push model weights, adapters, checkpoints, internal documents, secrets, raw data, or large experiment outputs.

Lightweight Markdown reports may be tracked under `results/reports/` when they are safe to share.
