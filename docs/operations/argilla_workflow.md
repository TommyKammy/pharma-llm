# Argilla Workflow

## Purpose

Use Argilla as the human review gate for SFT, DPO, and eval candidate data.

## Install SDK

```bash
uv sync --extra dev --extra review
```

## Environment

Set local connection values in `.env` or your shell:

```bash
export ARGILLA_API_URL=http://localhost:6900
export ARGILLA_API_KEY=...
```

## Server

Argilla requires a deployed UI/server before review workflows can run. For local experiments, prefer a self-hosted local service or a private hosted workspace. Do not upload internal documents or sensitive review data to public services.

The current host has Docker CLI available, but server-backed checks require a running Docker daemon or an existing private Argilla endpoint.

## Verify SDK

```bash
make env-check
uv run python -c "import argilla; print('argilla sdk ok')"
make argilla-smoke
make argilla-server-smoke
```

`make argilla-smoke` creates an offline review sample under:

```text
/Users/tsinfra/Dev/pharma-llm/local/argilla/phase1_review_sample.jsonl
```

This does not require a running server. Server-backed registration is a separate check once a local or private Argilla instance is available.

`make argilla-server-smoke` checks `ARGILLA_API_URL` and `ARGILLA_API_KEY`. If no API key is set, it records a skipped result under:

```text
/Users/tsinfra/Dev/pharma-llm/local/argilla/argilla_server_smoke.json
```

## Promotion Rule

Only records with approved review state may be exported to training datasets. AI-assisted candidates remain non-training data until human review or human editing is complete.

Later phases will add:

```text
scripts/export_to_argilla.py
scripts/import_from_argilla.py
```
