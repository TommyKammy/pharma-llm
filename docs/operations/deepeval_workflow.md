# DeepEval Workflow

## Purpose

Use DeepEval for Python-based custom metrics and pytest-style LLM evaluation tests.

## Install

```bash
uv sync --extra dev --extra eval
```

DeepEval is installed through the Python package `deepeval`.

## Verify

```bash
make env-check
uv run python -c "import deepeval; print('deepeval ok')"
make deepeval-smoke
```

The Phase 1 smoke test uses `ExactMatchMetric`, so it does not require an LLM judge or provider API key.

## Config Location

Keep DeepEval configs under:

```text
configs/deepeval/
```

Custom metrics should live under:

```text
src/pharma_llm_lab/eval/
```

## Local-First Evaluation

Prefer local model endpoints or explicitly configured evaluation providers. Keep provider secrets in `.env` or host-level secret storage, never in Git.
