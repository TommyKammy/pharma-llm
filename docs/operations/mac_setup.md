# Mac Setup

Initial setup targets one Mac Studio used as the combined A+B host.

```text
/Users/tsinfra/Dev/pharma-llm/
  pharma-llm-main/
  pharma-llm-worktrees/
```

Phase 1 will fill in concrete MLX LM, Argilla, promptfoo, and DeepEval setup commands.

## Current Host Role

This host is treated as both:

- Mac Studio A: training and adapter generation
- Mac Studio B: development, evaluation, review, reporting

The repository remains in:

```text
/Users/tsinfra/Dev/pharma-llm/pharma-llm-main
```

Issue-scoped worktrees should live under:

```text
/Users/tsinfra/Dev/pharma-llm/pharma-llm-worktrees
```

## Phase 1 Environment Check

Run:

```bash
make env-check
make phase1-smoke
```

The check verifies:

- macOS / Apple Silicon host
- `uv`
- `git`
- Node.js and npm for promptfoo
- optional Python packages for MLX LM, Argilla, and DeepEval
- optional `promptfoo` CLI

Optional packages are warnings until the relevant sub-workflow is being executed.

`make phase1-smoke` runs local smoke checks for MLX, Argilla sample generation, promptfoo mock-provider evaluation, and DeepEval exact-match evaluation.

## Recommended Local Artifact Roots

Keep large artifacts outside Git:

```text
/Users/tsinfra/Dev/pharma-llm/local/models
/Users/tsinfra/Dev/pharma-llm/local/adapters
/Users/tsinfra/Dev/pharma-llm/local/runs
```

Create them when training or evaluation runs begin:

```bash
mkdir -p /Users/tsinfra/Dev/pharma-llm/local/{models,adapters,runs,argilla}
```

## Install Optional Python Tooling

Install all Phase 1 Python-side tools:

```bash
uv sync --extra dev --extra phase1
```

Or install by workflow:

```bash
uv sync --extra dev --extra training
uv sync --extra dev --extra review
uv sync --extra dev --extra eval
```

`promptfoo` is Node-based and is handled separately in `promptfoo_workflow.md`.
