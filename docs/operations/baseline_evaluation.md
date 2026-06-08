# Baseline Evaluation Workflow

## Purpose

Phase 5 measures base model behavior before LoRA experiments. The first step is
a stable inference contract that can be tested without downloading Qwen or Gemma
weights in CI.

## Inference Contract

Baseline runners should use:

```text
src/pharma_llm_lab/inference/
```

The contract records:

- request id and prompt
- model id, provider, and optional adapter id
- generated text
- finish reason
- total latency, TTFT, tokens/sec, prompt tokens, and completion tokens
- optional raw output path for local-only artifacts

`MockMlxInferenceClient` is CI-safe and deterministic. Real MLX execution should
implement the same `MlxInferenceClient` protocol and return the same
`InferenceResponse` shape.

## Local Artifact Policy

Keep model weights, adapters, raw generations, and large run outputs outside Git:

```text
/Users/tsinfra/Dev/pharma-llm/local/models
/Users/tsinfra/Dev/pharma-llm/local/adapters
/Users/tsinfra/Dev/pharma-llm/local/runs
```

Small summarized reports may live under:

```text
results/reports/
```

Raw baseline prediction JSONL should stay in ignored local paths unless a small
synthetic fixture is intentionally added for tests.

## Real MLX Plug-in Point

Later Phase 5 work should add a real MLX client that:

1. loads a model from `local/models` or an explicit external path,
2. optionally loads an adapter from `local/adapters`,
3. generates from Phase 4 eval prompts,
4. writes raw outputs under `local/runs`,
5. emits only schema-checked summaries into tracked reports.

## Mock Baseline Runner

Use the CI-safe runner to validate the baseline output shape before real model
execution:

```bash
uv run python scripts/run_baseline_eval.py \
  --input evals/prompts/phase4_seed.jsonl \
  --output /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/mock_predictions.jsonl \
  --model-label qwen-base \
  --run-id phase5-baseline-mock
```

The runner emits JSONL records with:

- `run_id`
- `eval_id`
- `category`
- `prompt`
- `expected_points`
- model identity
- generated text
- timing metadata
- finish reason

Supported CI-safe labels are `qwen-base`, `gemma-base`, and
`endpoint-optional`. They are mock identities only; real model quality must not
be inferred from these outputs.

## Promptfoo Mock Config

The promptfoo smoke config lives at:

```text
configs/promptfoo/baseline_mock.yaml
```

It compares Qwen, Gemma, and optional endpoint labels through the existing mock
provider. This checks promptfoo wiring without downloading models or calling
external endpoints.
