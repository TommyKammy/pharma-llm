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

Deterministic local baseline summaries may also be written under the ignored
default path:

```text
results/baseline/
```

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

## Real Qwen Baseline Plan

Before Phase 6 LoRA training, prepare a real Qwen base baseline run plan:

```bash
uv run python scripts/run_qwen_baseline.py \
  --dry-run \
  --input evals/prompts/phase4_seed.jsonl \
  --local-root /Users/tsinfra/Dev/pharma-llm/local/runs/baseline \
  --model-path /Users/tsinfra/Dev/pharma-llm/local/models/qwen3.6-27b-base \
  --run-id phase6-qwen-base \
  --write-plan /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/plan.json
```

The dry-run validates accepted Phase 4 eval records, records the eval count and
ordered eval-id fingerprint, and fixes the local-only output paths:

```text
/Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/qwen_base_predictions.jsonl
/Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/summary.json
/Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/category_metrics.csv
```

The prediction JSONL is intentionally local-only. Use the same eval-id
fingerprint when comparing base and LoRA outputs; reports must not compare
different prompt subsets.

After confirming the local Qwen model path exists, generate the real local base
prediction JSONL with:

```bash
uv run --extra training python scripts/run_real_mlx_eval.py \
  --input evals/prompts/phase4_seed.jsonl \
  --output /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/qwen_base_predictions.jsonl \
  --run-id phase6-qwen-base \
  --model-id qwen/qwen3.6-27b-base \
  --model-path /Users/tsinfra/Dev/pharma-llm/local/models/qwen3.6-27b-base \
  --max-tokens 512
```

The runner invokes `mlx_lm.generate` locally and writes prediction records that
reuse the Phase 5 result schema. It is not used by CI and must not commit the
generated JSONL.

After the real prediction JSONL exists, run:

```bash
uv run python scripts/summarize_baseline_results.py \
  --input /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/qwen_base_predictions.jsonl \
  --summary-output /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/summary.json \
  --category-csv-output /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/category_metrics.csv

uv run python scripts/generate_baseline_report.py \
  --input /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/phase6-qwen-base/qwen_base_predictions.jsonl \
  --output results/reports/qwen_base_baseline_report.md \
  --mock-notice "Real Qwen base baseline run. Interpret quality only after confirming this uses the approved Phase 4 eval id set and local model path recorded in the run plan."
```

`scripts/run_qwen_baseline.py` does not download weights or execute the real
model in CI. It is the reproducible operator plan for the local Qwen baseline
artifact capture; real execution must write prediction records that satisfy the
Phase 5 result schema.

## Result Persistence

Validate and aggregate baseline prediction JSONL with:

```bash
uv run python scripts/summarize_baseline_results.py \
  --input /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/mock_predictions.jsonl \
  --summary-output /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/summary.json \
  --category-csv-output /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/category_metrics.csv
```

The summarizer checks required identifiers and timing values before writing
outputs. Required fields include `run_id`, `eval_id`, `category`, model id,
provider, generated text, and finite non-negative total latency. Empty
generated text is preserved as a valid baseline outcome. Optional TTFT and
tokens/sec values are aggregated when present.

The summary JSON includes the run id, model id, total record count, scoring
status counts, and per-category metrics. Provider and adapter identity are also
included so local MLX and endpoint-compatible runs cannot be silently blended.
The category CSV contains one row per Phase 4 category represented in the input,
with counts and average latency, TTFT, tokens/sec, and scoring status counts.

## Baseline Report

Generate a Markdown report from one or more baseline prediction JSONL files:

```bash
uv run python scripts/generate_baseline_report.py \
  --input \
    /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/qwen_predictions.jsonl \
    /Users/tsinfra/Dev/pharma-llm/local/runs/baseline/gemma_predictions.jsonl \
  --output results/reports/baseline_report.md
```

The report includes:

- model comparison by model id, provider, adapter, run id, eval count, and scoring status
- per-category counts and average latency, TTFT, and tokens/sec
- notable failure modes such as empty completions and non-default scoring statuses
- an Obsidian copy block for Phase notes
- limitations that distinguish CI-safe mock data from real baseline results

When multiple prediction files are supplied, each file must cover the same
`eval_id` set. This prevents side-by-side Qwen/Gemma reports from comparing
different prompt subsets. The default `results/reports/baseline_report.md`
output is intentionally trackable as a lightweight Markdown report.

Interpret the report as the Phase 6 pre-LoRA reference only when it was produced
from live baseline runs over the same eval ids. CI fixture reports validate the
workflow shape, not model quality.

## Promptfoo Mock Config

The promptfoo smoke config lives at:

```text
configs/promptfoo/baseline_mock.yaml
```

It compares Qwen, Gemma, and optional endpoint labels through the existing mock
provider. This checks promptfoo wiring without downloading models or calling
external endpoints.
