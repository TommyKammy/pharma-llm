# MLX LoRA Workflow

## Purpose

Use MLX LM for Apple Silicon inference and LoRA/fine-tuning experiments.

## Install

```bash
uv sync --extra dev --extra training
```

MLX LM is installed through the Python package `mlx-lm`.

## Verify

```bash
make env-check
uv run python -c "import mlx_lm; print('mlx-lm ok')"
make mlx-smoke
make mlx-generate-smoke
make mlx-lora-smoke
```

## Local Artifact Policy

Keep model weights, adapters, checkpoints, and training logs outside Git:

```text
/Users/tsinfra/Dev/pharma-llm/local/models
/Users/tsinfra/Dev/pharma-llm/local/adapters
/Users/tsinfra/Dev/pharma-llm/local/runs
```

## Phase 6 Training Dataset Input

Use the approved-only SFT v0.1 export as the initial Qwen LoRA training input:

```bash
uv run python scripts/export_sft_v0_1.py \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/phase6_reviewed_sft.jsonl \
  --output data/prepared/sft_v0_1.jsonl \
  --manifest data/prepared/sft_v0_1.manifest.json
```

The runner-facing JSONL shape is deterministic and contains one SFT record per
line. `response` preserves the project SFT schema, and `completion` mirrors it
for the default MLX LM completions format:

```json
{"id":"...","dataset_type":"sft","prompt":"...","response":"...","completion":"...","provenance":{...}}
```

Use `data/prepared/sft_v0_1.manifest.json` to audit the dataset version,
approved counts, output checksum, and Phase 4 eval-id fingerprint before
starting a LoRA run.

## Phase 6 Qwen LoRA Dry Run

Use the Phase 6 config and runner to validate the local paths and planned
`mlx_lm.lora` command before starting any 27B-class training job:

```bash
uv run python scripts/run_mlx_lora.py \
  --config configs/mlx/qwen_sft_lora_r16.toml \
  --dry-run \
  --write-plan /Users/tsinfra/Dev/pharma-llm/local/runs/qwen_sft_lora_r16_v1/run_plan.json
```

The dry run requires the prepared SFT JSONL to exist, allows the Qwen model
path to be absent for CI and pre-download checks, and refuses adapter/run output
destinations outside `/Users/tsinfra/Dev/pharma-llm/local`. It materializes the
MLX-facing local inputs under the configured run directory: the approved SFT
JSONL is copied to `mlx_data/train.jsonl`, and the reviewed LoRA settings are
written to `mlx_lora_config.yaml` with `lora_parameters.rank`, `scale`,
`dropout`, and `keys`. The Qwen attention keys use MLX module paths such as
`self_attn.q_proj`; unqualified projection names are not accepted as the Phase 6
recipe. `mask_prompt: true` is emitted for completion-style SFT so the prompt
tokens are not optimized as answer text.

Any stale `valid.jsonl` or `test.jsonl` split files in the local MLX data
directory are removed before writing the current `train.jsonl`. The planned
command points at that local YAML config, so the validated settings and the
eventual MLX invocation stay aligned. Training length is controlled by `iters`
in this runner contract. The JSON plan is written to the configured
`output.run_output_path`; if `--write-plan` is supplied, it must equal that path
and acts as an explicit assertion rather than an alternate output location.
The runner rejects path collisions across the JSON plan, generated YAML config,
MLX split files, and adapter destination before materializing any local inputs.
It also rejects unqualified LoRA target names; Phase 6 Qwen dry runs are limited
to the reviewed `self_attn.*_proj` key allowlist.
MLX-compatible numeric settings are preserved, including fractional
`lora_parameters.scale`, `num_layers = -1` for all layers, and `seed = 0`.
Real MLX LoRA execution is intentionally deferred to P6-004 after this command
contract is reviewed.

## Phase 6 Qwen LoRA Execution Metadata

P6-004 records small adapter metadata without committing adapter weights,
checkpoints, or raw logs. Use the dry-run plan first, then execute the generated
MLX command locally after confirming model weights and the approved SFT dataset
are present.

Operator checklist:

1. Run the P6-003 dry run and inspect `run_plan.json`,
   `mlx_lora_config.yaml`, and `mlx_data/train.jsonl`.
2. Confirm the Qwen model path and approved SFT dataset version are correct.
3. Execute the planned command locally:

```bash
mlx_lm.lora \
  --config /Users/tsinfra/Dev/pharma-llm/local/runs/qwen_sft_lora_r16_v1/mlx_lora_config.yaml
```

4. Keep adapter files under `/Users/tsinfra/Dev/pharma-llm/local/adapters`.
5. Record metadata after the run:

```bash
uv run python scripts/record_lora_adapter_metadata.py \
  --run-plan /Users/tsinfra/Dev/pharma-llm/local/runs/qwen_sft_lora_r16_v1/run_plan.json \
  --output /Users/tsinfra/Dev/pharma-llm/local/runs/qwen_sft_lora_r16_v1/adapter_metadata.json \
  --status executed \
  --started-at 2026-06-10T01:00:00Z \
  --ended-at 2026-06-10T03:00:00Z \
  --status-note "Local Qwen SFT LoRA v1 completed; adapter directory exists."
```

If training cannot be executed yet, record a `planned` placeholder instead of
pretending the adapter exists. The metadata validator rejects `executed` records
that still look like dry-run placeholders or point outside the local artifact
root.

Re-run behavior:

- Re-run `scripts/run_mlx_lora.py --dry-run` after changing config or dataset.
- Re-record adapter metadata after every completed training attempt.
- Use the metadata `config.source_sha256`, `config.generated_sha256`, and
  `dataset.training_input.sha256` fields to identify which config and
  materialized MLX input produced an adapter. Treat `dataset.sha256` as the
  source SFT JSONL digest, not as proof of the exact training split when local
  inputs may have been regenerated.
- Do not commit local `adapter_metadata.json`; it lives beside the local run
  plan and adapter artifacts for operator audit.

## Phase 6 Base vs LoRA Evaluation Report

P6-005 compares the Qwen base prediction JSONL and the Qwen SFT LoRA v1
prediction JSONL over the same Phase 4 eval IDs. CI can exercise the wiring with
mock predictions:

```bash
uv run python scripts/run_baseline_eval.py \
  --output results/baseline/qwen_base_predictions.jsonl \
  --model-label qwen-base \
  --run-id phase6-base-mock

uv run python scripts/run_lora_eval.py \
  --output results/lora/qwen_sft_lora_v1_predictions.jsonl \
  --adapter-id qwen_sft_lora_r16_v1 \
  --run-id phase6-lora-mock
```

For a reviewed local adapter, generate the LoRA prediction JSONL with the
recorded adapter metadata so the model and adapter identity stay aligned:

```bash
uv run python scripts/run_lora_eval.py \
  --adapter-metadata /Users/tsinfra/Dev/pharma-llm/local/runs/qwen_sft_lora_r16_v1/adapter_metadata.json \
  --output /Users/tsinfra/Dev/pharma-llm/local/runs/qwen_sft_lora_r16_v1/lora_predictions.jsonl
```

Then produce the Markdown comparison report:

```bash
uv run python scripts/generate_lora_comparison_report.py \
  --base-input results/baseline/qwen_base_predictions.jsonl \
  --lora-input results/lora/qwen_sft_lora_v1_predictions.jsonl \
  --adapter-metadata /Users/tsinfra/Dev/pharma-llm/local/runs/qwen_sft_lora_r16_v1/adapter_metadata.json \
  --output results/reports/lora_comparison_report.md
```

The report rejects mismatched eval ID sets, requires base predictions to have no
adapter ID, requires LoRA predictions to include an adapter ID, and surfaces
safety/style notes even when scoring remains manual-review-only.

## Initial Smoke Tests

Phase 1 should start with a small model before any 27B-class experiment.

Suggested order:

1. Verify MLX LM imports.
2. Run `make mlx-smoke` to verify the MLX runtime without downloading a model.
3. Run `make mlx-generate-smoke` for a tiny real-model inference test.
4. Run `make mlx-lora-smoke` for a one-iteration tiny LoRA training test.
5. Record runtime, memory behavior, and output location.
6. Only then move to Qwen3.6-27B and Gemma 4 26B A4B checks.

## Notes

The first Qwen and Gemma runs should be read-only inference checks. Training configs are added in later phases.

## Phase 1 Tiny Model

Phase 1 uses `mlx-community/SmolLM-135M-Instruct-4bit` for smoke checks. This is only a toolchain test model and is not a project target model.
