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
line:

```json
{"id":"...","dataset_type":"sft","prompt":"...","response":"...","provenance":{...}}
```

Use `data/prepared/sft_v0_1.manifest.json` to audit the dataset version,
approved counts, output checksum, and Phase 4 eval-id fingerprint before
starting a LoRA run.

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
