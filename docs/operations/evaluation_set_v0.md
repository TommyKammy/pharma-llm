# Evaluation Set v0

## Purpose

Phase 4 creates a stable evaluation set before baseline and LoRA experiments.
Evaluation records measure model behavior; they are never training records.

## Categories

|Range|Category|Purpose|
|---|---|---|
|001-050|`business_summary`|Japanese business-document summarization|
|051-100|`package_insert_reading`|Package-insert-style reading comprehension|
|101-150|`safety_information`|Safety information explanation|
|151-200|`gxp_qa_audit`|GxP, QA, and audit context|
|201-250|`di_inquiry`|Drug information inquiry style|
|251-300|`unsafe_refusal`|Unsafe-answer inducement and refusal tests|

## Seed Evaluation Set

The initial seed file lives at:

```text
evals/prompts/phase4_seed.jsonl
```

It contains 30 synthetic, non-confidential records: 5 records for each category.
Every record uses `dataset_type: eval`, `provenance.source_type: eval_only`,
and at least 3 `expected_points`. Safety, GxP, DI, and refusal prompts carry
`risk_flags` so reviewer attention is visible in the data itself.

## Expansion Manifest and Candidate Workflow

The Evaluation Set v0 manifest lives at:

```text
evals/manifest/evaluation_set_v0.json
```

It records the 300-case target, accepted prompt files, deterministic category ID
ranges, and current accepted coverage. Report current coverage with:

```bash
uv run python scripts/plan_eval_expansion.py
```

To prepare the next synthetic review batch:

```bash
uv run python scripts/plan_eval_expansion.py \
  --per-category 5 \
  --write-candidates evals/candidates/phase4_batch_001.jsonl
```

Candidate records are not final evaluation data. They carry
`candidate_status: review_candidate`, `review_status: unreviewed`, and
`source_document: synthetic_phase4_candidate`. Codex app may draft candidates,
Claude Code reviews category fit, scoring points, and safety boundaries, and
Antigravity checks the workflow before a human reviewer promotes accepted
records into `evals/prompts/*.jsonl`.

## Eval JSONL Contract

Accepted evaluation records live under `evals/prompts/` and use:

- `dataset_type: eval`
- `category`: one of the six Phase 4 categories
- `prompt`: synthetic evaluation prompt
- `expected_points`: non-empty scoring expectations
- `provenance.source_type: eval_only`
- provenance `risk_flags` when safety or regulated-context behavior matters

The existing Phase 2 validator parses `eval` records and keeps `eval_only`
records blocked from training validation.

## Scoring Rubric Contract

The rubric lives at:

```text
evals/expected/scoring_rubrics.yaml
```

For dependency-light CI, this file is JSON-compatible YAML and can be validated
with the standard library. It must include:

- all six Phase 4 categories
- `safety`
- `pharma_style`
- `factuality`

## Data Boundary

Evaluation prompts may include unsafe requests as tests, but expected behavior
must not provide medical advice, compliance decisions, production approvals, or
patient-specific recommendations. Evaluation records must not be promoted into
SFT, DPO, or CPT prepared datasets.

## Training Separation Guard

Before baseline or LoRA work uses prepared training JSONL, run the leakage guard
against accepted eval prompts and prepared training inputs:

```bash
uv run python scripts/check_eval_leakage.py \
  --eval evals/prompts/phase4_seed.jsonl \
  --training data/prepared/example_sft.jsonl
```

The guard fails on:

- exact ID reuse between eval records and prepared training records
- exact prompt/text reuse after whitespace normalization
- non-eval records passed through the eval side
- eval records passed through the prepared training side

The Phase 3 promotion path also reuses the dataset validator, so `eval_only`
records are skipped instead of being written into SFT, DPO, or CPT outputs. These
checks intentionally prefer false positives over silent evaluation leakage.
