# Argilla Workflow

## Purpose

Use Argilla as the human review gate for SFT, DPO, CPT, and eval candidate data.
Phase 3 keeps this workflow offline-safe first: CI validates the local schema and
synthetic sample payloads without requiring a live Argilla server.

Only records that have completed human review may be promoted into prepared
training datasets. Raw AI output, unreviewed AI candidates, and eval-only records
remain non-training material.

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

## Project Naming

Use a dedicated local/private workspace for this repository:

```text
workspace: pharma-llm-local-review
dataset prefix: pharma_llm_phase3
dataset pattern: pharma_llm_phase3_phase{phase}_{dataset_type}_review
```

Initial Phase 3 datasets:

|Dataset|Purpose|
|---|---|
|`pharma_llm_phase3_phase3_sft_review`|SFT candidate review|
|`pharma_llm_phase3_phase3_dpo_review`|DPO preference candidate review|
|`pharma_llm_phase3_phase3_cpt_review`|CPT corpus candidate review|
|`pharma_llm_phase3_phase3_eval_review`|Evaluation-only item review|

## Review Schema

Each review payload preserves the Phase 2 dataset fields and adds an `argilla`
section that identifies the target workspace, review dataset, visible fields,
and questions. The canonical provenance remains under `provenance`.

Required provenance fields from Phase 2:

|Field|Purpose|
|---|---|
|`source_type`|Structured source classification|
|`source_document`|Synthetic or source document identifier|
|`source_license`|Must be `synthetic_test_only` for local smoke samples|
|`review_status`|Human review state|
|`ai_assisted`|Whether an AI tool helped create the candidate|
|`ai_tool`|Required when `ai_assisted=true`|
|`raw_ai_output_used_as_training_target`|Must remain false for promotable data|
|`human_reviewer`|Reviewer identifier after approval or edit approval|
|`review_date`|Review date after approval or edit approval|
|`risk_flags`|Structured risk tags retained for audit and later filtering|

Argilla-visible questions:

|Question|Allowed values or shape|
|---|---|
|`review_status`|`approved`, `rejected`, `needs_edit`, `edited_and_approved`, `risk_flagged`|
|`risk_flags`|List of non-empty strings|
|`review_notes`|Optional free-text reviewer notes|

## Review Status Mapping

|Review status|Training eligible?|Use|
|---|---:|---|
|`approved`|Yes|Human-authored or document-derived data approved as-is|
|`edited_and_approved`|Yes|AI-assisted or human-edited data approved after human edits|
|`needs_edit`|No|Candidate requires more editing before training use|
|`rejected`|No|Candidate must not be promoted|
|`risk_flagged`|No|Candidate requires separate risk handling before reuse|
|`unreviewed`|No|Phase 2 provenance state for raw/unreviewed data; not an Argilla final label|

## Source Type Boundaries

|Source type|Training eligible after review?|Boundary|
|---|---:|---|
|`human_authored`|Yes, if approved|Human-created synthetic or approved source-derived record|
|`human_edited_ai_assisted`|Yes, if edited and approved|AI helped draft, but a human edited and approved the final target|
|`public_doc_derived`|Yes, if approved|Public source-derived content, subject to provenance and license checks|
|`internal_doc_derived`|Yes, if approved in a private workflow|Never use real internal content in this repo or public CI|
|`ai_candidate_unreviewed`|No|Review candidate only; cannot be training data|
|`raw_ai_output`|No|Never use raw AI output directly as a training target|
|`eval_only`|No for training|May be reviewed as evaluation data, but must remain separate from training|

## Synthetic Smoke Samples

`scripts/create_argilla_sample.py` produces only synthetic examples and marks each
record with:

```text
provenance.source_document = synthetic_argilla_phase3_sample
provenance.source_license = synthetic_test_only
```

The sample set covers:

- approved human-authored SFT data
- unreviewed AI candidate data requiring edits
- edited-and-approved human-edited AI-assisted DPO data
- eval-only data that must not be promoted to training
- raw AI output that must never be used as a training target

## Verify SDK

```bash
make env-check
uv run python -c "import argilla; print('argilla sdk ok')"
make argilla-smoke
make argilla-server-smoke
```

`make argilla-smoke` creates an offline review sample under:

```text
/Users/tsinfra/Dev/pharma-llm/local/argilla/phase3_review_sample.jsonl
```

This does not require a running server. Server-backed registration is a separate check once a local or private Argilla instance is available.

## Offline Export / Import

Before wiring a live Argilla server, Phase 3 uses local JSONL payloads to verify
the review contract.

Export synthetic candidates into an Argilla-friendly review payload:

```bash
uv run python scripts/export_to_argilla.py \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/phase3_review_sample.jsonl \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/phase3_review_payload.jsonl
```

The exported payload contains:

- `fields`: values reviewers should inspect
- `provenance`: Phase 2 provenance metadata preserved as-is
- `review`: editable review fields
- `original_record`: the full source record used for round-trip import

After review, import the reviewed payload back into dataset JSONL:

```bash
uv run python scripts/import_from_argilla.py \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/phase3_review_payload.jsonl \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/phase3_reviewed_dataset.jsonl
```

Import copies reviewed `fields` back onto the dataset record, updates provenance
review metadata, and removes the local `argilla` helper section. When an
`ai_candidate_unreviewed` record is marked `edited_and_approved`, import changes
`source_type` to `human_edited_ai_assisted` and clears
`raw_ai_output_used_as_training_target` because the reviewed fields now replace
the candidate target. An unreviewed AI candidate cannot be marked `approved`
as-is; it must have at least one reviewed target field changed and then be
marked `edited_and_approved`. Existing risk flags are preserved from the
top-level exported provenance when review payloads omit `review.risk_flags`.
Any AI-assisted record must use `edited_and_approved`, not plain `approved`, so
the review state retains the human-edit audit signal. For SFT and DPO records,
prompt-only edits are not enough to satisfy this requirement because the training
target is `response`, `chosen`, or `rejected`. Raw AI output cannot be marked
`approved` or `edited_and_approved`; it must be recreated as a
human-edited candidate before approval. Import rejects invalid review statuses,
unsupported reviewed field keys, missing reviewer metadata on approved records,
empty risk flags for `risk_flagged` records, malformed risk flags, and malformed
payloads with clear CLI errors. It also rejects payload identity mismatches,
missing reviewed content fields, and mutations to immutable provenance fields
between the top-level exported provenance and embedded `original_record`.

`make argilla-server-smoke` checks `ARGILLA_API_URL` and `ARGILLA_API_KEY`. If no API key is set, it records a skipped result under:

```text
/Users/tsinfra/Dev/pharma-llm/local/argilla/argilla_server_smoke.json
```

## Promotion Rule

Only records with approved review state may be exported to training datasets. AI-assisted candidates remain non-training data until human review or human editing is complete.

Phase 3 includes the offline foundations:

```text
scripts/export_to_argilla.py
scripts/import_from_argilla.py
scripts/promote_reviewed_dataset.py
```

Phase 3 promotion code must reuse the Phase 2 validator before writing prepared
training outputs. Promoted records must be eligible under both:

- `review_status in {"approved", "edited_and_approved"}`
- Phase 2 training policy checks for `source_type`, `dataset_type`, and
  `raw_ai_output_used_as_training_target`

Promote reviewed records into prepared training JSONL:

```bash
uv run python scripts/promote_reviewed_dataset.py \
  --dataset-type sft \
  --audit-output /Users/tsinfra/Dev/pharma-llm/local/argilla/phase3_promotion_audit.json \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/phase3_reviewed_dataset.jsonl \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/prepared_sft.jsonl
```

Promotion writes only records that pass the Phase 2 validator for the requested
training dataset type. It removes local `argilla` helper metadata from prepared
outputs, skips policy-blocked records with explicit reasons, and fails without
writing output when the reviewed input contains malformed records or when no
records can be promoted. Promotion targets are limited to `sft`, `dpo`, and `cpt`;
`eval` remains reviewable but cannot be promoted into training data.

## Phase 6 SFT v0.1 Export

Phase 6 wraps the approved-only promotion path with a versioned SFT export for
Qwen LoRA training:

```bash
uv run python scripts/export_sft_v0_1.py \
  /Users/tsinfra/Dev/pharma-llm/local/argilla/phase6_reviewed_sft.jsonl \
  --output data/prepared/sft_v0_1.jsonl \
  --manifest data/prepared/sft_v0_1.manifest.json \
  --eval-path evals/prompts/phase4_seed.jsonl
```

The export reuses `scripts/promote_reviewed_dataset.py` and adds a manifest with:

- dataset version and dataset type
- source count and promoted / skipped / failed counts
- approved and edited-and-approved counts
- output SHA-256
- Phase 4 eval count and ordered eval-id fingerprint
- local artifact policy for raw exports and model artifacts

The prepared SFT JSONL shape for the training runner is:

```json
{"id":"...","dataset_type":"sft","prompt":"...","response":"...","provenance":{...}}
```

The export rejects review candidates, duplicate ids, unapproved records,
eval-only records, raw AI output, and other training-policy violations before
writing the manifest.
