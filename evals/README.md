# Evaluation Set v0

Evaluation artifacts are not training data. They are synthetic, versioned inputs
for baseline and LoRA comparisons.

Phase 4 stores the contract in:

```text
evals/prompts/
evals/manifest/evaluation_set_v0.json
evals/expected/scoring_rubrics.yaml
```

All accepted evaluation records must use:

- `dataset_type: eval`
- `provenance.source_type: eval_only`
- one Phase 4 category
- non-empty `expected_points`
- provenance `risk_flags` when safety, medical advice, GxP, audit, or refusal
  behavior is part of the scoring expectation

AI-generated candidates are not final evaluation records until reviewed.

Use the expansion manifest to track current accepted coverage against the
300-record target:

```bash
uv run python scripts/plan_eval_expansion.py
```

Use the helper to write synthetic review candidates. Candidate records use
`candidate_status: review_candidate` and `review_status: unreviewed`; they must
be manually promoted into `evals/prompts/*.jsonl` before they count as accepted
Evaluation Set v0 records.

```bash
uv run python scripts/plan_eval_expansion.py \
  --per-category 5 \
  --write-candidates evals/candidates/phase4_batch_001.jsonl
```
