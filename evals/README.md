# Evaluation Set v0

Evaluation artifacts are not training data. They are synthetic, versioned inputs
for baseline and LoRA comparisons.

Phase 4 stores the contract in:

```text
evals/prompts/
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
