# Evaluation Prompt JSONL Contract

Each `.jsonl` record under this directory must follow the Phase 2 eval record
schema plus the Phase 4 category contract:

```json
{
  "id": "eval_001",
  "dataset_type": "eval",
  "category": "business_summary",
  "prompt": "Synthetic evaluation prompt.",
  "expected_points": ["Point the answer should satisfy."],
  "provenance": {
    "source_type": "eval_only",
    "source_document": "synthetic_phase4_seed",
    "source_license": "synthetic_test_only",
    "review_status": "approved",
    "ai_assisted": false,
    "ai_tool": null,
    "raw_ai_output_used_as_training_target": false,
    "human_reviewer": "reviewer_id",
    "review_date": "2026-06-08",
    "risk_flags": []
  }
}
```

Allowed categories:

- `business_summary`
- `package_insert_reading`
- `safety_information`
- `gxp_qa_audit`
- `di_inquiry`
- `unsafe_refusal`

## Seed Set

`phase4_seed.jsonl` contains 30 synthetic, non-confidential evaluation records:

- 5 records for each allowed category
- `dataset_type: eval`
- `provenance.source_type: eval_only`
- at least 3 `expected_points` per record
- `risk_flags` on regulated or safety-sensitive prompts
