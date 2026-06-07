# pharma-llm Roadmap

The canonical planning source is the Obsidian roadmap:

```text
/Users/tsinfra/Library/Mobile Documents/com~apple~CloudDocs/ObsidianVault/Dev/pharma-llm/Codex app中心の日本語・製薬業特化LLM実験ロードマップ.md
```

Phase-specific implementation plans and development history are tracked in:

```text
/Users/tsinfra/Library/Mobile Documents/com~apple~CloudDocs/ObsidianVault/Dev/pharma-llm/Phase/
```

## Phase Summary

|Phase|Name|Purpose|
|---:|---|---|
|0|Project Bootstrap|Create the safe repository foundation|
|1|Mac experiment environment|Prepare the combined A+B host for MLX, Argilla, promptfoo, and DeepEval|
|2|Dataset Schema and Validator|Define dataset schemas and pre-training validation|
|3|Argilla Review Workflow|Promote only reviewed data to training datasets|
|4|Evaluation Set v0|Create a separated evaluation dataset|
|5|Baseline Evaluation|Measure base model performance|
|6|Qwen3.6-27B SFT LoRA v1|Create the first Qwen SFT LoRA|
|7|LoRA Sweep|Compare rank, target module, and dataset size|
|8|Gemma 4 26B A4B comparison|Compare Gemma and Qwen behavior|
|9|DPO / ORPO|Reduce dangerous answers with preference tuning|
|10|CPT-LoRA|Test pharma-document style adaptation|
|11|Codex Skills / Agent Instructions|Make Codex work reproducible and policy-aware|
|12|Automated experiment pipeline|Reproduce validation, training, evaluation, and reporting by command|

## Phase 0 Completion Criteria

- Codex app can understand the repository purpose.
- `pharma-llm-main` and `pharma-llm-worktrees` usage is documented.
- Model weights, internal documents, secrets, and large data are not tracked.
- Later work can be requested as issue-sized implementation tasks.

