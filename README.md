# pharma-llm

日本語・製薬業特化LLMのLoRA/Fine-tuning実験基盤。

Japanese pharma-domain LLM experimentation lab using MLX LM, Argilla, promptfoo, DeepEval, and Codex app.

This repository is an experimental foundation for learning and comparing fine-tuning approaches for Japanese pharmaceutical workflows. The goal is not to ship a production chatbot. The goal is to build reproducible knowledge around LoRA, SFT, DPO/ORPO, CPT-LoRA, data review, and evaluation.

## Current Host Layout

For the initial phases, one Mac Studio is used as the combined Mac Studio A+B host.

```text
/Users/tsinfra/Dev/pharma-llm/
  pharma-llm-main/        # local main
  pharma-llm-worktrees/   # Codex app / issue worktree area
```

Use `pharma-llm-main` as the stable local main checkout. Use `pharma-llm-worktrees` for issue-scoped Codex app worktrees and experiments.

If this project later moves to a two-machine setup, keep the same Git and worktree convention. Mac Studio A can become the training host, and Mac Studio B can become the development, evaluation, and review host.

## What This Project Does

- MLX LM based LoRA / fine-tuning on Apple Silicon
- Qwen and Gemma model experiments for Japanese pharma workflows
- SFT for safer business-answer style and response structure
- DPO / ORPO experiments for dangerous-answer reduction
- CPT-LoRA experiments for Japanese pharma document adaptation
- Argilla-based human review before data promotion
- promptfoo and DeepEval based automated evaluation
- Markdown reports that can be read from Obsidian

## What This Project Does Not Do

- It does not build a production RAG application.
- It does not provide patient-facing medical advice.
- It does not automate medical judgment.
- It does not train on unreviewed AI output.
- It does not put model weights, internal documents, secrets, or large raw data in Git.

## Data Policy

Training data must be human-authored, human-edited, or human-approved. Raw AI output from Codex app, Claude Code, Google Antigravity, or other tools must not be used directly as a training target.

Every training record should carry provenance metadata and review state. Evaluation data must remain separate from training data.

## Repository Layout

```text
configs/        Training and evaluation configs
data/           Dataset README plus non-tracked local data areas
scripts/        CLI scripts for validation, training, evaluation, reporting
src/            Python package code
evals/          Evaluation prompts and scoring rubrics
results/        Local experiment outputs and report index
docs/           ADRs and operations notes
codex/          Codex app instructions, skills, and task templates
tests/          Unit tests
```

## First Commands

```bash
uv sync
uv run pytest
```

Phase 0 intentionally contains scaffolding and documentation first. Implementation-heavy code starts in later phases.
