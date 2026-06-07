# Codex Instructions for pharma-llm

## Project Role

This repository is an experimental foundation for Japanese pharmaceutical LLM fine-tuning and evaluation. Codex app is the primary implementation agent for repository code, scripts, tests, configs, and Markdown reports.

## Local Layout

```text
/Users/tsinfra/Dev/pharma-llm/
  pharma-llm-main/        # local main
  pharma-llm-worktrees/   # issue-scoped Codex app worktrees
```

Use `pharma-llm-main` as the stable main checkout. Use worktrees under `pharma-llm-worktrees` for implementation tasks when practical.

## Safety and Data Policy

- Do not commit model weights, secrets, internal documents, raw data, or large generated artifacts.
- Do not turn raw AI output into training targets.
- Require human review or human editing before AI-assisted content can enter training data.
- Keep eval-only data out of training data.
- Preserve provenance metadata and review state on training records.

## Engineering Defaults

- Prefer small, issue-sized changes.
- Prefer config-driven experiment scripts.
- Keep tests close to behavior.
- Record experiment outputs in structured files and summarize them in Markdown.
- Treat DeepSeek V4 Flash as an optional baseline, not an initial dependency.

