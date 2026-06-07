# ADR-001: Toolchain

## Status

Accepted

## Context

The project needs a reproducible Apple Silicon LLM experimentation stack for Japanese pharmaceutical workflows.

## Decision

Use the following primary tools:

- Codex app for repository implementation
- MLX LM for Apple Silicon inference and LoRA/fine-tuning experiments
- Argilla for human review workflow
- promptfoo for prompt and model comparison
- DeepEval for Python-based custom evaluation metrics
- Obsidian for planning, ADRs, experiment history, and reports
- Git for code, configuration, schema, and lightweight report history

Claude Code and Google Antigravity are secondary review and workflow validation tools.

## Consequences

The repository will favor config-driven scripts, Markdown reports, and explicit data provenance. Large artifacts and sensitive data remain outside Git.

