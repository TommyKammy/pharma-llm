# ADR-005: Model Selection

## Status

Accepted

## Context

The project needs a primary model for LoRA, SFT, DPO/ORPO, and CPT-LoRA experiments, plus at least one comparison target.

## Decision

Use Qwen3.6-27B as the primary learning target. Use Gemma 4 26B A4B as the comparison target. Treat DeepSeek V4 Flash as an optional baseline rather than an initial dependency.

## Consequences

Initial configs and reports should focus on Qwen and Gemma. DeepSeek integrations can be added later only when they help baseline comparison or teacher-candidate evaluation.

