# ADR-004: Evaluation Policy

## Status

Accepted

## Context

Fine-tuning experiments are only meaningful if evaluation data remains separate and repeatable.

## Decision

Maintain dedicated evaluation datasets for:

- Japanese business document summarization
- Package-insert-like document reading
- Safety information explanation
- GxP / QA / audit contexts
- Drug information inquiry style
- Dangerous-answer induction and refusal tests

Evaluation data must not be mixed into training datasets. Evaluation results should be saved in structured formats and summarized in Markdown.

## Consequences

Dataset validation will need leakage checks. Experiment reports will compare base and tuned models using stable evaluation sets.

