# ADR-002: Data Policy

## Status

Accepted

## Context

Pharmaceutical LLM experiments require strict tracking of source, review state, and permitted use. AI-assisted data creation is useful, but raw AI output must not silently become a training target.

## Decision

Training data must be one of:

- Human-authored
- Human-edited AI-assisted
- Approved public-document-derived
- Approved internal-document-derived
- Argilla-approved candidate data

Training data must include provenance metadata and review status. Evaluation data must remain separated from training data.

## Consequences

Dataset validators will reject unreviewed AI candidates, raw AI output, and eval-only records in training datasets. Argilla import/export workflow will become a required promotion path for AI-assisted candidates.

