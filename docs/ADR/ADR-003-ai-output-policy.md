# ADR-003: AI Output Policy

## Status

Accepted

## Context

Codex app, Claude Code, Google Antigravity, and other AI tools may help draft examples, rubrics, scripts, reports, and review candidates. Their raw outputs can be useful, but using raw AI output directly as training target creates provenance and quality risks.

## Decision

Raw AI output must not be used directly as a training target.

AI-assisted content may enter training datasets only after human review, human editing, and explicit approval. Records must preserve whether AI assistance was used and which tool was involved.

## Consequences

The repository will include validators and review workflow scripts that distinguish unreviewed AI candidates from approved training records.

