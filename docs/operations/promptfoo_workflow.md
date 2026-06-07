# promptfoo Workflow

## Purpose

Use promptfoo for model comparison, prompt evaluation, and safety-style regression checks.

## Install

promptfoo is Node-based. Use one of:

```bash
npx promptfoo@latest --version
npm install -g promptfoo
brew install promptfoo
```

The current promptfoo documentation requires Node.js `^20.20.0` or `>=22.22.0`.

## Verify

```bash
make env-check
npx promptfoo@latest --version
make promptfoo-smoke
```

If installed globally:

```bash
promptfoo --version
```

## Config Location

Keep promptfoo configs under:

```text
configs/promptfoo/
```

Phase 5 will add baseline comparison configs. Phase 6 and later will compare base vs LoRA outputs.

Phase 1 includes a local mock-provider config:

```text
configs/promptfoo/phase1_smoke.yaml
```

It uses Python providers only and does not call any external LLM API.

## Data Policy

Evaluation prompts are not training data. Do not promote promptfoo eval cases into training datasets without explicit review and provenance updates.
