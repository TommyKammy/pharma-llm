.PHONY: help test lint env-check mlx-smoke argilla-smoke promptfoo-smoke deepeval-smoke phase1-smoke validate-data

help:
	@echo "pharma-llm commands"
	@echo "  make test              Run unit tests"
	@echo "  make lint              Run ruff checks"
	@echo "  make env-check         Inspect local Phase 1 toolchain state"
	@echo "  make mlx-smoke         Run MLX runtime smoke check"
	@echo "  make argilla-smoke     Create offline Argilla review sample"
	@echo "  make promptfoo-smoke   Run promptfoo with local mock providers"
	@echo "  make deepeval-smoke    Run DeepEval exact-match sample"
	@echo "  make phase1-smoke      Run all Phase 1 smoke checks"
	@echo "  make validate-data     Placeholder for Phase 2 dataset validation"

test:
	uv run pytest

lint:
	uv run ruff check .

env-check:
	uv run python scripts/check_environment.py

mlx-smoke:
	uv run python scripts/run_mlx_smoke.py

argilla-smoke:
	uv run python scripts/create_argilla_sample.py

promptfoo-smoke:
	PROMPTFOO_PYTHON=$$(pwd)/.venv/bin/python npx promptfoo@latest eval -c configs/promptfoo/phase1_smoke.yaml --no-progress-bar --no-table --output /tmp/pharma-llm-promptfoo-smoke.json

deepeval-smoke:
	uv run python scripts/run_deepeval_smoke.py

phase1-smoke: env-check mlx-smoke argilla-smoke promptfoo-smoke deepeval-smoke

validate-data:
	@echo "Dataset validation is implemented in Phase 2."
