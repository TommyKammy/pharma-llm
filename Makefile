.PHONY: help test lint validate-data

help:
	@echo "pharma-llm commands"
	@echo "  make test           Run unit tests"
	@echo "  make lint           Run ruff checks"
	@echo "  make validate-data  Placeholder for Phase 2 dataset validation"

test:
	uv run pytest

lint:
	uv run ruff check .

validate-data:
	@echo "Dataset validation is implemented in Phase 2."

