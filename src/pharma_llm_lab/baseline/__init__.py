"""Baseline evaluation result schema and aggregation helpers."""

from pharma_llm_lab.baseline.results import (
    BaselineResult,
    BaselineResultError,
    CategoryMetrics,
    BaselineSummary,
    aggregate_results,
    load_baseline_results,
    write_category_metrics_csv,
    write_summary_json,
)

__all__ = [
    "BaselineResult",
    "BaselineResultError",
    "BaselineSummary",
    "CategoryMetrics",
    "aggregate_results",
    "load_baseline_results",
    "write_category_metrics_csv",
    "write_summary_json",
]
