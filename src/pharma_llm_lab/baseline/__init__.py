"""Baseline evaluation result schema and aggregation helpers."""

from pharma_llm_lab.baseline.reports import (
    BaselineReportInput,
    build_baseline_report,
    load_report_inputs,
    write_baseline_report,
)
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
    "BaselineReportInput",
    "BaselineResult",
    "BaselineResultError",
    "BaselineSummary",
    "CategoryMetrics",
    "aggregate_results",
    "build_baseline_report",
    "load_baseline_results",
    "load_report_inputs",
    "write_baseline_report",
    "write_category_metrics_csv",
    "write_summary_json",
]
