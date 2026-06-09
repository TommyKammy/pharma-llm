"""Markdown report generation for baseline evaluation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pharma_llm_lab.baseline.results import (
    BaselineResult,
    BaselineResultError,
    BaselineSummary,
    aggregate_results,
    load_baseline_results,
)


@dataclass(frozen=True)
class BaselineReportInput:
    source_path: Path
    summary: BaselineSummary


def load_report_inputs(paths: tuple[Path, ...]) -> tuple[BaselineReportInput, ...]:
    if not paths:
        raise BaselineResultError("at least one baseline prediction JSONL is required")

    report_inputs: list[BaselineReportInput] = []
    seen_model_keys: set[tuple[str, str, str | None]] = set()
    for path in paths:
        results = load_baseline_results(path)
        summary = aggregate_results(results)
        model_key = (summary.model_id, summary.provider, summary.adapter_id)
        if model_key in seen_model_keys:
            raise BaselineResultError(
                "baseline report inputs must contain unique model identities"
            )
        seen_model_keys.add(model_key)
        report_inputs.append(BaselineReportInput(source_path=path, summary=summary))

    return tuple(report_inputs)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_status_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))


def table_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def markdown_row(cells: list[object]) -> str:
    return "|" + "|".join(table_cell(cell) for cell in cells) + "|"


def notable_failure_modes(results: tuple[BaselineResult, ...]) -> tuple[str, ...]:
    failures: list[str] = []
    empty_count = sum(1 for result in results if result.generated_text == "")
    non_unscored = sorted(
        {
            result.scoring_status
            for result in results
            if result.scoring_status != "unscored"
        }
    )
    zero_tps_count = sum(
        1 for result in results if result.tokens_per_second is not None and result.tokens_per_second == 0
    )

    if empty_count:
        failures.append(f"{empty_count} empty completion(s) preserved for inspection")
    if non_unscored:
        failures.append("non-default scoring statuses: " + ", ".join(non_unscored))
    if zero_tps_count:
        failures.append(f"{zero_tps_count} record(s) with zero tokens/sec")

    if not failures:
        failures.append("No empty completions or non-default scoring statuses detected")
    return tuple(failures)


def build_baseline_report(
    report_inputs: tuple[BaselineReportInput, ...],
    *,
    title: str = "Baseline Evaluation Report",
    mock_notice: str | None = None,
) -> str:
    if not report_inputs:
        raise BaselineResultError("at least one baseline summary is required")

    notice = mock_notice or (
        "These results may come from CI-safe mock or fixture data. Use this report "
        "to validate evaluation wiring and reporting shape; do not claim real model "
        "quality until live baseline runs are attached."
    )

    lines: list[str] = [
        f"# {title}",
        "",
        "## Limitations",
        "",
        notice,
        "",
        "## Model Comparison",
        "",
        "|Model|Provider|Adapter|Run ID|Eval Count|Scoring Statuses|",
        "|---|---|---|---|---:|---|",
    ]
    for item in report_inputs:
        summary = item.summary
        lines.append(
            markdown_row(
                [
                    summary.model_id,
                    summary.provider,
                    summary.adapter_id or "none",
                    summary.run_id,
                    summary.total_count,
                    format_status_counts(summary.scoring_status_counts),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Category Breakdown",
            "",
            "|Model|Category|Count|Avg Latency ms|Avg TTFT ms|Avg Tokens/sec|Scoring Statuses|",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in report_inputs:
        for metrics in item.summary.category_metrics:
            lines.append(
                markdown_row(
                    [
                        item.summary.model_id,
                        metrics.category.value,
                        metrics.count,
                        format_optional_float(metrics.avg_total_latency_ms),
                        format_optional_float(metrics.avg_ttft_ms),
                        format_optional_float(metrics.avg_tokens_per_second),
                        format_status_counts(metrics.scoring_status_counts),
                    ]
                )
            )

    lines.extend(["", "## Latency Summary", ""])
    for item in report_inputs:
        summary = item.summary
        total_latency_values = [
            metrics.avg_total_latency_ms for metrics in summary.category_metrics
        ]
        ttft_values = [
            metrics.avg_ttft_ms
            for metrics in summary.category_metrics
            if metrics.avg_ttft_ms is not None
        ]
        tps_values = [
            metrics.avg_tokens_per_second
            for metrics in summary.category_metrics
            if metrics.avg_tokens_per_second is not None
        ]
        lines.extend(
            [
                f"### {summary.model_id}",
                "",
                f"- Total evals: {summary.total_count}",
                f"- Category count: {len(summary.category_metrics)}",
                f"- Mean category latency ms: {format_optional_float(mean(total_latency_values))}",
                f"- Mean category TTFT ms: {format_optional_float(mean(ttft_values))}",
                f"- Mean category tokens/sec: {format_optional_float(mean(tps_values))}",
                "",
            ]
        )

    lines.extend(["## Notable Failure Modes", ""])
    for item in report_inputs:
        results = load_baseline_results(item.source_path)
        lines.append(f"### {item.summary.model_id}")
        lines.append("")
        for failure in notable_failure_modes(results):
            lines.append(f"- {failure}")
        lines.append("")

    lines.extend(
        [
            "## Obsidian Copy Block",
            "",
            "```markdown",
        ]
    )
    for item in report_inputs:
        summary = item.summary
        lines.append(
            f"- {summary.model_id}: {summary.total_count} evals, "
            f"{len(summary.category_metrics)} categories, "
            f"statuses={format_status_counts(summary.scoring_status_counts)}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Phase 6 Interpretation",
            "",
            "Use this baseline as the pre-LoRA reference for latency, coverage, and "
            "failure-mode tracking. Compare later LoRA outputs against the same eval "
            "ids and categories before claiming improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def write_baseline_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
