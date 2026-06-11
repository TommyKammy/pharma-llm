"""Base-vs-LoRA comparison report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharma_llm_lab.baseline.reports import (
    BaselineReportInput,
    format_optional_float,
    format_status_counts,
    load_report_inputs,
    markdown_row,
    mean,
    notable_failure_modes,
)
from pharma_llm_lab.baseline.results import BaselineResult, BaselineResultError
from pharma_llm_lab.dataset import EvaluationCategory
from pharma_llm_lab.training.lora_metadata import (
    AdapterMetadataValidationError,
    validate_adapter_metadata,
)

SAFETY_REVIEW_CATEGORIES = {
    EvaluationCategory.SAFETY_INFORMATION,
    EvaluationCategory.UNSAFE_REFUSAL,
}


@dataclass(frozen=True)
class LoraComparisonInput:
    base: BaselineReportInput
    lora: BaselineReportInput
    adapter_metadata: dict[str, Any] | None = None
    adapter_metadata_source: Path | None = None


def load_adapter_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineResultError(f"{path}: adapter metadata path is not a file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BaselineResultError(f"{path}: could not read adapter metadata: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineResultError(f"{path}: malformed adapter metadata JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise BaselineResultError(f"{path}: adapter metadata root must be an object")
    try:
        metadata = validate_adapter_metadata(payload).to_mapping()
    except AdapterMetadataValidationError as exc:
        raise BaselineResultError(f"{path}: invalid adapter metadata: {exc}") from exc
    if metadata["status"] != "executed":
        raise BaselineResultError("adapter metadata must have status executed")
    return metadata


def eval_category_map(results: tuple[BaselineResult, ...]) -> dict[str, EvaluationCategory]:
    return {result.eval_id: result.category for result in results}


def validate_base_lora_pair(
    base: BaselineReportInput,
    lora: BaselineReportInput,
    adapter_metadata: dict[str, Any] | None,
) -> None:
    if base.summary.adapter_id is not None:
        raise BaselineResultError("base predictions must not include adapter_id")
    if lora.summary.adapter_id is None:
        raise BaselineResultError("LoRA predictions must include adapter_id")
    if base.summary.model_id != lora.summary.model_id:
        raise BaselineResultError("base and LoRA predictions must use the same model_id")
    if base.summary.provider != lora.summary.provider:
        raise BaselineResultError("base and LoRA predictions must use the same provider")

    base_categories = eval_category_map(base.results)
    lora_categories = eval_category_map(lora.results)
    if base_categories != lora_categories:
        raise BaselineResultError("base and LoRA inputs must use the same eval_id/category mapping")

    if adapter_metadata is None:
        return
    metadata_run_id = adapter_metadata["run_id"]
    if lora.summary.adapter_id != metadata_run_id:
        raise BaselineResultError("LoRA adapter_id must match adapter metadata run_id")
    metadata_model_id = adapter_metadata["model"]["id"]
    if lora.summary.model_id != metadata_model_id:
        raise BaselineResultError("LoRA model_id must match adapter metadata model.id")


def load_lora_comparison_inputs(
    *,
    base_path: Path,
    lora_path: Path,
    adapter_metadata_path: Path | None = None,
) -> LoraComparisonInput:
    base, lora = load_report_inputs((base_path, lora_path))
    adapter_metadata = (
        load_adapter_metadata(adapter_metadata_path) if adapter_metadata_path is not None else None
    )
    validate_base_lora_pair(base, lora, adapter_metadata)
    return LoraComparisonInput(
        base=base,
        lora=lora,
        adapter_metadata=adapter_metadata,
        adapter_metadata_source=adapter_metadata_path,
    )


def format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    formatted = format_optional_float(round(value, 3))
    if value > 0:
        return f"+{formatted}"
    return formatted


def metric_delta(lora_value: float | None, base_value: float | None) -> float | None:
    if lora_value is None or base_value is None:
        return None
    return round(lora_value - base_value, 3)


def category_metric_rows(comparison: LoraComparisonInput) -> list[str]:
    base_metrics = {
        metrics.category: metrics for metrics in comparison.base.summary.category_metrics
    }
    lora_metrics = {
        metrics.category: metrics for metrics in comparison.lora.summary.category_metrics
    }
    rows: list[str] = []
    for category in EvaluationCategory:
        if category not in base_metrics or category not in lora_metrics:
            continue
        base = base_metrics[category]
        lora = lora_metrics[category]
        rows.append(
            markdown_row(
                [
                    category.value,
                    base.count,
                    format_optional_float(base.avg_total_latency_ms),
                    format_optional_float(lora.avg_total_latency_ms),
                    format_delta(
                        metric_delta(lora.avg_total_latency_ms, base.avg_total_latency_ms)
                    ),
                    format_optional_float(base.avg_ttft_ms),
                    format_optional_float(lora.avg_ttft_ms),
                    format_delta(metric_delta(lora.avg_ttft_ms, base.avg_ttft_ms)),
                    format_optional_float(base.avg_tokens_per_second),
                    format_optional_float(lora.avg_tokens_per_second),
                    format_delta(
                        metric_delta(
                            lora.avg_tokens_per_second,
                            base.avg_tokens_per_second,
                        )
                    ),
                    format_status_counts(base.scoring_status_counts),
                    format_status_counts(lora.scoring_status_counts),
                ]
            )
        )
    return rows


def safety_status_note(results: tuple[BaselineResult, ...]) -> str:
    statuses = sorted(
        {
            result.scoring_status
            for result in results
            if result.category in SAFETY_REVIEW_CATEGORIES
        }
    )
    if not statuses:
        return "no safety/style eval records found"
    if statuses == ["passed"]:
        return "automated safety/style checks passed"
    return "manual-review-only or partial scoring: " + ", ".join(statuses)


def safety_category_summary(comparison: LoraComparisonInput) -> list[str]:
    rows: list[str] = [
        "|Category|Base Statuses|LoRA Statuses|Note|",
        "|---|---|---|---|",
    ]
    base_metrics = {
        metrics.category: metrics for metrics in comparison.base.summary.category_metrics
    }
    lora_metrics = {
        metrics.category: metrics for metrics in comparison.lora.summary.category_metrics
    }
    for category in EvaluationCategory:
        if category not in SAFETY_REVIEW_CATEGORIES:
            continue
        if category not in base_metrics or category not in lora_metrics:
            continue
        lora_statuses = lora_metrics[category].scoring_status_counts
        note = (
            "requires human review before improvement claim"
            if any(status != "passed" for status in lora_statuses)
            else "no safety regression signaled by automated statuses"
        )
        rows.append(
            markdown_row(
                [
                    category.value,
                    format_status_counts(base_metrics[category].scoring_status_counts),
                    format_status_counts(lora_statuses),
                    note,
                ]
            )
        )
    return rows


def append_adapter_metadata_section(lines: list[str], comparison: LoraComparisonInput) -> None:
    metadata = comparison.adapter_metadata
    if metadata is None:
        lines.extend(
            [
                "## Adapter Metadata",
                "",
                "No adapter metadata file was supplied. Treat the LoRA adapter identity "
                "as prediction-JSONL-only until local `adapter_metadata.json` is attached.",
                "",
            ]
        )
        return

    adapter = metadata["adapter"]
    config = metadata["config"]
    dataset = metadata["dataset"]
    lines.extend(
        [
            "## Adapter Metadata",
            "",
            "|Field|Value|",
            "|---|---|",
            markdown_row(["source", comparison.adapter_metadata_source or "n/a"]),
            markdown_row(["metadata run_id", metadata["run_id"]]),
            markdown_row(["status", metadata["status"]]),
            markdown_row(["adapter path", adapter["path"]]),
            markdown_row(["metadata path", adapter["metadata_path"]]),
            markdown_row(["dataset version", dataset["version"]]),
            markdown_row(["source config sha256", config["source_sha256"]]),
            "",
        ]
    )


def build_lora_comparison_report(
    comparison: LoraComparisonInput,
    *,
    title: str = "Base vs LoRA Evaluation and Safety Report",
    limitation_notice: str | None = None,
) -> str:
    base = comparison.base.summary
    lora = comparison.lora.summary
    notice = limitation_notice or (
        "This report compares already-generated prediction JSONL artifacts. It does "
        "not train an adapter or claim quality improvement by itself; safety/style "
        "findings remain manual-review-only unless reviewed scoring artifacts are attached."
    )

    lines: list[str] = [
        f"# {title}",
        "",
        "## Limitations",
        "",
        notice,
        "",
        "## Model and Adapter Identity",
        "",
        "|Role|Model|Provider|Adapter|Run ID|Source|Eval Count|Scoring Statuses|",
        "|---|---|---|---|---|---|---:|---|",
        markdown_row(
            [
                "base",
                base.model_id,
                base.provider,
                "none",
                base.run_id,
                comparison.base.source_path,
                base.total_count,
                format_status_counts(base.scoring_status_counts),
            ]
        ),
        markdown_row(
            [
                "lora",
                lora.model_id,
                lora.provider,
                lora.adapter_id or "none",
                lora.run_id,
                comparison.lora.source_path,
                lora.total_count,
                format_status_counts(lora.scoring_status_counts),
            ]
        ),
        "",
        "## Eval Coverage",
        "",
        f"- Matched eval count: {base.total_count}",
        "- Eval ID set: matched before report generation",
        f"- Category count: {len(base.category_metrics)}",
        "",
    ]
    append_adapter_metadata_section(lines, comparison)

    lines.extend(
        [
            "## Category Deltas",
            "",
            "|Category|Count|Base Latency ms|LoRA Latency ms|Delta Latency ms|"
            "Base TTFT ms|LoRA TTFT ms|Delta TTFT ms|Base Tokens/sec|"
            "LoRA Tokens/sec|Delta Tokens/sec|Base Statuses|LoRA Statuses|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            *category_metric_rows(comparison),
            "",
            "## Latency Summary",
            "",
            markdown_row(["Metric", "Base", "LoRA", "Delta"]),
            markdown_row(["---", "---:", "---:", "---:"]),
        ]
    )
    base_latency = mean([metrics.avg_total_latency_ms for metrics in base.category_metrics])
    lora_latency = mean([metrics.avg_total_latency_ms for metrics in lora.category_metrics])
    base_ttft = mean(
        [metrics.avg_ttft_ms for metrics in base.category_metrics if metrics.avg_ttft_ms is not None]
    )
    lora_ttft = mean(
        [metrics.avg_ttft_ms for metrics in lora.category_metrics if metrics.avg_ttft_ms is not None]
    )
    lines.extend(
        [
            markdown_row(
                [
                    "Mean category latency ms",
                    format_optional_float(base_latency),
                    format_optional_float(lora_latency),
                    format_delta(metric_delta(lora_latency, base_latency)),
                ]
            ),
            markdown_row(
                [
                    "Mean category TTFT ms",
                    format_optional_float(base_ttft),
                    format_optional_float(lora_ttft),
                    format_delta(metric_delta(lora_ttft, base_ttft)),
                ]
            ),
            "",
            "## Safety and Style Regression Notes",
            "",
            f"- Base safety/style scoring: {safety_status_note(comparison.base.results)}",
            f"- LoRA safety/style scoring: {safety_status_note(comparison.lora.results)}",
            "- Do not start Phase 7 from this report unless safety/style categories are "
            "reviewed and no regression is accepted by the operator.",
            "",
            *safety_category_summary(comparison),
            "",
            "## Notable Failure Modes",
            "",
            "### Base",
            "",
        ]
    )
    lines.extend(f"- {failure}" for failure in notable_failure_modes(comparison.base.results))
    lines.extend(["", "### LoRA", ""])
    lines.extend(f"- {failure}" for failure in notable_failure_modes(comparison.lora.results))
    lines.extend(
        [
            "",
            "## Promptfoo / DeepEval Entry Points",
            "",
            "- Promptfoo mock comparison: `promptfoo eval -c "
            "configs/promptfoo/lora_comparison_mock.yaml`",
            "- DeepEval or reviewed scoring artifacts may be attached later, but CI must "
            "not block on networked or model-hosted scoring.",
            "",
            "## Phase 7 Interpretation Rules",
            "",
            "- Phase 7 LoRA sweep can start only when base and LoRA eval IDs match.",
            "- Safety/style categories must not regress; manual-review-only statuses are "
            "not proof of improvement.",
            "- Category deltas and latency deltas should be reviewed before selecting "
            "sweep parameters.",
            "- Any limitation in this report must be resolved or copied into the Phase 7 "
            "experiment notes.",
            "",
            "## Obsidian Copy Block",
            "",
            "```markdown",
            f"- Base vs LoRA: {base.model_id}, evals={base.total_count}, "
            f"adapter={lora.adapter_id}, lora_statuses={format_status_counts(lora.scoring_status_counts)}",
            f"- Safety/style: {safety_status_note(comparison.lora.results)}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_lora_comparison_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
