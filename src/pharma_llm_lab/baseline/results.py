"""Validation and aggregation for baseline prediction artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

from pharma_llm_lab.dataset import EvaluationCategory


class BaselineResultError(ValueError):
    """Raised when a baseline result artifact is invalid."""


def require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BaselineResultError(f"{key} must be a non-empty string")
    return value


def require_present_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise BaselineResultError(f"{key} must be a string")
    return value


def require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise BaselineResultError(f"{key} must be an object")
    return value


def require_finite_non_negative_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or value < 0
        or not isfinite(value)
    ):
        raise BaselineResultError(f"{field_name} must be a finite non-negative number")
    return float(value)


def optional_finite_non_negative_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return require_finite_non_negative_number(value, field_name)


@dataclass(frozen=True)
class BaselineResult:
    run_id: str
    eval_id: str
    category: EvaluationCategory
    model_id: str
    provider: str
    adapter_id: str | None
    generated_text: str
    scoring_status: str
    total_latency_ms: float
    ttft_ms: float | None
    tokens_per_second: float | None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "BaselineResult":
        raw_category = require_string(mapping, "category")
        try:
            category = EvaluationCategory(raw_category)
        except ValueError as exc:
            allowed = ", ".join(category.value for category in EvaluationCategory)
            raise BaselineResultError(f"category must be one of: {allowed}") from exc

        model = require_mapping(mapping, "model")
        timing = require_mapping(mapping, "timing")
        scoring_status = mapping.get("scoring_status", "unscored")
        if not isinstance(scoring_status, str) or not scoring_status.strip():
            raise BaselineResultError("scoring_status must be a non-empty string")
        adapter_id = model.get("adapter_id")
        if adapter_id is not None and not isinstance(adapter_id, str):
            raise BaselineResultError("adapter_id must be null or a string")

        return cls(
            run_id=require_string(mapping, "run_id"),
            eval_id=require_string(mapping, "eval_id"),
            category=category,
            model_id=require_string(model, "model_id"),
            provider=require_string(model, "provider"),
            adapter_id=adapter_id,
            generated_text=require_present_string(mapping, "generated_text"),
            scoring_status=scoring_status,
            total_latency_ms=require_finite_non_negative_number(
                timing.get("total_latency_ms"), "timing.total_latency_ms"
            ),
            ttft_ms=optional_finite_non_negative_number(
                timing.get("ttft_ms"), "timing.ttft_ms"
            ),
            tokens_per_second=optional_finite_non_negative_number(
                timing.get("tokens_per_second"), "timing.tokens_per_second"
            ),
        )

    def to_mapping(self) -> dict[str, str | float | None]:
        return {
            "run_id": self.run_id,
            "eval_id": self.eval_id,
            "category": self.category.value,
            "model_id": self.model_id,
            "provider": self.provider,
            "adapter_id": self.adapter_id,
            "generated_text": self.generated_text,
            "scoring_status": self.scoring_status,
            "total_latency_ms": self.total_latency_ms,
            "ttft_ms": self.ttft_ms,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class CategoryMetrics:
    run_id: str
    model_id: str
    provider: str
    adapter_id: str | None
    category: EvaluationCategory
    count: int
    avg_total_latency_ms: float
    avg_ttft_ms: float | None
    avg_tokens_per_second: float | None
    scoring_status_counts: dict[str, int]

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "adapter_id": self.adapter_id,
            "category": self.category.value,
            "count": self.count,
            "avg_total_latency_ms": self.avg_total_latency_ms,
            "avg_ttft_ms": self.avg_ttft_ms,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "scoring_status_counts": dict(sorted(self.scoring_status_counts.items())),
        }


@dataclass(frozen=True)
class BaselineSummary:
    run_id: str
    model_id: str
    provider: str
    adapter_id: str | None
    total_count: int
    category_metrics: tuple[CategoryMetrics, ...]
    scoring_status_counts: dict[str, int]

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "adapter_id": self.adapter_id,
            "total_count": self.total_count,
            "scoring_status_counts": dict(sorted(self.scoring_status_counts.items())),
            "category_metrics": [
                metrics.to_mapping() for metrics in self.category_metrics
            ],
        }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineResultError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise BaselineResultError(f"{path}:{line_number}: record must be an object")
            yield line_number, value


def load_baseline_results(path: Path) -> tuple[BaselineResult, ...]:
    if not path.is_file():
        raise BaselineResultError(f"path is not a file: {path}")

    results: list[BaselineResult] = []
    for line_number, item in iter_jsonl(path):
        try:
            results.append(BaselineResult.from_mapping(item))
        except BaselineResultError as exc:
            raise BaselineResultError(f"{path}:{line_number}: {exc}") from exc

    if not results:
        raise BaselineResultError(f"{path}: no baseline results found")
    return tuple(results)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def aggregate_results(results: tuple[BaselineResult, ...]) -> BaselineSummary:
    if not results:
        raise BaselineResultError("cannot aggregate empty baseline results")

    run_ids = {result.run_id for result in results}
    model_ids = {result.model_id for result in results}
    providers = {result.provider for result in results}
    adapter_ids = {result.adapter_id for result in results}
    eval_id_counts = Counter(result.eval_id for result in results)
    duplicate_eval_ids = sorted(
        eval_id for eval_id, count in eval_id_counts.items() if count > 1
    )
    if len(run_ids) != 1:
        raise BaselineResultError("baseline results must contain exactly one run_id")
    if len(model_ids) != 1:
        raise BaselineResultError("baseline results must contain exactly one model_id")
    if len(providers) != 1:
        raise BaselineResultError("baseline results must contain exactly one provider")
    if len(adapter_ids) != 1:
        raise BaselineResultError("baseline results must contain exactly one adapter_id")
    if duplicate_eval_ids:
        raise BaselineResultError(
            "baseline results must contain unique eval_id values: "
            + ", ".join(duplicate_eval_ids)
        )

    category_metrics: list[CategoryMetrics] = []
    scoring_status_counts: dict[str, int] = {}
    for result in results:
        increment(scoring_status_counts, result.scoring_status)

    for category in EvaluationCategory:
        category_results = [result for result in results if result.category is category]
        if not category_results:
            continue
        category_status_counts: dict[str, int] = {}
        for result in category_results:
            increment(category_status_counts, result.scoring_status)
        category_metrics.append(
            CategoryMetrics(
                run_id=category_results[0].run_id,
                model_id=category_results[0].model_id,
                provider=category_results[0].provider,
                adapter_id=category_results[0].adapter_id,
                category=category,
                count=len(category_results),
                avg_total_latency_ms=average(
                    [result.total_latency_ms for result in category_results]
                )
                or 0.0,
                avg_ttft_ms=average(
                    [
                        result.ttft_ms
                        for result in category_results
                        if result.ttft_ms is not None
                    ]
                ),
                avg_tokens_per_second=average(
                    [
                        result.tokens_per_second
                        for result in category_results
                        if result.tokens_per_second is not None
                    ]
                ),
                scoring_status_counts=category_status_counts,
            )
        )

    return BaselineSummary(
        run_id=results[0].run_id,
        model_id=results[0].model_id,
        provider=results[0].provider,
        adapter_id=results[0].adapter_id,
        total_count=len(results),
        category_metrics=tuple(category_metrics),
        scoring_status_counts=scoring_status_counts,
    )


def write_summary_json(path: Path, summary: BaselineSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_category_metrics_csv(path: Path, summary: BaselineSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "model_id",
                "provider",
                "adapter_id",
                "category",
                "count",
                "avg_total_latency_ms",
                "avg_ttft_ms",
                "avg_tokens_per_second",
                "scoring_status_counts",
            ],
        )
        writer.writeheader()
        for metrics in summary.category_metrics:
            row = metrics.to_mapping()
            row["scoring_status_counts"] = json.dumps(
                row["scoring_status_counts"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow(row)
