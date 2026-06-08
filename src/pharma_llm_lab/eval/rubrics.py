"""Scoring rubric contract for Phase 4 evaluation sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharma_llm_lab.dataset import EvaluationCategory


class RubricError(ValueError):
    """Raised when a scoring rubric file violates the Phase 4 contract."""


@dataclass(frozen=True)
class RubricMetric:
    id: str
    description: str
    max_score: int


@dataclass(frozen=True)
class ScoringRubric:
    version: str
    categories: dict[EvaluationCategory, str]
    metrics: tuple[RubricMetric, ...]

    @property
    def metric_ids(self) -> tuple[str, ...]:
        return tuple(metric.id for metric in self.metrics)


REQUIRED_METRIC_IDS = ("safety", "pharma_style", "factuality")


def require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RubricError(f"{field_name} must be an object")
    return value


def require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RubricError(f"{field_name} must be a non-empty string")
    return value


def parse_metric(value: Any, *, index: int) -> RubricMetric:
    metric = require_mapping(value, f"metrics[{index}]")
    metric_id = require_string(metric.get("id"), f"metrics[{index}].id")
    description = require_string(
        metric.get("description"),
        f"metrics[{index}].description",
    )
    max_score = metric.get("max_score")
    if not isinstance(max_score, int) or max_score <= 0:
        raise RubricError(f"metrics[{index}].max_score must be a positive integer")
    return RubricMetric(id=metric_id, description=description, max_score=max_score)


def parse_categories(value: Any) -> dict[EvaluationCategory, str]:
    raw_categories = require_mapping(value, "categories")
    categories: dict[EvaluationCategory, str] = {}
    for category in EvaluationCategory:
        if category.value not in raw_categories:
            raise RubricError(f"categories must include {category.value!r}")
        categories[category] = require_string(
            raw_categories[category.value],
            f"categories.{category.value}",
        )

    unknown_categories = sorted(set(raw_categories) - {category.value for category in EvaluationCategory})
    if unknown_categories:
        raise RubricError(
            "categories contain unsupported key(s): "
            + ", ".join(repr(category) for category in unknown_categories)
        )
    return categories


def parse_scoring_rubric(value: dict[str, Any]) -> ScoringRubric:
    rubric = require_mapping(value.get("rubric"), "rubric")
    version = require_string(rubric.get("version"), "version")
    categories = parse_categories(rubric.get("categories"))

    raw_metrics = rubric.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise RubricError("metrics must be a non-empty list")
    metrics = tuple(parse_metric(metric, index=index) for index, metric in enumerate(raw_metrics))

    metric_ids = [metric.id for metric in metrics]
    duplicate_metric_ids = sorted({metric_id for metric_id in metric_ids if metric_ids.count(metric_id) > 1})
    if duplicate_metric_ids:
        raise RubricError(
            "metrics contain duplicate id(s): "
            + ", ".join(repr(metric_id) for metric_id in duplicate_metric_ids)
        )

    missing_metrics = [metric_id for metric_id in REQUIRED_METRIC_IDS if metric_id not in metric_ids]
    if missing_metrics:
        raise RubricError(
            "metrics must include required id(s): "
            + ", ".join(repr(metric_id) for metric_id in missing_metrics)
        )

    return ScoringRubric(version=version, categories=categories, metrics=metrics)


def load_scoring_rubric(path: Path) -> ScoringRubric:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RubricError(
            "scoring rubric must be JSON-compatible YAML for offline validation: "
            f"{exc.msg}"
        ) from exc
    return parse_scoring_rubric(value)
