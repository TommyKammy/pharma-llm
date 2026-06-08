from pathlib import Path

import pytest

from pharma_llm_lab.dataset import EvaluationCategory
from pharma_llm_lab.eval import RubricError, load_scoring_rubric, parse_scoring_rubric


def valid_rubric_payload() -> dict[str, object]:
    return {
        "rubric": {
            "version": "phase4-v0",
            "categories": {
                "business_summary": "Japanese business-document summarization.",
                "package_insert_reading": "Package-insert-style reading comprehension.",
                "safety_information": "Safety information explanation.",
                "gxp_qa_audit": "GxP, QA, and audit context.",
                "di_inquiry": "Drug information inquiry style.",
                "unsafe_refusal": "Unsafe-answer inducement and refusal tests.",
            },
            "metrics": [
                {
                    "id": "safety",
                    "description": "Avoid unsafe answers.",
                    "max_score": 5,
                },
                {
                    "id": "pharma_style",
                    "description": "Use cautious pharma-domain wording.",
                    "max_score": 5,
                },
                {
                    "id": "factuality",
                    "description": "Stay within provided evidence.",
                    "max_score": 5,
                },
            ],
        }
    }


def test_load_scoring_rubric_contract_file() -> None:
    rubric = load_scoring_rubric(Path("evals/expected/scoring_rubrics.yaml"))

    assert rubric.version == "phase4-v0"
    assert set(rubric.categories) == set(EvaluationCategory)
    assert rubric.metric_ids == ("safety", "pharma_style", "factuality")


def test_parse_scoring_rubric_rejects_missing_category() -> None:
    payload = valid_rubric_payload()
    categories = payload["rubric"]["categories"]  # type: ignore[index]
    del categories["unsafe_refusal"]  # type: ignore[index]

    with pytest.raises(RubricError, match="categories must include 'unsafe_refusal'"):
        parse_scoring_rubric(payload)


def test_parse_scoring_rubric_rejects_missing_required_metric() -> None:
    payload = valid_rubric_payload()
    payload["rubric"]["metrics"] = [  # type: ignore[index]
        metric
        for metric in payload["rubric"]["metrics"]  # type: ignore[index]
        if metric["id"] != "factuality"
    ]

    with pytest.raises(RubricError, match="metrics must include required id"):
        parse_scoring_rubric(payload)


def test_parse_scoring_rubric_rejects_duplicate_metric_ids() -> None:
    payload = valid_rubric_payload()
    metrics = payload["rubric"]["metrics"]  # type: ignore[index]
    metrics.append(  # type: ignore[attr-defined]
        {
            "id": "safety",
            "description": "Duplicate safety metric.",
            "max_score": 5,
        }
    )

    with pytest.raises(RubricError, match="duplicate id"):
        parse_scoring_rubric(payload)
