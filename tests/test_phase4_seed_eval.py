import json
import re
from collections import Counter
from pathlib import Path

from pharma_llm_lab.dataset import DatasetType, EvaluationCategory, EvalRecord, SourceType
from pharma_llm_lab.dataset.schema import parse_record
from pharma_llm_lab.dataset.validators import parse_dataset_type, validate_jsonl


SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")

CATEGORY_ID_RANGES = {
    EvaluationCategory.BUSINESS_SUMMARY: range(1, 51),
    EvaluationCategory.PACKAGE_INSERT_READING: range(51, 101),
    EvaluationCategory.SAFETY_INFORMATION: range(101, 151),
    EvaluationCategory.GXP_QA_AUDIT: range(151, 201),
    EvaluationCategory.DI_INQUIRY: range(201, 251),
    EvaluationCategory.UNSAFE_REFUSAL: range(251, 301),
}

RISK_FLAG_REQUIRED_CATEGORIES = {
    EvaluationCategory.SAFETY_INFORMATION,
    EvaluationCategory.GXP_QA_AUDIT,
    EvaluationCategory.DI_INQUIRY,
    EvaluationCategory.UNSAFE_REFUSAL,
}


def load_seed_records() -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for line in SEED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = parse_record(json.loads(line))
        assert isinstance(parsed, EvalRecord)
        records.append(parsed)
    return records


def parse_eval_id_number(record_id: str) -> int:
    match = re.fullmatch(r"eval_(\d{3})", record_id)
    assert match is not None
    return int(match.group(1))


def test_phase4_seed_eval_validates_as_eval_only_dataset() -> None:
    result = validate_jsonl(SEED_PATH, parse_dataset_type("eval"))

    assert result.ok, [error.format() for error in result.errors]
    assert result.record_count == 30
    assert all(isinstance(record, EvalRecord) for record in result.records)
    assert all(record.dataset_type is DatasetType.EVAL for record in result.records)
    assert all(
        record.provenance.source_type is SourceType.EVAL_ONLY
        for record in result.records
    )


def test_phase4_seed_eval_covers_all_categories_with_five_records_each() -> None:
    records = load_seed_records()

    assert Counter(record.category for record in records) == {
        category: 5 for category in EvaluationCategory
    }


def test_phase4_seed_eval_ids_match_phase4_category_ranges() -> None:
    for record in load_seed_records():
        id_number = parse_eval_id_number(record.id)

        assert id_number in CATEGORY_ID_RANGES[record.category]


def test_phase4_seed_eval_records_have_scoring_points_and_synthetic_provenance() -> None:
    for record in load_seed_records():
        assert len(record.expected_points) >= 3
        assert record.prompt.strip()
        assert record.provenance.source_document == "synthetic_phase4_seed"
        assert record.provenance.source_license == "synthetic_test_only"
        assert not record.provenance.ai_assisted
        assert not record.provenance.raw_ai_output_used_as_training_target


def test_phase4_seed_eval_regulated_context_records_have_risk_flags() -> None:
    for record in load_seed_records():
        if record.category in RISK_FLAG_REQUIRED_CATEGORIES:
            assert record.provenance.risk_flags


def test_phase4_seed_eval_is_rejected_from_training_validation() -> None:
    result = validate_jsonl(SEED_PATH, parse_dataset_type("sft"))

    assert not result.ok
    assert result.record_count == 30
    assert any("dataset_type mismatch" in error.message for error in result.errors)
    assert any("not eligible for training" in error.message for error in result.errors)
