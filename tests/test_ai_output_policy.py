import json
import subprocess
import sys
from pathlib import Path

from pharma_llm_lab.dataset import parse_record, parse_dataset_type, validate_jsonl
from pharma_llm_lab.dataset.validators import describe_training_policy_failures


def provenance(
    *,
    source_type: str = "human_authored",
    review_status: str = "approved",
    ai_assisted: bool = False,
    raw_ai_output_used_as_training_target: bool = False,
    risk_flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source_type": source_type,
        "source_document": "synthetic_policy_sample",
        "source_license": "synthetic_test_only",
        "review_status": review_status,
        "ai_assisted": ai_assisted,
        "ai_tool": "codex_app" if ai_assisted else None,
        "raw_ai_output_used_as_training_target": raw_ai_output_used_as_training_target,
        "human_reviewer": "tester",
        "review_date": "2026-06-08",
        "risk_flags": risk_flags or [],
    }


def sft_record(*, provenance_data: dict[str, object]) -> dict[str, object]:
    return {
        "id": "sft_policy_001",
        "dataset_type": "sft",
        "prompt": "安全性情報を要約してください。",
        "response": "根拠範囲を明示して要約します。",
        "provenance": provenance_data,
    }


def eval_record() -> dict[str, object]:
    return {
        "id": "eval_policy_001",
        "dataset_type": "eval",
        "category": "unsafe_refusal",
        "prompt": "適応外使用の助言を求める入力を評価してください。",
        "expected_points": ["推奨しない", "根拠範囲を明示する"],
        "provenance": provenance(source_type="eval_only"),
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_raw_ai_output_is_rejected_for_training_with_explicit_reason(
    tmp_path: Path,
) -> None:
    path = write_jsonl(
        tmp_path / "raw_ai_output.jsonl",
        [
            sft_record(
                provenance_data=provenance(
                    source_type="raw_ai_output",
                    review_status="unreviewed",
                    ai_assisted=True,
                    raw_ai_output_used_as_training_target=True,
                )
            )
        ],
    )

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    message = result.errors[0].message
    assert "source_type 'raw_ai_output' is blocked for training" in message
    assert "raw_ai_output_used_as_training_target is true" in message
    assert "review_status 'unreviewed' is not approved for training" in message


def test_unreviewed_ai_candidate_is_rejected_for_training(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "unreviewed_ai_candidate.jsonl",
        [
            sft_record(
                provenance_data=provenance(
                    source_type="ai_candidate_unreviewed",
                    review_status="needs_edit",
                    ai_assisted=True,
                )
            )
        ],
    )

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    assert "source_type 'ai_candidate_unreviewed' is blocked" in result.errors[0].message
    assert "review_status 'needs_edit' is not approved" in result.errors[0].message


def test_human_edited_ai_assisted_record_is_accepted_after_approval(
    tmp_path: Path,
) -> None:
    path = write_jsonl(
        tmp_path / "approved_ai_assisted.jsonl",
        [
            sft_record(
                provenance_data=provenance(
                    source_type="human_edited_ai_assisted",
                    review_status="edited_and_approved",
                    ai_assisted=True,
                )
            )
        ],
    )

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert result.ok
    assert result.records[0].provenance.is_ai_assisted_but_reviewed


def test_eval_only_record_is_rejected_when_validated_as_training_dataset(
    tmp_path: Path,
) -> None:
    path = write_jsonl(tmp_path / "eval_leakage.jsonl", [eval_record()])

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    messages = [error.message for error in result.errors]
    assert any("dataset_type mismatch" in message for message in messages)
    assert any("dataset_type 'eval' is reserved for evaluation" in message for message in messages)
    assert any("source_type 'eval_only' is blocked for training" in message for message in messages)


def test_risk_flags_are_preserved_in_provenance_metadata() -> None:
    record = parse_record(
        sft_record(
            provenance_data=provenance(
                source_type="internal_doc_derived",
                review_status="approved",
                risk_flags=["needs_medical_review", "contains_synthetic_text"],
            )
        )
    )

    assert record.provenance.risk_flags == (
        "needs_medical_review",
        "contains_synthetic_text",
    )


def test_public_doc_derived_approved_record_has_no_training_policy_failures() -> None:
    record = parse_record(
        sft_record(
            provenance_data=provenance(
                source_type="public_doc_derived",
                review_status="approved",
            )
        )
    )

    assert describe_training_policy_failures(record) == ()


def test_cli_reports_policy_failures_clearly(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "raw_ai_output.jsonl",
        [
            sft_record(
                provenance_data=provenance(
                    source_type="raw_ai_output",
                    review_status="unreviewed",
                    ai_assisted=True,
                    raw_ai_output_used_as_training_target=True,
                )
            )
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--dataset-type",
            "sft",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "source_type 'raw_ai_output' is blocked for training" in result.stderr
    assert "raw_ai_output_used_as_training_target is true" in result.stderr
