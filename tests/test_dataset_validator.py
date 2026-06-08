import json
import subprocess
import sys
from pathlib import Path

import pytest

from pharma_llm_lab.dataset.validators import parse_dataset_type, validate_jsonl


def provenance(
    *,
    source_type: str = "human_authored",
    review_status: str = "approved",
    raw_ai_output_used_as_training_target: bool = False,
) -> dict[str, object]:
    return {
        "source_type": source_type,
        "source_document": "synthetic_validator_sample",
        "source_license": "synthetic_test_only",
        "review_status": review_status,
        "ai_assisted": False,
        "raw_ai_output_used_as_training_target": raw_ai_output_used_as_training_target,
        "human_reviewer": "tester",
        "review_date": "2026-06-08",
        "risk_flags": [],
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def valid_sft_record() -> dict[str, object]:
    return {
        "id": "sft_001",
        "dataset_type": "sft",
        "prompt": "安全性情報を要約してください。",
        "response": "根拠範囲を明示して要約します。",
        "provenance": provenance(),
    }


def test_validate_jsonl_accepts_valid_sft(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "sft.jsonl", [valid_sft_record()])

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert result.ok
    assert result.record_count == 1


def test_validate_jsonl_reports_malformed_json_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\n{"id": \n', encoding="utf-8")

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    assert any("line 2" in error.format() for error in result.errors)
    assert any("invalid JSON" in error.message for error in result.errors)


def test_validate_jsonl_rejects_missing_provenance(tmp_path: Path) -> None:
    record = valid_sft_record()
    del record["provenance"]
    path = write_jsonl(tmp_path / "missing_provenance.jsonl", [record])

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    assert "provenance must be an object" in result.errors[0].message


def test_validate_jsonl_rejects_bad_review_status(tmp_path: Path) -> None:
    record = valid_sft_record()
    record["provenance"] = provenance(review_status="rubber_stamped")
    path = write_jsonl(tmp_path / "bad_review.jsonl", [record])

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    assert "invalid review_status" in result.errors[0].message


def test_validate_jsonl_rejects_directory_path(tmp_path: Path) -> None:
    result = validate_jsonl(tmp_path, parse_dataset_type("sft"))

    assert not result.ok
    assert "path is not a file" in result.errors[0].message


def test_validate_jsonl_rejects_eval_leakage_in_training_dataset(
    tmp_path: Path,
) -> None:
    record = {
        "id": "eval_001",
        "dataset_type": "eval",
        "category": "unsafe_refusal",
        "prompt": "評価してください。",
        "expected_points": ["拒否すべき"],
        "provenance": provenance(source_type="eval_only"),
    }
    path = write_jsonl(tmp_path / "leakage.jsonl", [record])

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    assert any("dataset_type mismatch" in error.message for error in result.errors)
    assert any("not eligible for training" in error.message for error in result.errors)


def test_validate_jsonl_rejects_unapproved_training_record(tmp_path: Path) -> None:
    record = valid_sft_record()
    record["provenance"] = provenance(review_status="needs_edit")
    path = write_jsonl(tmp_path / "needs_edit.jsonl", [record])

    result = validate_jsonl(path, parse_dataset_type("sft"))

    assert not result.ok
    assert "not eligible for training" in result.errors[0].message


def test_validate_dataset_cli_success(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "sft.jsonl", [valid_sft_record()])

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

    assert result.returncode == 0
    assert "OK: 1 sft record" in result.stdout


def test_validate_dataset_cli_works_from_uninstalled_source_checkout(
    tmp_path: Path,
) -> None:
    path = write_jsonl(tmp_path / "sft.jsonl", [valid_sft_record()])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/validate_dataset.py").resolve()),
            "--dataset-type",
            "sft",
            str(path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: 1 sft record" in result.stdout


def test_validate_dataset_cli_failure(tmp_path: Path) -> None:
    record = valid_sft_record()
    record["provenance"] = provenance(
        source_type="raw_ai_output",
        review_status="approved",
        raw_ai_output_used_as_training_target=True,
    )
    path = write_jsonl(tmp_path / "bad.jsonl", [record])

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
    assert "line 1:" in result.stderr
    assert "not eligible for training" in result.stderr


def test_parse_dataset_type_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="dataset type must be one of"):
        parse_dataset_type("ranking")
