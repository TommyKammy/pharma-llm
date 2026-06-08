import json
import subprocess
import sys
from pathlib import Path

from scripts.create_argilla_sample import sample_records
from scripts.export_to_argilla import export_records
from scripts.import_from_argilla import import_records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reviewed_payload(payload: dict[str, object], **review_updates: object) -> dict[str, object]:
    review = dict(payload["review"])  # type: ignore[arg-type]
    review.update(review_updates)
    return {**payload, "review": review}


def with_reviewed_fields(
    payload: dict[str, object],
    **field_updates: object,
) -> dict[str, object]:
    fields = dict(payload["fields"])  # type: ignore[arg-type]
    fields.update(field_updates)
    return {**payload, "fields": fields}


def test_export_to_argilla_preserves_review_payload_shape(tmp_path: Path) -> None:
    input_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    output_path = tmp_path / "review.jsonl"

    count = export_records(input_path, output_path)
    payloads = read_jsonl(output_path)

    assert count == len(sample_records())
    assert payloads[0]["id"] == "phase3_argilla_sample_001"
    assert payloads[0]["fields"] == {
        "prompt": "安全性情報をQA向けに要約してください。",
        "response": "原文確認を前提に、既知情報と未確認情報を分けて要約します。",
    }
    assert payloads[0]["provenance"]["source_license"] == "synthetic_test_only"
    assert payloads[0]["original_record"]["id"] == payloads[0]["id"]
    assert payloads[0]["argilla"]["workspace"] == "pharma-llm-local-review"


def test_import_from_argilla_applies_review_metadata(tmp_path: Path) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = read_jsonl(review_path)[1]
    reviewed = with_reviewed_fields(
        reviewed_payload(
            payload,
            review_status="edited_and_approved",
            human_reviewer="reviewer_a",
            review_date="2026-06-08",
            risk_flags=["medical_advice", "edited"],
            review_notes="Synthetic candidate was edited before approval.",
        ),
        response="服薬変更は担当医療者に確認し、一般情報に限定して説明します。",
    )
    write_jsonl(review_path, [reviewed])

    count = import_records(review_path, imported_path)
    imported = read_jsonl(imported_path)

    assert count == 1
    assert "argilla" not in imported[0]
    assert imported[0]["id"] == "phase3_argilla_sample_002"
    assert imported[0]["response"] == "服薬変更は担当医療者に確認し、一般情報に限定して説明します。"
    assert imported[0]["provenance"]["source_type"] == "human_edited_ai_assisted"
    assert imported[0]["provenance"]["review_status"] == "edited_and_approved"
    assert imported[0]["provenance"]["human_reviewer"] == "reviewer_a"
    assert imported[0]["provenance"]["review_date"] == "2026-06-08"
    assert imported[0]["provenance"]["risk_flags"] == ["medical_advice", "edited"]


def test_import_from_argilla_rejects_raw_ai_output_edited_approval(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[4],
        review_status="edited_and_approved",
        human_reviewer="reviewer_a",
        review_date="2026-06-08",
    )
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "raw_ai_output cannot be marked edited_and_approved" in result.stderr


def test_import_from_argilla_rejects_approved_unreviewed_ai_candidate(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[1],
        review_status="approved",
        human_reviewer="reviewer_a",
        review_date="2026-06-08",
    )
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "ai_candidate_unreviewed requires edited_and_approved review" in result.stderr


def test_import_from_argilla_rejects_unexpected_review_field_keys(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = with_reviewed_fields(read_jsonl(review_path)[0], id="corrupted_id")
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "review fields contain unsupported key(s): 'id'" in result.stderr


def test_import_from_argilla_requires_risk_flags_for_risk_flagged_reviews(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[0],
        review_status="risk_flagged",
        human_reviewer="reviewer_a",
        review_date="2026-06-08",
        risk_flags=[],
    )
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "risk_flagged reviews must include at least one risk flag" in result.stderr


def test_import_from_argilla_rejects_mismatched_payload_identity(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = read_jsonl(review_path)[0]
    payload["id"] = "wrong_review_row"
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "payload id does not match original_record id" in result.stderr


def test_import_from_argilla_rejects_mutated_immutable_provenance(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = read_jsonl(review_path)[4]
    payload["original_record"]["provenance"]["raw_ai_output_used_as_training_target"] = False
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "original_record provenance does not match exported provenance"
        in result.stderr
    )
    assert "raw_ai_output_used_as_training_target" in result.stderr


def test_import_from_argilla_rejects_missing_review_fields(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = read_jsonl(review_path)[1]
    del payload["fields"]["response"]
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "review fields are missing required key(s): 'response'" in result.stderr


def test_import_from_argilla_rejects_invalid_review_status(tmp_path: Path) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(read_jsonl(review_path)[0], review_status="rubber_stamped")
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "invalid review_status 'rubber_stamped'" in result.stderr


def test_import_from_argilla_requires_reviewer_for_approved_records(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[0],
        review_status="approved",
        human_reviewer=None,
        review_date="2026-06-08",
    )
    write_jsonl(review_path, [payload])

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "human_reviewer must be a non-empty string" in result.stderr


def test_export_and_import_cli_work_from_uninstalled_source_checkout(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", [sample_records()[0]])
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"

    export_result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/export_to_argilla.py").resolve()),
            str(candidate_path),
            str(review_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert export_result.returncode == 0
    assert "OK: exported 1 review record" in export_result.stdout

    import_result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/import_from_argilla.py").resolve()),
            str(review_path),
            str(imported_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert import_result.returncode == 0
    assert "OK: imported 1 reviewed record" in import_result.stdout
    assert read_jsonl(imported_path)[0]["id"] == "phase3_argilla_sample_001"
