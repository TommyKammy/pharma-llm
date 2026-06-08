import json
import subprocess
import sys
from pathlib import Path

from scripts.create_argilla_sample import sample_records
from scripts.export_to_argilla import export_records
from scripts.import_from_argilla import import_records
from scripts.promote_reviewed_dataset import promote_reviewed_dataset


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
    assert not imported[0]["provenance"]["raw_ai_output_used_as_training_target"]
    assert imported[0]["provenance"]["human_reviewer"] == "reviewer_a"
    assert imported[0]["provenance"]["review_date"] == "2026-06-08"
    assert imported[0]["provenance"]["risk_flags"] == ["medical_advice", "edited"]
    assert imported[0]["provenance"]["target_fields_edited"] is True


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


def test_import_from_argilla_rejects_raw_ai_output_approval(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[4],
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
    assert "raw_ai_output cannot be marked approved" in result.stderr


def test_import_from_argilla_preserves_existing_risk_flags_when_review_omits_them(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = with_reviewed_fields(
        reviewed_payload(
            read_jsonl(review_path)[1],
            review_status="edited_and_approved",
            human_reviewer="reviewer_a",
            review_date="2026-06-08",
        ),
        response="服薬変更は担当医療者に確認し、一般情報に限定して説明します。",
    )
    del payload["review"]["risk_flags"]
    write_jsonl(review_path, [payload])

    count = import_records(review_path, imported_path)
    imported = read_jsonl(imported_path)

    assert count == 1
    assert imported[0]["provenance"]["risk_flags"] == ["medical_advice"]


def test_import_from_argilla_falls_back_to_exported_risk_flags(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = with_reviewed_fields(
        reviewed_payload(
            read_jsonl(review_path)[1],
            review_status="edited_and_approved",
            human_reviewer="reviewer_a",
            review_date="2026-06-08",
        ),
        response="服薬変更は担当医療者に確認し、一般情報に限定して説明します。",
    )
    payload["original_record"]["provenance"]["risk_flags"] = ["tampered_flag"]
    del payload["review"]["risk_flags"]
    write_jsonl(review_path, [payload])

    count = import_records(review_path, imported_path)
    imported = read_jsonl(imported_path)

    assert count == 1
    assert imported[0]["provenance"]["risk_flags"] == ["medical_advice"]


def test_import_from_argilla_clears_raw_target_flag_for_edited_ai_candidates(
    tmp_path: Path,
) -> None:
    candidate = sample_records()[1]
    candidate["provenance"]["raw_ai_output_used_as_training_target"] = True
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", [candidate])
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = with_reviewed_fields(
        reviewed_payload(
            read_jsonl(review_path)[0],
            review_status="edited_and_approved",
            human_reviewer="reviewer_a",
            review_date="2026-06-08",
            risk_flags=["medical_advice", "edited"],
        ),
        response="服薬変更は担当医療者に確認し、一般情報に限定して説明します。",
    )
    write_jsonl(review_path, [payload])

    count = import_records(review_path, imported_path)
    imported = read_jsonl(imported_path)

    assert count == 1
    assert imported[0]["provenance"]["source_type"] == "human_edited_ai_assisted"
    assert not imported[0]["provenance"]["raw_ai_output_used_as_training_target"]


def test_import_from_argilla_requires_actual_edits_for_ai_candidates(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[1],
        review_status="edited_and_approved",
        human_reviewer="reviewer_a",
        review_date="2026-06-08",
        risk_flags=["medical_advice", "edited"],
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
    assert "ai_candidate_unreviewed requires edited target fields" in result.stderr


def test_import_from_argilla_rejects_prompt_only_edits_for_ai_candidates(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = with_reviewed_fields(
        reviewed_payload(
            read_jsonl(review_path)[1],
            review_status="edited_and_approved",
            human_reviewer="reviewer_a",
            review_date="2026-06-08",
            risk_flags=["medical_advice", "edited"],
        ),
        prompt="患者向け服薬相談への回答方針を確認してください。",
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
    assert "ai_candidate_unreviewed requires edited target fields" in result.stderr


def test_import_from_argilla_rejects_approved_human_edited_ai_assisted_records(
    tmp_path: Path,
) -> None:
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", sample_records())
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[2],
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
    assert "human_edited_ai_assisted requires edited_and_approved review" in result.stderr


def test_import_from_argilla_rejects_approved_ai_assisted_records(
    tmp_path: Path,
) -> None:
    record = sample_records()[0]
    record["provenance"]["ai_assisted"] = True
    record["provenance"]["ai_tool"] = "codex_app"
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", [record])
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[0],
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
    assert "ai_assisted records require edited_and_approved review" in result.stderr


def test_import_from_argilla_rejects_unedited_ai_assisted_edited_approval(
    tmp_path: Path,
) -> None:
    record = sample_records()[0]
    record["provenance"]["ai_assisted"] = True
    record["provenance"]["ai_tool"] = "codex_app"
    candidate_path = write_jsonl(tmp_path / "candidate.jsonl", [record])
    review_path = tmp_path / "review.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    export_records(candidate_path, review_path)
    payload = reviewed_payload(
        read_jsonl(review_path)[0],
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
    assert (
        "edited_and_approved ai_assisted records require edited target fields"
        in result.stderr
    )


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


def test_promote_reviewed_dataset_writes_only_training_eligible_records(
    tmp_path: Path,
) -> None:
    reviewed_records = [
        sample_records()[0],
        {
            **sample_records()[1],
            "original_record": sample_records()[1],
            "response": "服薬変更は担当医療者に確認し、一般情報に限定して説明します。",
            "provenance": {
                **sample_records()[1]["provenance"],
                "source_type": "human_edited_ai_assisted",
                "review_status": "edited_and_approved",
                "raw_ai_output_used_as_training_target": False,
                "human_reviewer": "reviewer_a",
                "review_date": "2026-06-08",
                "risk_flags": ["medical_advice", "edited"],
            },
        },
        {
            **sample_records()[4],
            "provenance": {
                **sample_records()[4]["provenance"],
                "review_status": "rejected",
            },
        },
        sample_records()[3],
    ]
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", reviewed_records)
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )
    prepared = read_jsonl(prepared_path)

    assert result.ok
    assert [record["id"] for record in prepared] == [
        "phase3_argilla_sample_001",
        "phase3_argilla_sample_002",
    ]
    assert all("argilla" not in record for record in prepared)
    assert all("original_record" not in record for record in prepared)
    assert len(result.promoted) == 2
    assert len(result.skipped) == 2
    assert not result.failed
    assert any(
        "source_type 'raw_ai_output' is blocked" in entry.reason
        for entry in result.skipped
    )
    assert any("dataset_type mismatch" in entry.reason for entry in result.skipped)


def test_promote_reviewed_dataset_fails_without_writing_on_schema_errors(
    tmp_path: Path,
) -> None:
    bad_record = sample_records()[0]
    del bad_record["provenance"]
    reviewed_path = write_jsonl(tmp_path / "reviewed_bad.jsonl", [bad_record])
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert result.failed[0].id == "phase3_argilla_sample_001"
    assert "provenance must be an object" in result.failed[0].reason


def test_promote_reviewed_dataset_does_not_audit_promoted_on_mixed_failure(
    tmp_path: Path,
) -> None:
    bad_record = {
        **sample_records()[1],
        "provenance": None,
    }
    reviewed_path = write_jsonl(
        tmp_path / "reviewed_mixed.jsonl",
        [sample_records()[0], bad_record],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert not result.promoted_records
    assert not result.promoted
    assert [entry.id for entry in result.failed] == [
        "phase3_argilla_sample_002",
        "phase3_argilla_sample_001",
    ]
    assert "provenance must be an object" in result.failed[0].reason
    assert "eligible record was not promoted" in result.failed[1].reason


def test_promote_reviewed_dataset_fails_when_no_records_are_promoted(
    tmp_path: Path,
) -> None:
    reviewed_path = write_jsonl(
        tmp_path / "reviewed_skipped.jsonl",
        [
            {
                **sample_records()[4],
                "provenance": {
                    **sample_records()[4]["provenance"],
                    "review_status": "rejected",
                },
            }
        ],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert not result.promoted
    assert result.skipped


def test_promote_reviewed_dataset_skips_plain_approved_ai_assisted_records(
    tmp_path: Path,
) -> None:
    ai_assisted_record = {
        **sample_records()[0],
        "provenance": {
            **sample_records()[0]["provenance"],
            "source_type": "public_doc_derived",
            "review_status": "approved",
            "ai_assisted": True,
            "ai_tool": "codex_app",
            "human_reviewer": "reviewer_a",
            "review_date": "2026-06-08",
        },
    }
    reviewed_path = write_jsonl(tmp_path / "reviewed_ai_assisted.jsonl", [ai_assisted_record])
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert not result.promoted
    assert result.skipped[0].id == "phase3_argilla_sample_001"
    assert "ai_assisted records require edited_and_approved review" in result.skipped[0].reason


def test_promote_reviewed_dataset_skips_approved_human_edited_ai_assisted_source(
    tmp_path: Path,
) -> None:
    human_edited_record = {
        **sample_records()[0],
        "provenance": {
            **sample_records()[0]["provenance"],
            "source_type": "human_edited_ai_assisted",
            "review_status": "approved",
            "ai_assisted": False,
            "ai_tool": None,
            "human_reviewer": "reviewer_a",
            "review_date": "2026-06-08",
        },
    }
    reviewed_path = write_jsonl(
        tmp_path / "reviewed_human_edited.jsonl",
        [human_edited_record],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert not result.promoted
    assert result.skipped[0].id == "phase3_argilla_sample_001"
    assert (
        "human_edited_ai_assisted requires edited_and_approved review"
        in result.skipped[0].reason
    )


def test_promote_reviewed_dataset_skips_reviewed_records_without_metadata(
    tmp_path: Path,
) -> None:
    missing_metadata_record = {
        **sample_records()[0],
        "provenance": {
            **sample_records()[0]["provenance"],
            "human_reviewer": None,
            "review_date": None,
        },
    }
    reviewed_path = write_jsonl(
        tmp_path / "reviewed_missing_metadata.jsonl",
        [missing_metadata_record],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert not result.promoted
    assert result.skipped[0].id == "phase3_argilla_sample_001"
    assert "reviewed records require human_reviewer" in result.skipped[0].reason


def test_promote_reviewed_dataset_skips_ai_assisted_without_human_edit_source(
    tmp_path: Path,
) -> None:
    ai_assisted_record = {
        **sample_records()[0],
        "provenance": {
            **sample_records()[0]["provenance"],
            "source_type": "public_doc_derived",
            "review_status": "edited_and_approved",
            "ai_assisted": True,
            "ai_tool": "codex_app",
            "human_reviewer": "reviewer_a",
            "review_date": "2026-06-08",
        },
    }
    reviewed_path = write_jsonl(
        tmp_path / "reviewed_ai_assisted_unedited.jsonl",
        [ai_assisted_record],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert not result.promoted
    assert result.skipped[0].id == "phase3_argilla_sample_001"
    assert "human_edited_ai_assisted source_type" in result.skipped[0].reason


def test_promote_reviewed_dataset_skips_ai_assisted_without_edit_evidence(
    tmp_path: Path,
) -> None:
    ai_assisted_record = {
        **sample_records()[0],
        "provenance": {
            **sample_records()[0]["provenance"],
            "source_type": "human_edited_ai_assisted",
            "review_status": "edited_and_approved",
            "ai_assisted": True,
            "ai_tool": "codex_app",
            "human_reviewer": "reviewer_a",
            "review_date": "2026-06-08",
        },
    }
    reviewed_path = write_jsonl(
        tmp_path / "reviewed_ai_assisted_without_edit_evidence.jsonl",
        [ai_assisted_record],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not prepared_path.exists()
    assert not result.promoted
    assert result.skipped[0].id == "phase3_argilla_sample_001"
    assert "edited target fields" in result.skipped[0].reason


def test_promote_reviewed_dataset_accepts_imported_ai_assisted_edit_evidence(
    tmp_path: Path,
) -> None:
    imported_record = {
        **sample_records()[0],
        "response": "人間が編集した安全な回答です。",
        "provenance": {
            **sample_records()[0]["provenance"],
            "source_type": "human_edited_ai_assisted",
            "review_status": "edited_and_approved",
            "ai_assisted": True,
            "ai_tool": "codex_app",
            "human_reviewer": "reviewer_a",
            "review_date": "2026-06-08",
            "target_fields_edited": True,
        },
    }
    reviewed_path = write_jsonl(
        tmp_path / "reviewed_imported_ai_assisted.jsonl",
        [imported_record],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )
    prepared = read_jsonl(prepared_path)

    assert result.ok
    assert [record["id"] for record in prepared] == ["phase3_argilla_sample_001"]


def test_promote_reviewed_dataset_removes_stale_output_on_failure(
    tmp_path: Path,
) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [sample_records()[0]])
    prepared_path = tmp_path / "prepared_sft.jsonl"

    first_result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )
    bad_reviewed_path = write_jsonl(
        tmp_path / "reviewed_skipped.jsonl",
        [
            {
                **sample_records()[4],
                "provenance": {
                    **sample_records()[4]["provenance"],
                    "review_status": "rejected",
                },
            }
        ],
    )

    second_result = promote_reviewed_dataset(
        bad_reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert first_result.ok
    assert not second_result.ok
    assert not prepared_path.exists()


def test_promote_reviewed_dataset_rejects_input_output_collision(
    tmp_path: Path,
) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [sample_records()[0]])
    original_reviewed_content = reviewed_path.read_text(encoding="utf-8")

    result = promote_reviewed_dataset(
        reviewed_path,
        reviewed_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert result.failed[0].reason == "input and output paths must differ"
    assert reviewed_path.read_text(encoding="utf-8") == original_reviewed_content


def test_promote_reviewed_dataset_cli_writes_audit_summary(
    tmp_path: Path,
) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [sample_records()[0]])
    prepared_path = tmp_path / "prepared_sft.jsonl"
    audit_path = tmp_path / "promotion_audit.json"

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/promote_reviewed_dataset.py").resolve()),
            "--dataset-type",
            "sft",
            "--audit-output",
            str(audit_path),
            str(reviewed_path),
            str(prepared_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert "OK: promoted 1 record" in result.stdout
    assert read_jsonl(prepared_path)[0]["id"] == "phase3_argilla_sample_001"
    assert audit["promoted"] == 1
    assert audit["skipped"] == 0
    assert audit["failed"] == 0


def test_promote_reviewed_dataset_cli_rejects_audit_output_collision(
    tmp_path: Path,
) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [sample_records()[0]])
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/promote_reviewed_dataset.py").resolve()),
            "--dataset-type",
            "sft",
            "--audit-output",
            str(prepared_path),
            str(reviewed_path),
            str(prepared_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 2
    assert "--audit-output must not be the same path as output" in result.stderr
    assert not prepared_path.exists()


def test_promote_reviewed_dataset_cli_rejects_audit_input_collision(
    tmp_path: Path,
) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [sample_records()[0]])
    original_reviewed_content = reviewed_path.read_text(encoding="utf-8")
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/promote_reviewed_dataset.py").resolve()),
            "--dataset-type",
            "sft",
            "--audit-output",
            str(reviewed_path),
            str(reviewed_path),
            str(prepared_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 2
    assert "--audit-output must not be the same path as input" in result.stderr
    assert reviewed_path.read_text(encoding="utf-8") == original_reviewed_content
    assert not prepared_path.exists()


def test_promote_reviewed_dataset_cli_rejects_input_output_collision(
    tmp_path: Path,
) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [sample_records()[0]])
    original_reviewed_content = reviewed_path.read_text(encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/promote_reviewed_dataset.py").resolve()),
            "--dataset-type",
            "sft",
            str(reviewed_path),
            str(reviewed_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 2
    assert "input and output paths must differ" in result.stderr
    assert reviewed_path.read_text(encoding="utf-8") == original_reviewed_content


def test_promote_reviewed_dataset_rejects_eval_dataset_type(tmp_path: Path) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [sample_records()[3]])
    prepared_path = tmp_path / "prepared_eval.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/promote_reviewed_dataset.py").resolve()),
            "--dataset-type",
            "eval",
            str(reviewed_path),
            str(prepared_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 2
    assert "promotion dataset type must be one of" in result.stderr
