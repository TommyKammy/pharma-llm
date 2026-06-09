import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.create_argilla_sample import sample_records
from scripts.export_sft_v0_1 import DATASET_VERSION, export_sft_v0_1


SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


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


def approved_records() -> list[dict[str, object]]:
    records = sample_records()
    return [
        records[0],
        {
            **records[1],
            "original_record": records[1],
            "response": "服薬変更は担当医療者に確認し、一般情報に限定して説明します。",
            "provenance": {
                **records[1]["provenance"],
                "source_type": "human_edited_ai_assisted",
                "review_status": "edited_and_approved",
                "raw_ai_output_used_as_training_target": False,
                "human_reviewer": "reviewer_a",
                "review_date": "2026-06-08",
                "risk_flags": ["medical_advice", "edited"],
            },
        },
    ]


def test_export_sft_v0_1_writes_prepared_dataset_and_manifest(tmp_path: Path) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", approved_records())
    output_path = tmp_path / "sft_v0_1.jsonl"
    manifest_path = tmp_path / "sft_v0_1.manifest.json"

    manifest = export_sft_v0_1(
        input_path=reviewed_path,
        output_path=output_path,
        manifest_path=manifest_path,
        eval_path=SEED_PATH,
    )
    prepared = read_jsonl(output_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [record["id"] for record in prepared] == [
        "phase3_argilla_sample_001",
        "phase3_argilla_sample_002",
    ]
    assert all(record["dataset_type"] == "sft" for record in prepared)
    assert all("argilla" not in record for record in prepared)
    assert all("original_record" not in record for record in prepared)
    assert manifest.dataset_version == DATASET_VERSION
    assert manifest_data["source_count"] == 2
    assert manifest_data["promoted_count"] == 2
    assert manifest_data["approved_count"] == 1
    assert manifest_data["edited_and_approved_count"] == 1
    assert manifest_data["eval_count"] == 30
    assert len(manifest_data["eval_id_sha256"]) == 64
    assert len(manifest_data["output_sha256"]) == 64
    assert "Raw exports" in manifest_data["local_artifact_policy"]


@pytest.mark.parametrize(
    ("record_update", "message"),
    [
        (
            {"candidate_status": "review_candidate"},
            "review candidates are not accepted",
        ),
        (
            {
                "provenance": {
                    **sample_records()[0]["provenance"],
                    "review_status": "unreviewed",
                }
            },
            "review_status 'unreviewed' is not approved",
        ),
        (
            {
                "dataset_type": "eval",
                "category": "business_summary",
                "expected_points": ["x"],
                "provenance": {
                    **sample_records()[0]["provenance"],
                    "source_type": "eval_only",
                },
            },
            "dataset_type mismatch",
        ),
    ],
)
def test_export_sft_v0_1_rejects_policy_violations(
    tmp_path: Path,
    record_update: dict[str, object],
    message: str,
) -> None:
    record = {**sample_records()[0], **record_update}
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [record])
    output_path = tmp_path / "sft_v0_1.jsonl"
    manifest_path = tmp_path / "sft_v0_1.manifest.json"

    with pytest.raises(ValueError, match=message):
        export_sft_v0_1(
            input_path=reviewed_path,
            output_path=output_path,
            manifest_path=manifest_path,
            eval_path=SEED_PATH,
        )

    assert not output_path.exists()
    assert not manifest_path.exists()


def test_export_sft_v0_1_rejects_duplicate_ids(tmp_path: Path) -> None:
    records = approved_records()
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", [records[0], records[0]])
    output_path = tmp_path / "sft_v0_1.jsonl"
    manifest_path = tmp_path / "sft_v0_1.manifest.json"

    with pytest.raises(ValueError, match="duplicate id"):
        export_sft_v0_1(
            input_path=reviewed_path,
            output_path=output_path,
            manifest_path=manifest_path,
            eval_path=SEED_PATH,
        )

    assert not output_path.exists()
    assert not manifest_path.exists()


def test_export_sft_v0_1_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input path is not a file"):
        export_sft_v0_1(
            input_path=tmp_path / "missing.jsonl",
            output_path=tmp_path / "sft_v0_1.jsonl",
            manifest_path=tmp_path / "sft_v0_1.manifest.json",
            eval_path=SEED_PATH,
        )


def test_export_sft_v0_1_cli_writes_manifest(tmp_path: Path) -> None:
    reviewed_path = write_jsonl(tmp_path / "reviewed.jsonl", approved_records())
    output_path = tmp_path / "sft_v0_1.jsonl"
    manifest_path = tmp_path / "sft_v0_1.manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_sft_v0_1.py",
            str(reviewed_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--eval-path",
            str(SEED_PATH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: exported 2 SFT v0.1 record" in result.stdout
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_version"] == (
        DATASET_VERSION
    )
    assert len(read_jsonl(output_path)) == 2
    assert manifest_path.exists()
