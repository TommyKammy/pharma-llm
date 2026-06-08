import json
import subprocess
import sys
from pathlib import Path

from scripts.check_eval_leakage import check_eval_leakage
from scripts.promote_reviewed_dataset import promote_reviewed_dataset


def provenance(
    *,
    source_type: str = "human_authored",
    review_status: str = "approved",
) -> dict[str, object]:
    return {
        "source_type": source_type,
        "source_document": "synthetic_leakage_guard_sample",
        "source_license": "synthetic_test_only",
        "review_status": review_status,
        "ai_assisted": False,
        "ai_tool": None,
        "raw_ai_output_used_as_training_target": False,
        "human_reviewer": "leakage_guard_tester",
        "review_date": "2026-06-09",
        "risk_flags": [],
    }


def eval_record(
    *,
    record_id: str = "eval_001",
    prompt: str = "評価専用プロンプトです。",
) -> dict[str, object]:
    return {
        "id": record_id,
        "dataset_type": "eval",
        "category": "business_summary",
        "prompt": prompt,
        "expected_points": ["評価観点1", "評価観点2", "評価観点3"],
        "provenance": provenance(source_type="eval_only"),
    }


def sft_record(
    *,
    record_id: str = "sft_001",
    prompt: str = "学習用プロンプトです。",
    response: str = "学習用回答です。",
    source_type: str = "human_authored",
) -> dict[str, object]:
    return {
        "id": record_id,
        "dataset_type": "sft",
        "prompt": prompt,
        "response": response,
        "provenance": provenance(source_type=source_type),
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_eval_leakage_guard_accepts_distinct_eval_and_training_records(
    tmp_path: Path,
) -> None:
    eval_path = write_jsonl(tmp_path / "eval.jsonl", [eval_record()])
    training_path = write_jsonl(tmp_path / "sft.jsonl", [sft_record()])

    assert check_eval_leakage(
        eval_paths=(eval_path,),
        training_paths=(training_path,),
    ) == ()


def test_eval_leakage_guard_detects_duplicate_ids(tmp_path: Path) -> None:
    eval_path = write_jsonl(tmp_path / "eval.jsonl", [eval_record(record_id="eval_001")])
    training_path = write_jsonl(
        tmp_path / "sft.jsonl",
        [sft_record(record_id="eval_001")],
    )

    findings = check_eval_leakage(
        eval_paths=(eval_path,),
        training_paths=(training_path,),
    )

    assert len(findings) == 1
    assert findings[0].kind == "duplicate_id"
    assert "eval_001" in findings[0].detail


def test_eval_leakage_guard_detects_exact_prompt_reuse_after_normalization(
    tmp_path: Path,
) -> None:
    eval_path = write_jsonl(
        tmp_path / "eval.jsonl",
        [eval_record(prompt="評価専用プロンプトです。  追加確認をしてください。")],
    )
    training_path = write_jsonl(
        tmp_path / "sft.jsonl",
        [sft_record(prompt="評価専用プロンプトです。\n追加確認をしてください。")],
    )

    findings = check_eval_leakage(
        eval_paths=(eval_path,),
        training_paths=(training_path,),
    )

    assert len(findings) == 1
    assert findings[0].kind == "duplicate_text"
    assert "eval prompt duplicates training prompt" in findings[0].detail


def test_eval_leakage_guard_rejects_eval_only_training_record_without_overlap(
    tmp_path: Path,
) -> None:
    eval_path = write_jsonl(tmp_path / "eval.jsonl", [eval_record()])
    training_path = write_jsonl(
        tmp_path / "sft.jsonl",
        [
            sft_record(
                record_id="sft_eval_only_001",
                prompt="別の学習用プロンプトです。",
                source_type="eval_only",
            )
        ],
    )

    try:
        check_eval_leakage(
            eval_paths=(eval_path,),
            training_paths=(training_path,),
        )
    except ValueError as exc:
        assert "source_type 'eval_only' is blocked for training" in str(exc)
    else:
        raise AssertionError("eval_only training record was accepted")


def test_eval_leakage_guard_cli_fails_on_prompt_leakage(tmp_path: Path) -> None:
    eval_path = write_jsonl(tmp_path / "eval.jsonl", [eval_record(prompt="漏洩入力")])
    training_path = write_jsonl(tmp_path / "sft.jsonl", [sft_record(prompt="漏洩入力")])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_eval_leakage.py",
            "--eval",
            str(eval_path),
            "--training",
            str(training_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "FAILED: eval/training leakage detected" in result.stderr
    assert "duplicate_text" in result.stderr


def test_eval_leakage_guard_cli_rejects_policy_invalid_training_input(
    tmp_path: Path,
) -> None:
    eval_path = write_jsonl(tmp_path / "eval.jsonl", [eval_record()])
    training_path = write_jsonl(
        tmp_path / "sft.jsonl",
        [sft_record(source_type="eval_only", prompt="重複していない入力")],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_eval_leakage.py",
            "--eval",
            str(eval_path),
            "--training",
            str(training_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "source_type 'eval_only' is blocked for training" in result.stderr


def test_eval_leakage_guard_cli_accepts_clean_inputs(tmp_path: Path) -> None:
    eval_path = write_jsonl(tmp_path / "eval.jsonl", [eval_record()])
    training_path = write_jsonl(tmp_path / "sft.jsonl", [sft_record()])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_eval_leakage.py",
            "--eval",
            str(eval_path),
            "--training",
            str(training_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: no eval/training leakage detected" in result.stdout


def test_phase3_promotion_path_skips_eval_only_training_record(tmp_path: Path) -> None:
    reviewed_path = write_jsonl(
        tmp_path / "reviewed.jsonl",
        [sft_record(record_id="sft_eval_only_001", source_type="eval_only")],
    )
    prepared_path = tmp_path / "prepared_sft.jsonl"

    result = promote_reviewed_dataset(
        reviewed_path,
        prepared_path,
        dataset_type_value="sft",
    )

    assert not result.ok
    assert not result.promoted
    assert not prepared_path.exists()
    assert "source_type 'eval_only' is blocked for training" in result.skipped[0].reason
