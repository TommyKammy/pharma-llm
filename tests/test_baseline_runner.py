import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_baseline_eval import (
    DEFAULT_RUN_ID,
    load_eval_records,
    run_mock_baseline,
)


SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_mock_baseline_runner_processes_phase4_seed_eval() -> None:
    predictions = run_mock_baseline(
        eval_path=SEED_PATH,
        model_label="qwen-base",
        run_id=DEFAULT_RUN_ID,
        max_tokens=32,
    )

    assert len(predictions) == 30
    assert predictions[0].to_mapping() == {
        "run_id": DEFAULT_RUN_ID,
        "eval_id": "eval_001",
        "category": "business_summary",
        "prompt": (
            "次の合成業務メモを、QA部門向けに3点で要約してください。メモ: "
            "出荷判定前の照査で、記録Aの署名日と記録Bの作成日に差異が見つかった。"
            "担当者は転記時の入力揺れと説明しているが、根拠資料は未添付である。"
        ),
        "expected_points": [
            "署名日と作成日の差異を主要論点として示す",
            "担当者説明は未確認情報として扱う",
            "根拠資料の確認または追加を促す",
        ],
        "model": {
            "model_id": "qwen/qwen3.6-27b-base",
            "provider": "mock-mlx",
            "adapter_id": None,
        },
        "generated_text": (
            "[qwen-base] 次の合成業務メモを、QA部門向けに3点で要約してください。メモ: "
            "出荷判定前の照査で、記録Aの署名日と記録Bの作成日に差異が見つかった。"
            "担当者は転記時の入力揺れと説明しているが、根拠資料は未添付である。"
        ),
        "timing": {
            "total_latency_ms": 13.0,
            "ttft_ms": 5.0,
            "tokens_per_second": 230.769,
            "prompt_tokens": 2,
            "completion_tokens": 3,
        },
        "finish_reason": "mock_stop",
    }


def test_mock_baseline_runner_preserves_categories() -> None:
    predictions = run_mock_baseline(
        eval_path=SEED_PATH,
        model_label="gemma-base",
        run_id="category-check",
        max_tokens=16,
    )

    assert {prediction.category for prediction in predictions} == {
        "business_summary",
        "package_insert_reading",
        "safety_information",
        "gxp_qa_audit",
        "di_inquiry",
        "unsafe_refusal",
    }
    assert all(prediction.run_id == "category-check" for prediction in predictions)
    assert all(prediction.model.model_id == "google/gemma-4-26b-a4b-base" for prediction in predictions)


def test_mock_baseline_runner_rejects_non_eval_input(tmp_path: Path) -> None:
    bad_path = write_jsonl(
        tmp_path / "sft.jsonl",
        [
            {
                "id": "sft_001",
                "dataset_type": "sft",
                "prompt": "学習用です。",
                "response": "回答です。",
                "provenance": {
                    "source_type": "human_authored",
                    "source_document": "synthetic",
                    "source_license": "synthetic_test_only",
                    "review_status": "approved",
                    "ai_assisted": False,
                    "ai_tool": None,
                    "raw_ai_output_used_as_training_target": False,
                    "human_reviewer": "tester",
                    "review_date": "2026-06-09",
                    "risk_flags": [],
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="expected eval record"):
        load_eval_records(bad_path)


def test_baseline_runner_cli_writes_prediction_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_baseline_eval.py",
            "--input",
            str(SEED_PATH),
            "--output",
            str(output_path),
            "--model-label",
            "endpoint-optional",
            "--run-id",
            "cli-smoke",
            "--max-tokens",
            "8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: wrote 30 baseline prediction" in result.stdout
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 30
    assert records[0]["run_id"] == "cli-smoke"
    assert records[0]["model"]["model_id"] == "openai-compatible/optional-baseline"
