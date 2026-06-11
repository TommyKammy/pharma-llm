import json
import subprocess
import sys
from pathlib import Path

import pytest

from pharma_llm_lab.baseline import (
    BaselineResultError,
    build_lora_comparison_report,
    load_lora_comparison_inputs,
)
from scripts.run_baseline_eval import run_mock_baseline, write_predictions
from scripts.run_lora_eval import DEFAULT_ADAPTER_ID, run_mock_lora_eval

SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


def write_mock_base_predictions(path: Path, *, run_id: str = "base-fixture") -> Path:
    write_predictions(
        path,
        run_mock_baseline(
            eval_path=SEED_PATH,
            model_label="qwen-base",
            run_id=run_id,
            max_tokens=8,
        ),
    )
    return path


def write_mock_lora_predictions(path: Path, *, run_id: str = "lora-fixture") -> Path:
    write_predictions(
        path,
        run_mock_lora_eval(
            eval_path=SEED_PATH,
            model_id="qwen/qwen3.6-27b-base",
            adapter_id=DEFAULT_ADAPTER_ID,
            run_id=run_id,
            max_tokens=8,
        ),
    )
    return path


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def prediction_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "run_id": "lora-fixture",
        "eval_id": "eval_001",
        "category": "business_summary",
        "model": {
            "model_id": "qwen/qwen3.6-27b-base",
            "provider": "mock-mlx",
            "adapter_id": DEFAULT_ADAPTER_ID,
        },
        "generated_text": "fixture answer",
        "timing": {
            "total_latency_ms": 20.0,
            "ttft_ms": 5.0,
            "tokens_per_second": 100.0,
        },
        "scoring_status": "unscored",
    }
    record.update(updates)
    return record


def test_lora_comparison_report_generates_markdown(tmp_path: Path) -> None:
    base_path = write_mock_base_predictions(tmp_path / "base.jsonl")
    lora_path = write_mock_lora_predictions(tmp_path / "lora.jsonl")

    comparison = load_lora_comparison_inputs(base_path=base_path, lora_path=lora_path)
    report = build_lora_comparison_report(comparison)

    assert report.startswith("# Base vs LoRA Evaluation and Safety Report\n")
    assert "|base|qwen/qwen3.6-27b-base|mock-mlx|none|base-fixture|" in report
    assert f"|lora|qwen/qwen3.6-27b-base|mock-mlx|{DEFAULT_ADAPTER_ID}|" in report
    assert "- Matched eval count: 30" in report
    assert "## Category Deltas" in report
    assert "|safety_information|5|" in report
    assert "Delta Latency ms" in report
    assert "## Safety and Style Regression Notes" in report
    assert "manual-review-only or partial scoring: unscored" in report
    assert "configs/promptfoo/lora_comparison_mock.yaml" in report
    assert "## Phase 7 Interpretation Rules" in report


def test_lora_comparison_rejects_mismatched_eval_id_sets(tmp_path: Path) -> None:
    base_path = write_mock_base_predictions(tmp_path / "base.jsonl")
    lora_path = write_jsonl(
        tmp_path / "lora-subset.jsonl",
        [prediction_record(eval_id="eval_999")],
    )

    with pytest.raises(BaselineResultError, match="same eval_id set"):
        load_lora_comparison_inputs(base_path=base_path, lora_path=lora_path)


def test_lora_comparison_rejects_base_adapter_identity(tmp_path: Path) -> None:
    base_path = write_jsonl(
        tmp_path / "base-with-adapter.jsonl",
        [
            prediction_record(
                run_id="base-fixture",
                model={
                    "model_id": "qwen/qwen3.6-27b-base",
                    "provider": "mock-mlx",
                    "adapter_id": "unexpected-adapter",
                },
            )
        ],
    )
    lora_path = write_jsonl(tmp_path / "lora.jsonl", [prediction_record()])

    with pytest.raises(BaselineResultError, match="base predictions must not include"):
        load_lora_comparison_inputs(base_path=base_path, lora_path=lora_path)


def test_lora_comparison_cli_writes_markdown(tmp_path: Path) -> None:
    base_path = write_mock_base_predictions(tmp_path / "base.jsonl", run_id="cli-base")
    lora_path = write_mock_lora_predictions(tmp_path / "lora.jsonl", run_id="cli-lora")
    report_path = tmp_path / "lora_report.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_lora_comparison_report.py",
            "--base-input",
            str(base_path),
            "--lora-input",
            str(lora_path),
            "--output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: wrote LoRA comparison report" in result.stdout
    report = report_path.read_text(encoding="utf-8")
    assert "# Base vs LoRA Evaluation and Safety Report" in report
    assert "cli-base" in report
    assert "cli-lora" in report


def test_default_lora_comparison_report_path_is_trackable() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "results/reports/lora_comparison_report.md"],
        check=False,
    )

    assert result.returncode == 1
