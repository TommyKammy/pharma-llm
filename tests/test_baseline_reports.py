import json
import subprocess
import sys
from pathlib import Path

import pytest

from pharma_llm_lab.baseline import (
    BaselineResultError,
    build_baseline_report,
    load_report_inputs,
)
from scripts.run_baseline_eval import run_mock_baseline, write_predictions

SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


def write_mock_predictions(
    path: Path,
    *,
    model_label: str,
    run_id: str,
) -> Path:
    write_predictions(
        path,
        run_mock_baseline(
            eval_path=SEED_PATH,
            model_label=model_label,
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
        "run_id": "fixture-run",
        "eval_id": "eval_001",
        "category": "business_summary",
        "model": {
            "model_id": "qwen/qwen3.6-27b-base",
            "provider": "mock-mlx",
            "adapter_id": None,
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


def test_baseline_report_generates_markdown_for_fixture_models(tmp_path: Path) -> None:
    qwen_path = write_mock_predictions(
        tmp_path / "qwen.jsonl",
        model_label="qwen-base",
        run_id="qwen-fixture",
    )
    gemma_path = write_mock_predictions(
        tmp_path / "gemma.jsonl",
        model_label="gemma-base",
        run_id="gemma-fixture",
    )

    report = build_baseline_report(load_report_inputs((qwen_path, gemma_path)))

    assert report.startswith("# Baseline Evaluation Report\n")
    assert "|qwen/qwen3.6-27b-base|mock-mlx|none|qwen-fixture|30|unscored: 30|" in report
    assert (
        "|google/gemma-4-26b-a4b-base|mock-mlx|none|gemma-fixture|30|unscored: 30|"
        in report
    )
    assert "|qwen/qwen3.6-27b-base|business_summary|5|" in report
    assert "## Notable Failure Modes" in report
    assert "CI-safe mock or fixture data" in report
    assert "```markdown\n- qwen/qwen3.6-27b-base: 30 evals" in report
    assert "## Phase 6 Interpretation" in report


def test_baseline_report_rejects_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(BaselineResultError, match="at least one"):
        load_report_inputs(())

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(BaselineResultError, match="no baseline results"):
        load_report_inputs((empty_path,))


def test_baseline_report_rejects_malformed_metrics(tmp_path: Path) -> None:
    malformed_path = write_jsonl(
        tmp_path / "bad-metrics.jsonl",
        [
            prediction_record(
                timing={
                    "total_latency_ms": True,
                    "ttft_ms": 1.0,
                    "tokens_per_second": 2.0,
                }
            )
        ],
    )

    with pytest.raises(BaselineResultError, match="timing.total_latency_ms"):
        load_report_inputs((malformed_path,))


def test_baseline_report_cli_writes_markdown(tmp_path: Path) -> None:
    qwen_path = write_mock_predictions(
        tmp_path / "qwen.jsonl",
        model_label="qwen-base",
        run_id="cli-qwen",
    )
    report_path = tmp_path / "baseline_report.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_baseline_report.py",
            "--input",
            str(qwen_path),
            "--output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: wrote baseline report for 1 model" in result.stdout
    report = report_path.read_text(encoding="utf-8")
    assert "# Baseline Evaluation Report" in report
    assert "qwen/qwen3.6-27b-base" in report
    assert "business_summary" in report
