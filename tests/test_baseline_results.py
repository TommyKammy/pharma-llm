import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

from pharma_llm_lab.baseline import (
    BaselineResultError,
    aggregate_results,
    load_baseline_results,
    write_category_metrics_csv,
    write_summary_json,
)
from scripts.run_baseline_eval import run_mock_baseline, write_predictions


SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


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
        "prompt": "評価用プロンプトです。",
        "expected_points": ["観点1", "観点2", "観点3"],
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
            "prompt_tokens": 4,
            "completion_tokens": 2,
        },
        "finish_reason": "mock_stop",
        "scoring_status": "unscored",
    }
    record.update(updates)
    return record


def test_baseline_results_validate_and_aggregate_seed_fixture(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    write_predictions(
        prediction_path,
        run_mock_baseline(
            eval_path=SEED_PATH,
            model_label="qwen-base",
            run_id="fixture-run",
            max_tokens=8,
        ),
    )

    results = load_baseline_results(prediction_path)
    summary = aggregate_results(results)

    assert summary.run_id == "fixture-run"
    assert summary.model_id == "qwen/qwen3.6-27b-base"
    assert summary.total_count == 30
    assert summary.scoring_status_counts == {"unscored": 30}
    assert {metrics.category.value: metrics.count for metrics in summary.category_metrics} == {
        "business_summary": 5,
        "package_insert_reading": 5,
        "safety_information": 5,
        "gxp_qa_audit": 5,
        "di_inquiry": 5,
        "unsafe_refusal": 5,
    }
    assert all(metrics.avg_total_latency_ms > 0 for metrics in summary.category_metrics)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.pop("eval_id"), "eval_id must be a non-empty string"),
        (lambda record: record.pop("category"), "category must be a non-empty string"),
        (lambda record: record["model"].pop("model_id"), "model_id must be a non-empty string"),
        (lambda record: record.pop("generated_text"), "generated_text must be a non-empty string"),
        (
            lambda record: record["timing"].update({"total_latency_ms": -1}),
            "timing.total_latency_ms must be a finite non-negative number",
        ),
    ],
)
def test_baseline_results_reject_invalid_prediction_records(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    record = prediction_record()
    mutate(record)
    path = write_jsonl(tmp_path / "bad.jsonl", [record])

    with pytest.raises(BaselineResultError, match=message):
        load_baseline_results(path)


def test_baseline_results_reject_mixed_run_or_model(tmp_path: Path) -> None:
    mixed_run_path = write_jsonl(
        tmp_path / "mixed.jsonl",
        [
            prediction_record(eval_id="eval_001"),
            prediction_record(eval_id="eval_002", run_id="other-run"),
        ],
    )

    with pytest.raises(BaselineResultError, match="exactly one run_id"):
        aggregate_results(load_baseline_results(mixed_run_path))

    mixed_model_path = write_jsonl(
        tmp_path / "mixed-model.jsonl",
        [
            prediction_record(eval_id="eval_001"),
            prediction_record(
                eval_id="eval_002",
                model={"model_id": "google/gemma-4-26b-a4b-base", "provider": "mock-mlx"},
            ),
        ],
    )

    with pytest.raises(BaselineResultError, match="exactly one model_id"):
        aggregate_results(load_baseline_results(mixed_model_path))


def test_baseline_results_write_summary_json_and_category_csv(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            prediction_record(eval_id="eval_001", category="business_summary"),
            prediction_record(
                eval_id="eval_101",
                category="safety_information",
                timing={
                    "total_latency_ms": 30.0,
                    "ttft_ms": 10.0,
                    "tokens_per_second": 50.0,
                },
            ),
        ],
    )
    summary = aggregate_results(load_baseline_results(path))
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "category_metrics.csv"

    write_summary_json(summary_path, summary)
    write_category_metrics_csv(csv_path, summary)

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_data["total_count"] == 2
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert [row["category"] for row in rows] == [
        "business_summary",
        "safety_information",
    ]


def test_baseline_summary_cli_writes_summary_and_csv(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "category_metrics.csv"
    write_predictions(
        prediction_path,
        run_mock_baseline(
            eval_path=SEED_PATH,
            model_label="gemma-base",
            run_id="cli-summary",
            max_tokens=8,
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_baseline_results.py",
            "--input",
            str(prediction_path),
            "--summary-output",
            str(summary_path),
            "--category-csv-output",
            str(csv_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: summarized 30 baseline result" in result.stdout
    assert json.loads(summary_path.read_text(encoding="utf-8"))["model_id"] == (
        "google/gemma-4-26b-a4b-base"
    )
    assert csv_path.exists()
