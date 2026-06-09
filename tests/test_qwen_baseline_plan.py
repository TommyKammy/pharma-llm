import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_qwen_baseline import (
    DEFAULT_RUN_ID,
    build_plan,
    eval_id_fingerprint,
)


SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


def test_qwen_baseline_plan_validates_phase4_eval_set(tmp_path: Path) -> None:
    local_root = tmp_path / "local" / "runs" / "baseline"
    plan = build_plan(
        eval_path=SEED_PATH,
        local_root=local_root,
        run_id=DEFAULT_RUN_ID,
        model_path=tmp_path / "local" / "models" / "qwen",
        report_output=Path("results/reports/qwen_base_baseline_report.md"),
        max_tokens=256,
        temperature=0.0,
    )

    assert plan.run_id == DEFAULT_RUN_ID
    assert plan.model.model_id == "qwen/qwen3.6-27b-base"
    assert plan.model.provider == "mlx"
    assert plan.eval_count == 30
    assert len(plan.eval_id_sha256) == 64
    assert plan.prediction_output == (
        local_root / DEFAULT_RUN_ID / "qwen_base_predictions.jsonl"
    ).resolve()
    assert plan.summary_output == (local_root / DEFAULT_RUN_ID / "summary.json").resolve()
    assert plan.category_csv_output == (
        local_root / DEFAULT_RUN_ID / "category_metrics.csv"
    ).resolve()

    commands = plan.to_mapping()["post_processing_commands"]
    assert commands[0][:4] == ["uv", "run", "python", "scripts/summarize_baseline_results.py"]
    assert commands[1][:4] == ["uv", "run", "python", "scripts/generate_baseline_report.py"]


def test_eval_id_fingerprint_is_order_sensitive() -> None:
    assert eval_id_fingerprint(("eval_001", "eval_002")) != eval_id_fingerprint(
        ("eval_002", "eval_001")
    )


def test_qwen_baseline_plan_rejects_empty_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        build_plan(
            eval_path=SEED_PATH,
            local_root=tmp_path / "baseline",
            run_id="",
            model_path=tmp_path / "model",
            report_output=Path("results/reports/qwen_base_baseline_report.md"),
            max_tokens=256,
            temperature=0.0,
        )


def test_qwen_baseline_cli_requires_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_qwen_baseline.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--dry-run is required" in result.stderr


def test_qwen_baseline_cli_writes_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "qwen_baseline_plan.json"
    local_root = tmp_path / "local" / "runs" / "baseline"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_qwen_baseline.py",
            "--dry-run",
            "--input",
            str(SEED_PATH),
            "--local-root",
            str(local_root),
            "--model-path",
            str(tmp_path / "local" / "models" / "qwen"),
            "--write-plan",
            str(plan_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    stdout_plan = json.loads(result.stdout)
    file_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert stdout_plan == file_plan
    assert file_plan["eval_count"] == 30
    assert file_plan["prediction_output"].endswith(
        "phase6-qwen-base/qwen_base_predictions.jsonl"
    )
