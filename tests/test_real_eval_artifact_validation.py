import json
import subprocess
import sys
from pathlib import Path

import pytest

from pharma_llm_lab.baseline.results import (
    aggregate_results,
    load_baseline_results,
    write_category_metrics_csv,
    write_summary_json,
)
from pharma_llm_lab.training.lora_metadata import METADATA_VERSION
from scripts.run_baseline_eval import run_mock_baseline, write_predictions
from scripts.run_lora_eval import DEFAULT_ADAPTER_ID, run_mock_lora_eval
from scripts.validate_real_eval_artifacts import (
    ArtifactValidationError,
    validate_real_eval_artifacts,
)

SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


def write_adapter_metadata(path: Path, *, local_root: Path, model_path: Path) -> Path:
    adapter_path = local_root / "adapters" / DEFAULT_ADAPTER_ID
    train_path = local_root / "runs" / DEFAULT_ADAPTER_ID / "mlx_data" / "train.jsonl"
    generated_config = local_root / "runs" / DEFAULT_ADAPTER_ID / "mlx_lora_config.yaml"
    source_dataset = local_root / "argilla" / "phase6_reviewed_sft.jsonl"
    source_config = local_root / "configs" / "qwen_sft_lora_r16.toml"

    adapter_path.mkdir(parents=True)
    train_path.parent.mkdir(parents=True)
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    source_dataset.parent.mkdir(parents=True, exist_ok=True)
    source_config.parent.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_path / "adapters.safetensors").write_text("weights", encoding="utf-8")
    train_path.write_text("{}", encoding="utf-8")
    generated_config.write_text("model: qwen\n", encoding="utf-8")
    source_dataset.write_text("{}", encoding="utf-8")
    source_config.write_text("[training]\n", encoding="utf-8")

    payload = {
        "metadata_version": METADATA_VERSION,
        "run_id": DEFAULT_ADAPTER_ID,
        "status": "executed",
        "model": {
            "id": "qwen/qwen3.6-27b-base",
            "provider": "mock-mlx",
            "path": str(model_path.resolve()),
        },
        "dataset": {
            "version": "sft-v0.1",
            "path": str(source_dataset.resolve()),
            "sha256": "a" * 64,
            "training_input": {
                "path": str(train_path.resolve()),
                "sha256": "b" * 64,
            },
        },
        "config": {
            "source_path": str(source_config.resolve()),
            "source_sha256": "c" * 64,
            "generated_path": str(generated_config.resolve()),
            "generated_sha256": "d" * 64,
        },
        "adapter": {
            "path": str(adapter_path.resolve()),
            "exists": True,
            "is_directory": True,
            "marker_files": ["adapter_config.json", "adapters.safetensors"],
            "metadata_path": str(path.resolve()),
        },
        "training": {
            "rank": 16,
            "scale": 32.0,
            "dropout": 0.0,
            "mask_prompt": True,
            "target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
            "epochs": None,
            "max_seq_length": 128,
            "iters": 2,
            "batch_size": 1,
            "learning_rate": 0.00001,
            "num_layers": -1,
            "seed": 0,
        },
        "timestamps": {
            "created_at": "2026-06-10T00:00:00Z",
            "started_at": "2026-06-10T01:00:00Z",
            "ended_at": "2026-06-10T03:00:00Z",
        },
        "validation": {
            "is_dry_run_placeholder": False,
            "status_note": "Local training completed.",
        },
        "local_artifact_policy": {
            "local_root": str(local_root.resolve()),
            "large_artifacts_ignored": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_artifact_set(tmp_path: Path) -> dict[str, Path]:
    local_root = tmp_path / "local"
    model_path = local_root / "models" / "qwen3.6-27b-base"
    base_dir = local_root / "runs" / "baseline" / "phase6-qwen-base"
    lora_dir = local_root / "runs" / DEFAULT_ADAPTER_ID
    model_path.mkdir(parents=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    run_plan = lora_dir / "run_plan.json"
    run_plan.write_text('{"run_id":"qwen_sft_lora_r16_v1"}\n', encoding="utf-8")

    base_predictions = base_dir / "qwen_base_predictions.jsonl"
    lora_predictions = lora_dir / "lora_predictions.jsonl"
    write_predictions(
        base_predictions,
        run_mock_baseline(
            eval_path=SEED_PATH,
            model_label="qwen-base",
            run_id="phase6-qwen-base",
            max_tokens=8,
        ),
    )
    write_predictions(
        lora_predictions,
        run_mock_lora_eval(
            eval_path=SEED_PATH,
            model_id="qwen/qwen3.6-27b-base",
            adapter_id=DEFAULT_ADAPTER_ID,
            run_id="phase6-qwen-sft-lora-r16-v1",
            max_tokens=8,
        ),
    )
    summary = aggregate_results(load_baseline_results(base_predictions))
    base_summary = base_dir / "summary.json"
    base_category_metrics = base_dir / "category_metrics.csv"
    write_summary_json(base_summary, summary)
    write_category_metrics_csv(base_category_metrics, summary)
    adapter_metadata = write_adapter_metadata(
        lora_dir / "adapter_metadata.json",
        local_root=local_root,
        model_path=model_path,
    )
    return {
        "model_path": model_path,
        "base_predictions": base_predictions,
        "base_summary": base_summary,
        "base_category_metrics": base_category_metrics,
        "lora_predictions": lora_predictions,
        "adapter_metadata": adapter_metadata,
        "run_plan": run_plan,
    }


def test_validate_real_eval_artifacts_accepts_complete_artifact_set(tmp_path: Path) -> None:
    paths = write_artifact_set(tmp_path)

    lines = validate_real_eval_artifacts(**paths)

    assert f"base_predictions={paths['base_predictions']}" in lines
    assert f"lora_predictions={paths['lora_predictions']}" in lines
    assert f"adapter_id={DEFAULT_ADAPTER_ID}" in lines
    assert "matched_eval_count=30" in lines


def test_validate_real_eval_artifacts_rejects_missing_model_path(tmp_path: Path) -> None:
    paths = write_artifact_set(tmp_path)

    with pytest.raises(ArtifactValidationError, match="Qwen base model path"):
        validate_real_eval_artifacts(**{**paths, "model_path": tmp_path / "missing-model"})


def test_validate_real_eval_artifacts_cli_reports_missing_file(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_real_eval_artifacts.py",
            "--model-path",
            str(tmp_path / "missing-model"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Qwen base model path" in result.stderr
