import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_lora_eval import (
    DEFAULT_ADAPTER_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_RUN_ID,
    run_mock_lora_eval,
)

SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


def test_mock_lora_runner_processes_phase4_seed_eval() -> None:
    predictions = run_mock_lora_eval(
        eval_path=SEED_PATH,
        model_id=DEFAULT_MODEL_ID,
        adapter_id=DEFAULT_ADAPTER_ID,
        run_id=DEFAULT_RUN_ID,
        max_tokens=32,
    )

    assert len(predictions) == 30
    first = predictions[0].to_mapping()
    assert first["run_id"] == DEFAULT_RUN_ID
    assert first["eval_id"] == "eval_001"
    assert first["category"] == "business_summary"
    assert first["model"] == {
        "model_id": DEFAULT_MODEL_ID,
        "provider": "mock-mlx",
        "adapter_id": DEFAULT_ADAPTER_ID,
    }
    assert first["generated_text"].startswith(f"[lora:{DEFAULT_ADAPTER_ID}]")


def test_mock_lora_runner_rejects_empty_adapter_id() -> None:
    with pytest.raises(ValueError, match="adapter_id must be a non-empty string"):
        run_mock_lora_eval(
            eval_path=SEED_PATH,
            model_id=DEFAULT_MODEL_ID,
            adapter_id="",
            run_id=DEFAULT_RUN_ID,
            max_tokens=8,
        )


def test_lora_runner_cli_writes_prediction_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "lora_predictions.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_lora_eval.py",
            "--input",
            str(SEED_PATH),
            "--output",
            str(output_path),
            "--adapter-id",
            "qwen_sft_lora_r16_v1",
            "--run-id",
            "cli-lora",
            "--max-tokens",
            "8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: wrote 30 LoRA prediction" in result.stdout
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 30
    assert records[0]["run_id"] == "cli-lora"
    assert records[0]["model"]["adapter_id"] == "qwen_sft_lora_r16_v1"


def test_lora_runner_cli_rejects_metadata_backed_mock(tmp_path: Path) -> None:
    metadata_path = tmp_path / "adapter_metadata.json"
    metadata_path.write_text("{}\n", encoding="utf-8")
    output_path = tmp_path / "lora_predictions.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_lora_eval.py",
            "--input",
            str(SEED_PATH),
            "--output",
            str(output_path),
            "--adapter-metadata",
            str(metadata_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--adapter-metadata requires real LoRA generation" in result.stderr
    assert not output_path.exists()
