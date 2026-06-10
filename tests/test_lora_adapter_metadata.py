import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pharma_llm_lab.training.lora_metadata import (
    METADATA_VERSION,
    AdapterMetadataValidationError,
    validate_adapter_metadata,
)
from scripts.record_lora_adapter_metadata import build_metadata
from scripts.run_mlx_lora import build_plan, materialize_local_inputs, write_plan


def write_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")


def write_config(path: Path, *, dataset_path: Path, local_root: Path) -> None:
    path.write_text(
        f"""
[run]
run_id = "phase6-test-lora"

[model]
path = "{local_root / "models" / "qwen"}"

[data]
dataset_path = "{dataset_path}"

[output]
adapter_path = "{local_root / "adapters" / "phase6-test"}"
run_output_path = "{local_root / "runs" / "phase6-test" / "run_plan.json"}"
mlx_data_dir = "{local_root / "runs" / "phase6-test" / "mlx_data"}"
mlx_config_path = "{local_root / "runs" / "phase6-test" / "mlx_lora_config.yaml"}"

[training]
rank = 16
scale = 32.0
dropout = 0.0
mask_prompt = true
target_modules = ["self_attn.q_proj", "self_attn.v_proj"]
max_seq_length = 128
batch_size = 1
learning_rate = 0.00001
iters = 2
num_layers = -1
seed = 0
steps_per_report = 1
steps_per_eval = 1
save_every = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )


def prepare_run_plan(tmp_path: Path) -> tuple[Path, Path, Path]:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "qwen_sft_lora_r16.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)
    plan = build_plan(config_path=config_path, local_root=local_root)
    materialize_local_inputs(plan)
    write_plan(plan.run_output_path, plan)
    return plan.run_output_path, local_root, plan.adapter_path


def test_build_metadata_records_planned_placeholder(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )

    validated = validate_adapter_metadata(metadata).to_mapping()
    assert validated["metadata_version"] == METADATA_VERSION
    assert validated["run_id"] == "phase6-test-lora"
    assert validated["status"] == "planned"
    assert validated["validation"]["is_dry_run_placeholder"] is True
    assert validated["adapter"]["exists"] is False
    assert validated["training"]["epochs"] is None
    assert validated["training"]["mask_prompt"] is True
    assert validated["training"]["num_layers"] == -1
    assert validated["training"]["seed"] == 0
    assert validated["dataset"]["training_input"]["path"].endswith("mlx_data/train.jsonl")


def test_build_metadata_records_executed_adapter(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="executed",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at="2026-06-10T01:00:00Z",
        ended_at="2026-06-10T03:00:00Z",
        status_note="Local training completed and adapter directory exists.",
    )

    validated = validate_adapter_metadata(metadata).to_mapping()
    assert validated["status"] == "executed"
    assert validated["validation"]["is_dry_run_placeholder"] is False
    assert validated["adapter"]["exists"] is True
    assert validated["adapter"]["is_directory"] is True
    assert "adapter_config.json" in validated["adapter"]["marker_files"]
    assert "adapters.safetensors" in validated["adapter"]["marker_files"]
    assert validated["timestamps"]["started_at"] == "2026-06-10T01:00:00Z"


def test_validate_adapter_metadata_rejects_executed_placeholder(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="executed",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at="2026-06-10T01:00:00Z",
        ended_at="2026-06-10T03:00:00Z",
        status_note="Local training completed.",
    )
    metadata["validation"]["is_dry_run_placeholder"] = True

    with pytest.raises(AdapterMetadataValidationError, match="must not be marked"):
        validate_adapter_metadata(metadata)


def test_build_metadata_rejects_output_outside_local_root(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)

    with pytest.raises(AdapterMetadataValidationError, match="adapter.metadata_path must be under"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=tmp_path / "tracked_metadata.json",
            status="planned",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at=None,
            ended_at=None,
            status_note="Operator checklist prepared; training not executed in CI.",
        )


def test_build_metadata_rejects_output_collision_with_run_plan(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)

    with pytest.raises(ValueError, match="metadata output must differ from"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=run_plan_path,
            status="planned",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at=None,
            ended_at=None,
            status_note="Operator checklist prepared; training not executed in CI.",
        )


def test_build_metadata_rejects_output_collision_with_adapter_path(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)

    with pytest.raises(ValueError, match="metadata output must differ from output.adapter_path"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=adapter_path,
            status="planned",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at=None,
            ended_at=None,
            status_note="Operator checklist prepared; training not executed in CI.",
        )


def test_build_metadata_rejects_output_under_adapter_path(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)

    with pytest.raises(ValueError, match="metadata output must not be under output.adapter_path"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=adapter_path / "adapters.safetensors",
            status="planned",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at=None,
            ended_at=None,
            status_note="Operator checklist prepared; training not executed in CI.",
        )


def test_validate_adapter_metadata_resolves_local_root_escape(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["adapter"]["metadata_path"] = str(local_root / ".." / "escaped_metadata.json")

    with pytest.raises(AdapterMetadataValidationError, match="adapter.metadata_path must be under"):
        validate_adapter_metadata(metadata)


def test_build_metadata_hashes_materialized_training_input(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    Path(plan["dataset_path"]).write_text('{"prompt":"new","completion":"new"}\n', encoding="utf-8")

    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )

    assert metadata["dataset"]["path"] == plan["dataset_path"]
    assert metadata["dataset"]["training_input"]["path"] == plan["train_data_path"]
    assert metadata["dataset"]["sha256"] != metadata["dataset"]["training_input"]["sha256"]


def test_build_metadata_rejects_executed_adapter_file(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.parent.mkdir(parents=True)
    adapter_path.write_text("not a directory\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="adapter.is_directory must be true"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="executed",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at="2026-06-10T01:00:00Z",
            ended_at="2026-06-10T03:00:00Z",
            status_note="Local training completed.",
        )


def test_build_metadata_rejects_executed_adapter_without_weights(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="adapter weights file"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="executed",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at="2026-06-10T01:00:00Z",
            ended_at="2026-06-10T03:00:00Z",
            status_note="Local training completed.",
        )


def test_validate_adapter_metadata_rejects_bad_executed_timestamps(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="executed",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at="2026-06-10T01:00:00Z",
        ended_at="2026-06-10T03:00:00Z",
        status_note="Local training completed.",
    )
    metadata["timestamps"]["started_at"] = "bad"

    with pytest.raises(AdapterMetadataValidationError, match="timestamps.started_at must be"):
        validate_adapter_metadata(metadata)


def test_build_metadata_rejects_date_only_executed_timestamps(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="timestamps.started_at must be"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="executed",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at="2026-06-10Z",
            ended_at="2026-06-10T03:00:00Z",
            status_note="Local training completed.",
        )


def test_build_metadata_rejects_bad_failed_timestamps(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="timestamps.started_at must be"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="failed",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at="bad",
            ended_at=None,
            status_note="Local training failed before adapter creation.",
        )


def test_build_metadata_rejects_reversed_executed_timestamps(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="ended_at must be at or after"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="executed",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at="2026-06-10T03:00:00Z",
            ended_at="2026-06-10T01:00:00Z",
            status_note="Local training completed.",
        )


def test_build_metadata_marks_failed_runs_as_attempted(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="failed",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at="2026-06-10T01:00:00Z",
        ended_at="2026-06-10T01:30:00Z",
        status_note="Local training failed before adapter creation.",
    )

    assert metadata["validation"]["is_dry_run_placeholder"] is False


def test_validate_adapter_metadata_rejects_fractional_rank(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["training"]["rank"] = 1.5

    with pytest.raises(AdapterMetadataValidationError, match="training.rank must be an integer"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_fractional_training_count(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["training"]["iters"] = 1.5

    with pytest.raises(AdapterMetadataValidationError, match="training.iters must be an integer"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_nonpositive_training_count(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["training"]["max_seq_length"] = 0

    with pytest.raises(AdapterMetadataValidationError, match="training.max_seq_length must be positive"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_unknown_target_module(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["training"]["target_modules"] = ["foo"]

    with pytest.raises(AdapterMetadataValidationError, match="unsupported Qwen MLX module"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_missing_mask_prompt(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["training"].pop("mask_prompt")

    with pytest.raises(AdapterMetadataValidationError, match="training.mask_prompt must be a boolean"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_bad_sha256(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["dataset"]["sha256"] = "bad"

    with pytest.raises(AdapterMetadataValidationError, match="dataset.sha256 must be"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_generated_config_outside_local_root(
    tmp_path: Path,
) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["config"]["generated_path"] = str(tmp_path / "generated.yaml")

    with pytest.raises(AdapterMetadataValidationError, match="config.generated_path must be under"):
        validate_adapter_metadata(metadata)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("scale", -1.0, "training.scale must be positive"),
        ("dropout", -0.1, "training.dropout must be non-negative"),
        ("batch_size", "1", "training.batch_size must be an integer"),
        ("learning_rate", 0.0, "training.learning_rate must be positive"),
        ("num_layers", -2, "training.num_layers must be >= -1"),
        ("seed", -1, "training.seed must be >= 0"),
    ],
)
def test_validate_adapter_metadata_rejects_invalid_recorded_hyperparameters(
    tmp_path: Path,
    field_name: str,
    value: object,
    message: str,
) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    metadata = build_metadata(
        run_plan_path=run_plan_path,
        metadata_output=metadata_path,
        status="planned",
        dataset_version="sft-v0.1",
        model_id="qwen/qwen3.6-27b-base",
        local_root=local_root,
        started_at=None,
        ended_at=None,
        status_note="Operator checklist prepared; training not executed in CI.",
    )
    metadata["training"][field_name] = value

    with pytest.raises(AdapterMetadataValidationError, match=message):
        validate_adapter_metadata(metadata)


def test_metadata_cli_writes_local_json(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/record_lora_adapter_metadata.py",
            "--run-plan",
            str(run_plan_path),
            "--output",
            str(metadata_path),
            "--local-root",
            str(local_root),
            "--status-note",
            "Operator checklist prepared; training not executed in CI.",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    stdout_payload = json.loads(result.stdout)
    file_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["status"] == "planned"


def test_metadata_cli_requires_status_note_for_attempted_runs(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/record_lora_adapter_metadata.py",
            "--run-plan",
            str(run_plan_path),
            "--output",
            str(metadata_path),
            "--local-root",
            str(local_root),
            "--status",
            "failed",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--status-note is required" in result.stderr


def test_metadata_cli_writes_expanded_output_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    local_root = home / "pharma-llm" / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "qwen_sft_lora_r16.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)
    plan = build_plan(config_path=config_path, local_root=local_root)
    materialize_local_inputs(plan)
    write_plan(plan.run_output_path, plan)
    metadata_path = "~/pharma-llm/local/runs/phase6-test/adapter_metadata.json"
    env = {**os.environ, "HOME": str(home)}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/record_lora_adapter_metadata.py",
            "--run-plan",
            str(plan.run_output_path),
            "--output",
            metadata_path,
            "--local-root",
            "~/pharma-llm/local",
            "--status-note",
            "Operator checklist prepared; training not executed in CI.",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    expected_path = home / "pharma-llm" / "local" / "runs" / "phase6-test" / "adapter_metadata.json"
    assert result.returncode == 0
    assert expected_path.exists()
    assert not (Path.cwd() / "~").exists()
    file_payload = json.loads(expected_path.read_text(encoding="utf-8"))
    assert file_payload["adapter"]["metadata_path"] == str(expected_path)
