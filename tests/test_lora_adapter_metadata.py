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
    assert validated["dataset"]["sha256"] == validated["dataset"]["training_input"]["sha256"]


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


def test_validate_adapter_metadata_rejects_failed_placeholder(tmp_path: Path) -> None:
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
    metadata["validation"]["is_dry_run_placeholder"] = True

    with pytest.raises(AdapterMetadataValidationError, match="failed metadata must not"):
        validate_adapter_metadata(metadata)


@pytest.mark.parametrize("field_name", ["is_directory", "marker_files"])
def test_validate_adapter_metadata_requires_adapter_marker_fields_for_all_statuses(
    tmp_path: Path,
    field_name: str,
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
    metadata["adapter"].pop(field_name)

    with pytest.raises(AdapterMetadataValidationError, match=f"adapter.{field_name} must be"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_relative_artifact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.chdir(tmp_path)
    metadata["adapter"]["metadata_path"] = "local/runs/phase6-test/adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="adapter.metadata_path must be an absolute"):
        validate_adapter_metadata(metadata)


def test_validate_adapter_metadata_rejects_relative_local_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.chdir(tmp_path)
    metadata["local_artifact_policy"]["local_root"] = "local"

    with pytest.raises(
        AdapterMetadataValidationError,
        match="local_artifact_policy.local_root must be an absolute",
    ):
        validate_adapter_metadata(metadata)


@pytest.mark.parametrize(
    ("section_name", "field_name", "message"),
    [
        ("model", "path", "model.path must be an absolute path"),
        ("dataset", "path", "dataset.path must be an absolute path"),
        ("config", "source_path", "config.source_path must be an absolute path"),
    ],
)
def test_validate_adapter_metadata_rejects_relative_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section_name: str,
    field_name: str,
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
    monkeypatch.chdir(tmp_path)
    metadata[section_name][field_name] = "relative/source-path"

    with pytest.raises(AdapterMetadataValidationError, match=message):
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


@pytest.mark.parametrize("plan_key", ["dataset_path", "config_path"])
def test_build_metadata_rejects_output_collision_with_source_inputs(
    tmp_path: Path,
    plan_key: str,
) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="metadata output must differ from"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=Path(plan[plan_key]),
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


def test_build_metadata_rejects_source_dataset_drift(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    Path(plan["dataset_path"]).write_text('{"prompt":"new","completion":"new"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source dataset must match run plan dataset_sha256"):
        build_metadata(
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


def test_build_metadata_rejects_materialized_training_input_drift(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    Path(plan["train_data_path"]).write_text('{"prompt":"stale","completion":"stale"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="materialized training input must match"):
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


def test_build_metadata_rejects_source_config_drift(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    Path(plan["config_path"]).write_text("[run]\nrun_id = \"changed\"\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source config must match run plan config_sha256"):
        build_metadata(
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


def test_build_metadata_rejects_plan_adapter_path_mismatch_with_mlx_config(
    tmp_path: Path,
) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    mismatched_adapter_path = local_root / "adapters" / "mismatched"
    mismatched_adapter_path.mkdir(parents=True)
    (mismatched_adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (mismatched_adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    plan["adapter_path"] = str(mismatched_adapter_path)
    run_plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_path must match mlx_config.adapter_path"):
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


def test_build_metadata_rejects_local_root_mismatch(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(ValueError, match="--local-root must equal run plan local_root"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="planned",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=tmp_path,
            started_at=None,
            ended_at=None,
            status_note="Operator checklist prepared; training not executed in CI.",
        )


def test_build_metadata_rejects_generated_config_drift(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    Path(plan["mlx_config_path"]).write_text("rank: 999\n", encoding="utf-8")

    with pytest.raises(ValueError, match="generated MLX config must match run plan"):
        build_metadata(
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


def test_build_metadata_rejects_disabled_mlx_training_flag(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    plan["mlx_config"]["train"] = False
    run_plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mlx_config_path = Path(plan["mlx_config_path"])
    mlx_config_path.write_text(
        mlx_config_path.read_text(encoding="utf-8").replace("train: true\n", "train: false\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mlx_config.train must be true"):
        build_metadata(
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


def test_build_metadata_rejects_plan_drift_from_source_config(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"
    plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    plan["training"]["rank"] = 32
    plan["mlx_config"]["lora_parameters"]["rank"] = 32
    run_plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mlx_config_path = Path(plan["mlx_config_path"])
    mlx_config_path.write_text(
        mlx_config_path.read_text(encoding="utf-8").replace("  rank: 16\n", "  rank: 32\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run plan training must match source config dry-run output"):
        build_metadata(
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


def test_build_metadata_rejects_executed_adapter_without_config(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapters.safetensors").write_text("weights\n", encoding="utf-8")
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="adapter_config\\.json"):
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


def test_build_metadata_rejects_executed_adapter_weight_directory(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapters.safetensors").mkdir()
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
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
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


def test_build_metadata_rejects_missing_failed_timestamps(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="failed metadata must include"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="failed",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at=None,
            ended_at=None,
            status_note="Local training failed before adapter creation.",
        )


def test_build_metadata_rejects_planned_execution_timestamps(tmp_path: Path) -> None:
    run_plan_path, local_root, _adapter_path = prepare_run_plan(tmp_path)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="planned metadata must not include"):
        build_metadata(
            run_plan_path=run_plan_path,
            metadata_output=metadata_path,
            status="planned",
            dataset_version="sft-v0.1",
            model_id="qwen/qwen3.6-27b-base",
            local_root=local_root,
            started_at="2026-06-10T01:00:00Z",
            ended_at=None,
            status_note="Operator checklist prepared; training not executed in CI.",
        )


def test_build_metadata_rejects_reversed_executed_timestamps(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
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


def test_build_metadata_rejects_planned_after_adapter_exists(tmp_path: Path) -> None:
    run_plan_path, local_root, adapter_path = prepare_run_plan(tmp_path)
    adapter_path.mkdir(parents=True)
    metadata_path = local_root / "runs" / "phase6-test" / "adapter_metadata.json"

    with pytest.raises(AdapterMetadataValidationError, match="planned metadata must not"):
        build_metadata(
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


def test_validate_adapter_metadata_rejects_negative_epochs(tmp_path: Path) -> None:
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
    metadata["training"]["epochs"] = -1

    with pytest.raises(AdapterMetadataValidationError, match="training.epochs must be >= 0"):
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


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("scale", float("nan")),
        ("dropout", float("inf")),
        ("learning_rate", float("-inf")),
    ],
)
def test_validate_adapter_metadata_rejects_nonfinite_hyperparameters(
    tmp_path: Path,
    field_name: str,
    value: float,
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

    with pytest.raises(AdapterMetadataValidationError, match=f"training.{field_name} must be finite"):
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
            "~/pharma-llm/local/runs/phase6-test/run_plan.json",
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
