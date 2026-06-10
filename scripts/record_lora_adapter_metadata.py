from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.training.lora_metadata import (  # noqa: E402
    METADATA_VERSION,
    AdapterMetadataValidationError,
    validate_adapter_metadata,
)
from scripts.run_mlx_lora import dump_simple_yaml  # noqa: E402

DEFAULT_LOCAL_ROOT = Path("/Users/tsinfra/Dev/pharma-llm/local")
DEFAULT_MODEL_ID = "qwen/qwen3.6-27b-base"
DEFAULT_DATASET_VERSION = "sft-v0.1"
DEFAULT_PLANNED_STATUS_NOTE = "Dry-run metadata placeholder; update to executed only after local training completes."
MLX_SPLIT_NAMES = ("train.jsonl", "valid.jsonl", "test.jsonl")
MLX_CONFIG_FIELD_ORDER = (
    "model",
    "mask_prompt",
    "train",
    "data",
    "adapter_path",
    "iters",
    "batch_size",
    "num_layers",
    "max_seq_length",
    "learning_rate",
    "steps_per_report",
    "steps_per_eval",
    "save_every",
    "seed",
    "lora_parameters",
)
LORA_PARAMETER_FIELD_ORDER = ("rank", "scale", "dropout", "keys")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_plan_path(plan: dict[str, Any], key: str) -> Path:
    value = plan.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"run plan missing path: {key}")
    return Path(value).expanduser().resolve()


def require_plan_digest(plan: dict[str, Any], key: str) -> str:
    value = plan.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"run plan missing SHA-256 digest: {key}")
    if any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"run plan invalid SHA-256 digest: {key}")
    return value


def require_plan_training(plan: dict[str, Any]) -> dict[str, Any]:
    training = plan.get("training")
    if not isinstance(training, dict):
        raise ValueError("run plan missing training object")
    return training


def require_plan_mapping(plan: dict[str, Any], key: str) -> dict[str, Any]:
    value = plan.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"run plan missing object: {key}")
    return value


def require_plan_local_root(plan: dict[str, Any], local_root: Path) -> Path:
    plan_local_root = require_plan_path(plan, "local_root")
    supplied_local_root = local_root.expanduser().resolve()
    if supplied_local_root != plan_local_root:
        raise ValueError(f"--local-root must equal run plan local_root: {plan_local_root}")
    return plan_local_root


def mlx_split_paths(plan: dict[str, Any]) -> tuple[Path, ...]:
    mlx_data_dir = require_plan_path(plan, "mlx_data_dir")
    return tuple(mlx_data_dir / split_name for split_name in MLX_SPLIT_NAMES)


def require_metadata_output_does_not_collide(
    *,
    plan: dict[str, Any],
    run_plan_path: Path,
    metadata_output: Path,
) -> Path:
    resolved_output = metadata_output.expanduser().resolve()
    reserved_paths = {
        "run plan": run_plan_path.expanduser().resolve(),
        "data.dataset_path": require_plan_path(plan, "dataset_path"),
        "config.source_path": require_plan_path(plan, "config_path"),
        "output.adapter_path": require_plan_path(plan, "adapter_path"),
        "output.run_output_path": require_plan_path(plan, "run_output_path"),
        "output.mlx_config_path": require_plan_path(plan, "mlx_config_path"),
        **{f"MLX split {path.name}": path for path in mlx_split_paths(plan)},
    }
    for label, path in reserved_paths.items():
        if resolved_output == path:
            raise ValueError(f"metadata output must differ from {label}: {path}")
    adapter_path = reserved_paths["output.adapter_path"]
    if resolved_output.is_relative_to(adapter_path):
        raise ValueError(f"metadata output must not be under output.adapter_path: {adapter_path}")
    return resolved_output


def ordered_mapping(mapping: dict[str, Any], field_order: tuple[str, ...]) -> dict[str, Any]:
    ordered = {key: mapping[key] for key in field_order if key in mapping}
    ordered.update({key: value for key, value in mapping.items() if key not in ordered})
    return ordered


def canonical_mlx_config_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    canonical = ordered_mapping(mapping, MLX_CONFIG_FIELD_ORDER)
    lora_parameters = canonical.get("lora_parameters")
    if isinstance(lora_parameters, dict):
        canonical["lora_parameters"] = ordered_mapping(lora_parameters, LORA_PARAMETER_FIELD_ORDER)
    return canonical


def require_generated_config_matches_plan(plan: dict[str, Any]) -> None:
    generated_config_path = require_plan_path(plan, "mlx_config_path")
    expected_config = require_plan_mapping(plan, "mlx_config")
    expected_yaml = dump_simple_yaml(canonical_mlx_config_mapping(expected_config))
    actual_yaml = generated_config_path.read_text(encoding="utf-8")
    if actual_yaml != expected_yaml:
        raise ValueError(
            "generated MLX config must match run plan mlx_config; rerun dry-run before recording metadata"
        )


def require_source_config_matches_plan(plan: dict[str, Any]) -> None:
    source_config_path = require_plan_path(plan, "config_path")
    expected_digest = require_plan_digest(plan, "config_sha256")
    actual_digest = file_sha256(source_config_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "source config must match run plan config_sha256; rerun dry-run before recording metadata"
        )


def require_source_dataset_matches_plan(plan: dict[str, Any]) -> None:
    source_dataset_path = require_plan_path(plan, "dataset_path")
    expected_digest = require_plan_digest(plan, "dataset_sha256")
    actual_digest = file_sha256(source_dataset_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "source dataset must match run plan dataset_sha256; rerun dry-run before recording metadata"
        )


def build_metadata(
    *,
    run_plan_path: Path,
    metadata_output: Path,
    status: str,
    dataset_version: str,
    model_id: str,
    local_root: Path,
    started_at: str | None,
    ended_at: str | None,
    status_note: str,
) -> dict[str, Any]:
    resolved_run_plan_path = run_plan_path.expanduser().resolve()
    plan = load_json(resolved_run_plan_path)
    training = require_plan_training(plan)
    local_root_path = require_plan_local_root(plan, local_root)
    require_generated_config_matches_plan(plan)
    require_source_config_matches_plan(plan)
    require_source_dataset_matches_plan(plan)
    resolved_metadata_output = require_metadata_output_does_not_collide(
        plan=plan,
        run_plan_path=resolved_run_plan_path,
        metadata_output=metadata_output,
    )
    dataset_path = require_plan_path(plan, "dataset_path")
    training_input_path = require_plan_path(plan, "train_data_path")
    source_config_path = require_plan_path(plan, "config_path")
    generated_config_path = require_plan_path(plan, "mlx_config_path")
    adapter_path = require_plan_path(plan, "adapter_path")
    model_path = require_plan_path(plan, "model_path")

    placeholder = status == "planned"
    adapter_exists = adapter_path.exists()
    adapter_is_directory = adapter_path.is_dir()
    adapter_markers = sorted(path.name for path in adapter_path.iterdir() if path.is_file()) if adapter_is_directory else []
    metadata = {
        "metadata_version": METADATA_VERSION,
        "run_id": plan.get("run_id"),
        "status": status,
        "model": {
            "id": model_id,
            "provider": "mlx",
            "path": str(model_path),
        },
        "dataset": {
            "version": dataset_version,
            "path": str(dataset_path),
            "sha256": file_sha256(dataset_path),
            "training_input": {
                "path": str(training_input_path),
                "sha256": file_sha256(training_input_path),
            },
        },
        "config": {
            "source_path": str(source_config_path),
            "source_sha256": file_sha256(source_config_path),
            "generated_path": str(generated_config_path),
            "generated_sha256": file_sha256(generated_config_path),
        },
        "adapter": {
            "path": str(adapter_path),
            "exists": adapter_exists,
            "is_directory": adapter_is_directory,
            "marker_files": adapter_markers,
            "metadata_path": str(resolved_metadata_output),
        },
        "training": {
            "rank": training.get("rank"),
            "scale": training.get("scale"),
            "dropout": training.get("dropout"),
            "mask_prompt": training.get("mask_prompt"),
            "target_modules": training.get("target_modules"),
            "epochs": None,
            "max_seq_length": training.get("max_seq_length"),
            "iters": training.get("iters"),
            "batch_size": training.get("batch_size"),
            "learning_rate": training.get("learning_rate"),
            "num_layers": training.get("num_layers"),
            "seed": training.get("seed"),
        },
        "timestamps": {
            "created_at": utc_now(),
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "validation": {
            "is_dry_run_placeholder": placeholder,
            "status_note": status_note,
        },
        "local_artifact_policy": {
            "local_root": str(local_root_path),
            "large_artifacts_ignored": True,
        },
    }
    validate_adapter_metadata(metadata)
    return metadata


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record Qwen SFT LoRA v1 adapter metadata from a local MLX run plan."
    )
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status",
        choices=("planned", "executed", "failed"),
        default="planned",
    )
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--started-at", default=None)
    parser.add_argument("--ended-at", default=None)
    parser.add_argument(
        "--status-note",
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.status == "planned":
        status_note = args.status_note or DEFAULT_PLANNED_STATUS_NOTE
    elif args.status_note is None:
        parser.error("--status-note is required when --status is executed or failed")
    else:
        status_note = args.status_note
    try:
        metadata = build_metadata(
            run_plan_path=args.run_plan,
            metadata_output=args.output,
            status=args.status,
            dataset_version=args.dataset_version,
            model_id=args.model_id,
            local_root=args.local_root,
            started_at=args.started_at,
            ended_at=args.ended_at,
            status_note=status_note,
        )
        write_metadata(Path(metadata["adapter"]["metadata_path"]), metadata)
    except (OSError, ValueError, AdapterMetadataValidationError) as exc:
        parser.error(str(exc))
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
