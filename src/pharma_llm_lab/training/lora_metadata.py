from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

MetadataStatus = Literal["planned", "executed", "failed"]

METADATA_VERSION = "qwen-sft-lora-v1-metadata"
QWEN_TARGET_MODULE_KEYS = frozenset(
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    }
)
REQUIRED_TOP_LEVEL_FIELDS = {
    "metadata_version",
    "run_id",
    "status",
    "model",
    "dataset",
    "config",
    "adapter",
    "training",
    "timestamps",
    "validation",
    "local_artifact_policy",
}
STATUS_VALUES = {"planned", "executed", "failed"}


class AdapterMetadataValidationError(ValueError):
    """Raised when LoRA adapter metadata is incomplete or inconsistent."""


@dataclass(frozen=True)
class AdapterMetadata:
    payload: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.payload)


def require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterMetadataValidationError(f"{field_name} must be an object")
    return value


def require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterMetadataValidationError(f"{field_name} must be a non-empty string")
    return value


def require_sha256(value: Any, field_name: str) -> str:
    digest = require_non_empty_string(value, field_name)
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise AdapterMetadataValidationError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return digest


def require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise AdapterMetadataValidationError(f"{field_name} must be a boolean")
    return value


def require_number(value: Any, field_name: str) -> int | float:
    if type(value) not in (int, float):
        raise AdapterMetadataValidationError(f"{field_name} must be a number")
    return value


def require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise AdapterMetadataValidationError(f"{field_name} must be an integer")
    if value < 1:
        raise AdapterMetadataValidationError(f"{field_name} must be positive")
    return value


def require_int_at_least(value: Any, field_name: str, minimum: int) -> int:
    if type(value) is not int:
        raise AdapterMetadataValidationError(f"{field_name} must be an integer")
    if value < minimum:
        raise AdapterMetadataValidationError(f"{field_name} must be >= {minimum}")
    return value


def require_positive_number(value: Any, field_name: str) -> int | float:
    parsed = require_number(value, field_name)
    if parsed <= 0:
        raise AdapterMetadataValidationError(f"{field_name} must be positive")
    return parsed


def require_non_negative_number(value: Any, field_name: str) -> int | float:
    parsed = require_number(value, field_name)
    if parsed < 0:
        raise AdapterMetadataValidationError(f"{field_name} must be non-negative")
    return parsed


def require_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AdapterMetadataValidationError(f"{field_name} must be a non-empty string list")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AdapterMetadataValidationError(f"{field_name} must be a non-empty string list")
        parsed.append(item)
    return tuple(parsed)


def require_qwen_target_modules(value: Any, field_name: str) -> tuple[str, ...]:
    modules = require_string_list(value, field_name)
    unknown_modules = sorted(set(modules) - QWEN_TARGET_MODULE_KEYS)
    if unknown_modules:
        allowed = ", ".join(sorted(QWEN_TARGET_MODULE_KEYS))
        unknown = ", ".join(unknown_modules)
        raise AdapterMetadataValidationError(
            f"{field_name} contains unsupported Qwen MLX module key(s): {unknown}; "
            f"allowed keys: {allowed}"
        )
    return modules


def require_local_artifact_path(value: Any, field_name: str, local_root: str) -> str:
    raw_path = require_non_empty_string(value, field_name)
    path = Path(raw_path).expanduser().resolve()
    if not path.is_absolute():
        raise AdapterMetadataValidationError(f"{field_name} must be an absolute local path")
    local_root_path = Path(local_root).expanduser().resolve()
    if not path.is_relative_to(local_root_path):
        raise AdapterMetadataValidationError(f"{field_name} must be under local root: {local_root}")
    return str(path)


def parse_utc_timestamp(value: Any, field_name: str) -> datetime:
    raw_value = require_non_empty_string(value, field_name)
    if "T" not in raw_value or not raw_value.endswith("Z"):
        raise AdapterMetadataValidationError(f"{field_name} must be an ISO-8601 UTC timestamp")
    try:
        return datetime.fromisoformat(raw_value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise AdapterMetadataValidationError(
            f"{field_name} must be an ISO-8601 UTC timestamp"
        ) from exc


def require_utc_timestamp(value: Any, field_name: str) -> str:
    raw_value = require_non_empty_string(value, field_name)
    parse_utc_timestamp(raw_value, field_name)
    return raw_value


def validate_adapter_metadata(payload: dict[str, Any]) -> AdapterMetadata:
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - payload.keys())
    if missing:
        raise AdapterMetadataValidationError("missing required field(s): " + ", ".join(missing))

    if payload["metadata_version"] != METADATA_VERSION:
        raise AdapterMetadataValidationError(
            f"metadata_version must be {METADATA_VERSION!r}"
        )

    require_non_empty_string(payload["run_id"], "run_id")
    status = require_non_empty_string(payload["status"], "status")
    if status not in STATUS_VALUES:
        raise AdapterMetadataValidationError(
            "status must be one of: " + ", ".join(sorted(STATUS_VALUES))
        )

    artifact_policy = require_mapping(payload["local_artifact_policy"], "local_artifact_policy")
    local_root = require_non_empty_string(
        artifact_policy.get("local_root"),
        "local_artifact_policy.local_root",
    )
    require_bool(
        artifact_policy.get("large_artifacts_ignored"),
        "local_artifact_policy.large_artifacts_ignored",
    )

    model = require_mapping(payload["model"], "model")
    require_non_empty_string(model.get("id"), "model.id")
    require_non_empty_string(model.get("provider"), "model.provider")
    require_non_empty_string(model.get("path"), "model.path")

    dataset = require_mapping(payload["dataset"], "dataset")
    require_non_empty_string(dataset.get("version"), "dataset.version")
    require_non_empty_string(dataset.get("path"), "dataset.path")
    require_sha256(dataset.get("sha256"), "dataset.sha256")
    training_input = require_mapping(dataset.get("training_input"), "dataset.training_input")
    require_local_artifact_path(
        training_input.get("path"),
        "dataset.training_input.path",
        local_root,
    )
    require_sha256(training_input.get("sha256"), "dataset.training_input.sha256")

    config = require_mapping(payload["config"], "config")
    require_non_empty_string(config.get("source_path"), "config.source_path")
    require_sha256(config.get("source_sha256"), "config.source_sha256")
    require_local_artifact_path(config.get("generated_path"), "config.generated_path", local_root)
    require_sha256(config.get("generated_sha256"), "config.generated_sha256")

    adapter = require_mapping(payload["adapter"], "adapter")
    require_local_artifact_path(adapter.get("path"), "adapter.path", local_root)
    require_local_artifact_path(adapter.get("metadata_path"), "adapter.metadata_path", local_root)
    if status == "executed":
        require_bool(adapter.get("exists"), "adapter.exists")
        if not adapter["exists"]:
            raise AdapterMetadataValidationError("adapter.exists must be true for executed metadata")
        require_bool(adapter.get("is_directory"), "adapter.is_directory")
        if not adapter["is_directory"]:
            raise AdapterMetadataValidationError("adapter.is_directory must be true for executed metadata")
        markers = require_string_list(adapter.get("marker_files"), "adapter.marker_files")
        expected_weight_markers = {"adapters.safetensors", "adapter.safetensors"}
        if not any(marker in expected_weight_markers for marker in markers):
            raise AdapterMetadataValidationError(
                "executed adapter metadata must include an adapter weights file"
            )

    training = require_mapping(payload["training"], "training")
    require_positive_int(training.get("rank"), "training.rank")
    require_positive_number(training.get("scale"), "training.scale")
    require_non_negative_number(training.get("dropout"), "training.dropout")
    require_bool(training.get("mask_prompt"), "training.mask_prompt")
    require_qwen_target_modules(training.get("target_modules"), "training.target_modules")
    require_positive_int(training.get("max_seq_length"), "training.max_seq_length")
    require_positive_int(training.get("iters"), "training.iters")
    require_positive_int(training.get("batch_size"), "training.batch_size")
    require_positive_number(training.get("learning_rate"), "training.learning_rate")
    require_int_at_least(training.get("num_layers"), "training.num_layers", -1)
    require_int_at_least(training.get("seed"), "training.seed", 0)
    if "epochs" not in training:
        raise AdapterMetadataValidationError("training.epochs must be present")
    if training["epochs"] is not None and type(training["epochs"]) is not int:
        raise AdapterMetadataValidationError("training.epochs must be null or an integer")

    timestamps = require_mapping(payload["timestamps"], "timestamps")
    require_utc_timestamp(timestamps.get("created_at"), "timestamps.created_at")
    started_at = None
    ended_at = None
    if timestamps.get("started_at") is not None:
        started_at = parse_utc_timestamp(timestamps.get("started_at"), "timestamps.started_at")
    if timestamps.get("ended_at") is not None:
        ended_at = parse_utc_timestamp(timestamps.get("ended_at"), "timestamps.ended_at")
    if started_at is not None and ended_at is not None and ended_at < started_at:
        raise AdapterMetadataValidationError("timestamps.ended_at must be at or after started_at")
    if status == "executed":
        require_utc_timestamp(timestamps.get("started_at"), "timestamps.started_at")
        require_utc_timestamp(timestamps.get("ended_at"), "timestamps.ended_at")

    validation = require_mapping(payload["validation"], "validation")
    require_bool(validation.get("is_dry_run_placeholder"), "validation.is_dry_run_placeholder")
    if status != "planned" and validation["is_dry_run_placeholder"]:
        raise AdapterMetadataValidationError(
            f"{status} metadata must not be marked as a dry-run placeholder"
        )
    require_non_empty_string(validation.get("status_note"), "validation.status_note")

    return AdapterMetadata(payload)
