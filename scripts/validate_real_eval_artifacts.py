from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.baseline import BaselineResultError, load_lora_comparison_inputs  # noqa: E402
from pharma_llm_lab.baseline.results import aggregate_results, load_baseline_results  # noqa: E402
from pharma_llm_lab.training import AdapterMetadataValidationError, validate_adapter_metadata  # noqa: E402

DEFAULT_LOCAL_ROOT = Path("/Users/tsinfra/Dev/pharma-llm/local")
DEFAULT_BASE_RUN_DIR = DEFAULT_LOCAL_ROOT / "runs" / "baseline" / "phase6-qwen-base"
DEFAULT_LORA_RUN_DIR = DEFAULT_LOCAL_ROOT / "runs" / "qwen_sft_lora_r16_v1"
DEFAULT_MODEL_PATH = DEFAULT_LOCAL_ROOT / "models" / "qwen3.6-27b-base"
DEFAULT_EVAL_PATH = Path("evals/prompts/phase4_seed.jsonl")
REAL_PROVIDER = "mlx"


class ArtifactValidationError(ValueError):
    """Raised when the real local evaluation artifact set is incomplete."""


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"{path}: malformed JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{path}: JSON root must be an object")
    return value


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ArtifactValidationError(f"{label} must be a file: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ArtifactValidationError(f"{label} must be a directory: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path: Path, expected_sha256: str, label: str) -> None:
    require_file(path, label)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ArtifactValidationError(
            f"{label} sha256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def resolve_metadata_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def validate_summary_matches_predictions(summary_path: Path, predictions_path: Path) -> None:
    require_file(summary_path, "base summary")
    summary_payload = load_json_object(summary_path)
    prediction_summary = aggregate_results(load_baseline_results(predictions_path)).to_mapping()
    if summary_payload != prediction_summary:
        raise ArtifactValidationError(f"{summary_path}: does not match base predictions")


def expected_category_metric_rows(predictions_path: Path) -> dict[str, dict[str, str]]:
    summary = aggregate_results(load_baseline_results(predictions_path))
    rows: dict[str, dict[str, str]] = {}
    for metrics in summary.category_metrics:
        row = metrics.to_mapping()
        rows[metrics.category.value] = {
            "run_id": str(row["run_id"]),
            "model_id": str(row["model_id"]),
            "provider": str(row["provider"]),
            "adapter_id": "" if row["adapter_id"] is None else str(row["adapter_id"]),
            "category": str(row["category"]),
            "count": str(row["count"]),
            "avg_total_latency_ms": str(row["avg_total_latency_ms"]),
            "avg_ttft_ms": "" if row["avg_ttft_ms"] is None else str(row["avg_ttft_ms"]),
            "avg_tokens_per_second": (
                "" if row["avg_tokens_per_second"] is None else str(row["avg_tokens_per_second"])
            ),
            "scoring_status_counts": json.dumps(
                row["scoring_status_counts"], ensure_ascii=False, sort_keys=True
            ),
        }
    return rows


def validate_category_metrics_csv(path: Path, predictions_path: Path) -> None:
    require_file(path, "base category metrics")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ArtifactValidationError(f"{path}: CSV header is missing")
        required = {
            "run_id",
            "model_id",
            "provider",
            "adapter_id",
            "category",
            "count",
            "avg_total_latency_ms",
            "avg_ttft_ms",
            "avg_tokens_per_second",
            "scoring_status_counts",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ArtifactValidationError(
                f"{path}: missing required column(s): " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise ArtifactValidationError(f"{path}: at least one category row is required")
    categories = [row.get("category", "") for row in rows]
    duplicate_categories = sorted(
        category for category in set(categories) if categories.count(category) > 1
    )
    if duplicate_categories:
        raise ArtifactValidationError(
            f"{path}: duplicate category row(s): " + ", ".join(duplicate_categories)
        )
    expected_rows = expected_category_metric_rows(predictions_path)
    actual_rows = {row.get("category", ""): {key: row.get(key, "") for key in required} for row in rows}
    if set(actual_rows) != set(expected_rows):
        raise ArtifactValidationError(f"{path}: category rows do not match base predictions")
    for category, expected_row in expected_rows.items():
        actual_row = actual_rows[category]
        mismatches = [
            field for field, expected_value in expected_row.items() if actual_row[field] != expected_value
        ]
        if mismatches:
            raise ArtifactValidationError(
                f"{path}: category {category} does not match base predictions for field(s): "
                + ", ".join(mismatches)
            )


def validate_adapter_metadata_file(path: Path) -> dict[str, Any]:
    require_file(path, "adapter metadata")
    payload = load_json_object(path)
    try:
        metadata = validate_adapter_metadata(payload).to_mapping()
    except AdapterMetadataValidationError as exc:
        raise ArtifactValidationError(f"{path}: invalid adapter metadata: {exc}") from exc
    if metadata["status"] != "executed":
        raise ArtifactValidationError(f"{path}: adapter metadata status must be executed")
    if metadata["model"]["provider"] != REAL_PROVIDER:
        raise ArtifactValidationError(
            f"{path}: adapter metadata model.provider must be {REAL_PROVIDER!r}"
        )
    return metadata


def validate_metadata_artifact_paths(metadata: dict[str, Any]) -> None:
    model_path = resolve_metadata_path(metadata["model"]["path"])
    adapter_path = resolve_metadata_path(metadata["adapter"]["path"])
    generated_config_path = resolve_metadata_path(metadata["config"]["generated_path"])
    training_input_path = resolve_metadata_path(metadata["dataset"]["training_input"]["path"])
    source_dataset_path = resolve_metadata_path(metadata["dataset"]["path"])
    source_config_path = resolve_metadata_path(metadata["config"]["source_path"])
    require_dir(model_path, "metadata model.path")
    require_dir(adapter_path, "metadata adapter.path")
    require_digest(source_dataset_path, metadata["dataset"]["sha256"], "metadata dataset.path")
    require_digest(
        training_input_path,
        metadata["dataset"]["training_input"]["sha256"],
        "metadata dataset.training_input.path",
    )
    require_digest(source_config_path, metadata["config"]["source_sha256"], "metadata config.source_path")
    require_digest(
        generated_config_path,
        metadata["config"]["generated_sha256"],
        "metadata config.generated_path",
    )
    for marker in metadata["adapter"]["marker_files"]:
        require_file(adapter_path / marker, f"adapter marker {marker}")


def validate_requested_model_path(model_path: Path, metadata: dict[str, Any]) -> None:
    expected_path = resolve_metadata_path(metadata["model"]["path"])
    require_path_matches(model_path, expected_path, "requested model path")


def require_path_matches(path: Path, expected_path: Path, label: str) -> None:
    if path.expanduser().resolve() != expected_path.expanduser().resolve():
        raise ArtifactValidationError(f"{label} must match {expected_path}: {path}")


def validate_run_plan_file(path: Path, metadata: dict[str, Any]) -> None:
    require_file(path, "LoRA run plan")
    run_plan = load_json_object(path)
    expected_paths = {
        "model_path": resolve_metadata_path(metadata["model"]["path"]),
        "adapter_path": resolve_metadata_path(metadata["adapter"]["path"]),
        "dataset_path": resolve_metadata_path(metadata["dataset"]["path"]),
        "train_data_path": resolve_metadata_path(metadata["dataset"]["training_input"]["path"]),
        "mlx_config_path": resolve_metadata_path(metadata["config"]["generated_path"]),
        "config_path": resolve_metadata_path(metadata["config"]["source_path"]),
    }
    if run_plan.get("run_id") != metadata["run_id"]:
        raise ArtifactValidationError(f"{path}: run_id must match adapter metadata")
    for field, expected_path in expected_paths.items():
        value = run_plan.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ArtifactValidationError(f"{path}: {field} must be a non-empty string")
        require_path_matches(Path(value), expected_path, f"{path}: {field}")
    if run_plan.get("dataset_sha256") != metadata["dataset"]["sha256"]:
        raise ArtifactValidationError(f"{path}: dataset_sha256 must match adapter metadata")
    if run_plan.get("config_sha256") != metadata["config"]["source_sha256"]:
        raise ArtifactValidationError(f"{path}: config_sha256 must match adapter metadata")
    training = run_plan.get("training")
    if not isinstance(training, dict):
        raise ArtifactValidationError(f"{path}: training must be an object")
    for field, expected_value in metadata["training"].items():
        if field == "epochs":
            continue
        if training.get(field) != expected_value:
            raise ArtifactValidationError(f"{path}: training.{field} must match adapter metadata")
    mlx_config = run_plan.get("mlx_config")
    if not isinstance(mlx_config, dict):
        raise ArtifactValidationError(f"{path}: mlx_config must be an object")
    require_path_matches(
        Path(str(mlx_config.get("model"))),
        expected_paths["model_path"],
        f"{path}: mlx_config.model",
    )
    require_path_matches(
        Path(str(mlx_config.get("adapter_path"))),
        expected_paths["adapter_path"],
        f"{path}: mlx_config.adapter_path",
    )


def validate_real_provider_identity(
    *,
    base_predictions: Path,
    lora_predictions: Path,
    adapter_metadata: Path,
) -> None:
    comparison = load_lora_comparison_inputs(
        base_path=base_predictions,
        lora_path=lora_predictions,
        adapter_metadata_path=adapter_metadata,
    )
    providers = {
        "base predictions": comparison.base.summary.provider,
        "LoRA predictions": comparison.lora.summary.provider,
    }
    for label, provider in providers.items():
        if provider != REAL_PROVIDER:
            raise ArtifactValidationError(f"{label} provider must be {REAL_PROVIDER!r}")


def expected_eval_mapping(eval_path: Path) -> dict[str, str]:
    require_file(eval_path, "Phase 4 eval set")
    mapping: dict[str, str] = {}
    with eval_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(
                    f"{eval_path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ArtifactValidationError(f"{eval_path}:{line_number}: record must be an object")
            eval_id = record.get("id")
            category = record.get("category")
            if not isinstance(eval_id, str) or not eval_id.strip():
                raise ArtifactValidationError(f"{eval_path}:{line_number}: id must be a non-empty string")
            if not isinstance(category, str) or not category.strip():
                raise ArtifactValidationError(
                    f"{eval_path}:{line_number}: category must be a non-empty string"
                )
            if eval_id in mapping:
                raise ArtifactValidationError(f"{eval_path}:{line_number}: duplicate eval id {eval_id}")
            mapping[eval_id] = category
    if not mapping:
        raise ArtifactValidationError(f"{eval_path}: no eval records found")
    return mapping


def prediction_eval_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for result in load_baseline_results(path):
        if result.eval_id in mapping:
            raise ArtifactValidationError(f"{path}: duplicate eval id {result.eval_id}")
        mapping[result.eval_id] = result.category.value
    return mapping


def validate_phase4_eval_coverage(
    *,
    eval_path: Path,
    base_predictions: Path,
    lora_predictions: Path,
) -> None:
    expected = expected_eval_mapping(eval_path)
    for label, path in (
        ("base predictions", base_predictions),
        ("LoRA predictions", lora_predictions),
    ):
        actual = prediction_eval_mapping(path)
        if actual != expected:
            raise ArtifactValidationError(
                f"{label} must match Phase 4 eval set {eval_path}: "
                f"expected {len(expected)} record(s), got {len(actual)}"
            )


def validate_real_eval_artifacts(
    *,
    model_path: Path,
    base_predictions: Path,
    base_summary: Path,
    base_category_metrics: Path,
    lora_predictions: Path,
    adapter_metadata: Path,
    run_plan: Path,
    eval_path: Path = DEFAULT_EVAL_PATH,
) -> tuple[str, ...]:
    errors = collect_real_eval_artifact_errors(
        model_path=model_path,
        eval_path=eval_path,
        base_predictions=base_predictions,
        base_summary=base_summary,
        base_category_metrics=base_category_metrics,
        lora_predictions=lora_predictions,
        adapter_metadata=adapter_metadata,
        run_plan=run_plan,
    )
    if errors:
        raise ArtifactValidationError(
            "real eval artifact validation failed:\n- " + "\n- ".join(errors)
        )

    comparison = load_lora_comparison_inputs(
        base_path=base_predictions,
        lora_path=lora_predictions,
        adapter_metadata_path=adapter_metadata,
    )
    return (
        f"base_predictions={base_predictions}",
        f"lora_predictions={lora_predictions}",
        f"adapter_id={comparison.lora.summary.adapter_id}",
        f"matched_eval_count={len(comparison.base.results)}",
    )


def collect_real_eval_artifact_errors(
    *,
    model_path: Path,
    eval_path: Path,
    base_predictions: Path,
    base_summary: Path,
    base_category_metrics: Path,
    lora_predictions: Path,
    adapter_metadata: Path,
    run_plan: Path,
) -> tuple[str, ...]:
    errors: list[str] = []

    def record_errors(fn: Any) -> None:
        try:
            fn()
        except (ArtifactValidationError, BaselineResultError) as exc:
            errors.append(str(exc))

    metadata: dict[str, Any] | None = None
    record_errors(lambda: require_dir(model_path.expanduser(), "Qwen base model path"))
    record_errors(lambda: require_file(base_predictions, "base predictions"))
    record_errors(lambda: require_file(lora_predictions, "LoRA predictions"))
    try:
        metadata = validate_adapter_metadata_file(adapter_metadata)
    except ArtifactValidationError as exc:
        errors.append(str(exc))
    if metadata is not None:
        record_errors(lambda: validate_requested_model_path(model_path.expanduser(), metadata))
        record_errors(lambda: validate_metadata_artifact_paths(metadata))
        record_errors(lambda: validate_run_plan_file(run_plan, metadata))
    if base_predictions.is_file():
        record_errors(lambda: validate_summary_matches_predictions(base_summary, base_predictions))
        record_errors(lambda: validate_category_metrics_csv(base_category_metrics, base_predictions))
    if base_predictions.is_file() and lora_predictions.is_file() and adapter_metadata.is_file():
        record_errors(
            lambda: validate_phase4_eval_coverage(
                eval_path=eval_path,
                base_predictions=base_predictions,
                lora_predictions=lora_predictions,
            )
        )
        record_errors(
            lambda: load_lora_comparison_inputs(
                base_path=base_predictions,
                lora_path=lora_predictions,
                adapter_metadata_path=adapter_metadata,
            )
        )
        record_errors(
            lambda: validate_real_provider_identity(
                base_predictions=base_predictions,
                lora_predictions=lora_predictions,
                adapter_metadata=adapter_metadata,
            )
        )
    return tuple(errors)


def summarize_real_eval_artifacts(
    *,
    base_predictions: Path,
    lora_predictions: Path,
    adapter_metadata: Path,
) -> tuple[str, ...]:
    comparison = load_lora_comparison_inputs(
        base_path=base_predictions,
        lora_path=lora_predictions,
        adapter_metadata_path=adapter_metadata,
    )
    return (
        f"base_predictions={base_predictions}",
        f"lora_predictions={lora_predictions}",
        f"adapter_id={comparison.lora.summary.adapter_id}",
        f"matched_eval_count={len(comparison.base.results)}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate host-local real Qwen base and LoRA evaluation artifacts."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=DEFAULT_BASE_RUN_DIR / "qwen_base_predictions.jsonl",
    )
    parser.add_argument(
        "--base-summary",
        type=Path,
        default=DEFAULT_BASE_RUN_DIR / "summary.json",
    )
    parser.add_argument(
        "--base-category-metrics",
        type=Path,
        default=DEFAULT_BASE_RUN_DIR / "category_metrics.csv",
    )
    parser.add_argument(
        "--lora-predictions",
        type=Path,
        default=DEFAULT_LORA_RUN_DIR / "lora_predictions.jsonl",
    )
    parser.add_argument(
        "--adapter-metadata",
        type=Path,
        default=DEFAULT_LORA_RUN_DIR / "adapter_metadata.json",
    )
    parser.add_argument(
        "--run-plan",
        type=Path,
        default=DEFAULT_LORA_RUN_DIR / "run_plan.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        lines = validate_real_eval_artifacts(
            model_path=args.model_path,
            eval_path=args.eval_path,
            base_predictions=args.base_predictions,
            base_summary=args.base_summary,
            base_category_metrics=args.base_category_metrics,
            lora_predictions=args.lora_predictions,
            adapter_metadata=args.adapter_metadata,
            run_plan=args.run_plan,
        )
    except ArtifactValidationError as exc:
        parser.error(str(exc))
    for line in lines:
        print(f"OK: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
