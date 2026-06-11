from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.baseline import BaselineResultError, load_lora_comparison_inputs  # noqa: E402
from pharma_llm_lab.baseline.results import aggregate_results, load_baseline_results  # noqa: E402
from pharma_llm_lab.training import AdapterMetadataValidationError, validate_adapter_metadata  # noqa: E402

DEFAULT_LOCAL_ROOT = Path("/Users/tsinfra/Dev/pharma-llm/local")
DEFAULT_BASE_RUN_DIR = DEFAULT_LOCAL_ROOT / "runs" / "baseline" / "phase6-qwen-base"
DEFAULT_LORA_RUN_DIR = DEFAULT_LOCAL_ROOT / "runs" / "qwen_sft_lora_r16_v1"
DEFAULT_MODEL_PATH = DEFAULT_LOCAL_ROOT / "models" / "qwen3.6-27b-base"


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


def validate_summary_matches_predictions(summary_path: Path, predictions_path: Path) -> None:
    require_file(summary_path, "base summary")
    summary_payload = load_json_object(summary_path)
    prediction_summary = aggregate_results(load_baseline_results(predictions_path)).to_mapping()
    expected_fields = ("run_id", "model_id", "provider", "adapter_id", "total_count")
    mismatches = [
        field
        for field in expected_fields
        if summary_payload.get(field) != prediction_summary[field]
    ]
    if mismatches:
        raise ArtifactValidationError(
            f"{summary_path}: does not match base predictions for field(s): "
            + ", ".join(mismatches)
        )


def validate_category_metrics_csv(path: Path) -> None:
    require_file(path, "base category metrics")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ArtifactValidationError(f"{path}: CSV header is missing")
        required = {"run_id", "model_id", "provider", "adapter_id", "category", "count"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ArtifactValidationError(
                f"{path}: missing required column(s): " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise ArtifactValidationError(f"{path}: at least one category row is required")


def validate_adapter_metadata_file(path: Path) -> dict[str, Any]:
    require_file(path, "adapter metadata")
    payload = load_json_object(path)
    try:
        metadata = validate_adapter_metadata(payload).to_mapping()
    except AdapterMetadataValidationError as exc:
        raise ArtifactValidationError(f"{path}: invalid adapter metadata: {exc}") from exc
    if metadata["status"] != "executed":
        raise ArtifactValidationError(f"{path}: adapter metadata status must be executed")
    return metadata


def validate_metadata_artifact_paths(metadata: dict[str, Any]) -> None:
    model_path = Path(metadata["model"]["path"]).expanduser()
    adapter_path = Path(metadata["adapter"]["path"]).expanduser()
    generated_config_path = Path(metadata["config"]["generated_path"]).expanduser()
    training_input_path = Path(metadata["dataset"]["training_input"]["path"]).expanduser()
    require_dir(model_path, "metadata model.path")
    require_dir(adapter_path, "metadata adapter.path")
    require_file(generated_config_path, "metadata config.generated_path")
    require_file(training_input_path, "metadata dataset.training_input.path")
    for marker in metadata["adapter"]["marker_files"]:
        require_file(adapter_path / marker, f"adapter marker {marker}")


def validate_real_eval_artifacts(
    *,
    model_path: Path,
    base_predictions: Path,
    base_summary: Path,
    base_category_metrics: Path,
    lora_predictions: Path,
    adapter_metadata: Path,
    run_plan: Path,
) -> tuple[str, ...]:
    errors = collect_real_eval_artifact_errors(
        model_path=model_path,
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
    record_errors(lambda: require_file(run_plan, "LoRA run plan"))
    try:
        metadata = validate_adapter_metadata_file(adapter_metadata)
    except ArtifactValidationError as exc:
        errors.append(str(exc))
    if metadata is not None:
        record_errors(lambda: validate_metadata_artifact_paths(metadata))
    if base_predictions.is_file():
        record_errors(lambda: validate_summary_matches_predictions(base_summary, base_predictions))
        record_errors(lambda: validate_category_metrics_csv(base_category_metrics))
    if base_predictions.is_file() and lora_predictions.is_file() and adapter_metadata.is_file():
        record_errors(
            lambda: load_lora_comparison_inputs(
                base_path=base_predictions,
                lora_path=lora_predictions,
                adapter_metadata_path=adapter_metadata,
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
