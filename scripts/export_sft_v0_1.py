from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.dataset.validators import validate_jsonl  # noqa: E402
from pharma_llm_lab.dataset.schema import DatasetType  # noqa: E402
from scripts.promote_reviewed_dataset import (  # noqa: E402
    PromotionResult,
    iter_jsonl,
    paths_collide,
    promote_reviewed_dataset,
)
from scripts.check_eval_leakage import check_eval_leakage  # noqa: E402
from scripts.run_qwen_baseline import eval_id_fingerprint  # noqa: E402
from scripts.run_baseline_eval import load_eval_records  # noqa: E402


DATASET_VERSION = "sft-v0.1"
DEFAULT_EVAL_PATH = Path("evals/prompts/phase4_seed.jsonl")
DEFAULT_OUTPUT = Path("data/prepared/sft_v0_1.jsonl")
DEFAULT_MANIFEST = Path("data/prepared/sft_v0_1.manifest.json")
LOCAL_ARTIFACT_POLICY = (
    "Prepared SFT JSONL and manifest are small reproducible artifacts. Raw exports, "
    "Argilla payloads, and model artifacts remain local or ignored."
)


@dataclass(frozen=True)
class SftExportManifest:
    dataset_version: str
    dataset_type: str
    source_path: Path
    output_path: Path
    source_count: int
    promoted_count: int
    skipped_count: int
    failed_count: int
    approved_count: int
    edited_and_approved_count: int
    output_sha256: str
    eval_path: Path
    eval_count: int
    eval_id_sha256: str
    local_artifact_policy: str = LOCAL_ARTIFACT_POLICY

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "dataset_type": self.dataset_type,
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "source_count": self.source_count,
            "promoted_count": self.promoted_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "approved_count": self.approved_count,
            "edited_and_approved_count": self.edited_and_approved_count,
            "output_sha256": self.output_sha256,
            "eval_path": str(self.eval_path),
            "eval_count": self.eval_count,
            "eval_id_sha256": self.eval_id_sha256,
            "local_artifact_policy": self.local_artifact_policy,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_count(path: Path) -> int:
    return sum(1 for _line_number, _record in iter_jsonl(path))


def validate_source_preflight(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    review_candidates: list[str] = []
    for line_number, record in iter_jsonl(path):
        raw_id = record.get("id")
        record_id = raw_id if isinstance(raw_id, str) and raw_id else f"line {line_number}"
        if record_id in seen_ids:
            duplicate_ids.add(record_id)
        seen_ids.add(record_id)
        if record.get("candidate_status") == "review_candidate":
            review_candidates.append(record_id)

    if duplicate_ids:
        raise ValueError("duplicate id value(s): " + ", ".join(sorted(duplicate_ids)))
    if review_candidates:
        raise ValueError(
            "review candidates are not accepted for SFT v0.1 export: "
            + ", ".join(review_candidates)
        )


def review_status_counts(records: tuple[dict[str, Any], ...]) -> tuple[int, int]:
    approved = 0
    edited_and_approved = 0
    for record in records:
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if provenance.get("review_status") == "approved":
            approved += 1
        if provenance.get("review_status") == "edited_and_approved":
            edited_and_approved += 1
    return approved, edited_and_approved


def build_manifest(
    *,
    input_path: Path,
    output_path: Path,
    eval_path: Path,
    result: PromotionResult,
) -> SftExportManifest:
    validation = validate_jsonl(output_path, DatasetType.SFT)
    if not validation.ok:
        formatted = "; ".join(error.format() for error in validation.errors)
        raise ValueError(f"exported SFT dataset did not validate: {formatted}")

    eval_records = load_eval_records(eval_path)
    approved_count, edited_and_approved_count = review_status_counts(result.promoted_records)
    return SftExportManifest(
        dataset_version=DATASET_VERSION,
        dataset_type="sft",
        source_path=input_path,
        output_path=output_path,
        source_count=source_count(input_path),
        promoted_count=len(result.promoted),
        skipped_count=len(result.skipped),
        failed_count=len(result.failed),
        approved_count=approved_count,
        edited_and_approved_count=edited_and_approved_count,
        output_sha256=file_sha256(output_path),
        eval_path=eval_path,
        eval_count=len(eval_records),
        eval_id_sha256=eval_id_fingerprint(tuple(record.id for record in eval_records)),
    )


def write_manifest(path: Path, manifest: SftExportManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def remove_export_artifacts(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def promotion_failure_reason(result: PromotionResult) -> str:
    failed_reasons = [entry.reason for entry in result.failed]
    skipped_reasons = [entry.reason for entry in result.skipped]
    return "; ".join(failed_reasons or skipped_reasons or ["no records promoted"])


def require_complete_promotion(result: PromotionResult) -> None:
    if not result.ok:
        raise ValueError(f"SFT v0.1 export failed: {promotion_failure_reason(result)}")
    if result.skipped:
        raise ValueError(
            "SFT v0.1 export failed: skipped source records are not allowed: "
            + "; ".join(entry.reason for entry in result.skipped)
        )


def require_no_eval_leakage(*, eval_path: Path, output_path: Path) -> None:
    findings = check_eval_leakage(
        eval_paths=(eval_path,),
        training_paths=(output_path,),
    )
    if findings:
        raise ValueError(
            "SFT v0.1 export failed: eval/training leakage detected: "
            + "; ".join(finding.format() for finding in findings)
        )


def export_sft_v0_1(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    eval_path: Path,
) -> SftExportManifest:
    if paths_collide(input_path, output_path):
        raise ValueError("input and output paths must differ")
    if paths_collide(input_path, manifest_path):
        raise ValueError("input and manifest paths must differ")
    if paths_collide(output_path, manifest_path):
        raise ValueError("output and manifest paths must differ")

    validate_source_preflight(input_path)
    result = promote_reviewed_dataset(
        input_path,
        output_path,
        dataset_type_value="sft",
    )

    try:
        require_complete_promotion(result)
        require_no_eval_leakage(eval_path=eval_path, output_path=output_path)
        manifest = build_manifest(
            input_path=input_path,
            output_path=output_path,
            eval_path=eval_path,
            result=result,
        )
        write_manifest(manifest_path, manifest)
        return manifest
    except ValueError:
        remove_export_artifacts(output_path, manifest_path)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export approved-only SFT dataset v0.1 for Qwen LoRA training."
    )
    parser.add_argument("input", type=Path, help="Reviewed SFT JSONL path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = export_sft_v0_1(
            input_path=args.input,
            output_path=args.output,
            manifest_path=args.manifest,
            eval_path=args.eval_path,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(json.dumps(manifest.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"OK: exported {manifest.promoted_count} SFT v0.1 record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
