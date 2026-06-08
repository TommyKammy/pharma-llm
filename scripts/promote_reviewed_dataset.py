from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.dataset.schema import SchemaError, parse_record  # noqa: E402
from pharma_llm_lab.dataset.provenance import ReviewStatus, SourceType  # noqa: E402
from pharma_llm_lab.dataset.validators import (  # noqa: E402
    ValidationError,
    parse_dataset_type,
    validate_jsonl,
    validate_record_policy,
)


TRAINING_DATASET_TYPES = {"sft", "dpo", "cpt"}


@dataclass(frozen=True)
class PromotionAuditEntry:
    id: str
    status: str
    reason: str


@dataclass(frozen=True)
class PromotionResult:
    promoted_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    promoted: tuple[PromotionAuditEntry, ...] = field(default_factory=tuple)
    skipped: tuple[PromotionAuditEntry, ...] = field(default_factory=tuple)
    failed: tuple[PromotionAuditEntry, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return bool(self.promoted_records) and not self.failed

    def audit_summary(self) -> dict[str, object]:
        return {
            "promoted": len(self.promoted),
            "skipped": len(self.skipped),
            "failed": len(self.failed),
            "entries": {
                "promoted": [entry.__dict__ for entry in self.promoted],
                "skipped": [entry.__dict__ for entry in self.skipped],
                "failed": [entry.__dict__ for entry in self.failed],
            },
        }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: record must be a JSON object")
            yield line_number, value


def record_id(raw_record: dict[str, Any], *, line_number: int) -> str:
    raw_id = raw_record.get("id")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    return f"line {line_number}"


def format_policy_errors(errors: list[ValidationError]) -> str:
    return "; ".join(error.message for error in errors)


def remove_stale_output(path: Path) -> None:
    if path.is_file():
        path.unlink()


def parse_training_dataset_type(value: str) -> str:
    dataset_type = parse_dataset_type(value)
    if dataset_type.value not in TRAINING_DATASET_TYPES:
        allowed = ", ".join(sorted(TRAINING_DATASET_TYPES))
        raise ValueError(f"promotion dataset type must be one of: {allowed}")
    return dataset_type.value


def prepared_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw_record.items() if key != "argilla"}


def review_workflow_policy_failure(record: Any) -> str | None:
    if record.provenance.is_reviewed_for_training:
        if not record.provenance.human_reviewer:
            return "reviewed records require human_reviewer before promotion"
        if not record.provenance.review_date:
            return "reviewed records require review_date before promotion"
    if (
        record.provenance.source_type is SourceType.HUMAN_EDITED_AI_ASSISTED
        and record.provenance.review_status is ReviewStatus.APPROVED
    ):
        return "human_edited_ai_assisted requires edited_and_approved review"
    if (
        record.provenance.ai_assisted
        and record.provenance.review_status is ReviewStatus.APPROVED
    ):
        return "ai_assisted records require edited_and_approved review"
    if (
        record.provenance.ai_assisted
        and record.provenance.review_status is ReviewStatus.EDITED_AND_APPROVED
        and record.provenance.source_type is not SourceType.HUMAN_EDITED_AI_ASSISTED
    ):
        return (
            "edited_and_approved ai_assisted records require "
            "human_edited_ai_assisted source_type"
        )
    return None


def paths_collide(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def evaluate_promotion(
    input_path: Path,
    *,
    dataset_type_value: str,
) -> PromotionResult:
    dataset_type = parse_dataset_type(parse_training_dataset_type(dataset_type_value))
    promoted_records: list[dict[str, Any]] = []
    promoted: list[PromotionAuditEntry] = []
    skipped: list[PromotionAuditEntry] = []
    failed: list[PromotionAuditEntry] = []

    if not input_path.is_file():
        return PromotionResult(
            failed=(
                PromotionAuditEntry(
                    id=str(input_path),
                    status="failed",
                    reason=f"input path is not a file: {input_path}",
                ),
            )
        )

    try:
        raw_records = list(iter_jsonl(input_path))
    except ValueError as exc:
        return PromotionResult(
            failed=(
                PromotionAuditEntry(
                    id=str(input_path),
                    status="failed",
                    reason=str(exc),
                ),
            )
        )

    if not raw_records:
        return PromotionResult(
            failed=(
                PromotionAuditEntry(
                    id=str(input_path),
                    status="failed",
                    reason="input file contains no records",
                ),
            )
        )

    for line_number, raw_record in raw_records:
        current_id = record_id(raw_record, line_number=line_number)
        try:
            record = parse_record(raw_record)
        except SchemaError as exc:
            failed.append(
                PromotionAuditEntry(
                    id=current_id,
                    status="failed",
                    reason=f"schema error: {exc}",
                )
            )
            continue

        policy_errors = validate_record_policy(
            record,
            expected_dataset_type=dataset_type,
            line_number=line_number,
        )
        if policy_errors:
            skipped.append(
                PromotionAuditEntry(
                    id=current_id,
                    status="skipped",
                    reason=format_policy_errors(policy_errors),
                )
            )
            continue

        review_policy_failure = review_workflow_policy_failure(record)
        if review_policy_failure:
            skipped.append(
                PromotionAuditEntry(
                    id=current_id,
                    status="skipped",
                    reason=review_policy_failure,
                )
            )
            continue

        promoted_records.append(prepared_record(raw_record))
        promoted.append(
            PromotionAuditEntry(
                id=record.id,
                status="promoted",
                reason="eligible for prepared training dataset",
            )
        )

    return PromotionResult(
        promoted_records=tuple(promoted_records),
        promoted=tuple(promoted),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )


def write_jsonl(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def promote_reviewed_dataset(
    input_path: Path,
    output_path: Path,
    *,
    dataset_type_value: str,
) -> PromotionResult:
    if paths_collide(input_path, output_path):
        return PromotionResult(
            failed=(
                PromotionAuditEntry(
                    id=str(output_path),
                    status="failed",
                    reason="input and output paths must differ",
                ),
            )
        )

    result = evaluate_promotion(input_path, dataset_type_value=dataset_type_value)
    if not result.ok:
        remove_stale_output(output_path)
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)

    write_jsonl(temp_path, result.promoted_records)
    validation = validate_jsonl(temp_path, parse_dataset_type(dataset_type_value))
    if not validation.ok:
        temp_path.unlink(missing_ok=True)
        remove_stale_output(output_path)
        failed = tuple(result.failed) + tuple(
            PromotionAuditEntry(
                id=str(temp_path),
                status="failed",
                reason=error.format(),
            )
            for error in validation.errors
        )
        return PromotionResult(
            promoted_records=result.promoted_records,
            promoted=result.promoted,
            skipped=result.skipped,
            failed=failed,
        )
    temp_path.replace(output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote approved reviewed records into prepared training JSONL."
    )
    parser.add_argument("input", type=Path, help="Reviewed dataset JSONL path.")
    parser.add_argument("output", type=Path, help="Prepared training JSONL output path.")
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="Prepared training dataset type: sft, dpo, or cpt.",
    )
    parser.add_argument("--audit-output", type=Path, help="Optional audit summary JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        parse_training_dataset_type(args.dataset_type)
    except ValueError as exc:
        parser.error(str(exc))

    if args.audit_output and paths_collide(args.audit_output, args.output):
        parser.error("--audit-output must not be the same path as output")
    if args.audit_output and paths_collide(args.audit_output, args.input):
        parser.error("--audit-output must not be the same path as input")
    if paths_collide(args.input, args.output):
        parser.error("input and output paths must differ")

    result = promote_reviewed_dataset(
        args.input,
        args.output,
        dataset_type_value=args.dataset_type,
    )
    summary = result.audit_summary()
    rendered_summary = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered_summary)

    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(rendered_summary + "\n", encoding="utf-8")

    if result.ok:
        print(f"OK: promoted {len(result.promoted)} record(s) to {args.output}")
        return 0

    print(
        "FAILED: no output written" if not result.promoted_records else "FAILED: output invalid",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
