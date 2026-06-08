from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.dataset.provenance import ReviewStatus, SourceType  # noqa: E402
from pharma_llm_lab.dataset.schema import parse_record  # noqa: E402

SUPPORTED_REVIEW_STATUSES = {
    ReviewStatus.APPROVED.value,
    ReviewStatus.REJECTED.value,
    ReviewStatus.NEEDS_EDIT.value,
    ReviewStatus.EDITED_AND_APPROVED.value,
    ReviewStatus.RISK_FLAGGED.value,
}
REVIEWED_STATUSES_REQUIRE_METADATA = {
    ReviewStatus.APPROVED.value,
    ReviewStatus.EDITED_AND_APPROVED.value,
    ReviewStatus.RISK_FLAGGED.value,
}
REVIEW_FIELD_NAMES_BY_DATASET_TYPE = {
    "sft": frozenset({"prompt", "response"}),
    "dpo": frozenset({"prompt", "chosen", "rejected"}),
    "cpt": frozenset({"text"}),
    "eval": frozenset({"prompt", "expected_points"}),
}
REVIEW_MUTABLE_PROVENANCE_FIELDS = {
    "review_status",
    "human_reviewer",
    "review_date",
    "risk_flags",
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


def require_review_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    review = payload.get("review")
    if not isinstance(review, dict):
        raise ValueError("review payload must include a review object")
    return review


def require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def require_risk_flags(value: Any, *, require_non_empty: bool = False) -> list[str]:
    if value is None:
        if require_non_empty:
            raise ValueError("risk_flagged reviews must include at least one risk flag")
        return []
    if not isinstance(value, list):
        raise ValueError("risk_flags must be a list")
    if any(not isinstance(flag, str) or not flag.strip() for flag in value):
        raise ValueError("risk_flags must contain only non-empty strings")
    if require_non_empty and not value:
        raise ValueError("risk_flagged reviews must include at least one risk flag")
    return value


def expected_review_field_names(original_record: dict[str, Any]) -> frozenset[str]:
    dataset_type = original_record.get("dataset_type")
    try:
        return REVIEW_FIELD_NAMES_BY_DATASET_TYPE[str(dataset_type)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dataset_type for reviewed fields: {dataset_type!r}"
        ) from exc


def require_fields_mapping(
    payload: dict[str, Any],
    *,
    original_record: dict[str, Any],
) -> dict[str, Any]:
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("review payload must include a non-empty fields object")
    allowed_fields = expected_review_field_names(original_record)
    unexpected_fields = sorted(set(fields) - allowed_fields)
    missing_fields = sorted(allowed_fields - set(fields))
    if unexpected_fields:
        raise ValueError(
            "review fields contain unsupported key(s): "
            + ", ".join(repr(field) for field in unexpected_fields)
        )
    if missing_fields:
        raise ValueError(
            "review fields are missing required key(s): "
            + ", ".join(repr(field) for field in missing_fields)
        )
    return fields


def merge_reviewed_fields(original_record: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    imported_record = {
        key: value
        for key, value in original_record.items()
        if key != "argilla"
    }
    imported_record.update(fields)
    return imported_record


def validate_payload_identity(
    payload: dict[str, Any],
    *,
    original_record: dict[str, Any],
) -> None:
    for field_name in ("id", "dataset_type"):
        if payload.get(field_name) != original_record.get(field_name):
            raise ValueError(
                f"payload {field_name} does not match original_record {field_name}"
            )


def require_top_level_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("review payload must include a provenance object")
    return provenance


def validate_immutable_provenance(
    *,
    exported_provenance: dict[str, Any],
    original_provenance: dict[str, Any],
) -> None:
    keys = sorted(set(exported_provenance) | set(original_provenance))
    mismatched_fields = [
        key
        for key in keys
        if key not in REVIEW_MUTABLE_PROVENANCE_FIELDS
        and exported_provenance.get(key) != original_provenance.get(key)
    ]
    if mismatched_fields:
        raise ValueError(
            "original_record provenance does not match exported provenance for "
            "immutable field(s): "
            + ", ".join(repr(field) for field in mismatched_fields)
        )


def apply_review(payload: dict[str, Any]) -> dict[str, Any]:
    original_record = payload.get("original_record")
    if not isinstance(original_record, dict):
        raise ValueError("review payload must include an original_record object")

    validate_payload_identity(payload, original_record=original_record)
    review = require_review_mapping(payload)
    fields = require_fields_mapping(payload, original_record=original_record)

    provenance = original_record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("original_record must include a provenance object")
    validate_immutable_provenance(
        exported_provenance=require_top_level_provenance(payload),
        original_provenance=provenance,
    )

    review_status = require_string(review.get("review_status"), "review_status")
    if review_status not in SUPPORTED_REVIEW_STATUSES:
        allowed = ", ".join(sorted(SUPPORTED_REVIEW_STATUSES))
        raise ValueError(f"invalid review_status {review_status!r}; expected one of: {allowed}")

    updated_provenance = dict(provenance)
    updated_provenance["review_status"] = review_status
    updated_provenance["risk_flags"] = require_risk_flags(
        review.get("risk_flags", []),
        require_non_empty=review_status == ReviewStatus.RISK_FLAGGED.value,
    )

    if (
        review_status == ReviewStatus.EDITED_AND_APPROVED.value
        and updated_provenance.get("source_type") == SourceType.AI_CANDIDATE_UNREVIEWED.value
    ):
        updated_provenance["source_type"] = SourceType.HUMAN_EDITED_AI_ASSISTED.value
    elif (
        review_status == ReviewStatus.APPROVED.value
        and updated_provenance.get("source_type") == SourceType.AI_CANDIDATE_UNREVIEWED.value
    ):
        raise ValueError("ai_candidate_unreviewed requires edited_and_approved review")
    elif (
        review_status == ReviewStatus.EDITED_AND_APPROVED.value
        and updated_provenance.get("source_type") == SourceType.RAW_AI_OUTPUT.value
    ):
        raise ValueError("raw_ai_output cannot be marked edited_and_approved")
    elif (
        review_status == ReviewStatus.APPROVED.value
        and updated_provenance.get("source_type") == SourceType.RAW_AI_OUTPUT.value
    ):
        raise ValueError("raw_ai_output cannot be marked approved")

    if review_status in REVIEWED_STATUSES_REQUIRE_METADATA:
        updated_provenance["human_reviewer"] = require_string(
            review.get("human_reviewer"), "human_reviewer"
        )
        updated_provenance["review_date"] = require_string(
            review.get("review_date"), "review_date"
        )
    else:
        updated_provenance["human_reviewer"] = review.get("human_reviewer")
        updated_provenance["review_date"] = review.get("review_date")

    imported_record = merge_reviewed_fields(original_record, fields)
    imported_record["provenance"] = updated_provenance
    parse_record(imported_record)
    return imported_record


def import_records(input_path: Path, output_path: Path) -> int:
    if not input_path.is_file():
        raise ValueError(f"input path is not a file: {input_path}")

    records = []
    for line_number, payload in iter_jsonl(input_path):
        try:
            records.append(apply_review(payload))
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc

    if not records:
        raise ValueError("input file contains no records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return len(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import offline Argilla review JSONL into dataset JSONL."
    )
    parser.add_argument("input", type=Path, help="Reviewed payload JSONL path.")
    parser.add_argument("output", type=Path, help="Imported dataset JSONL path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        count = import_records(args.input, args.output)
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"OK: imported {count} reviewed record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
