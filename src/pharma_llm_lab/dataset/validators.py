"""Dataset JSONL validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pharma_llm_lab.dataset.schema import (
    DatasetRecord,
    DatasetType,
    EvalRecord,
    SchemaError,
    parse_record,
)


class ValidationMode(str):
    TRAINING = "training"
    EVAL = "eval"


@dataclass(frozen=True)
class ValidationError:
    line_number: int
    message: str

    def format(self) -> str:
        return f"line {self.line_number}: {self.message}"


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    dataset_type: DatasetType
    records: tuple[DatasetRecord, ...] = field(default_factory=tuple)
    errors: tuple[ValidationError, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def record_count(self) -> int:
        return len(self.records)


def parse_dataset_type(value: str) -> DatasetType:
    try:
        return DatasetType(value)
    except ValueError as exc:
        allowed = ", ".join(dataset_type.value for dataset_type in DatasetType)
        raise ValueError(f"dataset type must be one of: {allowed}") from exc


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | ValidationError]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, ValidationError(
                    line_number, f"invalid JSON: {exc.msg}"
                )
                continue
            if not isinstance(value, dict):
                yield line_number, ValidationError(
                    line_number, "record must be a JSON object"
                )
                continue
            yield line_number, value


def validate_record_policy(
    record: DatasetRecord,
    *,
    expected_dataset_type: DatasetType,
    line_number: int,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if record.dataset_type is not expected_dataset_type:
        errors.append(
            ValidationError(
                line_number,
                "dataset_type mismatch: "
                f"expected {expected_dataset_type.value!r}, got {record.dataset_type.value!r}",
            )
        )

    if expected_dataset_type is DatasetType.EVAL:
        return errors

    if isinstance(record, EvalRecord) or record.provenance.is_blocked_for_training:
        errors.append(
            ValidationError(
                line_number,
                "record is not eligible for training "
                f"(source_type={record.provenance.source_type.value!r}, "
                f"review_status={record.provenance.review_status.value!r})",
            )
        )

    return errors


def validate_jsonl(path: Path, dataset_type: DatasetType) -> ValidationResult:
    records: list[DatasetRecord] = []
    errors: list[ValidationError] = []

    if not path.exists():
        return ValidationResult(
            path=path,
            dataset_type=dataset_type,
            errors=(ValidationError(0, f"file does not exist: {path}"),),
        )
    if not path.is_file():
        return ValidationResult(
            path=path,
            dataset_type=dataset_type,
            errors=(ValidationError(0, f"path is not a file: {path}"),),
        )

    for line_number, item in iter_jsonl(path):
        if isinstance(item, ValidationError):
            errors.append(item)
            continue

        try:
            record = parse_record(item)
        except SchemaError as exc:
            errors.append(ValidationError(line_number, str(exc)))
            continue

        errors.extend(
            validate_record_policy(
                record,
                expected_dataset_type=dataset_type,
                line_number=line_number,
            )
        )
        records.append(record)

    if not records and not errors:
        errors.append(ValidationError(0, "file contains no records"))

    return ValidationResult(
        path=path,
        dataset_type=dataset_type,
        records=tuple(records),
        errors=tuple(errors),
    )
