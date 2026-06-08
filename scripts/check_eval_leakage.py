from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.dataset import (  # noqa: E402
    CptRecord,
    DpoRecord,
    EvalRecord,
    SftRecord,
    parse_record,
)
from pharma_llm_lab.dataset.schema import SchemaError  # noqa: E402
from pharma_llm_lab.dataset.validators import validate_record_policy  # noqa: E402


@dataclass(frozen=True)
class LoadedRecord:
    path: Path
    line_number: int
    raw: dict[str, Any]
    parsed: Any

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line_number}"


@dataclass(frozen=True)
class LeakageFinding:
    kind: str
    eval_location: str
    training_location: str
    detail: str

    def format(self) -> str:
        return (
            f"{self.kind}: {self.detail} "
            f"(eval={self.eval_location}, training={self.training_location})"
        )


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            yield line_number, value


def load_records(paths: tuple[Path, ...]) -> tuple[LoadedRecord, ...]:
    records: list[LoadedRecord] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"path is not a file: {path}")
        for line_number, raw_record in iter_jsonl(path):
            try:
                parsed = parse_record(raw_record)
            except SchemaError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(
                LoadedRecord(
                    path=path,
                    line_number=line_number,
                    raw=raw_record,
                    parsed=parsed,
                )
            )
    return tuple(records)


def require_eval_records(records: tuple[LoadedRecord, ...]) -> None:
    for record in records:
        if not isinstance(record.parsed, EvalRecord):
            raise ValueError(f"{record.location}: expected eval record")


def require_training_records(records: tuple[LoadedRecord, ...]) -> None:
    for record in records:
        if not isinstance(record.parsed, SftRecord | DpoRecord | CptRecord):
            raise ValueError(f"{record.location}: expected sft, dpo, or cpt record")
        policy_errors = validate_record_policy(
            record.parsed,
            expected_dataset_type=record.parsed.dataset_type,
            line_number=record.line_number,
        )
        if policy_errors:
            formatted_errors = "; ".join(error.message for error in policy_errors)
            raise ValueError(f"{record.location}: {formatted_errors}")


def training_texts(record: LoadedRecord) -> tuple[tuple[str, str], ...]:
    parsed = record.parsed
    if isinstance(parsed, SftRecord):
        return (("prompt", parsed.prompt), ("response", parsed.response))
    if isinstance(parsed, DpoRecord):
        return (
            ("prompt", parsed.prompt),
            ("chosen", parsed.chosen),
            ("rejected", parsed.rejected),
        )
    if isinstance(parsed, CptRecord):
        return (("text", parsed.text),)
    return ()


def eval_texts(record: LoadedRecord) -> tuple[tuple[str, str], ...]:
    parsed = record.parsed
    if isinstance(parsed, EvalRecord):
        return (("prompt", parsed.prompt),)
    return ()


def check_eval_leakage(
    *,
    eval_paths: tuple[Path, ...],
    training_paths: tuple[Path, ...],
) -> tuple[LeakageFinding, ...]:
    eval_records = load_records(eval_paths)
    training_records = load_records(training_paths)
    require_eval_records(eval_records)
    require_training_records(training_records)

    findings: list[LeakageFinding] = []
    eval_records_by_id = {record.parsed.id: record for record in eval_records}
    for training_record in training_records:
        leaked_eval_record = eval_records_by_id.get(training_record.parsed.id)
        if leaked_eval_record is not None:
            findings.append(
                LeakageFinding(
                    kind="duplicate_id",
                    eval_location=leaked_eval_record.location,
                    training_location=training_record.location,
                    detail=f"id {training_record.parsed.id!r} appears in eval and training",
                )
            )

    eval_texts_by_normalized_value: dict[str, tuple[LoadedRecord, str]] = {}
    for eval_record in eval_records:
        for field_name, value in eval_texts(eval_record):
            normalized = normalize_text(value)
            if normalized:
                eval_texts_by_normalized_value[normalized] = (eval_record, field_name)

    for training_record in training_records:
        for training_field_name, training_value in training_texts(training_record):
            normalized = normalize_text(training_value)
            leaked = eval_texts_by_normalized_value.get(normalized)
            if leaked is None:
                continue
            eval_record, eval_field_name = leaked
            findings.append(
                LeakageFinding(
                    kind="duplicate_text",
                    eval_location=eval_record.location,
                    training_location=training_record.location,
                    detail=(
                        f"eval {eval_field_name} duplicates training "
                        f"{training_field_name}"
                    ),
                )
            )

    return tuple(findings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that eval prompt records are not reused in training JSONL."
    )
    parser.add_argument(
        "--eval",
        required=True,
        action="append",
        type=Path,
        dest="eval_paths",
        help="Evaluation prompt JSONL path. May be provided more than once.",
    )
    parser.add_argument(
        "--training",
        required=True,
        action="append",
        type=Path,
        dest="training_paths",
        help="Prepared training JSONL path. May be provided more than once.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        findings = check_eval_leakage(
            eval_paths=tuple(args.eval_paths),
            training_paths=tuple(args.training_paths),
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not findings:
        print("OK: no eval/training leakage detected")
        return 0

    print("FAILED: eval/training leakage detected", file=sys.stderr)
    for finding in findings:
        print(finding.format(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
