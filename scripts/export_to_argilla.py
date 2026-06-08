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

from pharma_llm_lab.dataset.schema import parse_record  # noqa: E402


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


def review_fields(record: dict[str, Any]) -> dict[str, Any]:
    dataset_type = record.get("dataset_type")
    if dataset_type == "sft":
        keys = ("prompt", "response")
    elif dataset_type == "dpo":
        keys = ("prompt", "chosen", "rejected")
    elif dataset_type == "cpt":
        keys = ("text",)
    elif dataset_type == "eval":
        keys = ("prompt", "expected_points")
    else:
        raise ValueError(f"unsupported dataset_type for review export: {dataset_type!r}")

    return {key: record[key] for key in keys if key in record}


def to_review_payload(record: dict[str, Any]) -> dict[str, Any]:
    parse_record(record)
    argilla = record.get("argilla")
    if not isinstance(argilla, dict):
        raise ValueError("record must include an argilla object")

    return {
        "id": record["id"],
        "dataset_type": record["dataset_type"],
        "fields": review_fields(record),
        "provenance": record["provenance"],
        "argilla": argilla,
        "review": {
            "review_status": record["provenance"]["review_status"],
            "human_reviewer": record["provenance"].get("human_reviewer"),
            "review_date": record["provenance"].get("review_date"),
            "risk_flags": record["provenance"].get("risk_flags", []),
            "review_notes": "",
        },
        "original_record": record,
    }


def export_records(input_path: Path, output_path: Path) -> int:
    if not input_path.is_file():
        raise ValueError(f"input path is not a file: {input_path}")

    payloads = [to_review_payload(record) for _, record in iter_jsonl(input_path)]
    if not payloads:
        raise ValueError("input file contains no records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads) + "\n",
        encoding="utf-8",
    )
    return len(payloads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export local candidate JSONL into offline Argilla review JSONL."
    )
    parser.add_argument("input", type=Path, help="Candidate JSONL path.")
    parser.add_argument("output", type=Path, help="Review payload JSONL path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        count = export_records(args.input, args.output)
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"OK: exported {count} review record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
