from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pharma_llm_lab.dataset.validators import parse_dataset_type, validate_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate pharma-llm JSONL datasets.")
    parser.add_argument("path", type=Path, help="Path to a JSONL dataset file.")
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="Expected dataset type: sft, dpo, cpt, or eval.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset_type = parse_dataset_type(args.dataset_type)
    except ValueError as exc:
        parser.error(str(exc))

    result = validate_jsonl(args.path, dataset_type)
    if result.ok:
        print(
            f"OK: {result.record_count} {dataset_type.value} record(s) "
            f"validated in {result.path}"
        )
        return 0

    print(f"FAILED: {result.path}", file=sys.stderr)
    for error in result.errors:
        print(error.format(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
