from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.dataset.validators import parse_dataset_type, validate_jsonl  # noqa: E402


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
