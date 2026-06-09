from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.baseline import (  # noqa: E402
    BaselineResultError,
    aggregate_results,
    load_baseline_results,
    write_category_metrics_csv,
    write_summary_json,
)


DEFAULT_INPUT = Path("results/baseline/mock_predictions.jsonl")
DEFAULT_SUMMARY = Path("results/baseline/summary.json")
DEFAULT_CATEGORY_CSV = Path("results/baseline/category_metrics.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate baseline prediction JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--category-csv-output", type=Path, default=DEFAULT_CATEGORY_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        results = load_baseline_results(args.input)
        summary = aggregate_results(results)
    except BaselineResultError as exc:
        parser.error(str(exc))

    write_summary_json(args.summary_output, summary)
    write_category_metrics_csv(args.category_csv_output, summary)
    print(
        f"OK: summarized {summary.total_count} baseline result(s) "
        f"to {args.summary_output} and {args.category_csv_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
