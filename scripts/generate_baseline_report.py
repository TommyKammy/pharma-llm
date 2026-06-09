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
    build_baseline_report,
    load_report_inputs,
    write_baseline_report,
)


DEFAULT_INPUT = (Path("results/baseline/mock_predictions.jsonl"),)
DEFAULT_OUTPUT = Path("results/reports/baseline_report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown baseline report from prediction JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUT,
        help="One or more baseline prediction JSONL files to compare.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="Baseline Evaluation Report")
    parser.add_argument(
        "--mock-notice",
        default=None,
        help="Override the report limitation notice for fixture or live runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report_inputs = load_report_inputs(tuple(args.input))
        report = build_baseline_report(
            report_inputs,
            title=args.title,
            mock_notice=args.mock_notice,
        )
    except BaselineResultError as exc:
        parser.error(str(exc))

    write_baseline_report(args.output, report)
    print(
        f"OK: wrote baseline report for {len(report_inputs)} model(s) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
