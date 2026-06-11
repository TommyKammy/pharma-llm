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
    build_lora_comparison_report,
    load_lora_comparison_inputs,
    write_lora_comparison_report,
)

DEFAULT_BASE_INPUT = Path("results/baseline/mock_predictions.jsonl")
DEFAULT_LORA_INPUT = Path("results/lora/mock_qwen_sft_lora_v1_predictions.jsonl")
DEFAULT_OUTPUT = Path("results/reports/lora_comparison_report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown base-vs-LoRA comparison and safety report."
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--lora-input", type=Path, default=DEFAULT_LORA_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-metadata", type=Path, default=None)
    parser.add_argument("--title", default="Base vs LoRA Evaluation and Safety Report")
    parser.add_argument(
        "--limitation-notice",
        default=None,
        help="Override the report limitation notice for local reviewed runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        comparison = load_lora_comparison_inputs(
            base_path=args.base_input,
            lora_path=args.lora_input,
            adapter_metadata_path=args.adapter_metadata,
        )
        report = build_lora_comparison_report(
            comparison,
            title=args.title,
            limitation_notice=args.limitation_notice,
        )
    except BaselineResultError as exc:
        parser.error(str(exc))

    write_lora_comparison_report(args.output, report)
    print(f"OK: wrote LoRA comparison report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
