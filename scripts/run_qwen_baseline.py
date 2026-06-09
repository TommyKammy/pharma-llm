from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.inference import ModelIdentity  # noqa: E402
from scripts.run_baseline_eval import load_eval_records  # noqa: E402


DEFAULT_INPUT = Path("evals/prompts/phase4_seed.jsonl")
DEFAULT_LOCAL_ROOT = Path("/Users/tsinfra/Dev/pharma-llm/local/runs/baseline")
DEFAULT_RUN_ID = "phase6-qwen-base"
DEFAULT_MODEL_PATH = Path("/Users/tsinfra/Dev/pharma-llm/local/models/qwen3.6-27b-base")
QWEN_BASE_MODEL = ModelIdentity(model_id="qwen/qwen3.6-27b-base", provider="mlx")


@dataclass(frozen=True)
class QwenBaselinePlan:
    run_id: str
    model: ModelIdentity
    model_path: Path
    eval_path: Path
    eval_count: int
    eval_id_sha256: str
    prediction_output: Path
    summary_output: Path
    category_csv_output: Path
    report_output: Path
    max_tokens: int
    temperature: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model.to_mapping(),
            "model_path": str(self.model_path),
            "eval_path": str(self.eval_path),
            "eval_count": self.eval_count,
            "eval_id_sha256": self.eval_id_sha256,
            "prediction_output": str(self.prediction_output),
            "summary_output": str(self.summary_output),
            "category_csv_output": str(self.category_csv_output),
            "report_output": str(self.report_output),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "post_processing_commands": [
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/summarize_baseline_results.py",
                    "--input",
                    str(self.prediction_output),
                    "--summary-output",
                    str(self.summary_output),
                    "--category-csv-output",
                    str(self.category_csv_output),
                ],
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/generate_baseline_report.py",
                    "--input",
                    str(self.prediction_output),
                    "--output",
                    str(self.report_output),
                    "--mock-notice",
                    "Real Qwen base baseline run. Interpret quality only after confirming this "
                    "uses the approved Phase 4 eval id set and local model path recorded here.",
                ],
            ],
        }


def require_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def require_non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def eval_id_fingerprint(eval_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(eval_ids).encode("utf-8")).hexdigest()


def ensure_local_output(path: Path, local_root: Path, field_name: str) -> Path:
    resolved_path = path.expanduser().resolve()
    resolved_root = local_root.expanduser().resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{field_name} must be under local baseline root: {resolved_root}")
    return resolved_path


def build_plan(
    *,
    eval_path: Path,
    local_root: Path,
    run_id: str,
    model_path: Path,
    report_output: Path,
    max_tokens: int,
    temperature: float,
) -> QwenBaselinePlan:
    if not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    records = load_eval_records(eval_path)
    eval_ids = tuple(record.id for record in records)
    run_root = local_root / run_id
    prediction_output = ensure_local_output(
        run_root / "qwen_base_predictions.jsonl",
        local_root,
        "prediction_output",
    )
    summary_output = ensure_local_output(run_root / "summary.json", local_root, "summary_output")
    category_csv_output = ensure_local_output(
        run_root / "category_metrics.csv",
        local_root,
        "category_csv_output",
    )

    return QwenBaselinePlan(
        run_id=run_id,
        model=QWEN_BASE_MODEL,
        model_path=model_path.expanduser().resolve(),
        eval_path=eval_path.expanduser().resolve(),
        eval_count=len(records),
        eval_id_sha256=eval_id_fingerprint(eval_ids),
        prediction_output=prediction_output,
        summary_output=summary_output,
        category_csv_output=category_csv_output,
        report_output=report_output,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def write_plan(path: Path, plan: QwenBaselinePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a real Qwen base baseline run plan without downloading weights."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("results/reports/qwen_base_baseline_report.md"),
    )
    parser.add_argument("--max-tokens", type=require_positive_int, default=512)
    parser.add_argument("--temperature", type=require_non_negative_float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the run plan. Required until real MLX execution is wired.",
    )
    parser.add_argument(
        "--write-plan",
        type=Path,
        default=None,
        help="Optional local JSON path for the generated run plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("--dry-run is required; real Qwen generation is not wired in this script yet")

    try:
        plan = build_plan(
            eval_path=args.input,
            local_root=args.local_root,
            run_id=args.run_id,
            model_path=args.model_path,
            report_output=args.report_output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.write_plan is not None:
        write_plan(args.write_plan, plan)
    print(json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
