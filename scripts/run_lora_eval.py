from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.inference import MockMlxInferenceClient, ModelIdentity  # noqa: E402
from scripts.run_baseline_eval import (  # noqa: E402
    BaselinePrediction,
    build_request,
    load_eval_records,
    write_predictions,
)

DEFAULT_INPUT = Path("evals/prompts/phase4_seed.jsonl")
DEFAULT_OUTPUT = Path("results/lora/mock_qwen_sft_lora_v1_predictions.jsonl")
DEFAULT_RUN_ID = "phase6-lora-mock"
DEFAULT_MODEL_ID = "qwen/qwen3.6-27b-base"
DEFAULT_ADAPTER_ID = "qwen_sft_lora_r16_v1"


def run_mock_lora_eval(
    *,
    eval_path: Path,
    model_id: str,
    adapter_id: str,
    run_id: str,
    max_tokens: int,
) -> tuple[BaselinePrediction, ...]:
    if not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if not adapter_id.strip():
        raise ValueError("adapter_id must be a non-empty string")

    model = ModelIdentity(model_id=model_id, provider="mock-mlx", adapter_id=adapter_id)
    client = MockMlxInferenceClient(model=model, response_prefix=f"[lora:{adapter_id}]")
    predictions: list[BaselinePrediction] = []
    for record in load_eval_records(eval_path):
        response = client.generate(
            build_request(
                record=record,
                model_label="qwen-lora",
                run_id=run_id,
                max_tokens=max_tokens,
            )
        )
        predictions.append(
            BaselinePrediction(
                run_id=run_id,
                eval_id=record.id,
                category=record.category.value,
                prompt=record.prompt,
                expected_points=record.expected_points,
                model=response.model,
                generated_text=response.generated_text,
                timing=response.timing.to_mapping(),
                finish_reason=response.finish_reason,
            )
        )
    return tuple(predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a CI-safe mock LoRA evaluation over accepted eval prompts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID)
    parser.add_argument(
        "--adapter-metadata",
        type=Path,
        default=None,
        help=(
            "Reserved for real local LoRA generation. The CI-safe mock runner refuses "
            "metadata-backed runs so executed adapters are not reported as mock outputs."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_id = args.model_id
    adapter_id = args.adapter_id
    if args.adapter_metadata is not None:
        parser.error(
            "--adapter-metadata requires real LoRA generation, which is not wired in "
            "this CI-safe mock runner; generate the local prediction JSONL with the "
            "executed adapter and pass it to generate_lora_comparison_report.py"
        )

    try:
        predictions = run_mock_lora_eval(
            eval_path=args.input,
            model_id=model_id,
            adapter_id=adapter_id,
            run_id=args.run_id,
            max_tokens=args.max_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))

    write_predictions(args.output, predictions)
    print(f"OK: wrote {len(predictions)} LoRA prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
