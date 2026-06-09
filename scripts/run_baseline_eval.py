from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.dataset import EvalRecord, parse_record  # noqa: E402
from pharma_llm_lab.dataset.provenance import ReviewStatus  # noqa: E402
from pharma_llm_lab.dataset.validators import iter_jsonl  # noqa: E402
from pharma_llm_lab.inference import (  # noqa: E402
    InferenceRequest,
    MockMlxInferenceClient,
    ModelIdentity,
)


DEFAULT_INPUT = Path("evals/prompts/phase4_seed.jsonl")
DEFAULT_OUTPUT = Path("results/baseline/mock_predictions.jsonl")
DEFAULT_RUN_ID = "phase5-baseline-mock"
ACCEPTED_REVIEW_STATUSES = {
    ReviewStatus.APPROVED,
    ReviewStatus.EDITED_AND_APPROVED,
}

BASELINE_MODEL_IDS = {
    "qwen-base": "qwen/qwen3.6-27b-base",
    "gemma-base": "google/gemma-4-26b-a4b-base",
    "endpoint-optional": "openai-compatible/optional-baseline",
}


@dataclass(frozen=True)
class BaselinePrediction:
    run_id: str
    eval_id: str
    category: str
    prompt: str
    expected_points: tuple[str, ...]
    model: ModelIdentity
    generated_text: str
    timing: dict[str, float | int | None]
    finish_reason: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "eval_id": self.eval_id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_points": list(self.expected_points),
            "model": self.model.to_mapping(),
            "generated_text": self.generated_text,
            "timing": self.timing,
            "finish_reason": self.finish_reason,
        }


def load_eval_records(path: Path) -> tuple[EvalRecord, ...]:
    records: list[EvalRecord] = []
    seen_ids: set[str] = set()
    for line_number, item in iter_jsonl(path):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: {item.message}")
        if item.get("candidate_status") == "review_candidate":
            raise ValueError(f"{path}:{line_number}: review candidates are not accepted eval records")
        record = parse_record(item)
        if not isinstance(record, EvalRecord):
            raise ValueError(f"{path}:{line_number}: expected eval record")
        if record.provenance.review_status not in ACCEPTED_REVIEW_STATUSES:
            raise ValueError(
                f"{path}:{line_number}: {record.id} review_status must be approved "
                "before baseline evaluation"
            )
        if record.id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate eval id {record.id}")
        seen_ids.add(record.id)
        records.append(record)

    if not records:
        raise ValueError(f"{path}: no eval records found")
    return tuple(records)


def build_request(
    *,
    record: EvalRecord,
    model_label: str,
    run_id: str,
    max_tokens: int,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=f"{run_id}:{model_label}:{record.id}",
        prompt=record.prompt,
        max_tokens=max_tokens,
        metadata={
            "run_id": run_id,
            "eval_id": record.id,
            "category": record.category.value,
            "model_label": model_label,
        },
    )


def run_mock_baseline(
    *,
    eval_path: Path,
    model_label: str,
    run_id: str,
    max_tokens: int,
) -> tuple[BaselinePrediction, ...]:
    if not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if model_label not in BASELINE_MODEL_IDS:
        allowed = ", ".join(sorted(BASELINE_MODEL_IDS))
        raise ValueError(f"model label must be one of: {allowed}")

    model = ModelIdentity(model_id=BASELINE_MODEL_IDS[model_label], provider="mock-mlx")
    client = MockMlxInferenceClient(model=model, response_prefix=f"[{model_label}]")
    predictions: list[BaselinePrediction] = []

    for record in load_eval_records(eval_path):
        response = client.generate(
            build_request(
                record=record,
                model_label=model_label,
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


def write_predictions(path: Path, predictions: tuple[BaselinePrediction, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(prediction.to_mapping(), ensure_ascii=False)
            for prediction in predictions
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a CI-safe mock baseline over accepted eval prompts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--model-label",
        default="qwen-base",
        choices=sorted(BASELINE_MODEL_IDS),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        predictions = run_mock_baseline(
            eval_path=args.input,
            model_label=args.model_label,
            run_id=args.run_id,
            max_tokens=args.max_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))

    write_predictions(args.output, predictions)
    print(f"OK: wrote {len(predictions)} baseline prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
