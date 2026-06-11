from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.inference import MlxInferenceClient, MlxLmCliClient, ModelIdentity  # noqa: E402
from pharma_llm_lab.training import validate_adapter_metadata  # noqa: E402
from scripts.run_baseline_eval import (  # noqa: E402
    BaselinePrediction,
    build_request,
    load_eval_records,
    write_predictions,
)

DEFAULT_INPUT = Path("evals/prompts/phase4_seed.jsonl")
DEFAULT_MODEL_ID = "qwen/qwen3.6-27b-base"
DEFAULT_PROVIDER = "mlx"


def require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def load_adapter_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"adapter metadata does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"adapter metadata is malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("adapter metadata must be a JSON object")
    metadata = validate_adapter_metadata(payload).to_mapping()
    if metadata["status"] != "executed":
        raise ValueError("adapter metadata status must be executed for real LoRA eval")
    return metadata


def metadata_model_identity(metadata: dict[str, Any]) -> ModelIdentity:
    model = metadata["model"]
    provider = model["provider"]
    if provider != DEFAULT_PROVIDER:
        raise ValueError(f"adapter metadata model.provider must be {DEFAULT_PROVIDER!r}")
    return ModelIdentity(
        model_id=model["id"],
        provider=provider,
        adapter_id=metadata["run_id"],
    )


def metadata_model_path(metadata: dict[str, Any]) -> Path:
    return Path(metadata["model"]["path"]).expanduser().resolve()


def metadata_adapter_path(metadata: dict[str, Any]) -> Path:
    return Path(metadata["adapter"]["path"]).expanduser().resolve()


def run_real_mlx_eval(
    *,
    eval_path: Path,
    client: MlxInferenceClient,
    run_id: str,
    model_label: str,
    max_tokens: int,
    mode: Literal["base", "lora"],
) -> tuple[BaselinePrediction, ...]:
    require_non_empty(run_id, "run_id")
    require_non_empty(model_label, "model_label")
    if mode == "base" and client.model.adapter_id is not None:
        raise ValueError("base real eval must not set model.adapter_id")
    if mode == "lora" and client.model.adapter_id is None:
        raise ValueError("LoRA real eval requires model.adapter_id")

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


def build_base_client(args: argparse.Namespace) -> MlxLmCliClient:
    if args.model_path is None:
        raise ValueError("--model-path is required for base real eval")
    return MlxLmCliClient(
        model=ModelIdentity(
            model_id=args.model_id,
            provider=DEFAULT_PROVIDER,
            adapter_id=None,
        ),
        model_path=args.model_path,
        adapter_path=None,
        command=tuple(shlex.split(args.generator_command)),
        timeout_seconds=args.timeout_seconds,
    )


def build_lora_client(args: argparse.Namespace) -> MlxLmCliClient:
    if args.adapter_metadata is None:
        raise ValueError("--adapter-metadata is required for LoRA real eval")
    metadata = load_adapter_metadata(args.adapter_metadata)
    model_path = metadata_model_path(metadata)
    if args.model_path is not None and args.model_path.expanduser().resolve() != model_path:
        raise ValueError("--model-path must match adapter metadata model.path")
    return MlxLmCliClient(
        model=metadata_model_identity(metadata),
        model_path=model_path,
        adapter_path=metadata_adapter_path(metadata),
        command=tuple(shlex.split(args.generator_command)),
        timeout_seconds=args.timeout_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real local MLX evaluation over accepted Phase 4 eval prompts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--adapter-metadata", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--generator-command",
        default="mlx_lm.generate",
        help=(
            "Command used before fixed MLX generate flags. Quote multi-token commands, "
            "for example: --generator-command 'python -m mlx_lm.generate'."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode: Literal["base", "lora"] = "lora" if args.adapter_metadata is not None else "base"

    try:
        client = build_lora_client(args) if mode == "lora" else build_base_client(args)
        predictions = run_real_mlx_eval(
            eval_path=args.input,
            client=client,
            run_id=args.run_id,
            model_label="qwen-lora" if mode == "lora" else "qwen-base",
            max_tokens=args.max_tokens,
            mode=mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    write_predictions(args.output, predictions)
    print(f"OK: wrote {len(predictions)} real {mode} prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
