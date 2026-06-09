from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/mlx/qwen_sft_lora_r16.toml")
DEFAULT_LOCAL_ROOT = Path("/Users/tsinfra/Dev/pharma-llm/local")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")


@dataclass(frozen=True)
class MlxLoraTrainingPlan:
    run_id: str
    config_path: Path
    local_root: Path
    model_path: Path
    dataset_path: Path
    adapter_path: Path
    run_output_path: Path
    rank: int
    target_modules: tuple[str, ...]
    epochs: int
    max_seq_length: int
    batch_size: int
    learning_rate: float
    iters: int
    num_layers: int
    seed: int
    steps_per_report: int
    steps_per_eval: int
    save_every: int

    def command(self) -> list[str]:
        return [
            "mlx_lm.lora",
            "--model",
            str(self.model_path),
            "--train",
            "--data",
            str(self.dataset_path.parent),
            "--adapter-path",
            str(self.adapter_path),
            "--iters",
            str(self.iters),
            "--batch-size",
            str(self.batch_size),
            "--num-layers",
            str(self.num_layers),
            "--max-seq-length",
            str(self.max_seq_length),
            "--learning-rate",
            str(self.learning_rate),
            "--steps-per-report",
            str(self.steps_per_report),
            "--steps-per-eval",
            str(self.steps_per_eval),
            "--save-every",
            str(self.save_every),
            "--seed",
            str(self.seed),
        ]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_path": str(self.config_path),
            "local_root": str(self.local_root),
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "adapter_path": str(self.adapter_path),
            "run_output_path": str(self.run_output_path),
            "training": {
                "rank": self.rank,
                "target_modules": list(self.target_modules),
                "epochs": self.epochs,
                "max_seq_length": self.max_seq_length,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "iters": self.iters,
                "num_layers": self.num_layers,
                "seed": self.seed,
                "steps_per_report": self.steps_per_report,
                "steps_per_eval": self.steps_per_eval,
                "save_every": self.save_every,
            },
            "planned_command": self.command(),
            "safety": {
                "mode": "dry_run",
                "large_artifacts_root": str(self.local_root),
                "model_exists": self.model_path.exists(),
                "dataset_exists": self.dataset_path.exists(),
                "executes_training": False,
            },
        }


def resolve_path(value: Any, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path.resolve()


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"missing [{name}] section")
    return section


def require_string(section: dict[str, Any], key: str, *, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{key} must be a non-empty string")
    return value


def require_positive_int(section: dict[str, Any], key: str, *, section_name: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{section_name}.{key} must be a positive integer")
    return value


def require_positive_float(section: dict[str, Any], key: str, *, section_name: str) -> float:
    value = section.get(key)
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{section_name}.{key} must be a positive number")
    return float(value)


def require_target_modules(section: dict[str, Any]) -> tuple[str, ...]:
    value = section.get("target_modules")
    if not isinstance(value, list) or not value:
        raise ValueError("training.target_modules must be a non-empty string list")
    modules: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("training.target_modules must be a non-empty string list")
        modules.append(item)
    return tuple(modules)


def require_under_root(path: Path, root: Path, field_name: str) -> Path:
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{field_name} must be under local artifact root: {resolved_root}")
    return resolved_path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_plan(
    *,
    config_path: Path,
    local_root: Path = DEFAULT_LOCAL_ROOT,
    require_model_exists: bool = False,
) -> MlxLoraTrainingPlan:
    resolved_config = config_path.expanduser().resolve()
    config = load_config(resolved_config)

    run = require_section(config, "run")
    model = require_section(config, "model")
    data = require_section(config, "data")
    output = require_section(config, "output")
    training = require_section(config, "training")

    run_id = require_string(run, "run_id", section_name="run")
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError("run.run_id must be 3-80 lowercase URL-safe characters")

    model_path = resolve_path(model.get("path"), field_name="model.path")
    dataset_path = resolve_path(
        data.get("dataset_path"),
        field_name="data.dataset_path",
    )
    adapter_path = require_under_root(
        resolve_path(output.get("adapter_path"), field_name="output.adapter_path"),
        local_root,
        "output.adapter_path",
    )
    run_output_path = require_under_root(
        resolve_path(
            output.get("run_output_path"),
            field_name="output.run_output_path",
        ),
        local_root,
        "output.run_output_path",
    )

    if not dataset_path.is_file():
        raise ValueError(f"data.dataset_path must exist and be a file: {dataset_path}")
    if require_model_exists and not model_path.exists():
        raise ValueError(f"model.path must exist before real execution: {model_path}")

    return MlxLoraTrainingPlan(
        run_id=run_id,
        config_path=resolved_config,
        local_root=local_root.expanduser().resolve(),
        model_path=model_path,
        dataset_path=dataset_path,
        adapter_path=adapter_path,
        run_output_path=run_output_path,
        rank=require_positive_int(training, "rank", section_name="training"),
        target_modules=require_target_modules(training),
        epochs=require_positive_int(training, "epochs", section_name="training"),
        max_seq_length=require_positive_int(training, "max_seq_length", section_name="training"),
        batch_size=require_positive_int(training, "batch_size", section_name="training"),
        learning_rate=require_positive_float(training, "learning_rate", section_name="training"),
        iters=require_positive_int(training, "iters", section_name="training"),
        num_layers=require_positive_int(training, "num_layers", section_name="training"),
        seed=require_positive_int(training, "seed", section_name="training"),
        steps_per_report=require_positive_int(training, "steps_per_report", section_name="training"),
        steps_per_eval=require_positive_int(training, "steps_per_eval", section_name="training"),
        save_every=require_positive_int(training, "save_every", section_name="training"),
    )


def write_plan(path: Path, plan: MlxLoraTrainingPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and render a dry-run MLX LoRA training plan."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print planned command without executing training.",
    )
    parser.add_argument(
        "--write-plan",
        type=Path,
        default=None,
        help="Optional JSON path for the generated run plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("--dry-run is required; real MLX LoRA execution is tracked in P6-004")

    try:
        plan = build_plan(config_path=args.config, local_root=args.local_root)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))

    if args.write_plan is not None:
        write_plan(args.write_plan, plan)
    print(json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
