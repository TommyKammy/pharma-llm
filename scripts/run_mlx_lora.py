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
MLX_SPLIT_NAMES = ("train.jsonl", "valid.jsonl", "test.jsonl")
QWEN_TARGET_MODULE_KEYS = frozenset(
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    }
)
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
    mlx_data_dir: Path
    mlx_config_path: Path
    rank: int
    scale: int
    dropout: float
    mask_prompt: bool
    target_modules: tuple[str, ...]
    max_seq_length: int
    batch_size: int
    learning_rate: float
    iters: int
    num_layers: int
    seed: int
    steps_per_report: int
    steps_per_eval: int
    save_every: int

    @property
    def train_data_path(self) -> Path:
        return self.mlx_data_dir / "train.jsonl"

    def command(self) -> list[str]:
        return [
            "mlx_lm.lora",
            "--config",
            str(self.mlx_config_path),
        ]

    def mlx_config_mapping(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path),
            "mask_prompt": self.mask_prompt,
            "train": True,
            "data": str(self.mlx_data_dir),
            "adapter_path": str(self.adapter_path),
            "iters": self.iters,
            "batch_size": self.batch_size,
            "num_layers": self.num_layers,
            "max_seq_length": self.max_seq_length,
            "learning_rate": self.learning_rate,
            "steps_per_report": self.steps_per_report,
            "steps_per_eval": self.steps_per_eval,
            "save_every": self.save_every,
            "seed": self.seed,
            "lora_parameters": {
                "rank": self.rank,
                "scale": self.scale,
                "dropout": self.dropout,
                "keys": list(self.target_modules),
            },
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_path": str(self.config_path),
            "local_root": str(self.local_root),
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "adapter_path": str(self.adapter_path),
            "run_output_path": str(self.run_output_path),
            "mlx_data_dir": str(self.mlx_data_dir),
            "train_data_path": str(self.train_data_path),
            "mlx_config_path": str(self.mlx_config_path),
            "mlx_config": self.mlx_config_mapping(),
            "training": {
                "rank": self.rank,
                "scale": self.scale,
                "dropout": self.dropout,
                "mask_prompt": self.mask_prompt,
                "target_modules": list(self.target_modules),
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
                "materializes_local_inputs": True,
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
    if type(value) is not int or value < 1:
        raise ValueError(f"{section_name}.{key} must be a positive integer")
    return value


def require_int_at_least(
    section: dict[str, Any],
    key: str,
    *,
    minimum: int,
    section_name: str,
) -> int:
    value = section.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{section_name}.{key} must be an integer >= {minimum}")
    return value


def require_positive_float(section: dict[str, Any], key: str, *, section_name: str) -> float:
    value = section.get(key)
    if type(value) not in (int, float) or value <= 0:
        raise ValueError(f"{section_name}.{key} must be a positive number")
    return float(value)


def require_non_negative_float(section: dict[str, Any], key: str, *, section_name: str) -> float:
    value = section.get(key)
    if type(value) not in (int, float) or value < 0:
        raise ValueError(f"{section_name}.{key} must be a non-negative number")
    return float(value)


def require_bool(section: dict[str, Any], key: str, *, section_name: str) -> bool:
    value = section.get(key)
    if type(value) is not bool:
        raise ValueError(f"{section_name}.{key} must be a boolean")
    return value


def require_qwen_target_modules(section: dict[str, Any]) -> tuple[str, ...]:
    value = section.get("target_modules")
    if not isinstance(value, list) or not value:
        raise ValueError("training.target_modules must be a non-empty string list")
    modules: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("training.target_modules must be a non-empty string list")
        modules.append(item)
    unknown_modules = sorted(set(modules) - QWEN_TARGET_MODULE_KEYS)
    if unknown_modules:
        allowed = ", ".join(sorted(QWEN_TARGET_MODULE_KEYS))
        unknown = ", ".join(unknown_modules)
        raise ValueError(
            "training.target_modules contains unsupported or unqualified Qwen MLX module "
            f"key(s): {unknown}; allowed keys: {allowed}"
        )
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


def mlx_split_paths(mlx_data_dir: Path) -> tuple[Path, ...]:
    return tuple(mlx_data_dir / split_name for split_name in MLX_SPLIT_NAMES)


def require_artifact_paths_do_not_collide(
    *,
    adapter_path: Path,
    run_output_path: Path,
    mlx_config_path: Path,
    mlx_data_dir: Path,
) -> None:
    split_paths = mlx_split_paths(mlx_data_dir)
    if run_output_path == mlx_data_dir:
        raise ValueError("output.run_output_path must differ from output.mlx_data_dir")
    if mlx_config_path == mlx_data_dir:
        raise ValueError("output.mlx_config_path must differ from output.mlx_data_dir")
    if mlx_config_path in split_paths:
        raise ValueError("output.mlx_config_path must differ from MLX split files")
    if run_output_path in split_paths:
        raise ValueError("output.run_output_path must differ from MLX split files")
    if adapter_path in split_paths:
        raise ValueError("output.adapter_path must differ from MLX split files")

    generated_file_paths = {
        "output.run_output_path": run_output_path,
        "output.mlx_config_path": mlx_config_path,
        **{f"mlx split {path.name}": path for path in split_paths},
    }

    seen: dict[Path, str] = {}
    for label, path in generated_file_paths.items():
        previous = seen.setdefault(path, label)
        if previous != label:
            raise ValueError(f"{label} must differ from {previous}")

    if adapter_path in generated_file_paths.values():
        raise ValueError("output.adapter_path must differ from generated MLX files")
    if adapter_path == mlx_data_dir:
        raise ValueError("output.adapter_path must differ from output.mlx_data_dir")


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
    mlx_data_dir = require_under_root(
        resolve_path(output.get("mlx_data_dir"), field_name="output.mlx_data_dir"),
        local_root,
        "output.mlx_data_dir",
    )
    mlx_config_path = require_under_root(
        resolve_path(output.get("mlx_config_path"), field_name="output.mlx_config_path"),
        local_root,
        "output.mlx_config_path",
    )

    if not dataset_path.is_file():
        raise ValueError(f"data.dataset_path must exist and be a file: {dataset_path}")
    if require_model_exists and not model_path.exists():
        raise ValueError(f"model.path must exist before real execution: {model_path}")
    require_artifact_paths_do_not_collide(
        adapter_path=adapter_path,
        run_output_path=run_output_path,
        mlx_config_path=mlx_config_path,
        mlx_data_dir=mlx_data_dir,
    )

    return MlxLoraTrainingPlan(
        run_id=run_id,
        config_path=resolved_config,
        local_root=local_root.expanduser().resolve(),
        model_path=model_path,
        dataset_path=dataset_path,
        adapter_path=adapter_path,
        run_output_path=run_output_path,
        mlx_data_dir=mlx_data_dir,
        mlx_config_path=mlx_config_path,
        rank=require_positive_int(training, "rank", section_name="training"),
        scale=require_positive_float(training, "scale", section_name="training"),
        dropout=require_non_negative_float(training, "dropout", section_name="training"),
        mask_prompt=require_bool(training, "mask_prompt", section_name="training"),
        target_modules=require_qwen_target_modules(training),
        max_seq_length=require_positive_int(training, "max_seq_length", section_name="training"),
        batch_size=require_positive_int(training, "batch_size", section_name="training"),
        learning_rate=require_positive_float(training, "learning_rate", section_name="training"),
        iters=require_positive_int(training, "iters", section_name="training"),
        num_layers=require_int_at_least(
            training,
            "num_layers",
            minimum=-1,
            section_name="training",
        ),
        seed=require_int_at_least(training, "seed", minimum=0, section_name="training"),
        steps_per_report=require_positive_int(training, "steps_per_report", section_name="training"),
        steps_per_eval=require_positive_int(training, "steps_per_eval", section_name="training"),
        save_every=require_positive_int(training, "save_every", section_name="training"),
    )


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def dump_simple_yaml(mapping: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, list):
                    lines.append(f"  {nested_key}:")
                    lines.extend(f"    - {yaml_scalar(item)}" for item in nested_value)
                else:
                    lines.append(f"  {nested_key}: {yaml_scalar(nested_value)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {yaml_scalar(item)}" for item in value)
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def materialize_local_inputs(plan: MlxLoraTrainingPlan) -> None:
    require_artifact_paths_do_not_collide(
        adapter_path=plan.adapter_path,
        run_output_path=plan.run_output_path,
        mlx_config_path=plan.mlx_config_path,
        mlx_data_dir=plan.mlx_data_dir,
    )
    train_data = plan.dataset_path.read_bytes()
    plan.mlx_data_dir.mkdir(parents=True, exist_ok=True)
    plan.mlx_config_path.parent.mkdir(parents=True, exist_ok=True)
    for split_path in mlx_split_paths(plan.mlx_data_dir):
        split_path.unlink(missing_ok=True)
    plan.train_data_path.write_bytes(train_data)
    plan.mlx_config_path.write_text(
        dump_simple_yaml(plan.mlx_config_mapping()),
        encoding="utf-8",
    )


def write_plan(path: Path, plan: MlxLoraTrainingPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_write_plan_path(path: Path, plan: MlxLoraTrainingPlan) -> Path:
    resolved_path = require_under_root(path, plan.local_root, "--write-plan")
    if resolved_path != plan.run_output_path:
        raise ValueError(f"--write-plan must equal output.run_output_path: {plan.run_output_path}")
    if resolved_path == plan.mlx_config_path:
        raise ValueError("--write-plan must not collide with output.mlx_config_path")
    if resolved_path == plan.train_data_path:
        raise ValueError("--write-plan must not collide with materialized train.jsonl")
    return resolved_path


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
        help=(
            "Optional assertion for the JSON run plan path. When supplied, it must equal "
            "output.run_output_path from the config."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("--dry-run is required; real MLX LoRA execution is tracked in P6-004")

    try:
        plan = build_plan(config_path=args.config, local_root=args.local_root)
        write_plan_path = plan.run_output_path
        if args.write_plan is not None:
            write_plan_path = require_write_plan_path(args.write_plan, plan)
        materialize_local_inputs(plan)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))

    write_plan(write_plan_path, plan)
    print(json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
