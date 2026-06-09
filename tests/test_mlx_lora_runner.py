import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_mlx_lora import build_plan


def write_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")


def write_config(
    path: Path,
    *,
    dataset_path: Path,
    local_root: Path,
    adapter_path: Path | None = None,
    run_id: str = "phase6-test-lora",
    rank: int = 16,
) -> None:
    adapter = adapter_path or local_root / "adapters" / "phase6-test"
    run_output = local_root / "runs" / "phase6-test" / "run_plan.json"
    path.write_text(
        f"""
[run]
run_id = "{run_id}"

[model]
path = "{local_root / "models" / "qwen"}"

[data]
dataset_path = "{dataset_path}"

[output]
adapter_path = "{adapter}"
run_output_path = "{run_output}"

[training]
rank = {rank}
target_modules = ["q_proj", "v_proj"]
epochs = 1
max_seq_length = 128
batch_size = 1
learning_rate = 0.00001
iters = 2
num_layers = 2
seed = 7
steps_per_report = 1
steps_per_eval = 1
save_every = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_build_plan_validates_config_and_planned_command(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)

    plan = build_plan(config_path=config_path, local_root=local_root)

    assert plan.run_id == "phase6-test-lora"
    assert plan.rank == 16
    assert plan.target_modules == ("q_proj", "v_proj")
    assert plan.adapter_path == (local_root / "adapters" / "phase6-test").resolve()
    command = plan.command()
    assert command[:3] == [
        "mlx_lm.lora",
        "--model",
        str((local_root / "models" / "qwen").resolve()),
    ]
    assert "--train" in command
    assert "--adapter-path" in command
    assert command[command.index("--data") + 1] == str(dataset_path.parent.resolve())


def test_build_plan_rejects_missing_dataset(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    config_path = tmp_path / "lora.toml"
    write_config(config_path, dataset_path=tmp_path / "missing.jsonl", local_root=local_root)

    with pytest.raises(ValueError, match="data.dataset_path must exist"):
        build_plan(config_path=config_path, local_root=local_root)


def test_build_plan_rejects_invalid_training_field(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root, rank=0)

    with pytest.raises(ValueError, match="training.rank must be a positive integer"):
        build_plan(config_path=config_path, local_root=local_root)


def test_build_plan_rejects_tracked_adapter_destination(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    write_dataset(dataset_path)
    write_config(
        config_path,
        dataset_path=dataset_path,
        local_root=local_root,
        adapter_path=Path("adapters/unsafe"),
    )

    with pytest.raises(ValueError, match="output.adapter_path must be under local artifact root"):
        build_plan(config_path=config_path, local_root=local_root)


def test_cli_requires_dry_run(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)

    result = subprocess.run(
        [sys.executable, "scripts/run_mlx_lora.py", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--dry-run is required" in result.stderr


def test_cli_dry_run_writes_plan(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    plan_path = tmp_path / "plan.json"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_mlx_lora.py",
            "--config",
            str(config_path),
            "--local-root",
            str(local_root),
            "--dry-run",
            "--write-plan",
            str(plan_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    stdout_plan = json.loads(result.stdout)
    file_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert stdout_plan == file_plan
    assert stdout_plan["safety"]["executes_training"] is False
    assert stdout_plan["planned_command"][0] == "mlx_lm.lora"
