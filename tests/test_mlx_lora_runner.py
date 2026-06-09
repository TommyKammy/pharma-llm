import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_mlx_lora import build_plan, materialize_local_inputs


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
mlx_data_dir = "{local_root / "runs" / "phase6-test" / "mlx_data"}"
mlx_config_path = "{local_root / "runs" / "phase6-test" / "mlx_lora_config.yaml"}"

[training]
rank = {rank}
scale = 32
dropout = 0.0
mask_prompt = true
target_modules = ["self_attn.q_proj", "self_attn.v_proj"]
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
    assert plan.target_modules == ("self_attn.q_proj", "self_attn.v_proj")
    assert plan.adapter_path == (local_root / "adapters" / "phase6-test").resolve()
    command = plan.command()
    assert command == [
        "mlx_lm.lora",
        "--config",
        str((local_root / "runs" / "phase6-test" / "mlx_lora_config.yaml").resolve()),
    ]
    assert plan.train_data_path == (
        local_root / "runs" / "phase6-test" / "mlx_data" / "train.jsonl"
    ).resolve()
    assert plan.mlx_config_mapping()["lora_parameters"] == {
        "rank": 16,
        "scale": 32,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    assert plan.mlx_config_mapping()["mask_prompt"] is True


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


def test_build_plan_rejects_bool_integer_field(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)
    text = config_path.read_text(encoding="utf-8").replace("iters = 2", "iters = true")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="training.iters must be a positive integer"):
        build_plan(config_path=config_path, local_root=local_root)


def test_build_plan_rejects_bool_float_field(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)
    text = config_path.read_text(encoding="utf-8").replace(
        "learning_rate = 0.00001",
        "learning_rate = true",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="training.learning_rate must be a positive number"):
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
    plan_path = local_root / "runs" / "phase6-test" / "run_plan.json"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)
    mlx_data_dir = local_root / "runs" / "phase6-test" / "mlx_data"
    mlx_data_dir.mkdir(parents=True)
    (mlx_data_dir / "valid.jsonl").write_text('{"prompt":"old","completion":"old"}\n')
    (mlx_data_dir / "test.jsonl").write_text('{"prompt":"old","completion":"old"}\n')

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
    assert stdout_plan["mlx_config"]["mask_prompt"] is True
    assert stdout_plan["planned_command"][0] == "mlx_lm.lora"
    assert (local_root / "runs" / "phase6-test" / "mlx_data" / "train.jsonl").is_file()
    assert not (local_root / "runs" / "phase6-test" / "mlx_data" / "valid.jsonl").exists()
    assert not (local_root / "runs" / "phase6-test" / "mlx_data" / "test.jsonl").exists()
    assert (local_root / "runs" / "phase6-test" / "mlx_lora_config.yaml").is_file()
    yaml_text = (local_root / "runs" / "phase6-test" / "mlx_lora_config.yaml").read_text(
        encoding="utf-8"
    )
    assert "rank: 16" in yaml_text
    assert "mask_prompt: true" in yaml_text
    assert "  scale: 32" in yaml_text
    assert "  dropout: 0.0" in yaml_text
    assert "  keys:" in yaml_text
    assert '    - "self_attn.q_proj"' in yaml_text


def test_cli_dry_run_writes_configured_plan_without_write_plan_arg(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
    plan_path = local_root / "runs" / "phase6-test" / "run_plan.json"
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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    stdout_plan = json.loads(result.stdout)
    file_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert stdout_plan == file_plan
    assert file_plan["run_output_path"] == str(plan_path.resolve())


def test_cli_rejects_write_plan_that_differs_from_configured_run_output(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
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
            str(local_root / "runs" / "phase6-test" / "other_plan.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--write-plan must equal output.run_output_path" in result.stderr
    assert not (local_root / "runs" / "phase6-test" / "mlx_data").exists()


def test_cli_rejects_write_plan_outside_local_root(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    dataset_path = tmp_path / "sft_v0_1.jsonl"
    config_path = tmp_path / "lora.toml"
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
            str(tmp_path / "run_plan.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--write-plan must be under local artifact root" in result.stderr
    assert not (local_root / "runs" / "phase6-test" / "mlx_data").exists()


def test_materialize_keeps_dataset_when_source_is_train_split(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    mlx_data_dir = local_root / "runs" / "phase6-test" / "mlx_data"
    dataset_path = mlx_data_dir / "train.jsonl"
    config_path = tmp_path / "lora.toml"
    write_dataset(dataset_path)
    write_config(config_path, dataset_path=dataset_path, local_root=local_root)
    (mlx_data_dir / "valid.jsonl").write_text('{"prompt":"old","completion":"old"}\n')

    plan = build_plan(config_path=config_path, local_root=local_root)
    materialize_local_inputs(plan)

    assert dataset_path.read_text(encoding="utf-8") == '{"prompt":"p","completion":"c"}\n'
    assert not (mlx_data_dir / "valid.jsonl").exists()
