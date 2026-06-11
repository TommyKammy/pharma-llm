import json
import subprocess
import sys
from pathlib import Path

import pytest

from pharma_llm_lab.inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceTiming,
    ModelIdentity,
    MlxLmCliClient,
    MlxLmPythonClient,
)
from pharma_llm_lab.training.lora_metadata import METADATA_VERSION
from scripts.run_real_mlx_eval import load_adapter_metadata, run_real_mlx_eval

SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


class FakeRealMlxClient:
    def __init__(self, model: ModelIdentity) -> None:
        self.model = model

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            generated_text=f"real local answer for {request.metadata['eval_id']}",
            model=self.model,
            timing=InferenceTiming(
                total_latency_ms=42.0,
                ttft_ms=None,
                tokens_per_second=100.0,
                prompt_tokens=len(request.prompt.split()),
                completion_tokens=5,
            ),
            finish_reason="stop",
        )


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


def write_adapter_metadata(path: Path, *, local_root: Path, status: str = "executed") -> Path:
    model_path = local_root / "models" / "qwen3.6-27b-base"
    adapter_path = local_root / "adapters" / "qwen_sft_lora_r16_v1"
    train_path = local_root / "runs" / "qwen_sft_lora_r16_v1" / "mlx_data" / "train.jsonl"
    generated_config = local_root / "runs" / "qwen_sft_lora_r16_v1" / "mlx_lora_config.yaml"
    source_dataset = local_root / "argilla" / "phase6_reviewed_sft.jsonl"
    source_config = local_root / "configs" / "qwen_sft_lora_r16.toml"

    model_path.mkdir(parents=True)
    if status == "executed":
        adapter_path.mkdir(parents=True)
    train_path.parent.mkdir(parents=True)
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    source_dataset.parent.mkdir(parents=True, exist_ok=True)
    source_config.parent.mkdir(parents=True, exist_ok=True)
    if status == "executed":
        (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_path / "adapters.safetensors").write_text("weights", encoding="utf-8")

    payload = {
        "metadata_version": METADATA_VERSION,
        "run_id": "qwen_sft_lora_r16_v1",
        "status": status,
        "model": {
            "id": "qwen/qwen3.6-27b-base",
            "provider": "mlx",
            "path": str(model_path.resolve()),
        },
        "dataset": {
            "version": "sft-v0.1",
            "path": str(source_dataset.resolve()),
            "sha256": "a" * 64,
            "training_input": {
                "path": str(train_path.resolve()),
                "sha256": "b" * 64,
            },
        },
        "config": {
            "source_path": str(source_config.resolve()),
            "source_sha256": "c" * 64,
            "generated_path": str(generated_config.resolve()),
            "generated_sha256": "d" * 64,
        },
        "adapter": {
            "path": str(adapter_path.resolve()),
            "exists": status == "executed",
            "is_directory": status == "executed",
            "marker_files": (
                ["adapter_config.json", "adapters.safetensors"]
                if status == "executed"
                else []
            ),
            "metadata_path": str(path.resolve()),
        },
        "training": {
            "rank": 16,
            "scale": 32.0,
            "dropout": 0.0,
            "mask_prompt": True,
            "target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
            "epochs": None,
            "max_seq_length": 128,
            "iters": 2,
            "batch_size": 1,
            "learning_rate": 0.0001,
            "num_layers": -1,
            "seed": 0,
        },
        "timestamps": {
            "created_at": "2026-06-11T01:00:00Z",
            "started_at": "2026-06-11T01:00:00Z" if status == "executed" else None,
            "ended_at": "2026-06-11T02:00:00Z" if status == "executed" else None,
        },
        "validation": {
            "is_dry_run_placeholder": status == "planned",
            "status_note": "Local Qwen SFT LoRA v1 completed.",
        },
        "local_artifact_policy": {
            "local_root": str(local_root.resolve()),
            "large_artifacts_ignored": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_run_real_mlx_eval_produces_schema_compatible_base_predictions() -> None:
    predictions = run_real_mlx_eval(
        eval_path=SEED_PATH,
        client=FakeRealMlxClient(
            ModelIdentity(model_id="qwen/qwen3.6-27b-base", provider="mlx")
        ),
        run_id="phase6-qwen-base-real",
        model_label="qwen-base",
        max_tokens=8,
        mode="base",
    )

    assert len(predictions) == 30
    first = predictions[0].to_mapping()
    assert first["run_id"] == "phase6-qwen-base-real"
    assert first["eval_id"] == "eval_001"
    assert first["model"] == {
        "model_id": "qwen/qwen3.6-27b-base",
        "provider": "mlx",
        "adapter_id": None,
    }
    assert first["generated_text"].startswith("real local answer")


def test_run_real_mlx_eval_requires_lora_adapter_identity() -> None:
    with pytest.raises(ValueError, match="LoRA real eval requires model.adapter_id"):
        run_real_mlx_eval(
            eval_path=SEED_PATH,
            client=FakeRealMlxClient(
                ModelIdentity(model_id="qwen/qwen3.6-27b-base", provider="mlx")
            ),
            run_id="phase6-qwen-lora-real",
            model_label="qwen-lora",
            max_tokens=8,
            mode="lora",
        )


def test_load_adapter_metadata_requires_executed_status(tmp_path: Path) -> None:
    metadata_path = write_adapter_metadata(
        tmp_path / "local" / "runs" / "qwen_sft_lora_r16_v1" / "adapter_metadata.json",
        local_root=tmp_path / "local",
        status="planned",
    )

    with pytest.raises(ValueError, match="status must be executed"):
        load_adapter_metadata(metadata_path)


def test_mlx_lm_cli_client_invokes_generator_command(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    client = MlxLmCliClient(
        model=ModelIdentity(model_id="qwen/qwen3.6-27b-base", provider="mlx"),
        model_path=model_path,
        command=(
            sys.executable,
            "-c",
            "import sys; prompt=sys.argv[sys.argv.index('--prompt') + 1]; print('generated:' + prompt[:6])",
        ),
    )

    response = client.generate(
        InferenceRequest(
            request_id="request-1",
            prompt="abcdefg prompt",
            max_tokens=4,
        )
    )

    assert response.generated_text == "generated:abcdef"
    assert response.model.adapter_id is None
    assert "--verbose" in client.build_command(
        InferenceRequest(request_id="request-2", prompt="prompt", max_tokens=4)
    )
    assert response.timing.prompt_tokens is None
    assert response.timing.completion_tokens is None
    assert response.timing.tokens_per_second is None
    assert response.finish_reason == "unknown"
    assert response.timing.total_latency_ms >= 0


def test_mlx_lm_python_client_loads_once_and_counts_tokenizer_tokens(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    adapter_path = tmp_path / "adapter"
    model_path.mkdir()
    adapter_path.mkdir()
    load_calls: list[dict[str, str]] = []
    generate_calls: list[dict[str, object]] = []

    def fake_load(**kwargs: str) -> tuple[str, FakeTokenizer]:
        load_calls.append(kwargs)
        return "loaded-model", FakeTokenizer()

    def fake_generate(**kwargs: object) -> str:
        generate_calls.append(kwargs)
        return "生成結果"

    client = MlxLmPythonClient(
        model=ModelIdentity(
            model_id="qwen/qwen3.6-27b-base",
            provider="mlx",
            adapter_id="qwen_sft_lora_r16_v1",
        ),
        model_path=model_path,
        adapter_path=adapter_path,
        load_fn=fake_load,
        generate_fn=fake_generate,
    )

    first = client.generate(
        InferenceRequest(request_id="request-1", prompt="安全性確認", max_tokens=4)
    )
    second = client.generate(
        InferenceRequest(request_id="request-2", prompt="業務文体", max_tokens=4)
    )

    assert len(load_calls) == 1
    assert load_calls[0] == {
        "path_or_hf_repo": str(model_path.resolve()),
        "adapter_path": str(adapter_path.resolve()),
    }
    assert len(generate_calls) == 2
    assert generate_calls[0]["verbose"] is False
    assert first.generated_text == "生成結果"
    assert first.timing.prompt_tokens == 5
    assert first.timing.completion_tokens == 4
    assert first.timing.tokens_per_second is not None
    assert first.finish_reason == "length"
    assert second.timing.prompt_tokens == 4


def test_real_eval_cli_writes_base_predictions_with_fake_generator(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    output_path = tmp_path / "predictions.jsonl"
    model_path.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_mlx_eval.py",
            "--input",
            str(SEED_PATH),
            "--output",
            str(output_path),
            "--run-id",
            "phase6-qwen-base-real",
            "--model-path",
            str(model_path),
            "--max-tokens",
            "4",
            "--client-backend",
            "cli",
            "--generator-command",
            (
                f"{sys.executable} -c "
                "\"import sys; prompt=sys.argv[sys.argv.index('--prompt') + 1]; "
                "print('real:' + prompt[:8])\""
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 30
    assert records[0]["model"]["provider"] == "mlx"
    assert records[0]["model"]["adapter_id"] is None
    assert records[0]["generated_text"].startswith("real:")
