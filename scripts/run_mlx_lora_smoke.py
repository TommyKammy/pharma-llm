from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path


MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
LOCAL_ROOT = Path("/Users/tsinfra/Dev/pharma-llm/local")
DATA_DIR = LOCAL_ROOT / "runs" / "mlx_lora_smoke_data"
ADAPTER_DIR = LOCAL_ROOT / "adapters" / "mlx_lora_smoke"
RUN_DIR = LOCAL_ROOT / "runs"


TRAIN_RECORDS = [
    {
        "prompt": "製薬業務LLMの学習データポリシーを一文で説明してください。",
        "completion": "人間レビュー済みのデータだけを学習対象にし、評価データは分離します。",
    },
    {
        "prompt": "安全性情報の回答で避けるべきことを一文で説明してください。",
        "completion": "根拠が確認できない効能や安全性を断定してはいけません。",
    },
]


def write_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        path = DATA_DIR / filename
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in TRAIN_RECORDS) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset()

    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)

    output_path = RUN_DIR / "mlx_lora_smoke.json"
    command = [
        "mlx_lm.lora",
        "--model",
        MODEL,
        "--train",
        "--data",
        str(DATA_DIR),
        "--adapter-path",
        str(ADAPTER_DIR),
        "--iters",
        "1",
        "--batch-size",
        "1",
        "--num-layers",
        "1",
        "--max-seq-length",
        "128",
        "--steps-per-report",
        "1",
        "--steps-per-eval",
        "1",
        "--val-batches",
        "1",
        "--save-every",
        "1",
        "--seed",
        "7",
    ]

    started = time.monotonic()
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900)
    elapsed_seconds = round(time.monotonic() - started, 3)

    payload = {
        "check": "mlx_lora_smoke",
        "model": MODEL,
        "data_dir": str(DATA_DIR),
        "adapter_dir": str(ADAPTER_DIR),
        "elapsed_seconds": elapsed_seconds,
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip().splitlines()[-40:],
        "stderr_tail": result.stderr.strip().splitlines()[-40:],
        "adapter_files": sorted(path.name for path in ADAPTER_DIR.glob("*")) if ADAPTER_DIR.exists() else [],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if result.returncode != 0:
        print(f"MLX LoRA smoke failed: {output_path}")
        print(result.stdout)
        print(result.stderr)
        return result.returncode

    print(f"MLX LoRA smoke ok: {output_path}")
    print(f"Adapter files: {', '.join(payload['adapter_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
