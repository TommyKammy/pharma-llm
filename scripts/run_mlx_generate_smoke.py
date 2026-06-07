from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
LOCAL_RUN_DIR = Path("/Users/tsinfra/Dev/pharma-llm/local/runs")
PROMPT = (
    "製薬業務の回答では、原文確認と人間レビューが必要です。"
    "この方針を短く英語で要約してください。"
)


def main() -> int:
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOCAL_RUN_DIR / "mlx_generate_smoke.json"

    command = [
        "mlx_lm.generate",
        "--model",
        MODEL,
        "--prompt",
        PROMPT,
        "--max-tokens",
        "24",
        "--temp",
        "0",
        "--seed",
        "7",
        "--verbose",
        "False",
    ]

    started = time.monotonic()
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)
    elapsed_seconds = round(time.monotonic() - started, 3)

    payload = {
        "check": "mlx_generate_smoke",
        "model": MODEL,
        "prompt": PROMPT,
        "elapsed_seconds": elapsed_seconds,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr_tail": result.stderr.strip().splitlines()[-20:],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if result.returncode != 0:
        print(f"MLX generate smoke failed: {output_path}")
        print(result.stderr)
        return result.returncode

    print(f"MLX generate smoke ok: {output_path}")
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
