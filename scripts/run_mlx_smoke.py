from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx_lm


LOCAL_RUN_DIR = Path("/Users/tsinfra/Dev/pharma-llm/local/runs")


def main() -> int:
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)

    values = mx.array([1.0, 2.0, 3.0])
    doubled = values * 2
    result = {
        "check": "mlx_smoke",
        "mlx_lm_module": getattr(mlx_lm, "__name__", "mlx_lm"),
        "input": values.tolist(),
        "output": doubled.tolist(),
    }

    output_path = LOCAL_RUN_DIR / "mlx_smoke.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MLX smoke ok: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
