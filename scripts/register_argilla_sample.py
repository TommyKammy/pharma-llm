from __future__ import annotations

import json
import os
from pathlib import Path

import argilla as rg

from create_argilla_sample import sample_records


LOCAL_ARGILLA_DIR = Path("/Users/tsinfra/Dev/pharma-llm/local/argilla")


def main() -> int:
    api_url = os.environ.get("ARGILLA_API_URL", "http://localhost:6900")
    api_key = os.environ.get("ARGILLA_API_KEY")
    output_path = LOCAL_ARGILLA_DIR / "argilla_server_smoke.json"
    LOCAL_ARGILLA_DIR.mkdir(parents=True, exist_ok=True)

    if not api_key:
        payload = {
            "check": "argilla_server_smoke",
            "status": "skipped",
            "reason": "ARGILLA_API_KEY is not set",
            "api_url": api_url,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Argilla server smoke skipped: {output_path}")
        return 0

    try:
        client = rg.Argilla(api_url=api_url, api_key=api_key)
        me = client.me
    except Exception as exc:
        payload = {
            "check": "argilla_server_smoke",
            "status": "failed",
            "api_url": api_url,
            "error": str(exc),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Argilla server smoke failed: {output_path}")
        return 1

    payload = {
        "check": "argilla_server_smoke",
        "status": "connected",
        "api_url": api_url,
        "user": repr(me),
        "sample_record_count": len(sample_records()),
        "note": "Connection verified. Dataset registration is implemented in Phase 3.",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Argilla server smoke ok: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
