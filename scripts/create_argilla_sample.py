from __future__ import annotations

import json
from pathlib import Path


LOCAL_ARGILLA_DIR = Path("/Users/tsinfra/Dev/pharma-llm/local/argilla")


def sample_records() -> list[dict[str, object]]:
    return [
        {
            "id": "phase1_argilla_sample_001",
            "dataset_type": "sft_candidate",
            "source_type": "human_authored",
            "review_status": "approved",
            "prompt": "安全性情報をQA向けに要約してください。",
            "response": "原文確認を前提に、既知情報と未確認情報を分けて要約します。",
            "metadata": {
                "phase": 1,
                "ai_assisted": False,
                "raw_ai_output_used_as_training_target": False,
            },
        },
        {
            "id": "phase1_argilla_sample_002",
            "dataset_type": "dpo_candidate",
            "source_type": "human_edited_ai_assisted",
            "review_status": "needs_edit",
            "prompt": "患者に服薬変更を助言してよいですか。",
            "chosen": "医療判断は担当医療者に確認する必要があるため、一般情報に限定して説明します。",
            "rejected": "自己判断で服薬を変更して構いません。",
            "metadata": {
                "phase": 1,
                "ai_assisted": True,
                "raw_ai_output_used_as_training_target": False,
                "risk_flags": ["medical_advice"],
            },
        },
    ]


def main() -> int:
    import argilla as rg

    LOCAL_ARGILLA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOCAL_ARGILLA_DIR / "phase1_review_sample.jsonl"

    records = sample_records()
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    print(f"Argilla SDK ok: {rg.__name__}")
    print(f"Offline review sample written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
