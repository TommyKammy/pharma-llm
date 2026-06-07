from __future__ import annotations

import json
from pathlib import Path

from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase


LOCAL_RUN_DIR = Path("/Users/tsinfra/Dev/pharma-llm/local/runs")


def main() -> int:
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)

    test_case = LLMTestCase(
        input="Return the review status marker.",
        actual_output="review_required",
        expected_output="review_required",
    )
    metric = ExactMatchMetric(threshold=1.0)
    score = metric.measure(test_case)

    result = {
        "check": "deepeval_smoke",
        "metric": "ExactMatchMetric",
        "score": score,
        "success": metric.is_successful(),
        "reason": metric.reason,
    }
    output_path = LOCAL_RUN_DIR / "deepeval_smoke.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not metric.is_successful():
        print(json.dumps(result, indent=2))
        return 1

    print(f"DeepEval smoke ok: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
