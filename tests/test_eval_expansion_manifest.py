import json
import subprocess
import sys
from pathlib import Path

from pharma_llm_lab.dataset import EvaluationCategory, EvalRecord, ReviewStatus, SourceType
from pharma_llm_lab.dataset.schema import parse_record

from scripts.plan_eval_expansion import (
    build_coverage,
    load_manifest,
    propose_candidate_records,
    validate_manifest_consistency,
)


MANIFEST_PATH = Path("evals/manifest/evaluation_set_v0.json")


def test_evaluation_set_v0_manifest_matches_seed_coverage() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    coverage = build_coverage(manifest)

    assert manifest.target_total == 300
    assert {plan.category for plan in manifest.categories} == set(EvaluationCategory)
    assert all(plan.target_count == 50 for plan in manifest.categories)
    assert coverage == {category: 5 for category in EvaluationCategory}
    validate_manifest_consistency(manifest)


def test_eval_expansion_candidates_use_deterministic_next_ids() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    candidates = propose_candidate_records(manifest, per_category=1)

    assert [candidate["id"] for candidate in candidates] == [
        "eval_006",
        "eval_056",
        "eval_106",
        "eval_156",
        "eval_206",
        "eval_256",
    ]
    assert [candidate["category"] for candidate in candidates] == [
        category.value for category in EvaluationCategory
    ]


def test_eval_expansion_candidates_remain_review_candidates() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    for candidate in propose_candidate_records(manifest, per_category=2):
        assert candidate["candidate_status"] == "review_candidate"
        record = parse_record(candidate)
        assert isinstance(record, EvalRecord)
        assert record.provenance.source_type is SourceType.EVAL_ONLY
        assert record.provenance.source_document == "synthetic_phase4_candidate"
        assert record.provenance.review_status is ReviewStatus.UNREVIEWED
        assert not record.provenance.raw_ai_output_used_as_training_target
        assert len(record.expected_points) == 3


def test_eval_expansion_cli_reports_current_vs_target() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/plan_eval_expansion.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "business_summary: 5/50" in result.stdout
    assert "unsafe_refusal: 5/50" in result.stdout
    assert "Total: 30/300" in result.stdout


def test_eval_expansion_cli_writes_review_candidate_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "candidates.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_eval_expansion.py",
            "--per-category",
            "1",
            "--write-candidates",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Wrote 6 review candidate" in result.stdout

    candidates = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(candidates) == 6
    assert all(candidate["candidate_status"] == "review_candidate" for candidate in candidates)
