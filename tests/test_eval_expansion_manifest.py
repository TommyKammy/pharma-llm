import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pharma_llm_lab.dataset import EvaluationCategory, EvalRecord, ReviewStatus, SourceType
from pharma_llm_lab.dataset.schema import parse_record

from scripts.plan_eval_expansion import (
    build_coverage,
    load_manifest,
    propose_candidate_records,
    validate_manifest_consistency,
)


MANIFEST_PATH = Path("evals/manifest/evaluation_set_v0.json")
SEED_PATH = Path("evals/prompts/phase4_seed.jsonl")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def first_seed_record() -> dict[str, object]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8").splitlines()[0])


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


def test_eval_expansion_rejects_review_candidates_in_accepted_prompts(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    accepted_path = write_jsonl(
        tmp_path / "accepted.jsonl",
        [propose_candidate_records(manifest, per_category=1)[0]],
    )
    test_manifest = replace(
        manifest,
        accepted_prompt_files=(accepted_path.relative_to(tmp_path),),
    )

    with pytest.raises(ValueError, match="review candidates are not accepted"):
        build_coverage(test_manifest, repo_root=tmp_path)


def test_eval_expansion_rejects_unapproved_records_in_accepted_prompts(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    candidate = dict(propose_candidate_records(manifest, per_category=1)[0])
    del candidate["candidate_status"]
    accepted_path = write_jsonl(tmp_path / "accepted.jsonl", [candidate])
    test_manifest = replace(
        manifest,
        accepted_prompt_files=(accepted_path.relative_to(tmp_path),),
    )

    with pytest.raises(ValueError, match="review_status must be approved"):
        build_coverage(test_manifest, repo_root=tmp_path)


def test_eval_expansion_rejects_duplicate_accepted_eval_ids(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    seed_record = first_seed_record()
    accepted_path = write_jsonl(tmp_path / "accepted.jsonl", [seed_record, seed_record])
    test_manifest = replace(
        manifest,
        accepted_prompt_files=(accepted_path.relative_to(tmp_path),),
    )

    with pytest.raises(ValueError, match="duplicate accepted eval id: eval_001"):
        build_coverage(test_manifest, repo_root=tmp_path)


def test_eval_expansion_reserves_pending_candidate_ids(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    candidate_dir = tmp_path / "candidates"
    write_jsonl(
        candidate_dir / "phase4_batch_001.jsonl",
        [propose_candidate_records(manifest, per_category=1)[0]],
    )
    test_manifest = replace(manifest, candidate_directory=candidate_dir)

    candidates = propose_candidate_records(test_manifest, per_category=1)

    assert [candidate["id"] for candidate in candidates] == [
        "eval_007",
        "eval_056",
        "eval_106",
        "eval_156",
        "eval_206",
        "eval_256",
    ]


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
