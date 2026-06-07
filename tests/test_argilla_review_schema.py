from scripts.create_argilla_sample import (
    ARGILLA_WORKSPACE,
    REQUIRED_REVIEW_METADATA,
    SUPPORTED_REVIEW_STATUSES,
    TRAINING_ELIGIBLE_REVIEW_STATUSES,
    argilla_dataset_name,
    sample_records,
)
from pharma_llm_lab.dataset import ReviewStatus, SourceType


def test_argilla_dataset_name_is_stable_for_phase3() -> None:
    assert (
        argilla_dataset_name(phase=3, dataset_type="sft")
        == "pharma_llm_phase3_phase3_sft_review"
    )


def test_review_status_mapping_matches_phase2_statuses() -> None:
    phase2_statuses = {status.value for status in ReviewStatus}

    assert set(SUPPORTED_REVIEW_STATUSES).issubset(phase2_statuses)
    assert set(TRAINING_ELIGIBLE_REVIEW_STATUSES) == {
        ReviewStatus.APPROVED.value,
        ReviewStatus.EDITED_AND_APPROVED.value,
    }
    assert ReviewStatus.UNREVIEWED.value not in SUPPORTED_REVIEW_STATUSES


def test_argilla_sample_records_cover_required_source_classes() -> None:
    records = sample_records()
    source_types = {record["provenance"]["source_type"] for record in records}
    review_statuses = {record["provenance"]["review_status"] for record in records}

    assert SourceType.HUMAN_AUTHORED.value in source_types
    assert SourceType.AI_CANDIDATE_UNREVIEWED.value in source_types
    assert SourceType.HUMAN_EDITED_AI_ASSISTED.value in source_types
    assert SourceType.EVAL_ONLY.value in source_types
    assert SourceType.RAW_AI_OUTPUT.value in source_types
    assert ReviewStatus.APPROVED.value in review_statuses
    assert ReviewStatus.NEEDS_EDIT.value in review_statuses
    assert ReviewStatus.EDITED_AND_APPROVED.value in review_statuses


def test_argilla_sample_records_preserve_phase2_provenance_metadata() -> None:
    for record in sample_records():
        provenance = record["provenance"]

        assert provenance["source_document"] == "synthetic_argilla_phase3_sample"
        assert provenance["source_license"] == "synthetic_test_only"
        assert isinstance(provenance["ai_assisted"], bool)
        assert isinstance(provenance["raw_ai_output_used_as_training_target"], bool)
        assert isinstance(provenance["risk_flags"], list)
        assert set(REQUIRED_REVIEW_METADATA).issubset(provenance)


def test_argilla_sample_records_include_review_target_metadata() -> None:
    for record in sample_records():
        argilla = record["argilla"]

        assert argilla["workspace"] == ARGILLA_WORKSPACE
        assert str(argilla["dataset"]).startswith("pharma_llm_phase3_phase3_")
        assert "review_status" in argilla["questions"]
        assert "risk_flags" in argilla["questions"]


def test_promotable_sample_records_have_reviewer_metadata() -> None:
    promotable_statuses = set(TRAINING_ELIGIBLE_REVIEW_STATUSES)

    for record in sample_records():
        provenance = record["provenance"]
        if provenance["review_status"] not in promotable_statuses:
            continue

        assert provenance["human_reviewer"] == "synthetic_reviewer"
        assert provenance["review_date"] == "2026-06-08"
        assert not provenance["raw_ai_output_used_as_training_target"]
