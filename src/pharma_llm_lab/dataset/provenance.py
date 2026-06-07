"""Provenance and review-state schema helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SourceType(StrEnum):
    HUMAN_AUTHORED = "human_authored"
    HUMAN_EDITED_AI_ASSISTED = "human_edited_ai_assisted"
    PUBLIC_DOC_DERIVED = "public_doc_derived"
    INTERNAL_DOC_DERIVED = "internal_doc_derived"
    AI_CANDIDATE_UNREVIEWED = "ai_candidate_unreviewed"
    EVAL_ONLY = "eval_only"
    RAW_AI_OUTPUT = "raw_ai_output"


class ReviewStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"
    EDITED_AND_APPROVED = "edited_and_approved"
    RISK_FLAGGED = "risk_flagged"
    UNREVIEWED = "unreviewed"


TRAINING_APPROVED_STATUSES = {
    ReviewStatus.APPROVED,
    ReviewStatus.EDITED_AND_APPROVED,
}

TRAINING_BLOCKED_SOURCE_TYPES = {
    SourceType.AI_CANDIDATE_UNREVIEWED,
    SourceType.EVAL_ONLY,
    SourceType.RAW_AI_OUTPUT,
}


@dataclass(frozen=True)
class ProvenanceMetadata:
    source_type: SourceType
    source_document: str
    source_license: str
    review_status: ReviewStatus
    ai_assisted: bool = False
    ai_tool: str | None = None
    raw_ai_output_used_as_training_target: bool = False
    human_reviewer: str | None = None
    review_date: str | None = None
    risk_flags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ProvenanceMetadata":
        required = ("source_type", "source_document", "source_license", "review_status")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"missing provenance field(s): {', '.join(missing)}")

        try:
            source_type = SourceType(value["source_type"])
        except ValueError as exc:
            raise ValueError(f"invalid source_type: {value['source_type']!r}") from exc

        try:
            review_status = ReviewStatus(value["review_status"])
        except ValueError as exc:
            raise ValueError(f"invalid review_status: {value['review_status']!r}") from exc

        risk_flags = value.get("risk_flags", ())
        if risk_flags is None:
            risk_flags = ()
        if not isinstance(risk_flags, list | tuple):
            raise ValueError("risk_flags must be a list or tuple")

        return cls(
            source_type=source_type,
            source_document=str(value["source_document"]),
            source_license=str(value["source_license"]),
            review_status=review_status,
            ai_assisted=bool(value.get("ai_assisted", False)),
            ai_tool=value.get("ai_tool"),
            raw_ai_output_used_as_training_target=bool(
                value.get("raw_ai_output_used_as_training_target", False)
            ),
            human_reviewer=value.get("human_reviewer"),
            review_date=value.get("review_date"),
            risk_flags=tuple(str(flag) for flag in risk_flags),
        )

    @property
    def is_reviewed_for_training(self) -> bool:
        return self.review_status in TRAINING_APPROVED_STATUSES

    @property
    def is_blocked_for_training(self) -> bool:
        return (
            self.source_type in TRAINING_BLOCKED_SOURCE_TYPES
            or self.raw_ai_output_used_as_training_target
            or not self.is_reviewed_for_training
        )

    @property
    def is_ai_assisted_but_reviewed(self) -> bool:
        return self.ai_assisted and self.is_reviewed_for_training
