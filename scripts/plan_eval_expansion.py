from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pharma_llm_lab.dataset import (  # noqa: E402
    EvaluationCategory,
    EvalRecord,
    ReviewStatus,
    SourceType,
    parse_record,
)
from pharma_llm_lab.dataset.validators import iter_jsonl  # noqa: E402


DEFAULT_MANIFEST = Path("evals/manifest/evaluation_set_v0.json")

ACCEPTED_REVIEW_STATUSES = {
    ReviewStatus.APPROVED,
    ReviewStatus.EDITED_AND_APPROVED,
}


@dataclass(frozen=True)
class CategoryPlan:
    category: EvaluationCategory
    id_start: int
    id_end: int
    target_count: int
    current_count: int

    @property
    def id_range(self) -> range:
        return range(self.id_start, self.id_end + 1)

    @property
    def remaining_count(self) -> int:
        return max(self.target_count - self.current_count, 0)


@dataclass(frozen=True)
class ExpansionManifest:
    name: str
    target_total: int
    accepted_prompt_files: tuple[Path, ...]
    candidate_directory: Path
    candidate_status: str
    review_workflow: tuple[str, ...]
    categories: tuple[CategoryPlan, ...]


def require_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def load_manifest(path: Path = DEFAULT_MANIFEST) -> ExpansionManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    categories = []
    for raw_category, raw_plan in raw["categories"].items():
        if not isinstance(raw_plan, dict):
            raise ValueError(f"{raw_category} plan must be an object")
        categories.append(
            CategoryPlan(
                category=EvaluationCategory(raw_category),
                id_start=require_int(raw_plan, "id_start"),
                id_end=require_int(raw_plan, "id_end"),
                target_count=require_int(raw_plan, "target_count"),
                current_count=require_int(raw_plan, "current_count"),
            )
        )

    return ExpansionManifest(
        name=raw["name"],
        target_total=require_int(raw, "target_total"),
        accepted_prompt_files=tuple(Path(item) for item in raw["accepted_prompt_files"]),
        candidate_directory=Path(raw["candidate_directory"]),
        candidate_status=raw["candidate_status"],
        review_workflow=tuple(raw["review_workflow"]),
        categories=tuple(categories),
    )


def eval_id(number: int) -> str:
    return f"eval_{number:03d}"


def eval_id_number(record_id: str) -> int:
    match = re.fullmatch(r"eval_(\d{3})", record_id)
    if match is None:
        raise ValueError(f"invalid eval id: {record_id}")
    return int(match.group(1))


def load_accepted_records(manifest: ExpansionManifest, *, repo_root: Path) -> tuple[EvalRecord, ...]:
    records: list[EvalRecord] = []
    for relative_path in manifest.accepted_prompt_files:
        path = repo_root / relative_path
        for line_number, item in iter_jsonl(path):
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: {item.message}")
            if item.get("candidate_status") == manifest.candidate_status:
                raise ValueError(
                    f"{path}:{line_number}: review candidates are not accepted eval records"
                )
            record = parse_record(item)
            if not isinstance(record, EvalRecord):
                raise ValueError(f"{path}:{line_number}: expected an eval record")
            if record.provenance.review_status not in ACCEPTED_REVIEW_STATUSES:
                raise ValueError(
                    f"{path}:{line_number}: {record.id} review_status must be approved "
                    "before accepted coverage counts it"
                )
            records.append(record)
    return tuple(records)


def load_pending_candidate_ids(
    manifest: ExpansionManifest, *, repo_root: Path = REPO_ROOT
) -> set[int]:
    candidate_dir = repo_root / manifest.candidate_directory
    if not candidate_dir.exists():
        return set()
    if not candidate_dir.is_dir():
        raise ValueError(f"candidate_directory is not a directory: {candidate_dir}")

    candidate_ids: set[int] = set()
    for path in sorted(candidate_dir.glob("*.jsonl")):
        for line_number, item in iter_jsonl(path):
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: {item.message}")
            if item.get("candidate_status") != manifest.candidate_status:
                continue
            raw_id = item.get("id")
            if not isinstance(raw_id, str):
                raise ValueError(f"{path}:{line_number}: candidate id must be a string")
            id_number = eval_id_number(raw_id)
            if id_number in candidate_ids:
                raise ValueError(f"{path}:{line_number}: duplicate pending candidate id {raw_id}")
            candidate_ids.add(id_number)

    return candidate_ids


def build_coverage(
    manifest: ExpansionManifest, *, repo_root: Path = REPO_ROOT
) -> dict[EvaluationCategory, int]:
    coverage = {plan.category: 0 for plan in manifest.categories}
    plans_by_category = {plan.category: plan for plan in manifest.categories}
    seen_ids: set[str] = set()

    for record in load_accepted_records(manifest, repo_root=repo_root):
        if record.id in seen_ids:
            raise ValueError(f"duplicate accepted eval id: {record.id}")
        seen_ids.add(record.id)

        plan = plans_by_category[record.category]
        id_number = eval_id_number(record.id)
        if id_number not in plan.id_range:
            raise ValueError(f"{record.id} is outside {record.category.value} range")
        if record.provenance.source_type is not SourceType.EVAL_ONLY:
            raise ValueError(f"{record.id} must use eval_only source_type")
        coverage[record.category] += 1

    return coverage


def validate_manifest_consistency(
    manifest: ExpansionManifest, *, repo_root: Path = REPO_ROOT
) -> None:
    if sum(plan.target_count for plan in manifest.categories) != manifest.target_total:
        raise ValueError("category target_count values must sum to target_total")

    coverage = build_coverage(manifest, repo_root=repo_root)
    for plan in manifest.categories:
        if coverage[plan.category] != plan.current_count:
            raise ValueError(
                f"{plan.category.value} manifest current_count={plan.current_count} "
                f"but accepted records contain {coverage[plan.category]}"
            )


def propose_candidate_records(
    manifest: ExpansionManifest,
    *,
    repo_root: Path = REPO_ROOT,
    per_category: int = 1,
) -> tuple[dict[str, Any], ...]:
    accepted_records = load_accepted_records(manifest, repo_root=repo_root)
    used_ids = {
        *{eval_id_number(record.id) for record in accepted_records},
        *load_pending_candidate_ids(manifest, repo_root=repo_root),
    }
    candidates: list[dict[str, Any]] = []

    for plan in manifest.categories:
        limit = min(per_category, plan.remaining_count)
        next_ids = [number for number in plan.id_range if number not in used_ids][:limit]
        for id_number in next_ids:
            record_id = eval_id(id_number)
            candidates.append(
                {
                    "id": record_id,
                    "dataset_type": "eval",
                    "category": plan.category.value,
                    "candidate_status": manifest.candidate_status,
                    "prompt": (
                        f"[REVIEW CANDIDATE] Draft a synthetic {plan.category.value} "
                        f"evaluation prompt for {record_id}."
                    ),
                    "expected_points": [
                        "Reviewer replaces this placeholder with a concrete scoring point.",
                        "Reviewer confirms category fit and evidence boundary.",
                        "Reviewer confirms safety, pharma style, and factuality expectations.",
                    ],
                    "provenance": {
                        "source_type": "eval_only",
                        "source_document": "synthetic_phase4_candidate",
                        "source_license": "synthetic_test_only",
                        "review_status": "unreviewed",
                        "ai_assisted": False,
                        "ai_tool": None,
                        "raw_ai_output_used_as_training_target": False,
                        "human_reviewer": None,
                        "review_date": None,
                        "risk_flags": risk_flags_for_category(plan.category),
                    },
                }
            )

    return tuple(candidates)


def risk_flags_for_category(category: EvaluationCategory) -> list[str]:
    return {
        EvaluationCategory.BUSINESS_SUMMARY: [],
        EvaluationCategory.PACKAGE_INSERT_READING: ["medical_advice_boundary"],
        EvaluationCategory.SAFETY_INFORMATION: ["safety_signal", "medical_advice_boundary"],
        EvaluationCategory.GXP_QA_AUDIT: ["gxp_context", "regulated_decision"],
        EvaluationCategory.DI_INQUIRY: ["di_context", "medical_advice_boundary"],
        EvaluationCategory.UNSAFE_REFUSAL: ["unsafe_request"],
    }[category]


def format_coverage_report(
    manifest: ExpansionManifest, *, repo_root: Path = REPO_ROOT
) -> str:
    coverage = build_coverage(manifest, repo_root=repo_root)
    lines = [f"{manifest.name} coverage"]
    accepted_total = 0
    for plan in manifest.categories:
        accepted = coverage[plan.category]
        accepted_total += accepted
        remaining = plan.target_count - accepted
        lines.append(f"- {plan.category.value}: {accepted}/{plan.target_count} ({remaining} remaining)")
    lines.append(f"Total: {accepted_total}/{manifest.target_total}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report Phase 4 eval coverage or propose review candidates."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the Evaluation Set v0 manifest.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=1,
        help="Number of review candidates to propose per category.",
    )
    parser.add_argument(
        "--write-candidates",
        type=Path,
        help="Write JSONL review candidates to this path instead of printing a report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.per_category < 1:
        parser.error("--per-category must be at least 1")

    manifest = load_manifest(args.manifest)
    validate_manifest_consistency(manifest)

    if not args.write_candidates:
        print(format_coverage_report(manifest))
        return 0

    if args.write_candidates.exists():
        parser.error(f"candidate output already exists: {args.write_candidates}")

    candidates = propose_candidate_records(manifest, per_category=args.per_category)
    args.write_candidates.parent.mkdir(parents=True, exist_ok=True)
    args.write_candidates.write_text(
        "\n".join(json.dumps(candidate, ensure_ascii=False) for candidate in candidates)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(candidates)} review candidate(s) to {args.write_candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
