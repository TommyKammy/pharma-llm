"""Typed dataset record schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pharma_llm_lab.dataset.provenance import ProvenanceMetadata


class SchemaError(ValueError):
    """Raised when a dataset record does not match the expected schema."""


class DatasetType(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    CPT = "cpt"
    EVAL = "eval"


class DatasetRecord(Protocol):
    id: str
    dataset_type: DatasetType
    provenance: ProvenanceMetadata


def require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{key} must be a non-empty string")
    return value


def require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise SchemaError(f"{key} must be an object")
    return value


def parse_provenance(mapping: dict[str, Any]) -> ProvenanceMetadata:
    try:
        return ProvenanceMetadata.from_mapping(require_mapping(mapping, "provenance"))
    except ValueError as exc:
        raise SchemaError(str(exc)) from exc


def require_dataset_type(mapping: dict[str, Any], expected: DatasetType) -> None:
    raw_dataset_type = mapping.get("dataset_type")
    if raw_dataset_type != expected.value:
        raise SchemaError(
            f"dataset_type must be {expected.value!r}, got {raw_dataset_type!r}"
        )


@dataclass(frozen=True)
class SftRecord:
    id: str
    prompt: str
    response: str
    provenance: ProvenanceMetadata
    dataset_type: DatasetType = DatasetType.SFT

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SftRecord":
        require_dataset_type(mapping, DatasetType.SFT)
        return cls(
            id=require_string(mapping, "id"),
            prompt=require_string(mapping, "prompt"),
            response=require_string(mapping, "response"),
            provenance=parse_provenance(mapping),
        )


@dataclass(frozen=True)
class DpoRecord:
    id: str
    prompt: str
    chosen: str
    rejected: str
    provenance: ProvenanceMetadata
    dataset_type: DatasetType = DatasetType.DPO

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "DpoRecord":
        require_dataset_type(mapping, DatasetType.DPO)
        return cls(
            id=require_string(mapping, "id"),
            prompt=require_string(mapping, "prompt"),
            chosen=require_string(mapping, "chosen"),
            rejected=require_string(mapping, "rejected"),
            provenance=parse_provenance(mapping),
        )


@dataclass(frozen=True)
class CptRecord:
    id: str
    text: str
    provenance: ProvenanceMetadata
    dataset_type: DatasetType = DatasetType.CPT

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "CptRecord":
        require_dataset_type(mapping, DatasetType.CPT)
        return cls(
            id=require_string(mapping, "id"),
            text=require_string(mapping, "text"),
            provenance=parse_provenance(mapping),
        )


@dataclass(frozen=True)
class EvalRecord:
    id: str
    prompt: str
    expected_points: tuple[str, ...]
    provenance: ProvenanceMetadata
    category: str | None = None
    dataset_type: DatasetType = DatasetType.EVAL

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "EvalRecord":
        require_dataset_type(mapping, DatasetType.EVAL)
        expected_points = mapping.get("expected_points")
        if not isinstance(expected_points, list | tuple) or not expected_points:
            raise SchemaError("expected_points must be a non-empty list")

        category = mapping.get("category")
        if category is not None and not isinstance(category, str):
            raise SchemaError("category must be a string when provided")

        return cls(
            id=require_string(mapping, "id"),
            prompt=require_string(mapping, "prompt"),
            expected_points=tuple(str(point) for point in expected_points),
            provenance=parse_provenance(mapping),
            category=category,
        )


RECORD_TYPES = {
    DatasetType.SFT: SftRecord,
    DatasetType.DPO: DpoRecord,
    DatasetType.CPT: CptRecord,
    DatasetType.EVAL: EvalRecord,
}


def parse_record(mapping: dict[str, Any]) -> DatasetRecord:
    raw_dataset_type = mapping.get("dataset_type")
    try:
        dataset_type = DatasetType(raw_dataset_type)
    except ValueError as exc:
        raise SchemaError(f"invalid dataset_type: {raw_dataset_type!r}") from exc

    return RECORD_TYPES[dataset_type].from_mapping(mapping)
