"""Shared inference request and response contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class InferenceContractError(ValueError):
    """Raised when inference contract values are invalid."""


def require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceContractError(f"{field_name} must be a non-empty string")
    return value


def require_non_negative_number(value: float | int | None, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or value < 0:
        raise InferenceContractError(f"{field_name} must be a non-negative number")
    return float(value)


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    provider: str = "mlx"
    adapter_id: str | None = None

    def __post_init__(self) -> None:
        require_non_empty_string(self.model_id, "model_id")
        require_non_empty_string(self.provider, "provider")
        if self.adapter_id is not None:
            require_non_empty_string(self.adapter_id, "adapter_id")

    def to_mapping(self) -> dict[str, str | None]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "adapter_id": self.adapter_id,
        }


@dataclass(frozen=True)
class InferenceTiming:
    total_latency_ms: float
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        require_non_negative_number(self.total_latency_ms, "total_latency_ms")
        require_non_negative_number(self.ttft_ms, "ttft_ms")
        require_non_negative_number(self.tokens_per_second, "tokens_per_second")
        require_non_negative_number(self.prompt_tokens, "prompt_tokens")
        require_non_negative_number(self.completion_tokens, "completion_tokens")

    def to_mapping(self) -> dict[str, float | int | None]:
        return {
            "total_latency_ms": self.total_latency_ms,
            "ttft_ms": self.ttft_ms,
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    request_id: str
    max_tokens: int = 512
    temperature: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        require_non_empty_string(self.prompt, "prompt")
        require_non_empty_string(self.request_id, "request_id")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise InferenceContractError("max_tokens must be a positive integer")
        if not isinstance(self.temperature, int | float) or self.temperature < 0:
            raise InferenceContractError("temperature must be a non-negative number")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": float(self.temperature),
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    generated_text: str
    model: ModelIdentity
    timing: InferenceTiming
    finish_reason: str = "stop"
    raw_output_path: str | None = None

    def __post_init__(self) -> None:
        require_non_empty_string(self.request_id, "request_id")
        if not isinstance(self.generated_text, str):
            raise InferenceContractError("generated_text must be a string")
        require_non_empty_string(self.finish_reason, "finish_reason")
        if self.raw_output_path is not None:
            require_non_empty_string(self.raw_output_path, "raw_output_path")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "generated_text": self.generated_text,
            "model": self.model.to_mapping(),
            "timing": self.timing.to_mapping(),
            "finish_reason": self.finish_reason,
            "raw_output_path": self.raw_output_path,
        }
