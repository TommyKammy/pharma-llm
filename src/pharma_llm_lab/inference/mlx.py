"""MLX-oriented inference client contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pharma_llm_lab.inference.contracts import (
    InferenceRequest,
    InferenceResponse,
    InferenceTiming,
    ModelIdentity,
)


class MlxInferenceClient(Protocol):
    """Small interface shared by mock and real MLX inference clients."""

    @property
    def model(self) -> ModelIdentity:
        """Return the model identity used for responses."""

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate a response for a single inference request."""


@dataclass(frozen=True)
class MockMlxInferenceClient:
    """Deterministic MLX-shaped client for CI and local contract tests."""

    model: ModelIdentity
    response_prefix: str = "[mock-mlx]"

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        prompt_tokens = len(request.prompt.split())
        generated_text = f"{self.response_prefix} {request.prompt}"
        completion_tokens = len(generated_text.split())
        total_latency_ms = float(10 + completion_tokens)
        tokens_per_second = round(completion_tokens / (total_latency_ms / 1000), 3)

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            model=self.model,
            timing=InferenceTiming(
                total_latency_ms=total_latency_ms,
                ttft_ms=5.0,
                tokens_per_second=tokens_per_second,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason="mock_stop",
        )
