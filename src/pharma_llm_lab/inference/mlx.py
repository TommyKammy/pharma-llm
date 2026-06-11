"""MLX-oriented inference client contracts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Protocol

from pharma_llm_lab.inference.contracts import (
    InferenceContractError,
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
        generated_tokens = [self.response_prefix, *request.prompt.split()][
            : request.max_tokens
        ]
        generated_text = " ".join(generated_tokens)
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


def tokenizer_token_count(tokenizer: Any, text: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None
    tokens = encode(text)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if not hasattr(tokens, "__len__"):
        return None
    return len(tokens)


def finish_reason_from_completion_tokens(
    completion_tokens: int | None,
    max_tokens: int,
) -> str:
    if completion_tokens is None:
        return "unknown"
    if completion_tokens >= max_tokens:
        return "length"
    return "stop"


@dataclass(frozen=True)
class MlxLmPythonClient:
    """Persistent real local MLX LM client backed by the Python API."""

    model: ModelIdentity
    model_path: Path
    adapter_path: Path | None = None
    load_fn: Callable[..., tuple[Any, Any]] | None = None
    generate_fn: Callable[..., str] | None = None

    def __post_init__(self) -> None:
        if not self.model_path.expanduser().exists():
            raise InferenceContractError(f"model_path does not exist: {self.model_path}")
        if self.adapter_path is not None and not self.adapter_path.expanduser().exists():
            raise InferenceContractError(f"adapter_path does not exist: {self.adapter_path}")
        if self.adapter_path is None and self.model.adapter_id is not None:
            raise InferenceContractError("base MLX client must not set adapter_id")
        if self.adapter_path is not None and self.model.adapter_id is None:
            raise InferenceContractError("LoRA MLX client requires model.adapter_id")

        load_fn = self.load_fn
        generate_fn = self.generate_fn
        if load_fn is None or generate_fn is None:
            try:
                from mlx_lm import generate, load
            except ImportError as exc:
                raise InferenceContractError(
                    "mlx-lm is required for the Python real MLX client; "
                    "install the training extra or use --client-backend cli"
                ) from exc
            load_fn = load
            generate_fn = generate

        load_kwargs: dict[str, str] = {
            "path_or_hf_repo": str(self.model_path.expanduser().resolve())
        }
        if self.adapter_path is not None:
            load_kwargs["adapter_path"] = str(self.adapter_path.expanduser().resolve())
        loaded_model, tokenizer = load_fn(**load_kwargs)
        object.__setattr__(self, "_loaded_model", loaded_model)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_generate_fn", generate_fn)

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        start = perf_counter()
        generated_text = self._generate_fn(
            model=self._loaded_model,
            tokenizer=self._tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            verbose=False,
        )
        total_latency_ms = round((perf_counter() - start) * 1000, 3)
        prompt_tokens = tokenizer_token_count(self._tokenizer, request.prompt)
        completion_tokens = tokenizer_token_count(self._tokenizer, generated_text)
        tokens_per_second = None
        if completion_tokens is not None and total_latency_ms > 0:
            tokens_per_second = round(completion_tokens / (total_latency_ms / 1000), 3)

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            model=self.model,
            timing=InferenceTiming(
                total_latency_ms=total_latency_ms,
                ttft_ms=None,
                tokens_per_second=tokens_per_second,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason=finish_reason_from_completion_tokens(
                completion_tokens,
                request.max_tokens,
            ),
        )


@dataclass(frozen=True)
class MlxLmCliClient:
    """Real local MLX LM client backed by the `mlx_lm.generate` CLI."""

    model: ModelIdentity
    model_path: Path
    adapter_path: Path | None = None
    command: tuple[str, ...] = ("mlx_lm.generate",)
    timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        if not self.command:
            raise InferenceContractError("command must contain at least one executable")
        if not self.model_path.expanduser().exists():
            raise InferenceContractError(f"model_path does not exist: {self.model_path}")
        if self.adapter_path is not None and not self.adapter_path.expanduser().exists():
            raise InferenceContractError(f"adapter_path does not exist: {self.adapter_path}")
        if self.adapter_path is None and self.model.adapter_id is not None:
            raise InferenceContractError("base MLX client must not set adapter_id")
        if self.adapter_path is not None and self.model.adapter_id is None:
            raise InferenceContractError("LoRA MLX client requires model.adapter_id")

    def build_command(self, request: InferenceRequest) -> list[str]:
        command = [
            *self.command,
            "--model",
            str(self.model_path.expanduser().resolve()),
            "--prompt",
            request.prompt,
            "--max-tokens",
            str(request.max_tokens),
            "--temp",
            str(float(request.temperature)),
            "--verbose",
            "False",
        ]
        if self.adapter_path is not None:
            command.extend(["--adapter-path", str(self.adapter_path.expanduser().resolve())])
        return command

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        start = perf_counter()
        result = subprocess.run(
            self.build_command(request),
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        total_latency_ms = round((perf_counter() - start) * 1000, 3)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise InferenceContractError(
                f"mlx_lm.generate failed with exit code {result.returncode}: {stderr}"
            )

        generated_text = result.stdout.strip()
        return InferenceResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            model=self.model,
            timing=InferenceTiming(
                total_latency_ms=total_latency_ms,
                ttft_ms=None,
                tokens_per_second=None,
                prompt_tokens=None,
                completion_tokens=None,
            ),
            finish_reason="unknown",
        )
