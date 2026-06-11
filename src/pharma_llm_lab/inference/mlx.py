"""MLX-oriented inference client contracts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

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
    try:
        tokens = encode(text, add_special_tokens=False)
    except TypeError:
        tokens = encode(text)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if not hasattr(tokens, "__len__"):
        return None
    return len(tokens)


def normalize_generated_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InferenceContractError("generated_text must be a string or None")
    return value


def stream_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    text = getattr(response, "text", "")
    if text is None:
        return ""
    if not isinstance(text, str):
        raise InferenceContractError("stream response text must be a string or None")
    return text


def stream_response_token_count(response: Any) -> int | None:
    token = getattr(response, "token", None)
    if token is None:
        token = getattr(response, "tokens", None)
    if token is None:
        return None
    if isinstance(token, int):
        return 1
    if hasattr(token, "tolist"):
        token = token.tolist()
    if hasattr(token, "__len__"):
        return len(token)
    return 1


def normalize_finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    raw_reason = str(value).strip().lower()
    if not raw_reason:
        return None
    if raw_reason in {"length", "max_tokens", "max token", "max-tokens"}:
        return "length"
    if raw_reason in {"stop", "eos", "end", "end_turn"}:
        return "stop"
    return raw_reason


def stream_response_finish_reason(response: Any) -> str | None:
    for field_name in ("finish_reason", "stop_reason", "finishReason"):
        reason = normalize_finish_reason(getattr(response, field_name, None))
        if reason is not None:
            return reason
    return None


def finish_reason_from_generated_tokens(
    generated_token_count: int | None,
    max_tokens: int,
) -> str:
    if generated_token_count is None:
        return "unknown"
    if generated_token_count >= max_tokens:
        return "length"
    return "stop"


def strip_mlx_cli_diagnostics(stdout: str) -> str:
    diagnostic_prefixes = (
        "========",
        "--------",
        "prompt:",
        "generation:",
        "peak memory:",
        "prompt tokens:",
        "generation tokens:",
        "tokens per second:",
        "wired memory",
        "warning:",
        "[warning]",
    )
    response_lines: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if any(lowered.startswith(prefix) for prefix in diagnostic_prefixes):
            continue
        response_lines.append(line)
    return "\n".join(response_lines).strip()


@dataclass(frozen=True)
class MlxLmPythonClient:
    """Persistent real local MLX LM client backed by the Python API."""

    model: ModelIdentity
    model_path: Path
    adapter_path: Path | None = None
    load_fn: Callable[..., tuple[Any, Any]] | None = None
    stream_generate_fn: Callable[..., Iterable[Any]] | None = None
    make_sampler_fn: Callable[..., Any] | None = None

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
        stream_generate_fn = self.stream_generate_fn
        make_sampler_fn = self.make_sampler_fn
        if load_fn is None or stream_generate_fn is None or make_sampler_fn is None:
            try:
                from mlx_lm import load, stream_generate
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise InferenceContractError(
                    "mlx-lm is required for the Python real MLX client; "
                    "install the training extra or use --client-backend cli"
                ) from exc
            if load_fn is None:
                load_fn = load
            if stream_generate_fn is None:
                stream_generate_fn = stream_generate
            if make_sampler_fn is None:
                make_sampler_fn = make_sampler

        load_kwargs: dict[str, str] = {
            "path_or_hf_repo": str(self.model_path.expanduser().resolve())
        }
        if self.adapter_path is not None:
            load_kwargs["adapter_path"] = str(self.adapter_path.expanduser().resolve())
        loaded_model, tokenizer = load_fn(**load_kwargs)
        object.__setattr__(self, "_loaded_model", loaded_model)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_stream_generate_fn", stream_generate_fn)
        object.__setattr__(self, "_make_sampler_fn", make_sampler_fn)

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        start = perf_counter()
        generated_text = ""
        generated_token_count: int | None = None
        finish_reason: str | None = None
        sampler = self._make_sampler_fn(temp=float(request.temperature))
        for response in self._stream_generate_fn(
            model=self._loaded_model,
            tokenizer=self._tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False,
        ):
            generated_text += stream_response_text(response)
            token_count = stream_response_token_count(response)
            if token_count is not None:
                generated_token_count = (generated_token_count or 0) + token_count
            response_finish_reason = stream_response_finish_reason(response)
            if response_finish_reason is not None:
                finish_reason = response_finish_reason
        total_latency_ms = round((perf_counter() - start) * 1000, 3)
        prompt_tokens = tokenizer_token_count(self._tokenizer, request.prompt)
        completion_tokens = generated_token_count
        tokens_per_second = None
        if completion_tokens is not None and total_latency_ms > 0:
            tokens_per_second = round(completion_tokens / (total_latency_ms / 1000), 3)
        if finish_reason is None:
            finish_reason = finish_reason_from_generated_tokens(
                generated_token_count,
                request.max_tokens,
            )

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
            finish_reason=finish_reason,
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

        generated_text = strip_mlx_cli_diagnostics(result.stdout)
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
