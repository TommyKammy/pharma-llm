import pytest

from pharma_llm_lab.inference import (
    InferenceRequest,
    InferenceTiming,
    MockMlxInferenceClient,
    ModelIdentity,
)
from pharma_llm_lab.inference.contracts import InferenceContractError


def test_inference_request_and_response_mapping_are_stable() -> None:
    request = InferenceRequest(
        request_id="eval_001:qwen-base",
        prompt="安全性情報を要約してください。",
        max_tokens=128,
        metadata={"eval_id": "eval_001", "category": "safety_information"},
    )
    model = ModelIdentity(model_id="mlx-community/mock-qwen", adapter_id=None)
    response = MockMlxInferenceClient(model=model).generate(request)

    assert request.to_mapping() == {
        "request_id": "eval_001:qwen-base",
        "prompt": "安全性情報を要約してください。",
        "max_tokens": 128,
        "temperature": 0.0,
        "metadata": {"eval_id": "eval_001", "category": "safety_information"},
    }
    assert response.to_mapping() == {
        "request_id": "eval_001:qwen-base",
        "generated_text": "[mock-mlx] 安全性情報を要約してください。",
        "model": {
            "model_id": "mlx-community/mock-qwen",
            "provider": "mlx",
            "adapter_id": None,
        },
        "timing": {
            "total_latency_ms": 12.0,
            "ttft_ms": 5.0,
            "tokens_per_second": 166.667,
            "prompt_tokens": 1,
            "completion_tokens": 2,
        },
        "finish_reason": "mock_stop",
        "raw_output_path": None,
    }


def test_mock_mlx_inference_client_is_deterministic() -> None:
    request = InferenceRequest(request_id="deterministic", prompt="同じ入力")
    client = MockMlxInferenceClient(
        model=ModelIdentity(model_id="mlx-community/mock-gemma")
    )

    first = client.generate(request)
    second = client.generate(request)

    assert first == second
    assert first.generated_text == "[mock-mlx] 同じ入力"
    assert first.model.adapter_id is None
    assert first.timing.total_latency_ms > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"request_id": "", "prompt": "入力"}, "request_id"),
        ({"request_id": "bad", "prompt": ""}, "prompt"),
        ({"request_id": "bad", "prompt": "入力", "max_tokens": 0}, "max_tokens"),
        ({"request_id": "bad", "prompt": "入力", "temperature": -0.1}, "temperature"),
    ],
)
def test_inference_request_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(InferenceContractError, match=message):
        InferenceRequest(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("timing", "message"),
    [
        ({"total_latency_ms": -1}, "total_latency_ms"),
        ({"total_latency_ms": 1, "ttft_ms": -1}, "ttft_ms"),
        ({"total_latency_ms": 1, "tokens_per_second": -1}, "tokens_per_second"),
        ({"total_latency_ms": 1, "prompt_tokens": -1}, "prompt_tokens"),
        ({"total_latency_ms": 1, "completion_tokens": -1}, "completion_tokens"),
    ],
)
def test_inference_timing_rejects_negative_values(
    timing: dict[str, float], message: str
) -> None:
    with pytest.raises(InferenceContractError, match=message):
        InferenceTiming(**timing)
