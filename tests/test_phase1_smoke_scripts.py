from scripts.create_argilla_sample import sample_records
from scripts.promptfoo_mock_provider import call_api
from scripts.run_mlx_lora_smoke import TRAIN_RECORDS


def test_argilla_sample_records_preserve_review_metadata() -> None:
    records = sample_records()
    assert records
    assert all("review_status" in record for record in records)
    assert all(
        record["metadata"]["raw_ai_output_used_as_training_target"] is False
        for record in records
    )


def test_promptfoo_mock_provider_marks_review_required() -> None:
    response = call_api(
        "Check this safety note",
        {"config": {"style": "concise"}},
        {"vars": {"audience": "QA"}},
    )
    assert "review_required" in response["output"]
    assert "QA" in response["output"]


def test_mlx_lora_smoke_records_use_completion_format() -> None:
    assert TRAIN_RECORDS
    assert all(set(record) == {"prompt", "completion"} for record in TRAIN_RECORDS)
