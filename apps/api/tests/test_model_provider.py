import asyncio
import traceback

import pytest
from pydantic import BaseModel

from debate_api.providers.model import (
    FakeModelOutcome,
    FakeModelStep,
    ModelChatMessage,
    ModelErrorCategory,
    ModelIdentity,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelUsage,
    ProgrammableFakeModelProvider,
)


class AgentOutput(BaseModel):
    answer: str
    score: int


def request(identifier: str = "req_1", *, repair_attempts: int = 1) -> ModelRequest:
    return ModelRequest(
        request_id=identifier,
        operation="test_generation",
        input_text="private prompt that must not appear in provider errors",
        output_schema_name="AgentOutput",
        timeout_seconds=0.05,
        repair_attempts=repair_attempts,
    )


def text_request(identifier: str = "text_1") -> ModelRequest:
    return ModelRequest(
        request_id=identifier,
        operation="test_generation",
        input_text="Return a plain answer.",
        output_schema_name=None,
        timeout_seconds=0.05,
        repair_attempts=0,
    )


def test_fake_provider_is_protocol_compatible_and_unknown_usage_is_preserved() -> None:
    provider = ProgrammableFakeModelProvider(
        [FakeModelStep(operation="test_generation", output={"answer": "bounded", "score": 3})],
        identity=ModelIdentity(provider="fake", model="qwen-test", revision="r1"),
    )
    assert isinstance(provider, ModelProvider)
    response = asyncio.run(provider.generate_structured(request(), AgentOutput))
    assert response.output == AgentOutput(answer="bounded", score=3)
    assert response.model.model == "qwen-test"
    assert response.model.revision == "r1"
    assert response.usage == ModelUsage()
    assert response.latency_ms >= 0
    assert provider.calls[0].operation == "test_generation"


def test_fake_provider_supports_plain_text_without_a_schema() -> None:
    provider = ProgrammableFakeModelProvider(
        [FakeModelStep(operation="test_generation", output="Reasoning. Final answer: 4")]
    )

    response = asyncio.run(provider.generate_text(text_request()))

    assert response.output == "Reasoning. Final answer: 4"
    assert provider.calls[0].output_schema_name is None


def test_model_request_accepts_a_role_preserving_conversation() -> None:
    conversation = (
        ModelChatMessage(role="user", content="Solve this problem."),
        ModelChatMessage(role="assistant", content="My first solution."),
        ModelChatMessage(role="user", content="Review the other solution."),
    )

    model_request = text_request().model_copy(update={"conversation": conversation})

    assert model_request.conversation == conversation


def test_usage_supports_partial_and_fully_reported_values() -> None:
    partial = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="test_generation",
                output={"answer": "partial", "score": 1},
                completion_tokens=7,
            )
        ]
    )
    assert asyncio.run(partial.generate_structured(request(), AgentOutput)).usage == ModelUsage(
        completion_tokens=7
    )
    full = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="test_generation",
                output={"answer": "full", "score": 1},
                prompt_tokens=0,
                completion_tokens=7,
                total_tokens=7,
            )
        ]
    )
    assert asyncio.run(full.generate_structured(request("full"), AgentOutput)).usage == ModelUsage(
        prompt_tokens=0, completion_tokens=7, total_tokens=7
    )


@pytest.mark.parametrize(
    "step",
    [
        FakeModelStep(
            operation="test_generation", output={"answer": "negative", "score": 1}, prompt_tokens=-1
        ),
        FakeModelStep(
            operation="test_generation",
            output={"answer": "inconsistent", "score": 1},
            prompt_tokens=2,
            completion_tokens=3,
            total_tokens=99,
        ),
    ],
)
def test_invalid_usage_is_rejected_as_safe_normalization_error(step: FakeModelStep) -> None:
    with pytest.raises(ModelProviderError) as error:
        asyncio.run(
            ProgrammableFakeModelProvider([step]).generate_structured(request(), AgentOutput)
        )
    assert error.value.category == ModelErrorCategory.NORMALIZATION_ERROR
    assert "negative" not in str(error.value)


def test_fake_provider_programs_call_sequence_and_operation_type() -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(operation="planning", output={"answer": "first", "score": 1}),
            FakeModelStep(operation="research", output={"answer": "second", "score": 2}),
        ]
    )
    first = asyncio.run(
        provider.generate_structured(
            request("first").model_copy(update={"operation": "planning"}), AgentOutput
        )
    )
    second = asyncio.run(
        provider.generate_structured(
            request("second").model_copy(update={"operation": "research"}), AgentOutput
        )
    )
    assert [first.output.answer, second.output.answer] == ["first", "second"]
    with pytest.raises(ModelProviderError) as error:
        asyncio.run(provider.generate_structured(request("extra"), AgentOutput))
    assert error.value.category == ModelErrorCategory.UNEXPECTED_CALL


def test_schema_name_mismatch_is_rejected_by_provider() -> None:
    provider = ProgrammableFakeModelProvider(
        [FakeModelStep(operation="test_generation", output={"answer": "x", "score": 1})]
    )
    with pytest.raises(ModelProviderError) as error:
        asyncio.run(
            provider.generate_structured(
                request().model_copy(update={"output_schema_name": "WrongSchema"}), AgentOutput
            )
        )
    assert error.value.category == ModelErrorCategory.NORMALIZATION_ERROR


def test_malformed_and_schema_failures_are_safe_and_distinct() -> None:
    sentinel = "RAW_SENTINEL_123"
    malformed = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="test_generation",
                outcome=FakeModelOutcome.MALFORMED_OUTPUT,
                output=[sentinel],
            )
        ]
    )
    with pytest.raises(ModelProviderError) as malformed_error:
        asyncio.run(malformed.generate_structured(request(repair_attempts=0), AgentOutput))
    assert malformed_error.value.category == ModelErrorCategory.MALFORMED_OUTPUT
    assert sentinel not in str(malformed_error.value)
    schema_failure = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="test_generation",
                outcome=FakeModelOutcome.SCHEMA_FAILURE,
                output={"answer": sentinel},
            )
        ]
    )
    with pytest.raises(ModelProviderError) as schema_error:
        asyncio.run(schema_failure.generate_structured(request(repair_attempts=0), AgentOutput))
    assert schema_error.value.category == ModelErrorCategory.SCHEMA_VALIDATION
    assert sentinel not in str(schema_error.value)
    assert schema_error.value.__cause__ is None
    assert sentinel not in repr(schema_error.value.__context__)
    assert sentinel not in "".join(traceback.format_exception(schema_error.value))


def test_bounded_repair_is_one_attempt_and_does_not_leak_raw_output() -> None:
    repaired = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="test_generation",
                outcome=FakeModelOutcome.SCHEMA_FAILURE,
                output={"answer": "bad"},
                repair_output={"answer": "fixed", "score": 4},
            )
        ]
    )
    response = asyncio.run(repaired.generate_structured(request(), AgentOutput))
    assert response.output.answer == "fixed"
    assert response.repair_attempted is True
    sentinel = "REPAIR_RAW_SENTINEL_456"
    failed_repair = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="test_generation",
                outcome=FakeModelOutcome.SCHEMA_FAILURE,
                output={"answer": "bad"},
                repair_output={"answer": sentinel},
            )
        ]
    )
    with pytest.raises(ModelProviderError) as error:
        asyncio.run(failed_repair.generate_structured(request(), AgentOutput))
    assert error.value.category == ModelErrorCategory.REPAIR_FAILED
    assert error.value.repair_attempted is True
    assert sentinel not in str(error.value)
    assert error.value.__cause__ is None
    assert sentinel not in repr(error.value.__context__)
    assert sentinel not in "".join(traceback.format_exception(error.value))


def test_timeout_and_provider_reported_cancellation_are_normalized() -> None:
    timeout = ProgrammableFakeModelProvider(
        [FakeModelStep(operation="test_generation", outcome=FakeModelOutcome.TIMEOUT)]
    )
    with pytest.raises(ModelProviderError) as timeout_error:
        asyncio.run(timeout.generate_structured(request(), AgentOutput))
    assert timeout_error.value.category == ModelErrorCategory.TIMEOUT
    assert timeout_error.value.retryable is True
    cancelled = ProgrammableFakeModelProvider(
        [FakeModelStep(operation="test_generation", outcome=FakeModelOutcome.CANCELLED)]
    )
    with pytest.raises(ModelProviderError) as cancelled_error:
        asyncio.run(cancelled.generate_structured(request("cancel"), AgentOutput))
    assert cancelled_error.value.category == ModelErrorCategory.CANCELLED


def test_external_task_cancellation_preserves_asyncio_cancellation() -> None:
    provider = ProgrammableFakeModelProvider(
        [FakeModelStep(operation="test_generation", delay_seconds=1)]
    )

    async def execute() -> None:
        task = asyncio.create_task(provider.generate_structured(request("external"), AgentOutput))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(execute())
