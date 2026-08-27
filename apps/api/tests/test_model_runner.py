import asyncio

import pytest
from pydantic import BaseModel

from debate_api.orchestration import StructuredGenerationRunner
from debate_api.providers.model import (
    FakeModelOutcome,
    FakeModelStep,
    ModelErrorCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ProgrammableFakeModelProvider,
)


class AgentOutput(BaseModel):
    answer: str
    score: int


def request(identifier: str = "runner_1", timeout: float = 0.05) -> ModelRequest:
    return ModelRequest(
        request_id=identifier,
        operation="planning",
        input_text="private runner input",
        output_schema_name="UntrustedName",
        timeout_seconds=timeout,
    )


def provider(step: FakeModelStep) -> ProgrammableFakeModelProvider:
    return ProgrammableFakeModelProvider([step])


def test_runner_uses_protocol_and_authoritative_schema_name() -> None:
    fake = provider(FakeModelStep(operation="planning", output={"answer": "ok", "score": 2}))
    assert isinstance(fake, ModelProvider)
    response = asyncio.run(StructuredGenerationRunner(fake).run(request(), AgentOutput))
    assert isinstance(response, ModelResponse)
    assert response.output.answer == "ok"
    assert fake.calls[0].output_schema_name == "AgentOutput"


@pytest.mark.parametrize(
    ("outcome", "category"),
    [
        (FakeModelOutcome.MALFORMED_OUTPUT, ModelErrorCategory.MALFORMED_OUTPUT),
        (FakeModelOutcome.SCHEMA_FAILURE, ModelErrorCategory.SCHEMA_VALIDATION),
        (FakeModelOutcome.REPAIR_FAILURE, ModelErrorCategory.REPAIR_FAILED),
        (FakeModelOutcome.CANCELLED, ModelErrorCategory.CANCELLED),
    ],
)
def test_runner_preserves_normalized_provider_failures(
    outcome: FakeModelOutcome, category: ModelErrorCategory
) -> None:
    step = FakeModelStep(
        operation="planning",
        outcome=outcome,
        output=[] if outcome == FakeModelOutcome.MALFORMED_OUTPUT else {"answer": "bad"},
        repair_output={"answer": "also bad"},
    )
    call_request = request().model_copy(
        update={"repair_attempts": 0 if outcome != FakeModelOutcome.REPAIR_FAILURE else 1}
    )
    with pytest.raises(ModelProviderError) as error:
        asyncio.run(StructuredGenerationRunner(provider(step)).run(call_request, AgentOutput))
    assert error.value.category == category


def test_runner_enforces_timeout_at_calling_boundary() -> None:
    fake = provider(FakeModelStep(operation="planning", delay_seconds=1))
    with pytest.raises(ModelProviderError) as error:
        asyncio.run(StructuredGenerationRunner(fake).run(request(timeout=0.01), AgentOutput))
    assert error.value.category == ModelErrorCategory.TIMEOUT


def test_runner_retrieves_late_provider_exception_after_timeout() -> None:
    class LateFailureProvider:
        request_timeout_seconds = 1.0

        def __init__(self) -> None:
            self.cancelled = asyncio.Event()

        async def generate_structured(
            self, request: ModelRequest, output_schema: type[AgentOutput]
        ) -> ModelResponse[AgentOutput]:
            del request, output_schema
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise RuntimeError("late provider failure") from None

    async def execute() -> tuple[ModelProviderError, LateFailureProvider, list[dict[str, object]]]:
        provider = LateFailureProvider()
        loop = asyncio.get_running_loop()
        contexts: list[dict[str, object]] = []
        loop.set_exception_handler(lambda _, context: contexts.append(context))
        try:
            with pytest.raises(ModelProviderError) as error:
                await StructuredGenerationRunner(provider).run(
                    request(timeout=0.01), AgentOutput
                )
            return error.value, provider, contexts
        finally:
            loop.set_exception_handler(None)

    error, provider_instance, contexts = asyncio.run(execute())
    assert error.category == ModelErrorCategory.TIMEOUT
    assert provider_instance.cancelled.is_set()
    assert contexts == []


def test_runner_preserves_external_cancellation() -> None:
    class SlowProvider:
        async def generate_structured(
            self, request: ModelRequest, output_schema: type[AgentOutput]
        ) -> ModelResponse[AgentOutput]:
            del request, output_schema
            await asyncio.sleep(1)
            raise AssertionError("unreachable")

    async def execute() -> None:
        task = asyncio.create_task(
            StructuredGenerationRunner(SlowProvider()).run(request(), AgentOutput)
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(execute())
