from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

from debate_api.domain.models import InvestigatorDirectedUpdateOutput, InvestigatorOpeningOutput
from debate_api.providers.model import (
    ModelChatMessage,
    ModelErrorCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
)
from debate_api.providers.openai_compatible import (
    OpenAICompatibleModelProvider,
    ReasoningEffort,
    _normalize_strict_json_schema,
)


class AgentOutput(BaseModel):
    answer: str = Field(min_length=1)


class NestedOutput(BaseModel):
    label: str
    optional_note: str | None = None


class EnvelopeOutput(BaseModel):
    nested: NestedOutput
    optional_title: str | None = None


Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


def request(*, timeout_seconds: float = 1) -> ModelRequest:
    return ModelRequest(
        request_id="request-1",
        operation="test",
        input_text="bounded user input",
        output_schema_name=AgentOutput.__name__,
        timeout_seconds=timeout_seconds,
        max_output_tokens=123,
        repair_attempts=1,
    )


def text_request(*, timeout_seconds: float = 1) -> ModelRequest:
    return request(timeout_seconds=timeout_seconds).model_copy(
        update={"output_schema_name": None, "repair_attempts": 0}
    )


def completion(
    content: str | None,
    *,
    usage: dict[str, int] | None = None,
    refusal: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def provider(
    handler: Handler,
    *,
    connection_mode: Literal["openai", "ollama"] = "openai",
    base_url: str = "https://api.openai.com/v1",
    max_concurrency: int = 1,
    reasoning_effort: ReasoningEffort | None = None,
    temperature: float | None = None,
) -> tuple[OpenAICompatibleModelProvider, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = openai.AsyncOpenAI(
        api_key="synthetic-key",
        base_url=base_url,
        timeout=1,
        max_retries=0,
        http_client=http_client,
    )
    adapter = OpenAICompatibleModelProvider(
        "synthetic-key",
        model="test-model",
        connection_mode=connection_mode,
        base_url=base_url,
        max_concurrency=max_concurrency,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        client=client,
    )
    return adapter, http_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "connection_mode",
        "base_url",
        "strict",
        "token_parameter",
        "other_parameter",
        "reasoning_effort",
    ),
    [
        (
            "openai",
            "https://api.openai.com/v1",
            True,
            "max_completion_tokens",
            "max_tokens",
            None,
        ),
        (
            "ollama",
            "http://127.0.0.1:11434/v1",
            False,
            "max_tokens",
            "max_completion_tokens",
            None,
        ),
    ],
)
async def test_both_connections_use_one_chat_completions_schema_path(
    connection_mode: Literal["openai", "ollama"],
    base_url: str,
    strict: bool,
    token_parameter: str,
    other_parameter: str,
    reasoning_effort: str | None,
) -> None:
    calls: list[httpx.Request] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request)
        return httpx.Response(
            200,
            json=completion(
                '{"answer":"valid"}',
                usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            ),
        )

    adapter, http_client = provider(
        handler,
        connection_mode=connection_mode,
        base_url=base_url,
    )
    adapter.structured_output_strict = strict
    assert isinstance(adapter, ModelProvider)
    async with adapter:
        result = await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()

    assert len(calls) == 1
    body = json.loads(calls[0].content)
    assert calls[0].url.path == "/v1/chat/completions"
    assert body["model"] == "test-model"
    assert body["messages"][1] == {"role": "user", "content": "bounded user input"}
    assert body[token_parameter] == 123
    assert other_parameter not in body
    if reasoning_effort is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == reasoning_effort
    assert "temperature" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is strict
    assert body["response_format"]["json_schema"]["schema"]["title"] == "AgentOutput"
    if strict:
        assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        assert body["response_format"]["json_schema"]["schema"]["required"] == ["answer"]
    assert result.output == AgentOutput(answer="valid")
    assert result.model.provider == connection_mode
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 5
    assert result.repair_attempted is False


@pytest.mark.asyncio
async def test_plain_text_path_omits_response_format() -> None:
    calls: list[httpx.Request] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request)
        return httpx.Response(200, json=completion("Reasoning. Final answer: 4"))

    adapter, http_client = provider(handler)
    async with adapter:
        result = await adapter.generate_text(text_request())
    await http_client.aclose()

    body = json.loads(calls[0].content)
    assert "response_format" not in body
    assert "plain text, not JSON" in body["messages"][0]["content"]
    assert result.output == "Reasoning. Final answer: 4"


@pytest.mark.asyncio
async def test_role_preserving_conversation_omits_developer_message() -> None:
    calls: list[httpx.Request] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request)
        return httpx.Response(200, json=completion("Updated. Final answer: 4"))

    adapter, http_client = provider(handler)
    model_request = text_request().model_copy(
        update={
            "conversation": (
                ModelChatMessage(role="user", content="Original problem"),
                ModelChatMessage(role="assistant", content="Initial reasoning"),
                ModelChatMessage(role="user", content="Other agent reasoning"),
            )
        }
    )
    async with adapter:
        result = await adapter.generate_text(model_request)
    await http_client.aclose()

    body = json.loads(calls[0].content)
    assert body["messages"] == [
        {"role": "user", "content": "Original problem"},
        {"role": "assistant", "content": "Initial reasoning"},
        {"role": "user", "content": "Other agent reasoning"},
    ]
    assert all(message["role"] != "developer" for message in body["messages"])
    assert "response_format" not in body
    assert result.output == "Updated. Final answer: 4"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection_mode", "base_url"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("ollama", "http://127.0.0.1:11434/v1"),
    ],
)
async def test_configured_reasoning_effort_is_forwarded_for_both_connections(
    connection_mode: Literal["openai", "ollama"], base_url: str
) -> None:
    calls: list[httpx.Request] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request)
        return httpx.Response(200, json=completion('{"answer":"valid"}'))

    adapter, http_client = provider(
        handler,
        connection_mode=connection_mode,
        base_url=base_url,
        reasoning_effort="medium",
    )
    async with adapter:
        await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()

    assert json.loads(calls[0].content)["reasoning_effort"] == "medium"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection_mode", "base_url"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("ollama", "http://127.0.0.1:11434/v1"),
    ],
)
async def test_configured_temperature_is_forwarded_for_both_connections(
    connection_mode: Literal["openai", "ollama"], base_url: str
) -> None:
    calls: list[httpx.Request] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request)
        return httpx.Response(200, json=completion('{"answer":"valid"}'))

    adapter, http_client = provider(
        handler,
        connection_mode=connection_mode,
        base_url=base_url,
        temperature=0.7,
    )
    async with adapter:
        await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()

    assert json.loads(calls[0].content)["temperature"] == 0.7


def test_strict_schema_normalizer_handles_nested_and_nullable_models() -> None:
    schema = _normalize_strict_json_schema(EnvelopeOutput)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["nested", "optional_title"]
    assert schema["properties"]["optional_title"]["anyOf"][-1] == {"type": "null"}
    assert "default" not in schema["properties"]["optional_title"]
    nested = schema["$defs"]["NestedOutput"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["label", "optional_note"]
    assert nested["properties"]["optional_note"]["anyOf"][-1] == {"type": "null"}


@pytest.mark.parametrize(
    "output_schema",
    [InvestigatorOpeningOutput, InvestigatorDirectedUpdateOutput],
)
def test_investigator_schema_has_direct_allowlisted_claim_type(
    output_schema: type[BaseModel],
) -> None:
    schema = _normalize_strict_json_schema(output_schema)

    def assert_no_ref_siblings(value: object) -> None:
        if isinstance(value, dict):
            if "$ref" in value:
                assert set(value) == {"$ref"}
            for child in value.values():
                assert_no_ref_siblings(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_ref_siblings(child)

    assert_no_ref_siblings(schema)
    claim_type_schema = schema["$defs"]["InvestigatorClaimDraft"]["properties"]["claim_type"]
    assert "$ref" not in claim_type_schema
    assert claim_type_schema["enum"] == ["fact", "inference", "recommendation"]
    if output_schema is InvestigatorOpeningOutput:
        assert "target_option" not in schema["properties"]
        assert "interaction_kind" not in schema["properties"]
    else:
        assert schema["required"] == ["claims", "target_option", "interaction_kind"]
        target_option_schema = schema["properties"]["target_option"]
        assert target_option_schema["minimum"] == 0
        assert target_option_schema["maximum"] == 69


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "category"),
    [
        (None, ModelErrorCategory.MALFORMED_OUTPUT),
        ("", ModelErrorCategory.MALFORMED_OUTPUT),
        ("not json", ModelErrorCategory.MALFORMED_OUTPUT),
        (
            '{"answer":"one"} and {"answer":"two"}',
            ModelErrorCategory.MALFORMED_OUTPUT,
        ),
        ('{"answer":"truncated"', ModelErrorCategory.MALFORMED_OUTPUT),
        ('prefix [{"answer":"valid"}] suffix', ModelErrorCategory.MALFORMED_OUTPUT),
        ('{"wrong":"shape"}', ModelErrorCategory.SCHEMA_VALIDATION),
    ],
)
async def test_invalid_output_is_rejected_without_adapter_retry(
    content: str | None,
    category: ModelErrorCategory,
) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion(content))

    adapter, http_client = provider(handler)
    async with adapter:
        with pytest.raises(ModelProviderError) as caught:
            await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()
    assert caught.value.category == category
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"answer":"valid"}', "valid"),
        ('```json\n{"answer":"valid"}\n```', "valid"),
        ('```\n{"answer":"valid"}\n```', "valid"),
        (
            'The answer is: {"answer":"nested {brace} and \\"quoted\\""}.',
            'nested {brace} and "quoted"',
        ),
    ],
)
async def test_tolerant_content_extraction_accepts_one_object(
    content: str, expected: str
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(content))

    adapter, http_client = provider(handler)
    async with adapter:
        result = await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()

    assert result.output == AgentOutput(answer=expected)


@pytest.mark.asyncio
async def test_tolerant_content_extraction_rejects_multiple_fences_and_arrays() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("```json\n[1, 2]\n```"))

    adapter, http_client = provider(handler)
    async with adapter:
        with pytest.raises(ModelProviderError) as caught:
            await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()
    assert caught.value.category == ModelErrorCategory.MALFORMED_OUTPUT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"answer":"outside"}\n```json\n{"answer":"inside"}\n```',
        '```json\n{"answer":"first"}\n```\n```\n{"answer":"second"}\n```',
    ],
)
async def test_tolerant_content_extraction_rejects_ambiguous_fences(content: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(content))

    adapter, http_client = provider(handler)
    async with adapter:
        with pytest.raises(ModelProviderError) as caught:
            await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()
    assert caught.value.category == ModelErrorCategory.MALFORMED_OUTPUT


@pytest.mark.asyncio
async def test_refusal_is_a_safe_provider_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(None, refusal="private refusal"))

    adapter, http_client = provider(handler)
    async with adapter:
        with pytest.raises(ModelProviderError) as caught:
            await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()
    assert caught.value.category == ModelErrorCategory.PROVIDER_ERROR
    assert "private refusal" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (401, False), (429, True), (500, True)],
)
async def test_http_errors_are_normalized(status: int, retryable: bool) -> None:
    async def handler(request_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request_, json={"error": {"message": "private"}})

    adapter, http_client = provider(handler)
    async with adapter:
        with pytest.raises(ModelProviderError) as caught:
            await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()
    assert caught.value.category == ModelErrorCategory.PROVIDER_ERROR
    assert caught.value.retryable is retryable
    assert "private" not in str(caught.value)


@pytest.mark.asyncio
async def test_timeout_is_normalized_and_cancellation_propagates() -> None:
    async def timeout_handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private", request=http_request)

    adapter, http_client = provider(timeout_handler)
    async with adapter:
        with pytest.raises(ModelProviderError) as caught:
            await adapter.generate_structured(request(), AgentOutput)
    await http_client.aclose()
    assert caught.value.category == ModelErrorCategory.TIMEOUT
    assert caught.value.retryable is True

    started = asyncio.Event()

    async def slow_handler(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(60)
        return httpx.Response(200, json=completion('{"answer":"late"}'))

    slow_adapter, slow_client = provider(slow_handler)
    async with slow_adapter:
        task = asyncio.create_task(
            slow_adapter.generate_structured(request(timeout_seconds=30), AgentOutput)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await slow_client.aclose()


@pytest.mark.asyncio
async def test_concurrency_and_owned_client_lifecycle() -> None:
    active = 0
    peak = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json=completion('{"answer":"ok"}'))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = openai.AsyncOpenAI(
        api_key="synthetic-key",
        base_url="https://api.openai.com/v1",
        max_retries=0,
        http_client=http_client,
    )
    adapter = OpenAICompatibleModelProvider(
        "synthetic-key",
        model="test-model",
        connection_mode="openai",
        max_concurrency=2,
        client=client,
        client_owned=True,
    )
    async with adapter:
        results = await asyncio.gather(
            *(adapter.generate_structured(request(), AgentOutput) for _ in range(6))
        )
        assert adapter._client is client
    assert len(results) == 6
    assert peak == 2
    assert adapter._client is None
    assert http_client.is_closed


def test_custom_openai_compatible_base_url_is_allowed() -> None:
    adapter = OpenAICompatibleModelProvider(
        "synthetic-key",
        model="proxy-model",
        connection_mode="openai",
        base_url="https://models.example.test/gateway/openai/v1/",
    )
    assert adapter.base_url == "https://models.example.test/gateway/openai/v1"


@pytest.mark.parametrize("temperature", [-0.1, 2.1, float("nan"), float("inf"), True])
def test_invalid_temperatures_are_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        OpenAICompatibleModelProvider(
            "synthetic-key",
            model="synthetic-model",
            connection_mode="openai",
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://models.example.test/v1",
        "https://user:password@models.example.test/v1",
        "https://models.example.test/v1?token=private",
        "https://models.example.test/v1#fragment",
        "https://models.example.test/api",
        "https://models.example.test/v1\n",
    ],
)
def test_unsafe_or_non_v1_base_urls_are_rejected(base_url: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleModelProvider(
            "synthetic-key",
            model="synthetic-model",
            connection_mode="openai",
            base_url=base_url,
        )
