from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from debate_api.providers.search import (
    SearchResponse,
    TavilyErrorCategory,
    TavilySearchError,
    TavilySearchProvider,
    normalize_tavily_response,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tavily_basic_response.json"


def fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_basic_request_and_recorded_response_are_normalized() -> None:
    seen: list[tuple[httpx.Request, dict[str, object]]] = []

    async def request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request, body))
        return httpx.Response(200, json=fixture_payload())

    async with TavilySearchProvider(
        "fixture-api-key",
        max_results=3,
        transport=httpx.MockTransport(request),
    ) as provider:
        result = await provider.search("  bounded   query ")

    assert isinstance(result, SearchResponse)
    assert [hit.rank for hit in result.hits] == [1, 2, 3]
    assert result.hits[1].url == "https://docs.example.org/secondary"
    assert result.provider.credits == 1
    assert result.provider.request_id == "fixture-request-id"
    assert seen[0][0].url == TavilySearchProvider.endpoint
    assert seen[0][0].headers["authorization"] == "Bearer fixture-api-key"
    assert seen[0][1] == {
        "query": "bounded query",
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "max_results": 3,
        "include_usage": True,
    }
    serialized = result.model_dump_json()
    assert "raw_content" not in serialized
    assert "images" not in serialized


def test_normalizer_rejects_hostile_shapes_without_leaking_large_fields() -> None:
    payload = {
        "results": [
            {
                "url": "ftp://invalid.example/",
                "title": "x" * 10_000,
                "content": "x" * 10_000,
                "score": float("nan"),
            },
            {"url": "https://./x", "title": "empty host"},
            {"url": "https://bücher.example/ok", "title": "unicode host", "score": 10**1000},
            {
                "url": "https://valid.example/ok",
                "title": 123,
                "content": {"raw": "secret"},
                "score": 2,
            },
        ],
        "request_id": "r" * 10_000,
        "response_time": 10**1000,
        "usage": {"credits": -1},
    }
    result = normalize_tavily_response(payload)
    assert len(result.hits) == 2
    assert result.hits[0].url.startswith("https://xn--")
    assert result.hits[1].title == ""
    assert result.hits[1].snippet == ""
    assert result.hits[1].score is None
    assert result.provider.request_id is None
    assert result.provider.response_time_seconds is None
    assert len(result.warnings) >= 3
    assert "secret" not in result.model_dump_json()


@pytest.mark.parametrize("payload", [None, [], {"results": {}}, {"results": ["bad"]}])
def test_malformed_response_shapes_have_safe_taxonomy(payload: object) -> None:
    if payload == {"results": ["bad"]}:
        result = normalize_tavily_response(payload)
        assert result.hits == []
        return
    with pytest.raises(TavilySearchError) as exc_info:
        normalize_tavily_response(payload)
    assert exc_info.value.category == TavilyErrorCategory.MALFORMED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,category",
    [
        (401, TavilyErrorCategory.AUTH),
        (402, TavilyErrorCategory.QUOTA),
        (432, TavilyErrorCategory.QUOTA),
        (433, TavilyErrorCategory.QUOTA),
        (400, TavilyErrorCategory.PERMANENT),
    ],
)
async def test_http_error_taxonomy(status: int, category: TavilyErrorCategory) -> None:
    async def request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "secret body must not escape"})

    api_key = "tvly-secret-key"
    async with TavilySearchProvider(api_key, transport=httpx.MockTransport(request)) as provider:
        with pytest.raises(TavilySearchError) as exc_info:
            await provider.search("query")
    assert exc_info.value.category == category
    assert api_key not in str(exc_info.value)
    assert "secret body" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_rate_limit_retry_honors_bounded_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    async def request(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "999"})
        return httpx.Response(200, json=fixture_payload())

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    async with TavilySearchProvider(
        "key", max_retry_after_seconds=0.25, transport=httpx.MockTransport(request)
    ) as provider:
        result = await provider.search("query")
    assert result.hits
    assert calls == 2
    assert sleeps == [0.25]


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", ["nan", "inf", "-1"])
async def test_malformed_retry_after_uses_bounded_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch, retry_after: str
) -> None:
    sleeps: list[float] = []
    calls = 0

    async def request(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": retry_after})
        return httpx.Response(200, json=fixture_payload())

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    async with TavilySearchProvider(
        "key", backoff_seconds=0.125, transport=httpx.MockTransport(request)
    ) as provider:
        await provider.search("query")
    assert sleeps == [0.125]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": 121},
        {"max_retries": 4},
        {"max_retry_after_seconds": 61},
        {"backoff_seconds": 31},
        {"max_response_bytes": 4_095},
        {"max_response_bytes": 5_000_001},
    ],
)
def test_provider_numeric_limits_are_bounded(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TavilySearchProvider("key", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_results": 1.5},
        {"max_retries": 1.5},
        {"max_response_bytes": 4096.0},
        {"max_results": True},
        {"max_retries": False},
        {"max_response_bytes": True},
        {"timeout_seconds": "1"},
        {"max_retry_after_seconds": float("nan")},
        {"backoff_seconds": float("inf")},
    ],
)
def test_provider_rejects_wrong_numeric_types_at_construction(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TavilySearchProvider("key", **kwargs)


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_success_response_is_bounded_before_json_parse() -> None:
    async def declared_large(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": "999999"},
            stream=ChunkStream([b"{}"]),
        )

    async with TavilySearchProvider(
        "key", max_response_bytes=4_096, transport=httpx.MockTransport(declared_large)
    ) as provider:
        with pytest.raises(TavilySearchError) as exc_info:
            await provider.search("query")
    assert exc_info.value.category == TavilyErrorCategory.MALFORMED

    async def streamed_large(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkStream([b"x" * 3_000, b"y" * 2_000]))

    async with TavilySearchProvider(
        "key", max_response_bytes=4_096, transport=httpx.MockTransport(streamed_large)
    ) as provider:
        with pytest.raises(TavilySearchError) as exc_info:
            await provider.search("query")
    assert exc_info.value.category == TavilyErrorCategory.MALFORMED


@pytest.mark.asyncio
async def test_transient_5xx_has_bounded_retries_and_no_retry_for_4xx() -> None:
    calls = 0

    async def request(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with TavilySearchProvider(
        "key", max_retries=2, backoff_seconds=0, transport=httpx.MockTransport(request)
    ) as provider:
        with pytest.raises(TavilySearchError) as exc_info:
            await provider.search("query")
    assert exc_info.value.category == TavilyErrorCategory.TRANSIENT
    assert calls == 3


@pytest.mark.asyncio
async def test_timeout_and_cancellation_are_distinct_control_flow() -> None:
    async def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated")

    async with TavilySearchProvider("key", transport=httpx.MockTransport(timeout)) as provider:
        with pytest.raises(TavilySearchError) as exc_info:
            await provider.search("query")
    assert exc_info.value.category == TavilyErrorCategory.TIMEOUT

    started = asyncio.Event()

    async def slow(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(60)
        return httpx.Response(200, json=fixture_payload())

    async with TavilySearchProvider("key", transport=httpx.MockTransport(slow)) as provider:
        task = asyncio.create_task(provider.search("query"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
