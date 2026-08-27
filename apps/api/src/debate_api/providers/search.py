"""Tavily basic-search adapter and offline normalization contract."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from enum import StrEnum
from numbers import Real
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import Field, field_validator

from debate_api.domain.models import StrictModel


class TavilyErrorCategory(StrEnum):
    AUTH = "auth"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PERMANENT = "permanent"


class SearchQuery(StrictModel):
    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def normalize(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchProviderMetadata(StrictModel):
    request_id: str | None = Field(default=None, max_length=200)
    response_time_seconds: float | None = Field(default=None, ge=0, le=3_600)
    credits: int | None = Field(default=None, ge=0, le=1_000_000)


class SearchHit(StrictModel):
    url: str = Field(min_length=1, max_length=2_048)
    title: str = Field(default="", max_length=500)
    snippet: str = Field(default="", max_length=2_000)
    rank: int = Field(ge=1, le=100)
    score: float | None = Field(default=None, ge=0, le=1)


class SearchResponse(StrictModel):
    hits: list[SearchHit] = Field(default_factory=list, max_length=20)
    provider: SearchProviderMetadata = Field(default_factory=SearchProviderMetadata)
    warnings: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("warnings")
    @classmethod
    def bounded_warnings(cls, values: list[str]) -> list[str]:
        return [value[:300] for value in values if value][:32]


class SearchProvider(Protocol):
    """Provider-neutral discovery boundary.

    Implementations return only bounded, normalized candidates.  Search result
    text is still untrusted source data; callers must fetch and validate it
    before treating it as evidence.
    """

    async def search(self, query: SearchQuery | str) -> SearchResponse: ...


class TavilySearchError(RuntimeError):
    """Safe normalized error; response bodies and authorization are never retained."""

    def __init__(
        self,
        category: TavilyErrorCategory,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.request_id = request_id
        self.retry_after_seconds = retry_after_seconds


def _canonical_result_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2_048:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return None
        if parts.username is not None or parts.password is not None:
            return None
        if parts.port not in {None, 80, 443}:
            return None
        host = parts.hostname.rstrip(".")
        if not host:
            return None
        host = host.lower() if ":" in host else host.encode("idna").decode("ascii").lower()
        netloc = f"[{host}]" if ":" in host else host
        if parts.port is not None and not (
            (parts.scheme.lower() == "http" and parts.port == 80)
            or (parts.scheme.lower() == "https" and parts.port == 443)
        ):
            return None
        canonical = urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))
        return canonical if len(canonical) <= 2_048 else None
    except (ValueError, UnicodeError):
        return None


def _bounded_text(value: object, maximum: int) -> str:
    return value[:maximum] if isinstance(value, str) else ""


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            converted = float(value)
            if math.isfinite(converted) and 0 <= converted <= 1:
                return converted
        except (OverflowError, TypeError, ValueError):
            return None
    return None


def _provider_metadata(payload: Mapping[str, object]) -> tuple[SearchProviderMetadata, list[str]]:
    warnings: list[str] = []
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or len(request_id) > 200:
        request_id = None
        if payload.get("request_id") is not None:
            warnings.append("provider request_id was malformed")
    response_time = payload.get("response_time")
    normalized_response_time: float | None = None
    valid_response_time = False
    if isinstance(response_time, (int, float)) and not isinstance(response_time, bool):
        try:
            normalized_response_time = float(response_time)
            valid_response_time = (
                math.isfinite(normalized_response_time) and 0 <= normalized_response_time <= 3_600
            )
        except (OverflowError, TypeError, ValueError):
            valid_response_time = False
    if not valid_response_time:
        normalized_response_time = None
        if payload.get("response_time") is not None:
            warnings.append("provider response_time was malformed")
    credits_obj = payload.get("usage")
    credits = credits_obj.get("credits") if isinstance(credits_obj, Mapping) else None
    if not isinstance(credits, int) or isinstance(credits, bool) or not 0 <= credits <= 1_000_000:
        credits = None
        if credits_obj is not None:
            warnings.append("provider usage.credits was malformed")
    return SearchProviderMetadata(
        request_id=request_id,
        response_time_seconds=normalized_response_time,
        credits=credits,
    ), warnings


def normalize_tavily_response(payload: object, *, max_results: int = 10) -> SearchResponse:
    """Normalize only public search fields; raw_content/images are intentionally ignored."""

    if not isinstance(payload, Mapping):
        raise TavilySearchError(TavilyErrorCategory.MALFORMED, "search response was not an object")
    metadata, warnings = _provider_metadata(payload)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise TavilySearchError(TavilyErrorCategory.MALFORMED, "search results were not a list")
    hits: list[SearchHit] = []
    for raw in raw_results[:max_results]:
        if not isinstance(raw, Mapping):
            warnings.append("one search result was malformed")
            continue
        url = _canonical_result_url(raw.get("url"))
        if url is None:
            warnings.append("one search result had an invalid URL")
            continue
        hits.append(
            SearchHit(
                url=url,
                title=_bounded_text(raw.get("title"), 500),
                snippet=_bounded_text(raw.get("content"), 2_000),
                rank=len(hits) + 1,
                score=_safe_float(raw.get("score")),
            )
        )
    if len(raw_results) > max_results:
        warnings.append("provider returned more results than the configured bound")
    return SearchResponse(hits=hits, provider=metadata, warnings=warnings[:32])


def _finite_real(value: Real) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


class TavilySearchProvider:
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        *,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        max_retry_after_seconds: float = 30.0,
        backoff_seconds: float = 0.5,
        max_response_bytes: int = 512_000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Tavily API key must not be blank")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise ValueError("max_results must be an integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be an integer")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int):
            raise ValueError("max_response_bytes must be an integer")
        for name, value in (
            ("timeout_seconds", timeout_seconds),
            ("max_retry_after_seconds", max_retry_after_seconds),
            ("backoff_seconds", backoff_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not _finite_real(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")
        if not 0 < timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 0 and 120")
        if not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3")
        if not 0 <= max_retry_after_seconds <= 60:
            raise ValueError("max_retry_after_seconds must be between 0 and 60")
        if not 0 <= backoff_seconds <= 30:
            raise ValueError("backoff_seconds must be between 0 and 30")
        if not 4_096 <= max_response_bytes <= 5_000_000:
            raise ValueError("max_response_bytes must be between 4096 and 5000000")
        self._api_key = api_key
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_retry_after_seconds = max_retry_after_seconds
        self.backoff_seconds = backoff_seconds
        self.max_response_bytes = max_response_bytes
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> TavilySearchProvider:
        self._client = httpx.AsyncClient(
            transport=self._transport,
            timeout=httpx.Timeout(self.timeout_seconds),
            trust_env=False,
        )
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search(self, query: SearchQuery | str) -> SearchResponse:
        if self._client is None:
            raise RuntimeError("TavilySearchProvider must be used as an async context manager")
        normalized = query if isinstance(query, SearchQuery) else SearchQuery(query=query)
        payload = {
            "query": normalized.query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "max_results": self.max_results,
            "include_usage": True,
        }
        for attempt in range(self.max_retries + 1):
            try:
                async with self._client.stream(
                    "POST",
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    status_code = response.status_code
                    if 200 <= response.status_code < 300:
                        body_bytes = await self._bounded_success_body(response)
                    else:
                        body_bytes = None
            except asyncio.CancelledError:
                raise
            except (httpx.TimeoutException, TimeoutError) as error:
                raise TavilySearchError(
                    TavilyErrorCategory.TIMEOUT, "Tavily search timed out"
                ) from error
            except httpx.HTTPError as error:
                raise TavilySearchError(
                    TavilyErrorCategory.TRANSIENT, "Tavily transport failed"
                ) from error
            if 200 <= status_code < 300:
                try:
                    body = json.loads(body_bytes or b"")
                except (TypeError, ValueError) as error:
                    raise TavilySearchError(
                        TavilyErrorCategory.MALFORMED, "Tavily response was not valid JSON"
                    ) from error
                return normalize_tavily_response(body, max_results=self.max_results)
            category = self._error_category(status_code)
            retryable = status_code == 429 or 500 <= status_code <= 599
            if retryable and attempt < self.max_retries:
                await asyncio.sleep(self._retry_delay(response, attempt))
                continue
            raise TavilySearchError(
                category,
                "Tavily search request failed",
                status_code=status_code,
                retry_after_seconds=self._retry_after(response),
            )
        raise AssertionError("bounded retry loop did not return")

    async def _bounded_success_body(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_response_bytes:
                    raise TavilySearchError(
                        TavilyErrorCategory.MALFORMED,
                        "Tavily response exceeded the configured byte limit",
                    )
            except ValueError:
                pass
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(chunk) > self.max_response_bytes - len(body):
                raise TavilySearchError(
                    TavilyErrorCategory.MALFORMED,
                    "Tavily response exceeded the configured byte limit",
                )
            body.extend(chunk)
        return bytes(body)

    @staticmethod
    def _error_category(status_code: int) -> TavilyErrorCategory:
        if status_code in {401, 403}:
            return TavilyErrorCategory.AUTH
        if status_code in {402, 432, 433}:
            return TavilyErrorCategory.QUOTA
        if status_code == 429:
            return TavilyErrorCategory.RATE_LIMIT
        if status_code == 408:
            return TavilyErrorCategory.TIMEOUT
        if 500 <= status_code <= 599:
            return TavilyErrorCategory.TRANSIENT
        return TavilyErrorCategory.PERMANENT

    def _retry_after(self, response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0:
                return None
            cap = float(self.max_retry_after_seconds)
            return min(max(parsed, 0.0), cap)
        except (TypeError, ValueError, OverflowError):
            return None

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = self._retry_after(response)
        if retry_after is not None:
            return retry_after
        cap = float(self.max_retry_after_seconds)
        base = float(self.backoff_seconds)
        delay = base * float(2**attempt)
        return cap if cap < delay else delay


__all__ = [
    "SearchHit",
    "SearchProvider",
    "SearchProviderMetadata",
    "SearchQuery",
    "SearchResponse",
    "TavilyErrorCategory",
    "TavilySearchError",
    "TavilySearchProvider",
    "normalize_tavily_response",
]
