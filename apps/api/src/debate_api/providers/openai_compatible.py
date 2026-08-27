"""One OpenAI SDK Chat Completions adapter for hosted and local models."""

from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from time import perf_counter
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
import openai
from pydantic import BaseModel, SecretStr, ValidationError

from debate_api.providers.model import (
    ModelErrorCategory,
    ModelIdentity,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelTextResponse,
    ModelUsage,
    validate_structured_output,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"
DEFAULT_QWEN_MODEL_REVISION = "bc640142c66e1fdd12af0bd68f40445458f3869b"
MAX_TOKEN_COUNT = 1_000_000_000

ConnectionMode = Literal["openai", "ollama"]
ReasoningEffort = Literal["none", "low", "medium", "high"]


def _normalized_base_url(value: str | None, mode: ConnectionMode) -> str:
    candidate = value or (
        DEFAULT_OPENAI_BASE_URL if mode == "openai" else DEFAULT_OLLAMA_BASE_URL
    )
    if any(ord(character) < 32 or ord(character) == 127 for character in candidate):
        raise ValueError("Model base URL contains control characters")
    parsed = urlsplit(candidate)
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Model base URL has an invalid port") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Model base URL must be an HTTP(S) endpoint without credentials")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        raise ValueError("Model base URL must end in /v1")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _normalize_strict_json_schema(output_schema: type[BaseModel]) -> dict[str, Any]:
    """Return an OpenAI strict schema without relying on private SDK helpers."""

    schema = deepcopy(output_schema.model_json_schema())

    def normalize(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                normalize(item)
            return
        if not isinstance(value, dict):
            return
        if value.get("type") == "object":
            properties = value.get("properties")
            value["additionalProperties"] = False
            if isinstance(properties, dict):
                value["required"] = list(properties)
        if "default" in value and value["default"] is None:
            value.pop("default")
        for child in value.values():
            normalize(child)

    normalize(schema)
    return schema


def _safe_token(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not 0 <= value <= MAX_TOKEN_COUNT:
        return None
    return value


def _usage_from_response(response: object, request_id: str) -> ModelUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return ModelUsage()
    raw_values = (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )
    values = tuple(_safe_token(value) for value in raw_values)
    if any(
        raw is not None and value is None
        for raw, value in zip(raw_values, values, strict=True)
    ):
        raise ModelProviderError(
            ModelErrorCategory.NORMALIZATION_ERROR,
            "The model provider returned invalid usage metadata.",
            request_id=request_id,
        )
    prompt_tokens, completion_tokens, total_tokens = values
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    try:
        return ModelUsage.normalized(prompt_tokens, completion_tokens, total_tokens)
    except (TypeError, ValueError, ValidationError):
        raise ModelProviderError(
            ModelErrorCategory.NORMALIZATION_ERROR,
            "The model provider returned invalid usage metadata.",
            request_id=request_id,
        ) from None


def _balanced_container_end(
    content: str, start: int, opening: str, closing: str
) -> int | None:
    """Return the end of a string-aware balanced JSON container."""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        character = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def _balanced_object_end(content: str, start: int) -> int | None:
    return _balanced_container_end(content, start, "{", "}")


def _balanced_array_end(content: str, start: int) -> int | None:
    return _balanced_container_end(content, start, "[", "]")


def _contains_external_json_value(content: str) -> bool:
    """Identify complete or unclosed JSON containers in fence surroundings."""

    stripped = content.strip()
    if not stripped:
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return True

    decoder = json.JSONDecoder()
    for opening, balanced_end in (("{", _balanced_object_end), ("[", _balanced_array_end)):
        for match in re.finditer(re.escape(opening), content):
            start = match.start()
            try:
                value, _ = decoder.raw_decode(content, start)
            except json.JSONDecodeError:
                if balanced_end(content, start) is None:
                    return True
                continue
            if isinstance(value, (dict, list)):
                return True
    return False


def _extract_structured_object(content: str) -> dict[str, Any]:
    """Extract one object from model content without interpreting hidden reasoning."""

    stripped = content.strip()
    if not stripped:
        raise ValueError

    # Keep the provider's normal response path strict and deterministic.
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    else:
        if isinstance(direct, dict):
            return direct
        raise ValueError

    # A fenced response is only accepted when there is exactly one complete fence.
    # The JSON decoder, rather than a brace regex, handles nesting and escapes.
    fence_markers = list(re.finditer(r"(?m)^```[^\r\n]*(?:\r?\n|$)", content))
    if fence_markers:
        if len(fence_markers) != 2:
            raise ValueError
        opening = fence_markers[0].group()[3:].strip()
        closing = fence_markers[1].group()[3:].strip()
        if opening.lower() not in {"", "json"} or closing:
            raise ValueError
        prefix = content[: fence_markers[0].start()]
        suffix = content[fence_markers[1].end() :]
        if _contains_external_json_value(prefix) or _contains_external_json_value(suffix):
            raise ValueError
        try:
            fenced = json.loads(content[fence_markers[0].end() : fence_markers[1].start()])
        except json.JSONDecodeError:
            raise ValueError from None
        if not isinstance(fenced, dict):
            raise ValueError
        return fenced

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"{", content):
        start = match.start()
        try:
            value, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((start, end, value))

    # Nested object starts belong to their containing candidate; distinct outer
    # candidates are ambiguous and must not be selected arbitrarily.
    outer_candidates = [
        candidate
        for candidate in candidates
        if not any(
            other[0] < candidate[0] and candidate[1] <= other[1]
            for other in candidates
        )
    ]
    if len(outer_candidates) != 1:
        raise ValueError
    start, end, value = outer_candidates[0]

    # An object nested in an array is not a bounded structured response; reject
    # it instead of selecting an inner value from an otherwise invalid array.
    for match in re.finditer(r"\[", content):
        array_start = match.start()
        try:
            array_value, array_end = decoder.raw_decode(content, array_start)
        except json.JSONDecodeError:
            continue
        if isinstance(array_value, list) and array_start < start < array_end:
            raise ValueError

    # An unclosed brace outside the selected object (including one wrapping it)
    # indicates truncated JSON rather than harmless surrounding prose.
    for match in re.finditer(r"{", content):
        brace_start = match.start()
        if start <= brace_start < end:
            continue
        if _balanced_object_end(content, brace_start) is None:
            raise ValueError
    for match in re.finditer(r"\[", content):
        array_start = match.start()
        if start <= array_start < end:
            continue
        if _balanced_array_end(content, array_start) is None:
            raise ValueError
    return value


class OpenAICompatibleModelProvider:
    """Validated structured generation through one ``AsyncOpenAI`` client path."""

    def __init__(
        self,
        api_key: str | SecretStr,
        *,
        model: str,
        connection_mode: ConnectionMode,
        base_url: str | None = None,
        revision: str | None = None,
        max_concurrency: int = 1,
        request_timeout_seconds: float = 30.0,
        max_output_tokens: int = 32_768,
        reasoning_effort: ReasoningEffort | None = None,
        temperature: float | None = None,
        structured_output_strict: bool = True,
        client: openai.AsyncOpenAI | None = None,
        client_owned: bool = False,
    ) -> None:
        secret = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not secret.strip():
            raise ValueError("Model API key must not be blank")
        normalized_model = model.strip()
        if (
            not normalized_model
            or len(normalized_model) > 160
            or any(ord(character) < 32 for character in normalized_model)
        ):
            raise ValueError("Model name is invalid")
        if isinstance(max_concurrency, bool) or not 1 <= max_concurrency <= 16:
            raise ValueError("max_concurrency must be between 1 and 16")
        if not 0 < request_timeout_seconds <= 900:
            raise ValueError("request_timeout_seconds must be between 0 and 900")
        if isinstance(max_output_tokens, bool) or not 1 <= max_output_tokens <= 32_768:
            raise ValueError("max_output_tokens must be between 1 and 32768")
        if reasoning_effort not in {None, "none", "low", "medium", "high"}:
            raise ValueError("reasoning_effort must be one of none, low, medium, or high")
        if isinstance(temperature, bool) or (
            temperature is not None and not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        self.model = normalized_model
        self.connection_mode = connection_mode
        self.base_url = _normalized_base_url(base_url, connection_mode)
        self.revision = revision
        self.request_timeout_seconds = request_timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.structured_output_strict = structured_output_strict
        self._api_key = SecretStr(secret)
        self._client = client
        self._owns_client = client is None or client_owned
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(connection_mode={self.connection_mode!r}, "
            f"model={self.model!r}, base_url={self.base_url!r})"
        )

    async def __aenter__(self) -> OpenAICompatibleModelProvider:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key.get_secret_value(),
                base_url=self.base_url,
                timeout=httpx.Timeout(self.request_timeout_seconds),
                max_retries=0,
            )
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        await self.close()

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    async def generate_structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
    ) -> ModelResponse[T]:
        if self._client is None:
            raise RuntimeError(
                "OpenAICompatibleModelProvider must be used as an async context manager"
            )
        if request.output_schema_name != output_schema.__name__:
            raise ModelProviderError(
                ModelErrorCategory.NORMALIZATION_ERROR,
                "The requested output schema metadata did not match the schema.",
                request_id=request.request_id,
            )
        async with self._semaphore:
            return await self._generate(request, output_schema)

    async def generate_text(self, request: ModelRequest) -> ModelTextResponse:
        if self._client is None:
            raise RuntimeError(
                "OpenAICompatibleModelProvider must be used as an async context manager"
            )
        if request.output_schema_name is not None:
            raise ModelProviderError(
                ModelErrorCategory.NORMALIZATION_ERROR,
                "Plain-text generation cannot request an output schema.",
                request_id=request.request_id,
            )
        async with self._semaphore:
            return await self._generate_text(request)

    async def _generate(
        self,
        request: ModelRequest,
        output_schema: type[T],
    ) -> ModelResponse[T]:
        started = perf_counter()
        try:
            response = await asyncio.wait_for(
                self._request(request, output_schema),
                timeout=min(request.timeout_seconds, self.request_timeout_seconds),
            )
            output = self._parse_output(response, request, output_schema)
            usage = _usage_from_response(response, request.request_id)
            return ModelResponse(
                request_id=request.request_id,
                output=output,
                model=ModelIdentity(
                    provider=self.connection_mode,
                    model=self.model,
                    revision=self.revision,
                ),
                latency_ms=(perf_counter() - started) * 1_000,
                usage=usage,
                repair_attempted=False,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "The model provider exceeded the request deadline.",
                request_id=request.request_id,
                retryable=True,
            ) from None

    async def _generate_text(self, request: ModelRequest) -> ModelTextResponse:
        started = perf_counter()
        try:
            response = await asyncio.wait_for(
                self._request(request, None),
                timeout=min(request.timeout_seconds, self.request_timeout_seconds),
            )
            output = self._parse_text(response, request)
            return ModelTextResponse(
                request_id=request.request_id,
                output=output,
                model=ModelIdentity(
                    provider=self.connection_mode,
                    model=self.model,
                    revision=self.revision,
                ),
                latency_ms=(perf_counter() - started) * 1_000,
                usage=_usage_from_response(response, request.request_id),
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "The model provider exceeded the request deadline.",
                request_id=request.request_id,
                retryable=True,
            ) from None

    async def _request(
        self, request: ModelRequest, output_schema: type[T] | None
    ) -> object:
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI-compatible model client is closed")
        token_parameter = (
            "max_completion_tokens" if self.connection_mode == "openai" else "max_tokens"
        )
        if request.conversation is not None:
            if output_schema is not None:
                raise ModelProviderError(
                    ModelErrorCategory.NORMALIZATION_ERROR,
                    "A role-preserving conversation cannot request structured output.",
                    request_id=request.request_id,
                )
            messages = [message.model_dump(mode="json") for message in request.conversation]
        else:
            messages = [
                {
                    "role": "developer",
                    "content": (
                        "Return exactly one JSON object that satisfies the supplied schema. "
                        "Do not include markdown or hidden reasoning."
                        if output_schema is not None
                        else (
                            "Return plain text, not JSON or a markdown code fence. Follow the "
                            "user's requested answer format."
                        )
                    ),
                },
                {"role": "user", "content": request.input_text},
            ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "timeout": min(request.timeout_seconds, self.request_timeout_seconds),
        }
        if output_schema is not None:
            schema = (
                _normalize_strict_json_schema(output_schema)
                if self.structured_output_strict
                else output_schema.model_json_schema()
            )
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__[:64],
                    "strict": self.structured_output_strict,
                    "schema": schema,
                },
            }
        payload[token_parameter] = min(request.max_output_tokens, self.max_output_tokens)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        try:
            return await client.chat.completions.create(**cast(Any, payload))
        except asyncio.CancelledError:
            raise
        except (openai.APITimeoutError, httpx.TimeoutException, TimeoutError):
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "The model provider exceeded the request deadline.",
                request_id=request.request_id,
                retryable=True,
            ) from None
        except openai.AuthenticationError:
            raise self._provider_error(
                request.request_id, "rejected authentication", False
            ) from None
        except openai.RateLimitError:
            raise self._provider_error(
                request.request_id, "rate limit was reached", True
            ) from None
        except openai.BadRequestError:
            raise self._provider_error(
                request.request_id, "rejected the request", False
            ) from None
        except openai.APIConnectionError:
            raise self._provider_error(request.request_id, "connection failed", True) from None
        except openai.InternalServerError:
            raise self._provider_error(
                request.request_id, "returned a server error", True
            ) from None
        except openai.APIStatusError as error:
            retryable = error.status_code >= 500
            raise self._provider_error(
                request.request_id, "returned an unsuccessful status", retryable
            ) from None
        except openai.APIError:
            raise self._provider_error(
                request.request_id, "returned an API error", False
            ) from None

    def _parse_output(
        self,
        response: object,
        request: ModelRequest,
        output_schema: type[T],
    ) -> T:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ModelProviderError(
                ModelErrorCategory.MALFORMED_OUTPUT,
                "The model returned malformed structured output.",
                request_id=request.request_id,
            )
        message = getattr(choices[0], "message", None)
        if message is None or getattr(message, "refusal", None):
            raise self._provider_error(request.request_id, "refused the structured request", False)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ModelProviderError(
                ModelErrorCategory.MALFORMED_OUTPUT,
                "The model returned malformed structured output.",
                request_id=request.request_id,
            )
        try:
            raw = _extract_structured_object(content)
        except ValueError:
            raise ModelProviderError(
                ModelErrorCategory.MALFORMED_OUTPUT,
                "The model returned malformed structured output.",
                request_id=request.request_id,
            ) from None
        return validate_structured_output(raw, output_schema, request.request_id)

    def _parse_text(self, response: object, request: ModelRequest) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ModelProviderError(
                ModelErrorCategory.MALFORMED_OUTPUT,
                "The model returned malformed plain-text output.",
                request_id=request.request_id,
            )
        message = getattr(choices[0], "message", None)
        if message is None or getattr(message, "refusal", None):
            raise self._provider_error(request.request_id, "refused the text request", False)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ModelProviderError(
                ModelErrorCategory.MALFORMED_OUTPUT,
                "The model returned malformed plain-text output.",
                request_id=request.request_id,
            )
        return content.strip()

    @staticmethod
    def _provider_error(
        request_id: str,
        detail: str,
        retryable: bool,
    ) -> ModelProviderError:
        return ModelProviderError(
            ModelErrorCategory.PROVIDER_ERROR,
            f"The model provider {detail}.",
            request_id=request_id,
            retryable=retryable,
        )


__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_QWEN_MODEL_REVISION",
    "OpenAICompatibleModelProvider",
    "ReasoningEffort",
]
