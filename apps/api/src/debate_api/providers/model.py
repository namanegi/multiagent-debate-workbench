"""Provider-neutral structured model contract and deterministic fake adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from enum import StrEnum
from time import perf_counter
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ProviderModel(BaseModel):
    """Internal provider models reject fields that could drift silently."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ModelErrorCategory(StrEnum):
    """Stable safe categories surfaced to orchestration and structured logs."""

    MALFORMED_OUTPUT = "malformed_output"
    SCHEMA_VALIDATION = "schema_validation"
    REPAIR_FAILED = "repair_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PROVIDER_ERROR = "provider_error"
    NORMALIZATION_ERROR = "normalization_error"
    UNEXPECTED_CALL = "unexpected_call"


class ModelIdentity(ProviderModel):
    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=160)
    revision: str | None = Field(default=None, max_length=160)


class ModelUsage(ProviderModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @classmethod
    def normalized(
        cls,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None = None,
    ) -> ModelUsage:
        values = (prompt_tokens, completion_tokens, total_tokens)
        if any(value is not None and value < 0 for value in values):
            raise ValueError("usage token counts cannot be negative")
        if (
            prompt_tokens is not None
            and completion_tokens is not None
            and total_tokens is not None
            and total_tokens != prompt_tokens + completion_tokens
        ):
            raise ValueError("usage total does not match component token counts")
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class ModelChatMessage(ProviderModel):
    """One role-preserving message for a provider chat conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ModelRequest(ProviderModel):
    """Provider-neutral request; prompt text stays inside the adapter boundary."""

    request_id: str = Field(min_length=1, max_length=160)
    operation: str = Field(min_length=1, max_length=80)
    input_text: str = Field(min_length=1, max_length=100_000)
    conversation: tuple[ModelChatMessage, ...] | None = Field(
        default=None, min_length=1, max_length=16
    )
    output_schema_name: str | None = Field(default=None, min_length=1, max_length=160)
    timeout_seconds: float = Field(default=30.0, gt=0, le=900)
    max_output_tokens: int = Field(default=2_048, ge=1, le=32_768)
    repair_attempts: int = Field(default=1, ge=0, le=1)


T = TypeVar("T", bound=BaseModel)


class ModelResponse(ProviderModel, Generic[T]):
    """Only validated structured output crosses the provider boundary."""

    request_id: str
    output: T
    model: ModelIdentity
    latency_ms: float = Field(ge=0)
    usage: ModelUsage
    repair_attempted: bool = False


class ModelTextResponse(ProviderModel):
    """One bounded plain-text provider response with normalized usage."""

    request_id: str
    output: str = Field(min_length=1, max_length=100_000)
    model: ModelIdentity
    latency_ms: float = Field(ge=0)
    usage: ModelUsage


class ModelProviderError(RuntimeError):
    """Safe normalized provider failure; never stores prompt or raw provider data."""

    def __init__(
        self,
        category: ModelErrorCategory,
        message: str,
        *,
        request_id: str,
        retryable: bool = False,
        repair_attempted: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.request_id = request_id
        self.retryable = retryable
        self.repair_attempted = repair_attempted


def validate_structured_output(raw: object, output_schema: type[T], request_id: str) -> T:
    """Validate at the adapter boundary and classify malformed vs schema-invalid data."""

    if isinstance(raw, output_schema):
        return raw
    if not isinstance(raw, Mapping):
        raise ModelProviderError(
            ModelErrorCategory.MALFORMED_OUTPUT,
            "The model returned malformed structured output.",
            request_id=request_id,
        )
    try:
        validated = output_schema.model_validate(raw)
    except ValidationError:
        # The validation detail can contain model text; keep it out of public errors/logs.
        validated = None
    if validated is None:
        raise ModelProviderError(
            ModelErrorCategory.SCHEMA_VALIDATION,
            "The model output did not satisfy the requested schema.",
            request_id=request_id,
        ) from None
    return validated


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def request_timeout_seconds(self) -> float:
        """Configured upper bound for one provider request."""

    async def generate_structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
    ) -> ModelResponse[T]:
        """Generate one validated structured response."""


@runtime_checkable
class TextModelProvider(Protocol):
    @property
    def request_timeout_seconds(self) -> float:
        """Configured upper bound for one provider request."""

    async def generate_text(self, request: ModelRequest) -> ModelTextResponse:
        """Generate one non-empty plain-text response without an output schema."""


class FakeModelOutcome(StrEnum):
    SUCCESS = "success"
    MALFORMED_OUTPUT = "malformed_output"
    SCHEMA_FAILURE = "schema_failure"
    REPAIR_FAILURE = "repair_failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PROVIDER_ERROR = "provider_error"


class FakeModelStep(ProviderModel):
    """One programmable fake call, selected in declaration order."""

    operation: str = Field(min_length=1, max_length=80)
    outcome: FakeModelOutcome = FakeModelOutcome.SUCCESS
    output: Any = None
    repair_output: Any = None
    delay_seconds: float = Field(default=0, ge=0, le=900)
    retryable: bool = True
    # Provider-reported usage is normalized by ModelUsage.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ProgrammableFakeModelProvider:
    """Deterministic provider fake with safe normalized outcomes and call inspection."""

    def __init__(
        self,
        steps: Sequence[FakeModelStep],
        *,
        identity: ModelIdentity | None = None,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        if not 0 < request_timeout_seconds <= 900:
            raise ValueError("request_timeout_seconds must be between 0 and 900")
        self._steps = list(steps)
        self._identity = identity or ModelIdentity(provider="fake", model="deterministic")
        self.request_timeout_seconds = request_timeout_seconds
        self.calls: list[ModelRequest] = []

    async def generate_structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
    ) -> ModelResponse[T]:
        call_index = len(self.calls)
        self.calls.append(request)
        if call_index >= len(self._steps):
            raise ModelProviderError(
                ModelErrorCategory.UNEXPECTED_CALL,
                "The deterministic provider received an unexpected call.",
                request_id=request.request_id,
            )
        step = self._steps[call_index]
        if step.operation != request.operation:
            raise ModelProviderError(
                ModelErrorCategory.UNEXPECTED_CALL,
                "The deterministic provider received an unexpected operation.",
                request_id=request.request_id,
            )
        if request.output_schema_name != output_schema.__name__:
            raise ModelProviderError(
                ModelErrorCategory.NORMALIZATION_ERROR,
                "The requested output schema metadata did not match the schema.",
                request_id=request.request_id,
            )
        started = perf_counter()
        try:
            if step.outcome == FakeModelOutcome.TIMEOUT:
                await asyncio.wait_for(
                    asyncio.sleep(request.timeout_seconds + 1), request.timeout_seconds
                )
            elif step.delay_seconds:
                await asyncio.wait_for(asyncio.sleep(step.delay_seconds), request.timeout_seconds)
        except TimeoutError:
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "The model provider exceeded the request deadline.",
                request_id=request.request_id,
                retryable=True,
            ) from None
        except asyncio.CancelledError:
            # Cancellation of the caller's task is control flow, not a provider result.
            raise

        if step.outcome == FakeModelOutcome.CANCELLED:
            raise ModelProviderError(
                ModelErrorCategory.CANCELLED,
                "The model request was cancelled.",
                request_id=request.request_id,
            )
        if step.outcome == FakeModelOutcome.PROVIDER_ERROR:
            raise ModelProviderError(
                ModelErrorCategory.PROVIDER_ERROR,
                "The model provider returned an unavailable result.",
                request_id=request.request_id,
                retryable=step.retryable,
            )
        if step.outcome == FakeModelOutcome.REPAIR_FAILURE:
            raise ModelProviderError(
                ModelErrorCategory.REPAIR_FAILED,
                "The bounded model-output repair failed.",
                request_id=request.request_id,
                repair_attempted=request.repair_attempts > 0,
            )

        repair_attempted = False
        raw = step.output
        if step.outcome == FakeModelOutcome.MALFORMED_OUTPUT:
            try:
                return self._response(
                    request,
                    output_schema,
                    validate_structured_output(raw, output_schema, request.request_id),
                    started,
                    step,
                    repair_attempted,
                )
            except ModelProviderError as error:
                if request.repair_attempts <= 0 or step.repair_output is None:
                    raise error
                repair_attempted = True
                raw = step.repair_output
        elif step.outcome == FakeModelOutcome.SCHEMA_FAILURE:
            if request.repair_attempts <= 0 or step.repair_output is None:
                try:
                    validate_structured_output(raw, output_schema, request.request_id)
                except ModelProviderError as error:
                    raise error
                raise ModelProviderError(
                    ModelErrorCategory.SCHEMA_VALIDATION,
                    "The model output did not satisfy the requested schema.",
                    request_id=request.request_id,
                )
            repair_attempted = True
            raw = step.repair_output

        try:
            output = validate_structured_output(raw, output_schema, request.request_id)
        except ModelProviderError:
            if repair_attempted:
                raise ModelProviderError(
                    ModelErrorCategory.REPAIR_FAILED,
                    "The bounded model-output repair failed.",
                    request_id=request.request_id,
                    repair_attempted=True,
                ) from None
            raise
        return self._response(request, output_schema, output, started, step, repair_attempted)

    async def generate_text(self, request: ModelRequest) -> ModelTextResponse:
        call_index = len(self.calls)
        self.calls.append(request)
        if call_index >= len(self._steps):
            raise ModelProviderError(
                ModelErrorCategory.UNEXPECTED_CALL,
                "The deterministic provider received an unexpected call.",
                request_id=request.request_id,
            )
        step = self._steps[call_index]
        if step.operation != request.operation:
            raise ModelProviderError(
                ModelErrorCategory.UNEXPECTED_CALL,
                "The deterministic provider received an unexpected operation.",
                request_id=request.request_id,
            )
        if request.output_schema_name is not None:
            raise ModelProviderError(
                ModelErrorCategory.NORMALIZATION_ERROR,
                "Plain-text generation cannot request an output schema.",
                request_id=request.request_id,
            )
        started = perf_counter()
        try:
            if step.outcome == FakeModelOutcome.TIMEOUT:
                await asyncio.wait_for(
                    asyncio.sleep(request.timeout_seconds + 1), request.timeout_seconds
                )
            elif step.delay_seconds:
                await asyncio.wait_for(asyncio.sleep(step.delay_seconds), request.timeout_seconds)
        except TimeoutError:
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "The model provider exceeded the request deadline.",
                request_id=request.request_id,
                retryable=True,
            ) from None
        if step.outcome == FakeModelOutcome.CANCELLED:
            raise ModelProviderError(
                ModelErrorCategory.CANCELLED,
                "The model request was cancelled.",
                request_id=request.request_id,
            )
        if step.outcome != FakeModelOutcome.SUCCESS:
            raise ModelProviderError(
                ModelErrorCategory.PROVIDER_ERROR,
                "The model provider did not return usable plain text.",
                request_id=request.request_id,
                retryable=step.retryable,
            )
        if not isinstance(step.output, str) or not step.output.strip():
            raise ModelProviderError(
                ModelErrorCategory.MALFORMED_OUTPUT,
                "The model returned malformed plain-text output.",
                request_id=request.request_id,
            )
        return ModelTextResponse(
            request_id=request.request_id,
            output=step.output.strip(),
            model=self._identity,
            latency_ms=(perf_counter() - started) * 1_000,
            usage=self._normalized_usage(request.request_id, step),
        )

    def _response(
        self,
        request: ModelRequest,
        output_schema: type[T],
        output: T,
        started: float,
        step: FakeModelStep,
        repair_attempted: bool = False,
    ) -> ModelResponse[T]:
        del output_schema
        return ModelResponse(
            request_id=request.request_id,
            output=output,
            model=self._identity,
            latency_ms=(perf_counter() - started) * 1_000,
            usage=self._normalized_usage(request.request_id, step),
            repair_attempted=repair_attempted,
        )

    @staticmethod
    def _normalized_usage(request_id: str, step: FakeModelStep) -> ModelUsage:
        try:
            return ModelUsage.normalized(
                step.prompt_tokens,
                step.completion_tokens,
                step.total_tokens,
            )
        except (TypeError, ValueError, ValidationError):
            raise ModelProviderError(
                ModelErrorCategory.NORMALIZATION_ERROR,
                "The model provider returned invalid usage metadata.",
                request_id=request_id,
            ) from None
