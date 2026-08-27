"""Small provider-injected boundary for future structured-generation orchestration."""

from __future__ import annotations

import asyncio
from typing import TypeVar

from pydantic import BaseModel

from debate_api.providers.model import (
    ModelErrorCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelTextResponse,
    TextModelProvider,
)

T = TypeVar("T", bound=BaseModel)


class StructuredGenerationRunner:
    """Apply orchestration timeout and canonical schema metadata around a provider call."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    @property
    def request_timeout_seconds(self) -> float:
        """Expose the provider's configured per-request deadline to orchestration."""

        return self._provider.request_timeout_seconds

    async def run(self, request: ModelRequest, output_schema: type[T]) -> ModelResponse[T]:
        canonical_request = request.model_copy(
            update={"output_schema_name": output_schema.__name__}
        )
        provider_task = asyncio.create_task(
            self._provider.generate_structured(canonical_request, output_schema)
        )
        try:
            done, _ = await asyncio.wait(
                {provider_task}, timeout=canonical_request.timeout_seconds
            )
            if not done:
                provider_task.cancel()
                # Retrieve any exception raised while the provider task unwinds after
                # cancellation. This also keeps timeout classification authoritative.
                await asyncio.gather(provider_task, return_exceptions=True)
                raise TimeoutError
            return provider_task.result()
        except asyncio.CancelledError:
            # A caller cancelling its task must retain asyncio cancellation semantics.
            if not provider_task.done():
                provider_task.cancel()
            await asyncio.gather(provider_task, return_exceptions=True)
            raise
        except TimeoutError:
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "The model provider exceeded the request deadline.",
                request_id=canonical_request.request_id,
                retryable=True,
            ) from None
        except ModelProviderError:
            raise
        except Exception:
            # Adapter-specific exceptions never cross the orchestration boundary.
            raise ModelProviderError(
                ModelErrorCategory.PROVIDER_ERROR,
                "The model provider failed to produce a result.",
                request_id=canonical_request.request_id,
            ) from None

    async def run_text(self, request: ModelRequest) -> ModelTextResponse:
        """Run one plain-text request without attaching structured-output metadata."""

        if not isinstance(self._provider, TextModelProvider):
            raise ModelProviderError(
                ModelErrorCategory.PROVIDER_ERROR,
                "The configured model provider does not support plain-text generation.",
                request_id=request.request_id,
            )
        canonical_request = request.model_copy(update={"output_schema_name": None})
        provider_task = asyncio.create_task(
            self._provider.generate_text(canonical_request)
        )
        try:
            done, _ = await asyncio.wait(
                {provider_task}, timeout=canonical_request.timeout_seconds
            )
            if not done:
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                raise TimeoutError
            return provider_task.result()
        except asyncio.CancelledError:
            if not provider_task.done():
                provider_task.cancel()
            await asyncio.gather(provider_task, return_exceptions=True)
            raise
        except TimeoutError:
            raise ModelProviderError(
                ModelErrorCategory.TIMEOUT,
                "The model provider exceeded the request deadline.",
                request_id=canonical_request.request_id,
                retryable=True,
            ) from None
        except ModelProviderError:
            raise
        except Exception:
            raise ModelProviderError(
                ModelErrorCategory.PROVIDER_ERROR,
                "The model provider failed to produce a result.",
                request_id=canonical_request.request_id,
            ) from None
