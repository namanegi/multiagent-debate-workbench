"""Application-owned model provider selection and bounded runtime metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from debate_api.providers.model import ModelProvider
from debate_api.providers.openai_compatible import OpenAICompatibleModelProvider
from debate_api.settings import Settings


class ManagedModelProvider(ModelProvider, Protocol):
    """Provider contract for clients owned by the application lifespan."""

    async def __aenter__(self) -> ManagedModelProvider: ...

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class ModelProviderRuntime:
    """Provider instance plus the safe metadata used by the API lifecycle."""

    provider: ManagedModelProvider | None
    mode: str
    identity: str


def build_model_provider(settings: Settings) -> ModelProviderRuntime:
    """Construct the configured provider without opening its async client."""

    mode = settings.model_provider
    if mode == "fake":
        # None is deliberate: the existing orchestrator's deterministic
        # catalog/fallback remains the fake provider's zero-secret behavior.
        return ModelProviderRuntime(
            provider=None,
            mode="fake",
            identity="fake/deterministic",
        )
    if mode == "openai":
        key = settings.openai_api_key
        if key is None or not key.get_secret_value().strip():
            raise ValueError("MODEL_PROVIDER=openai requires a nonblank OPENAI_API_KEY")
        model = settings.openai_model.strip()
        if not model:
            raise ValueError("MODEL_PROVIDER=openai requires a nonblank OPENAI_MODEL")
        provider: ManagedModelProvider = OpenAICompatibleModelProvider(
            key,
            model=model,
            connection_mode="openai",
            max_concurrency=settings.openai_max_concurrency,
            base_url=settings.openai_base_url,
            request_timeout_seconds=settings.openai_request_timeout_seconds,
            reasoning_effort=settings.model_reasoning_effort,
            temperature=settings.model_temperature,
        )
        return ModelProviderRuntime(
            provider=provider,
            mode="openai",
            identity=f"openai/{model}",
        )
    if mode == "ollama":
        model = settings.ollama_model.strip()
        provider = OpenAICompatibleModelProvider(
            settings.ollama_api_key,
            base_url=settings.ollama_base_url,
            model=model,
            revision=settings.ollama_revision,
            connection_mode="ollama",
            max_concurrency=settings.ollama_max_concurrency,
            request_timeout_seconds=settings.ollama_request_timeout_seconds,
            reasoning_effort=settings.model_reasoning_effort,
            temperature=settings.model_temperature,
            structured_output_strict=False,
        )
        return ModelProviderRuntime(
            provider=provider,
            mode="ollama",
            identity=f"ollama/{model}",
        )
    # Settings validation makes this unreachable, but retaining a clear error
    # protects callers that construct a duck-typed settings object.
    raise ValueError(f"Unsupported MODEL_PROVIDER mode: {mode}")


__all__ = [
    "ManagedModelProvider",
    "ModelProviderRuntime",
    "build_model_provider",
]
