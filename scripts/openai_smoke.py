"""Explicit opt-in request through the shared OpenAI-compatible adapter.

Only bounded status, identity, usage, and latency metadata are printed or
persisted. Prompt, structured output, reasoning, raw responses, and secrets
never cross the smoke artifact boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from debate_api.providers.model import ModelProviderError, ModelRequest, ModelResponse
from debate_api.providers.openai_compatible import (
    DEFAULT_OPENAI_BASE_URL,
    OpenAICompatibleModelProvider,
)


class _SmokeSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="", validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(
        default=DEFAULT_OPENAI_BASE_URL, validation_alias="OPENAI_BASE_URL"
    )


_SmokeSettings.model_rebuild(_types_namespace={"SecretStr": SecretStr})


def load_settings(repo_root: Path | None = None) -> _SmokeSettings:
    """Load only OpenAI smoke settings from repo-root dotenv files."""

    root = repo_root or Path(__file__).resolve().parent.parent
    dotenv_files = tuple(path for path in (root / ".env", root / ".env.local") if path.is_file())
    return _SmokeSettings(_env_file=dotenv_files or None)


class SmokeOutput(BaseModel):
    status: Literal["ok"]


SmokeOutput.model_rebuild(_types_namespace={"Literal": Literal})


class _SmokeProvider(Protocol):
    async def __aenter__(self) -> _SmokeProvider: ...

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    async def generate_structured(
        self, request: ModelRequest, output_schema: type[SmokeOutput]
    ) -> ModelResponse[SmokeOutput]: ...


class _SmokeSummary(TypedDict, total=False):
    status: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    repair_attempted: bool
    category: str
    retryable: bool


async def run_smoke(
    settings: _SmokeSettings,
    *,
    provider_factory: type[OpenAICompatibleModelProvider] = OpenAICompatibleModelProvider,
) -> _SmokeSummary:
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        raise ValueError("OPENAI_API_KEY is required for the opt-in smoke")
    if not settings.openai_model.strip():
        raise ValueError("OPENAI_MODEL is required for the opt-in smoke")
    started = perf_counter()
    try:
        async with provider_factory(
            settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            connection_mode="openai",
            max_concurrency=1,
        ) as provider:
            response = await provider.generate_structured(
                ModelRequest(
                    request_id="openai-smoke",
                    operation="smoke",
                    input_text="Return the status object.",
                    output_schema_name=SmokeOutput.__name__,
                    timeout_seconds=30,
                    max_output_tokens=64,
                    repair_attempts=0,
                ),
                SmokeOutput,
            )
        return {
            "status": "ok",
            "provider": response.model.provider,
            "model": response.model.model,
            "latency_ms": round(response.latency_ms, 2),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "repair_attempted": response.repair_attempted,
        }
    except ModelProviderError as error:
        return {
            "status": "error",
            "category": error.category.value,
            "retryable": error.retryable,
            "repair_attempted": error.repair_attempted,
            "latency_ms": round((perf_counter() - started) * 1_000, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one opt-in OpenAI Chat Completions smoke")
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly opt into one live request and credit consumption",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/openai-smoke.json"),
        help="local gitignored bounded metadata artifact",
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("pass --live to opt into live OpenAI calls")
    settings = load_settings()
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        parser.error("OPENAI_API_KEY is required in repo-root .env or .env.local")
    if not settings.openai_model.strip():
        parser.error("OPENAI_MODEL is required in repo-root .env or .env.local")
    try:
        summary = asyncio.run(run_smoke(settings))
    except ValueError as error:
        parser.error(str(error))
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={summary['status']} category={summary.get('category', 'none')} "
        f"latency_ms={summary.get('latency_ms', 'none')} "
        f"total_tokens={summary.get('total_tokens', 'none')}"
    )
    print(f"metrics artifact: {args.artifact}")


if __name__ == "__main__":
    main()
