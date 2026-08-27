"""Explicit opt-in structured request through the shared Ollama model adapter."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Literal, TypedDict

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from debate_api.providers.model import ModelProviderError, ModelRequest
from debate_api.providers.openai_compatible import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_QWEN_MODEL_REVISION,
    OpenAICompatibleModelProvider,
    ReasoningEffort,
)


class _SmokeSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    ollama_base_url: str = Field(
        default=DEFAULT_OLLAMA_BASE_URL,
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_api_key: SecretStr = Field(
        default=SecretStr("ollama"),
        validation_alias="OLLAMA_API_KEY",
    )
    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        validation_alias="OLLAMA_MODEL",
    )
    ollama_model_revision: str = Field(
        default=DEFAULT_QWEN_MODEL_REVISION,
        validation_alias="OLLAMA_MODEL_REVISION",
    )
    ollama_request_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=900,
        validation_alias=AliasChoices(
            "OLLAMA_REQUEST_TIMEOUT_SECONDS", "DEBATE_API_OLLAMA_REQUEST_TIMEOUT_SECONDS"
        ),
    )
    model_reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        validation_alias="MODEL_REASONING_EFFORT",
    )


_SmokeSettings.model_rebuild(
    _types_namespace={"ReasoningEffort": ReasoningEffort, "SecretStr": SecretStr}
)


class SmokeOutput(BaseModel):
    status: Literal["ok"]


SmokeOutput.model_rebuild(_types_namespace={"Literal": Literal})


class _SmokeSummary(TypedDict, total=False):
    status: str
    provider: str
    model: str
    revision: str | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    category: str
    retryable: bool


def load_settings(repo_root: Path | None = None) -> _SmokeSettings:
    root = repo_root or Path(__file__).resolve().parent.parent
    dotenv_files = tuple(path for path in (root / ".env", root / ".env.local") if path.is_file())
    return _SmokeSettings(_env_file=dotenv_files or None)


async def run_smoke(
    settings: _SmokeSettings,
    *,
    provider_factory: type[OpenAICompatibleModelProvider] = OpenAICompatibleModelProvider,
    reasoning_effort: ReasoningEffort | None = None,
    max_output_tokens: int = 2_048,
) -> _SmokeSummary:
    started = perf_counter()
    effective_reasoning_effort = reasoning_effort or settings.model_reasoning_effort
    try:
        async with provider_factory(
            settings.ollama_api_key,
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            revision=settings.ollama_model_revision,
            connection_mode="ollama",
            max_concurrency=1,
            request_timeout_seconds=settings.ollama_request_timeout_seconds,
            reasoning_effort=effective_reasoning_effort,
            structured_output_strict=False,
        ) as provider:
            response = await provider.generate_structured(
                ModelRequest(
                    request_id="ollama-smoke",
                    operation="smoke",
                    input_text="Return the status object.",
                    output_schema_name=SmokeOutput.__name__,
                    timeout_seconds=settings.ollama_request_timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    repair_attempts=0,
                ),
                SmokeOutput,
            )
        return {
            "status": "ok",
            "provider": response.model.provider,
            "model": response.model.model,
            "revision": response.model.revision,
            "latency_ms": round(response.latency_ms, 2),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    except ModelProviderError as error:
        return {
            "status": "error",
            "category": error.category.value,
            "retryable": error.retryable,
            "latency_ms": round((perf_counter() - started) * 1_000, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one opt-in local Ollama model smoke")
    parser.add_argument("--live", action="store_true", help="opt into one local model request")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/ollama-smoke.json"),
        help="local gitignored bounded metadata artifact",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        help="Optional shared Chat Completions reasoning effort",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2_048,
        help="Maximum completion tokens for the smoke request (default: 2048)",
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("pass --live to opt into a local Ollama request")
    try:
        summary = asyncio.run(
            run_smoke(
                load_settings(),
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )
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
