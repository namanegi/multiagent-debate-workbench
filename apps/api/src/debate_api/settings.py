"""Application settings loaded from environment variables and optional dotenv files."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ReasoningEffort = Literal["none", "low", "medium", "high"]


class Settings(BaseSettings):
    """Runtime configuration for the API.

    Environment variables use the ``DEBATE_API_`` prefix. No setting is
    required for the local scaffold, so a clean checkout can start without
    secrets or external services.
    """

    model_config = SettingsConfigDict(
        env_file=(REPOSITORY_ROOT / ".env", REPOSITORY_ROOT / ".env.local"),
        env_prefix="DEBATE_API_",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "Open Debate Workbench API"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    database_url: str = "sqlite:///./data/debate.db"

    # Provider settings intentionally accept the repo-root .env names.  The
    # DEBATE_API_* aliases preserve the existing prefixed settings convention
    # for deployments that inject environment variables directly.
    model_provider: Literal["fake", "openai", "ollama"] = Field(
        default="fake",
        validation_alias=AliasChoices("MODEL_PROVIDER", "DEBATE_API_MODEL_PROVIDER"),
    )
    search_provider: Literal["fake", "tavily"] = Field(
        default="fake",
        validation_alias=AliasChoices("SEARCH_PROVIDER", "DEBATE_API_SEARCH_PROVIDER"),
    )
    fetch_provider: Literal["fake", "safe_httpx"] = Field(
        default="fake",
        validation_alias=AliasChoices("FETCH_PROVIDER", "DEBATE_API_FETCH_PROVIDER"),
    )
    model_reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MODEL_REASONING_EFFORT", "DEBATE_API_MODEL_REASONING_EFFORT"
        ),
    )
    model_temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        allow_inf_nan=False,
        validation_alias=AliasChoices(
            "MODEL_TEMPERATURE", "DEBATE_API_MODEL_TEMPERATURE"
        ),
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY", "DEBATE_API_TAVILY_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "DEBATE_API_OPENAI_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "DEBATE_API_OPENAI_BASE_URL"),
    )
    openai_model: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_MODEL", "DEBATE_API_OPENAI_MODEL"),
    )
    openai_max_concurrency: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "OPENAI_MAX_CONCURRENCY", "DEBATE_API_OPENAI_MAX_CONCURRENCY"
        ),
    )
    openai_request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices(
            "OPENAI_REQUEST_TIMEOUT_SECONDS", "DEBATE_API_OPENAI_REQUEST_TIMEOUT_SECONDS"
        ),
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434/v1",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "DEBATE_API_OLLAMA_BASE_URL"),
    )
    ollama_api_key: SecretStr = Field(
        default=SecretStr("ollama"),
        validation_alias=AliasChoices("OLLAMA_API_KEY", "DEBATE_API_OLLAMA_API_KEY"),
    )
    ollama_model: str = Field(
        default="hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M",
        validation_alias=AliasChoices("OLLAMA_MODEL", "DEBATE_API_OLLAMA_MODEL"),
    )
    ollama_revision: str = Field(
        default="bc640142c66e1fdd12af0bd68f40445458f3869b",
        validation_alias=AliasChoices(
            "OLLAMA_MODEL_REVISION", "DEBATE_API_OLLAMA_MODEL_REVISION"
        ),
    )
    ollama_max_concurrency: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "OLLAMA_MAX_CONCURRENCY", "DEBATE_API_OLLAMA_MAX_CONCURRENCY"
        ),
    )
    ollama_request_timeout_seconds: float = Field(
        default=120.0,
        validation_alias=AliasChoices(
            "OLLAMA_REQUEST_TIMEOUT_SECONDS", "DEBATE_API_OLLAMA_REQUEST_TIMEOUT_SECONDS"
        ),
    )
    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return ""
        return f"/{normalized.strip('/')}"


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""

    return Settings()
