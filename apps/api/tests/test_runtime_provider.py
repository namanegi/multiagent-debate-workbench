from typing import cast

import pytest
from fastapi.testclient import TestClient

import debate_api.main as main_module
from debate_api.main import create_app
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.openai_compatible import (
    DEFAULT_OLLAMA_MODEL,
    OpenAICompatibleModelProvider,
)
from debate_api.providers.runtime import (
    ManagedModelProvider,
    ModelProviderRuntime,
)
from debate_api.settings import REPOSITORY_ROOT, Settings


def _settings(tmp_path, **values: object) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        **values,
    )


def test_fake_is_the_zero_secret_default_and_ready(tmp_path) -> None:
    app = create_app(
        _settings(tmp_path), event_store=EventStore(f"sqlite:///{tmp_path / 'events.db'}")
    )

    assert app.state.model_provider is None
    with TestClient(app) as client:
        payload = client.get("/health/ready").json()

    assert payload["status"] == "ready"
    assert payload["checks"]["model_provider"]["detail"] == (
        "Provider fake/deterministic is ready."
    )


def test_openai_requires_a_nonblank_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="nonblank OPENAI_API_KEY"):
        create_app(_settings(tmp_path, model_provider="openai", openai_api_key="  "))


def test_openai_requires_an_explicit_nonblank_model(tmp_path) -> None:
    with pytest.raises(ValueError, match="nonblank OPENAI_MODEL"):
        create_app(
            _settings(
                tmp_path,
                model_provider="openai",
                openai_api_key="synthetic-test-key",
                openai_model="  ",
            )
        )


def test_openai_provider_is_entered_and_closed_by_lifespan(tmp_path) -> None:
    app = create_app(
        _settings(
            tmp_path,
            model_provider="openai",
            openai_api_key="synthetic-test-key",
            openai_model="test-model",
            openai_max_concurrency=2,
            model_reasoning_effort="medium",
            model_temperature=0.7,
        )
    )
    provider = app.state.model_provider
    assert isinstance(provider, OpenAICompatibleModelProvider)
    assert provider.connection_mode == "openai"
    assert provider.reasoning_effort == "medium"
    assert provider.temperature == 0.7
    assert app.state.model_provider_identity == "openai/test-model"
    assert app.state.run_launcher._provider is provider
    assert provider._client is None

    with TestClient(app) as client:
        assert client.get("/health/ready").json()["checks"]["model_provider"]["status"] == "ok"
        assert provider._client is not None

    assert provider._client is None


def test_ollama_uses_the_same_provider_class_and_lifespan(tmp_path) -> None:
    app = create_app(
        _settings(
            tmp_path,
            model_provider="ollama",
            ollama_base_url="http://127.0.0.1:11434/v1",
            ollama_model="test-local-model",
            ollama_revision="test-revision",
            model_reasoning_effort="low",
            model_temperature=0.4,
        )
    )
    provider = app.state.model_provider
    assert isinstance(provider, OpenAICompatibleModelProvider)
    assert provider.connection_mode == "ollama"
    assert provider.reasoning_effort == "low"
    assert provider.temperature == 0.4
    assert provider.structured_output_strict is False
    assert app.state.model_provider_identity == "ollama/test-local-model"
    assert app.state.run_launcher._provider is provider
    assert provider._client is None

    with TestClient(app) as client:
        assert client.get("/health/ready").json()["checks"]["model_provider"]["status"] == "ok"
        assert provider._client is not None

    assert provider._client is None


def test_ollama_defaults_to_the_direct_hugging_face_identifier(tmp_path) -> None:
    app = create_app(_settings(tmp_path, model_provider="ollama"))

    provider = app.state.model_provider
    assert isinstance(provider, OpenAICompatibleModelProvider)
    assert provider.model == DEFAULT_OLLAMA_MODEL
    assert provider.model == "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"


def test_openai_custom_base_url_is_runtime_configuration(tmp_path) -> None:
    app = create_app(
        _settings(
            tmp_path,
            model_provider="openai",
            openai_api_key="synthetic-test-key",
            openai_model="proxy-model",
            openai_base_url="https://models.example.test/v1",
        )
    )

    provider = app.state.model_provider
    assert isinstance(provider, OpenAICompatibleModelProvider)
    assert provider.base_url == "https://models.example.test/v1"


def test_injected_launcher_is_preserved(tmp_path) -> None:
    class Launcher:
        async def run(self, run_id: str) -> None:
            del run_id

    launcher = Launcher()
    app = create_app(_settings(tmp_path), run_launcher=launcher)

    assert app.state.run_launcher is launcher


def test_injected_launcher_skips_live_provider_configuration(tmp_path) -> None:
    class Launcher:
        async def run(self, run_id: str) -> None:
            del run_id

    app = create_app(
        _settings(tmp_path, model_provider="openai", openai_api_key=None),
        run_launcher=Launcher(),
    )

    assert app.state.model_provider is None
    with TestClient(app) as client:
        payload = client.get("/health/ready").json()
    assert payload["checks"]["model_provider"]["detail"] == (
        "Provider injected/run_launcher is ready."
    )


def test_dotenv_defaults_are_repo_root_absolute_and_cwd_independent(monkeypatch, tmp_path) -> None:
    assert Settings.model_config["env_file"] == (
        REPOSITORY_ROOT / ".env",
        REPOSITORY_ROOT / ".env.local",
    )

    safe_dotenv = tmp_path / "safe.env"
    safe_dotenv.write_text("MODEL_PROVIDER=fake\n", encoding="utf-8")

    class SafeSettings(Settings):
        model_config = Settings.model_config | {"env_file": safe_dotenv}

    away = tmp_path / "away"
    away.mkdir()
    monkeypatch.chdir(away)
    assert SafeSettings().model_provider == "fake"


def test_lifespan_closes_provider_after_partial_enter_failure(monkeypatch, tmp_path) -> None:
    class PartialProvider:
        def __init__(self) -> None:
            self.closed = False

        async def __aenter__(self) -> object:
            raise RuntimeError("synthetic enter failure")

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback

        async def close(self) -> None:
            self.closed = True

        async def generate_structured(self, request: object, output_schema: object) -> object:
            del request, output_schema
            raise AssertionError("not reached")

    provider = PartialProvider()
    runtime = ModelProviderRuntime(
        provider=cast(ManagedModelProvider, provider),
        mode="synthetic",
        identity="synthetic/partial",
    )
    monkeypatch.setattr(main_module, "build_model_provider", lambda settings: runtime)
    app = create_app(_settings(tmp_path))

    with pytest.raises(RuntimeError, match="failed startup: RuntimeError"):
        with TestClient(app):
            pass
    assert provider.closed


def test_repo_root_provider_environment_names_are_supported(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("MODEL_REASONING_EFFORT", "high")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.9")

    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'env.db'}",
    )

    assert settings.model_provider == "openai"
    assert settings.openai_model == "env-model"
    assert settings.openai_api_key is not None
    assert settings.model_reasoning_effort == "high"
    assert settings.model_temperature == 0.9
