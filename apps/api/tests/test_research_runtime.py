from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import debate_api.main as main_module
from debate_api.main import create_app
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.research import SafeDocumentFetcher
from debate_api.providers.research_runtime import (
    ResearchProviderRuntime,
    build_research_provider,
)
from debate_api.providers.search import TavilySearchProvider
from debate_api.settings import REPOSITORY_ROOT, Settings


def _settings(tmp_path, **values: object) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        **values,
    )


def test_fake_pair_has_no_service(tmp_path) -> None:
    runtime = build_research_provider(
        _settings(tmp_path, search_provider="fake", fetch_provider="fake"),
        EventStore(f"sqlite:///{tmp_path / 'events.db'}"),
    )
    assert runtime.service is None
    assert runtime.search_provider is None
    assert runtime.identity == "fake/deterministic"


def test_live_pair_constructs_and_lifecycles_without_network(tmp_path) -> None:
    app = create_app(
        _settings(
            tmp_path,
            search_provider="tavily",
            fetch_provider="safe_httpx",
            tavily_api_key="synthetic-key",
        ),
    )
    assert isinstance(app.state.search_provider, TavilySearchProvider)
    assert isinstance(app.state.fetch_provider, SafeDocumentFetcher)
    assert app.state.research_service is not None
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["checks"]["research_provider"] == {
            "status": "ok",
            "detail": "Provider tavily/safe_httpx is ready.",
        }
        assert app.state.search_provider._client is not None
        assert app.state.fetch_provider._client is not None
    assert app.state.search_provider._client is None
    assert app.state.fetch_provider._client is None


def test_live_pair_requires_key_and_rejects_mixed_pair(tmp_path) -> None:
    with pytest.raises(ValueError, match="nonblank TAVILY_API_KEY"):
        create_app(
            _settings(
                tmp_path,
                search_provider="tavily",
                fetch_provider="safe_httpx",
            )
        )
    with pytest.raises(ValueError, match="supported pairs"):
        create_app(
            _settings(
                tmp_path,
                search_provider="tavily",
                fetch_provider="fake",
                tavily_api_key="synthetic-key",
            )
        )


def test_injected_launcher_skips_research_provider_construction(tmp_path, monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> ResearchProviderRuntime:
        raise AssertionError("research runtime should not be constructed")

    monkeypatch.setattr(main_module, "build_research_provider", fail)

    class Launcher:
        async def run(self, run_id: str) -> None:
            del run_id

    app = create_app(
        _settings(
            tmp_path,
            search_provider="tavily",
            fetch_provider="safe_httpx",
        ),
        run_launcher=Launcher(),
    )
    assert app.state.research_service is None
    assert app.state.research_provider_identity == "injected/run_launcher"


def test_partial_research_startup_closes_entered_providers(monkeypatch, tmp_path) -> None:
    class Tracked:
        def __init__(self, fail_enter: bool = False) -> None:
            self.fail_enter = fail_enter
            self.closed = False

        async def __aenter__(self) -> Tracked:
            if self.fail_enter:
                raise RuntimeError("synthetic enter failure")
            return self

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback
            self.closed = True

    search = Tracked()
    fetch = Tracked(fail_enter=True)
    runtime = ResearchProviderRuntime(
        search_provider=search,
        fetch_provider=fetch,
        service=None,
        mode="synthetic",
        identity="synthetic/research",
    )
    monkeypatch.setattr(main_module, "build_research_provider", lambda *args, **kwargs: runtime)
    app = create_app(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="failed startup: RuntimeError"):
        with TestClient(app):
            pass
    assert search.closed
    assert fetch.closed


def test_cleanup_closes_all_providers_and_preserves_startup_error(monkeypatch, tmp_path) -> None:
    class Model:
        def __init__(self) -> None:
            self.closed = False

        async def __aenter__(self) -> Model:
            return self

        async def close(self) -> None:
            self.closed = True
            raise RuntimeError("synthetic model close failure")

        async def generate_structured(self, request: object, output_schema: object) -> object:
            del request, output_schema
            raise AssertionError("not reached")

    class Tracked:
        def __init__(self, fail_enter: bool = False) -> None:
            self.fail_enter = fail_enter
            self.closed = False

        async def __aenter__(self) -> Tracked:
            if self.fail_enter:
                raise RuntimeError("synthetic research enter failure")
            return self

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback
            self.closed = True

    model = Model()
    search = Tracked()
    fetch = Tracked(fail_enter=True)
    from debate_api.providers.runtime import ModelProviderRuntime

    monkeypatch.setattr(
        main_module,
        "build_model_provider",
        lambda settings: ModelProviderRuntime(
            provider=model,
            mode="synthetic",
            identity="synthetic/model",
        ),
    )
    monkeypatch.setattr(
        main_module,
        "build_research_provider",
        lambda *args, **kwargs: ResearchProviderRuntime(
            search_provider=search,
            fetch_provider=fetch,
            service=None,
            mode="synthetic",
            identity="synthetic/research",
        ),
    )
    app = create_app(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="Configured research provider failed startup"):
        with TestClient(app):
            pass
    assert model.closed
    assert search.closed
    assert fetch.closed


def test_settings_dotenv_path_is_repo_root_absolute() -> None:
    assert Settings.model_config["env_file"] == (
        REPOSITORY_ROOT / ".env",
        REPOSITORY_ROOT / ".env.local",
    )
