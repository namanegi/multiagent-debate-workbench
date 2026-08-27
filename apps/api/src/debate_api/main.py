"""FastAPI application entry point."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from debate_api.api.runs import RunLauncher
from debate_api.api.runs import router as runs_router
from debate_api.orchestration.debate import DebateOrchestrator
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.research_runtime import (
    ManagedResearchProvider,
    ResearchProviderRuntime,
    build_research_provider,
)
from debate_api.providers.runtime import (
    ModelProviderRuntime,
    build_model_provider,
)
from debate_api.settings import Settings, get_settings


class HealthCheck(BaseModel):
    """Status for one readiness dependency."""

    status: Literal["ok", "not_ready"]
    detail: str


class HealthResponse(BaseModel):
    """Stable health response shared by live and ready endpoints."""

    status: Literal["ok", "ready", "not_ready"]
    service: str
    version: str
    environment: str
    checks: dict[str, HealthCheck]


async def _close_research_provider(provider: ManagedResearchProvider) -> None:
    """Close one research adapter through its async context lifecycle."""

    await provider.__aexit__(None, None, None)


def create_app(
    settings: Settings | None = None,
    *,
    event_store: EventStore | None = None,
    run_launcher: RunLauncher | None = None,
) -> FastAPI:
    """Build the API application, allowing tests to inject settings."""

    resolved_settings = settings or get_settings()
    resolved_event_store = event_store or EventStore(resolved_settings.database_url)
    provider_runtime = (
        build_model_provider(resolved_settings)
        if run_launcher is None
        else ModelProviderRuntime(
            provider=None,
            mode="injected",
            identity="injected/run_launcher",
        )
    )
    research_runtime = (
        build_research_provider(
            resolved_settings,
            resolved_event_store,
        )
        if run_launcher is None
        else ResearchProviderRuntime(
            search_provider=None,
            fetch_provider=None,
            service=None,
            mode="injected",
            identity="injected/run_launcher",
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        active_error: BaseException | None = None

        async def cleanup() -> None:
            cleanup_error: BaseException | None = None
            for callback in reversed(cleanup_callbacks):
                try:
                    await callback()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if active_error is None and cleanup_error is not None:
                raise cleanup_error

        try:
            provider = provider_runtime.provider
            if provider is not None:
                # Register before entering: a provider may allocate a client
                # and then fail during __aenter__.
                cleanup_callbacks.append(provider.close)
                try:
                    await provider.__aenter__()
                except Exception as error:
                    app.state.model_provider_ready = False
                    raise RuntimeError(
                        f"Configured {provider_runtime.mode} model provider failed startup: "
                        f"{type(error).__name__}"
                    ) from error
                app.state.model_provider_ready = True
            search_provider = research_runtime.search_provider
            if search_provider is not None:

                async def close_search() -> None:
                    await _close_research_provider(search_provider)

                cleanup_callbacks.append(close_search)
                try:
                    await search_provider.__aenter__()
                except Exception as error:
                    app.state.research_provider_ready = False
                    raise RuntimeError(
                        f"Configured research provider failed startup: {type(error).__name__}"
                    ) from error
            fetch_provider = research_runtime.fetch_provider
            if fetch_provider is not None:

                async def close_fetch() -> None:
                    await _close_research_provider(fetch_provider)

                cleanup_callbacks.append(close_fetch)
                try:
                    await fetch_provider.__aenter__()
                except Exception as error:
                    app.state.research_provider_ready = False
                    raise RuntimeError(
                        f"Configured research provider failed startup: {type(error).__name__}"
                    ) from error
            app.state.research_provider_ready = (
                search_provider is None and fetch_provider is None
            ) or (search_provider is not None and fetch_provider is not None)
            yield
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                await cleanup()
            finally:
                app.state.model_provider_ready = False
                app.state.research_provider_ready = False

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Inspectable, evidence-backed AI debate runtime.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.event_store = resolved_event_store
    app.state.model_provider = provider_runtime.provider
    app.state.model_provider_mode = provider_runtime.mode
    app.state.model_provider_identity = provider_runtime.identity
    app.state.model_provider_ready = provider_runtime.provider is None
    app.state.search_provider = research_runtime.search_provider
    app.state.fetch_provider = research_runtime.fetch_provider
    app.state.research_service = research_runtime.service
    app.state.research_provider_mode = research_runtime.mode
    app.state.research_provider_identity = research_runtime.identity
    app.state.research_provider_ready = (
        research_runtime.search_provider is None and research_runtime.fetch_provider is None
    )
    app.state.run_launcher = (
        run_launcher
        if run_launcher is not None
        else DebateOrchestrator(
            app.state.event_store,
            provider=provider_runtime.provider,
            research_service=research_runtime.service,
        )
    )
    app.state.run_tasks = {}
    app.include_router(runs_router, prefix=resolved_settings.api_prefix)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    def live() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=resolved_settings.app_name,
            version=resolved_settings.app_version,
            environment=resolved_settings.environment,
            checks={"process": HealthCheck(status="ok", detail="Process is running.")},
        )

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        tags=["health"],
        responses={503: {"description": "The service is not ready."}},
    )
    def ready() -> HealthResponse:
        provider_ready = bool(app.state.model_provider_ready)
        checks = {
            "configuration": HealthCheck(
                status="ok",
                detail=(
                    "Configuration loaded; live provider credentials are configured."
                    if provider_runtime.mode not in {"fake", "injected"}
                    or research_runtime.mode not in {"fake", "injected"}
                    else "Configuration loaded without external secrets."
                ),
            ),
            "database": HealthCheck(status="ok", detail="SQLite event store is available."),
            "model_provider": HealthCheck(
                status="ok" if provider_ready else "not_ready",
                detail=(
                    f"Provider {provider_runtime.identity} is ready."
                    if provider_ready
                    else f"Provider {provider_runtime.identity} is not ready."
                ),
            ),
            "research_provider": HealthCheck(
                status="ok" if app.state.research_provider_ready else "not_ready",
                detail=(
                    f"Provider {research_runtime.identity} is ready."
                    if app.state.research_provider_ready
                    else f"Provider {research_runtime.identity} is not ready."
                ),
            ),
        }
        all_ready = provider_ready and bool(app.state.research_provider_ready)
        return HealthResponse(
            status="ready" if all_ready else "not_ready",
            service=resolved_settings.app_name,
            version=resolved_settings.app_version,
            environment=resolved_settings.environment,
            checks=checks,
        )

    return app


app = create_app()
