"""Application-owned research provider selection and lifecycle metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from debate_api.persistence.sqlite import EventStore
from debate_api.providers.research import SafeDocumentFetcher
from debate_api.providers.search import TavilySearchProvider
from debate_api.research import ResearchService
from debate_api.settings import Settings


class ManagedResearchProvider(Protocol):
    """Provider contract for clients owned by the application lifespan."""

    async def __aenter__(self) -> ManagedResearchProvider: ...

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


@dataclass(frozen=True)
class ResearchProviderRuntime:
    """Research adapters and the service that composes them."""

    search_provider: ManagedResearchProvider | None
    fetch_provider: ManagedResearchProvider | None
    service: ResearchService | None
    mode: str
    identity: str


def build_research_provider(
    settings: Settings,
    store: EventStore,
) -> ResearchProviderRuntime:
    """Construct the configured research path without opening network clients."""

    search_mode = settings.search_provider
    fetch_mode = settings.fetch_provider
    if (search_mode, fetch_mode) == ("fake", "fake"):
        return ResearchProviderRuntime(
            search_provider=None,
            fetch_provider=None,
            service=None,
            mode="fake",
            identity="fake/deterministic",
        )
    if (search_mode, fetch_mode) != ("tavily", "safe_httpx"):
        raise ValueError(
            "Unsupported research provider pair: "
            f"SEARCH_PROVIDER={search_mode!r}, FETCH_PROVIDER={fetch_mode!r}; "
            "supported pairs are fake/fake and tavily/safe_httpx"
        )
    key = settings.tavily_api_key
    if key is None or not key.get_secret_value().strip():
        raise ValueError("SEARCH_PROVIDER=tavily requires a nonblank TAVILY_API_KEY")
    search_impl = TavilySearchProvider(key.get_secret_value())
    fetch_impl = SafeDocumentFetcher()
    search: ManagedResearchProvider = cast(ManagedResearchProvider, search_impl)
    fetch: ManagedResearchProvider = cast(ManagedResearchProvider, fetch_impl)
    service = ResearchService(
        search_provider=search_impl,
        fetch_provider=fetch_impl,
        store=store,
    )
    return ResearchProviderRuntime(
        search_provider=search,
        fetch_provider=fetch,
        service=service,
        mode="tavily/safe_httpx",
        identity="tavily/safe_httpx",
    )


__all__ = [
    "ManagedResearchProvider",
    "ResearchProviderRuntime",
    "build_research_provider",
]
