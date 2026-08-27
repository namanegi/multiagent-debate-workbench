"""Provider-neutral bounded research path."""

from __future__ import annotations

import asyncio
import ipaddress
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, HttpUrl

from debate_api.domain.models import (
    Evidence,
    EvidenceStatus,
    RunEventType,
    RunPhase,
    StrictModel,
    new_id,
    utc_now,
)
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.research import (
    DocumentStatus,
    ExtractedDocument,
    FetchProvider,
)
from debate_api.providers.search import (
    SearchHit,
    SearchProvider,
    SearchResponse,
    TavilySearchError,
)


class ResearchResultStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SEARCH_FAILED = "search_failed"


class ResearchBudget(StrictModel):
    """Server-owned search ceilings; callers cannot raise them per request."""

    max_searches_per_investigator: int = Field(default=2, ge=1, le=2)
    max_verification_searches: int = Field(default=1, ge=0, le=1)


class ResearchResult(StrictModel):
    status: ResearchResultStatus
    investigator_id: str
    query: str = Field(min_length=1, max_length=500)
    search_hit_count: int = Field(default=0, ge=0, le=20)
    duplicate_url_count: int = Field(default=0, ge=0, le=20)
    rejected_hit_count: int = Field(default=0, ge=0, le=20)
    persistence_failure_count: int = Field(default=0, ge=0, le=20)
    evidence: list[Evidence] = Field(default_factory=list, max_length=20)
    warnings: list[str] = Field(default_factory=list, max_length=32)
    error_category: str | None = Field(default=None, max_length=80)
    error_message: str | None = Field(default=None, max_length=300)


def _canonical_candidate_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2_048:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        return None
    try:
        parts = urlsplit(value)
        scheme = parts.scheme.lower()
        if scheme not in {"http", "https"} or parts.username is not None:
            return None
        if parts.password is not None or not parts.hostname:
            return None
        hostname = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        if not hostname or hostname in {"localhost", "metadata", "metadata.google.internal"}:
            return None
        port = parts.port or (443 if scheme == "https" else 80)
        if port != (443 if scheme == "https" else 80):
            return None
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            return None
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        canonical = urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))
        return canonical if len(canonical) <= 2_048 else None
    except (UnicodeError, ValueError):
        return None


class EvidenceSink(Protocol):
    def reserve_research_search(
        self,
        run_id: str,
        agent_id: str,
        *,
        verification: bool = False,
        max_searches: int = 2,
    ) -> bool: ...

    def append_event(
        self,
        run_id: str,
        event_type: RunEventType,
        phase: RunPhase,
        payload: dict[str, object],
        actor_id: str | None = None,
    ) -> object: ...


class ResearchService:
    """Search, safely fetch, and persist bounded provenance records.

    The service has no default provider construction and therefore cannot make
    a network call unless a caller explicitly injects a live provider.  It
    does not retry provider failures; the search adapter owns its taxonomy and
    bounded retry policy.
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        fetch_provider: FetchProvider,
        store: EventStore,
        *,
        budget: ResearchBudget | None = None,
        max_concurrency: int = 4,
        search_timeout_seconds: float = 30.0,
        fetch_timeout_seconds: float = 40.0,
    ) -> None:
        if not 1 <= max_concurrency <= 32:
            raise ValueError("max_concurrency must be between 1 and 32")
        if not 0 < search_timeout_seconds <= 120:
            raise ValueError("search_timeout_seconds must be between 0 and 120")
        if not 0 < fetch_timeout_seconds <= 180:
            raise ValueError("fetch_timeout_seconds must be between 0 and 180")
        self.search_provider = search_provider
        self.fetch_provider = fetch_provider
        self.store: EvidenceSink = store
        self.budget = budget or ResearchBudget()
        self.search_timeout_seconds = search_timeout_seconds
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self._search_semaphore = asyncio.Semaphore(max_concurrency)
        self._fetch_semaphore = asyncio.Semaphore(max_concurrency)
        self._known_evidence: dict[tuple[str, str], Evidence] = {}
        self._inflight: dict[tuple[str, str], asyncio.Future[Evidence | None]] = {}
        self._evidence_lock = asyncio.Lock()

    async def search_and_fetch(
        self,
        run_id: str,
        investigator_id: str,
        query: str,
    ) -> ResearchResult:
        """Use one investigator search slot and fetch its bounded candidates."""

        if not await self._reserve_investigator_search(run_id, investigator_id):
            return self._budget_result(investigator_id, query)
        return await self._execute_search(
            run_id,
            investigator_id,
            query,
            phase=RunPhase.RESEARCHING,
        )

    async def verification_search(self, run_id: str, query: str) -> ResearchResult:
        """Use the one run-level verification slot, independent of investigators."""

        if not await self._reserve_verification_search(run_id):
            return self._budget_result("run-verification", query)
        return await self._execute_search(
            run_id,
            "run-verification",
            query,
            phase=RunPhase.RESEARCHING,
        )

    async def _reserve_investigator_search(self, run_id: str, investigator_id: str) -> bool:
        return self.store.reserve_research_search(
            run_id,
            investigator_id,
            max_searches=self.budget.max_searches_per_investigator,
        )

    async def _reserve_verification_search(self, run_id: str) -> bool:
        if self.budget.max_verification_searches == 0:
            return False
        return self.store.reserve_research_search(run_id, "run-verification", verification=True)

    @staticmethod
    def _budget_result(investigator_id: str, query: str) -> ResearchResult:
        return ResearchResult(
            status=ResearchResultStatus.BUDGET_EXHAUSTED,
            investigator_id=investigator_id,
            query=" ".join(query.split())[:500] or "(blank query)",
            error_category=ResearchResultStatus.BUDGET_EXHAUSTED.value,
            error_message="server-owned search budget is exhausted",
        )

    async def _execute_search(
        self,
        run_id: str,
        investigator_id: str,
        query: str,
        *,
        phase: RunPhase,
    ) -> ResearchResult:
        normalized_query = " ".join(query.split())[:500]
        try:
            async with self._search_semaphore:
                response = await asyncio.wait_for(
                    self.search_provider.search(normalized_query), self.search_timeout_seconds
                )
        except asyncio.CancelledError:
            raise
        except TavilySearchError as error:
            result = ResearchResult(
                status=ResearchResultStatus.SEARCH_FAILED,
                investigator_id=investigator_id,
                query=normalized_query or "(blank query)",
                error_category=error.category.value,
                error_message="search provider request failed",
            )
            return result
        except TimeoutError:
            result = ResearchResult(
                status=ResearchResultStatus.SEARCH_FAILED,
                investigator_id=investigator_id,
                query=normalized_query or "(blank query)",
                error_category="timeout",
                error_message="search provider request timed out",
            )
            return result
        except Exception:
            result = ResearchResult(
                status=ResearchResultStatus.SEARCH_FAILED,
                investigator_id=investigator_id,
                query=normalized_query or "(blank query)",
                error_category="transient",
                error_message="search provider request failed",
            )
            return result
        result = await self._fetch_results(
            run_id,
            investigator_id,
            normalized_query,
            response,
            phase=phase,
        )
        return result

    async def _fetch_results(
        self,
        run_id: str,
        investigator_id: str,
        query: str,
        response: SearchResponse,
        *,
        phase: RunPhase,
    ) -> ResearchResult:
        unique_hits: list[SearchHit] = []
        duplicate_count = 0
        rejected_count = 0
        warnings: list[str] = []
        local_urls: set[str] = set()
        for hit in response.hits:
            canonical_url = _canonical_candidate_url(hit.url)
            if canonical_url is None:
                rejected_count += 1
                warnings.append("one search result had an unsafe or malformed URL")
                continue
            if canonical_url in local_urls:
                duplicate_count += 1
                continue
            local_urls.add(canonical_url)
            unique_hits.append(hit.model_copy(update={"url": canonical_url}))
        fetched = await asyncio.gather(
            *(
                self._fetch_one(
                    run_id,
                    investigator_id,
                    hit,
                    phase=phase,
                )
                for hit in unique_hits
            )
        )
        evidence: list[Evidence] = []
        persistence_failures = 0
        for item, duplicate in fetched:
            if duplicate:
                duplicate_count += 1
            if item is None:
                persistence_failures += 1
            if item is not None and item not in evidence:
                evidence.append(item)
        if persistence_failures:
            warnings.append("evidence persistence failed; retry is allowed")
        return ResearchResult(
            status=ResearchResultStatus.COMPLETED,
            investigator_id=investigator_id,
            query=query or "(blank query)",
            search_hit_count=len(response.hits),
            duplicate_url_count=duplicate_count,
            rejected_hit_count=rejected_count,
            persistence_failure_count=persistence_failures,
            evidence=evidence,
            warnings=warnings[:32],
        )

    async def _fetch_one(
        self,
        run_id: str,
        investigator_id: str,
        hit: SearchHit,
        *,
        phase: RunPhase = RunPhase.RESEARCHING,
    ) -> tuple[Evidence | None, bool]:
        key = (run_id, hit.url)
        owner = False
        async with self._evidence_lock:
            known = self._known_evidence.get(key)
            if known is not None:
                return known, True
            future = self._inflight.get(key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._inflight[key] = future
                owner = True
        if not owner:
            return await asyncio.shield(future), True
        document: ExtractedDocument | None = None
        fetch_status = "unavailable"
        fetch_warning = "fetch failed"
        try:
            async with self._fetch_semaphore:
                document = await asyncio.wait_for(
                    self.fetch_provider.fetch(hit.url), self.fetch_timeout_seconds
                )
        except asyncio.CancelledError:
            await self._cancel_inflight(key, future)
            raise
        except TimeoutError:
            fetch_status = "timeout"
            fetch_warning = "fetch timed out"
        except Exception:
            fetch_status = "unavailable"
            fetch_warning = "fetch failed"
        try:
            try:
                evidence = (
                    self._evidence_from_document(investigator_id, hit, document)
                    if document is not None
                    else self._evidence_from_failure(
                        investigator_id, hit, fetch_status, fetch_warning
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                evidence = self._evidence_from_failure(
                    investigator_id, hit, "malformed", "normalized evidence was malformed"
                )
            persisted = False
            try:
                payload: dict[str, object] = {"evidence": evidence.model_dump(mode="json")}
                self.store.append_event(
                    run_id,
                    RunEventType.EVIDENCE_CREATED,
                    phase,
                    payload,
                    actor_id=investigator_id,
                )
                persisted = True
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._finish_inflight(key, future, None, False)
                return None, False
            try:
                await self._finish_inflight(key, future, evidence, persisted)
            except asyncio.CancelledError:
                # A cancellation arriving after persistence must not leave a
                # stale owner entry or a waiter blocked forever.
                await self._cancel_inflight(key, future)
                raise
            return evidence, False
        except asyncio.CancelledError:
            await self._cancel_inflight(key, future)
            raise
        except Exception:
            # Unexpected failures cannot prove that an Evidence event exists;
            # complete waiters with a private retryable absence instead.
            try:
                await self._finish_inflight(key, future, None, False)
            except asyncio.CancelledError:
                await self._cancel_inflight(key, future)
                raise
            return None, False
        finally:
            # Defensive finalization covers unexpected failures between the
            # normal owner paths as well as cancellation during persistence.
            await self._cleanup_inflight(key, future)

    async def _cancel_inflight(
        self, key: tuple[str, str], future: asyncio.Future[Evidence | None]
    ) -> None:
        await self._cleanup_inflight(key, future)

    async def _cleanup_inflight(
        self, key: tuple[str, str], future: asyncio.Future[Evidence | None]
    ) -> None:
        async with self._evidence_lock:
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)
            if not future.done():
                future.cancel()

    async def _finish_inflight(
        self,
        key: tuple[str, str],
        future: asyncio.Future[Evidence | None],
        evidence: Evidence | None,
        persisted: bool,
    ) -> None:
        async with self._evidence_lock:
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)
            if persisted and evidence is not None:
                self._known_evidence[key] = evidence
            if not future.done():
                future.set_result(evidence)

    def _evidence_from_document(
        self, investigator_id: str, hit: SearchHit, document: ExtractedDocument
    ) -> Evidence:
        available = document.status == DocumentStatus.EXTRACTED
        status_value = (
            document.status.value
            if isinstance(document.status, DocumentStatus)
            else str(document.status)
        )
        document_type_value = str(document.document_type) if document.document_type else None
        title = self._title(hit, document)
        publisher = self._publisher(hit.url)
        excerpt = (document.text or hit.snippet).strip()[:2_000] or "Source unavailable."
        return Evidence(
            id=new_id("evidence"),
            agent_id=None if investigator_id == "run-verification" else investigator_id,
            canonical_url=HttpUrl(hit.url),
            final_url=self._http_url(document.final_url),
            title=title,
            publisher=publisher,
            excerpt=excerpt,
            source_type=document_type_value or "web",
            status=EvidenceStatus.AVAILABLE if available else EvidenceStatus.UNAVAILABLE,
            unavailable_reason=None
            if available
            else (document.warnings[0] if document.warnings else status_value),
            fetch_status=status_value,
            document_type=document_type_value,
            extraction_warnings=document.warnings,
            extractor_name=document.extractor.name if document.extractor else None,
            extractor_version=document.extractor.version if document.extractor else None,
            fetched_at=utc_now(),
            content_hash=document.content_hash,
        )

    def _evidence_from_failure(
        self,
        investigator_id: str,
        hit: SearchHit,
        fetch_status: str,
        warning: str,
    ) -> Evidence:
        return Evidence(
            id=new_id("evidence"),
            agent_id=None if investigator_id == "run-verification" else investigator_id,
            canonical_url=HttpUrl(hit.url),
            title=hit.title[:300] or self._publisher(hit.url),
            publisher=self._publisher(hit.url),
            excerpt=hit.snippet[:2_000] or "Source unavailable.",
            status=EvidenceStatus.UNAVAILABLE,
            unavailable_reason=warning,
            fetch_status=fetch_status,
            fetched_at=utc_now(),
        )

    @staticmethod
    def _title(hit: SearchHit, document: ExtractedDocument) -> str:
        title = document.metadata.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()[:300]
        return hit.title.strip()[:300] or ResearchService._publisher(hit.url)

    @staticmethod
    def _publisher(url: str) -> str:
        return (urlsplit(url).hostname or "unknown publisher")[:160]

    @staticmethod
    def _http_url(value: str) -> HttpUrl | None:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        return HttpUrl(value)


__all__ = [
    "EvidenceSink",
    "ResearchBudget",
    "ResearchResult",
    "ResearchResultStatus",
    "ResearchService",
]
