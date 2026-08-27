"""Offline research service, budget, provenance, and replay tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from debate_api.domain.models import (
    CreateRunRequest,
    EvidenceStatus,
    PlanningOutcome,
    PlanningOutcomeCategory,
    RunEventType,
    RunPhase,
)
from debate_api.domain.validation import InvariantViolation
from debate_api.orchestration.debate import DebateOrchestrator
from debate_api.orchestration.topic import TopicOrchestrator
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.research import (
    DocumentStatus,
    ExtractedDocument,
    ExtractorIdentity,
)
from debate_api.providers.search import (
    SearchHit,
    SearchQuery,
    SearchResponse,
    TavilyErrorCategory,
    TavilySearchError,
)
from debate_api.research import ResearchResultStatus, ResearchService


def _document(
    url: str,
    status: DocumentStatus = DocumentStatus.EXTRACTED,
    *,
    document_type: str | None = "html",
    text: str | None = "A bounded synthetic finding.",
) -> ExtractedDocument:
    return ExtractedDocument(
        source_url=url,
        final_url=url,
        document_type=document_type,
        status=status,
        text=text,
        metadata={"title": "Synthetic source"} if document_type else {},
        warnings=[],
        extractor=ExtractorIdentity(name="fixture", version="1.0") if document_type else None,
        content_hash="a" * 64 if text else None,
    )


class FakeSearch:
    def __init__(self, response: SearchResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    async def search(self, query: SearchQuery | str) -> SearchResponse:
        self.calls.append(query.query if isinstance(query, SearchQuery) else query)
        return self.response


class FakeFetch:
    def __init__(
        self,
        documents: Mapping[str, ExtractedDocument],
        delay: float = 0.0,
        *,
        started: asyncio.Event | None = None,
    ) -> None:
        self.documents = documents
        self.delay = delay
        self.started = started
        self.calls: list[str] = []
        self.active = 0
        self.peak = 0

    async def fetch(self, source_url: str) -> ExtractedDocument:
        self.calls.append(source_url)
        if self.started is not None:
            self.started.set()
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.documents[source_url]
        finally:
            self.active -= 1


class MultiEvidenceSearch:
    """Return two unique offline hits for each concurrent investigator call."""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: SearchQuery | str) -> SearchResponse:
        del query
        self.calls += 1
        prefix = f"https://source.test/investigator-{self.calls}"
        return search_response(f"{prefix}/one", f"{prefix}/two")


class MultiEvidenceFetch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, source_url: str) -> ExtractedDocument:
        self.calls.append(source_url)
        return _document(source_url)


class ErrorSearch:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: SearchQuery | str) -> SearchResponse:
        del query
        self.calls += 1
        raise TavilySearchError(TavilyErrorCategory.RATE_LIMIT, "synthetic rate limit")


class FlakyStore:
    """Delegate durable reservations while injecting one append failure."""

    def __init__(self, store: EventStore) -> None:
        self.store = store
        self.fail_next_append = True

    def reserve_research_search(
        self,
        run_id: str,
        agent_id: str,
        *,
        verification: bool = False,
        max_searches: int = 2,
    ) -> bool:
        return self.store.reserve_research_search(
            run_id,
            agent_id,
            verification=verification,
            max_searches=max_searches,
        )

    def append_event(self, *args: object, **kwargs: object) -> object:
        if self.fail_next_append:
            self.fail_next_append = False
            raise RuntimeError("synthetic append failure")
        return self.store.append_event(*args, **kwargs)


class TypeErrorReservationStore:
    def reserve_research_search(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise TypeError("synthetic sink implementation failure")


def prepare_research_run(store: EventStore, topic: str) -> str:
    run, _ = store.create_run(CreateRunRequest(topic=topic))
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    briefs = [
        brief.model_copy(update={"id": f"brief_{run.id}_{index + 1}"})
        for index, brief in enumerate(TopicOrchestrator.catalog_briefs(topic, 3))
    ]
    store.commit_plan(
        run.id,
        PlanningOutcome(
            planner_id="test-planner",
            category=PlanningOutcomeCategory.FALLBACK,
            message="synthetic plan",
            brief_count=3,
        ),
        briefs,
        3,
    )
    return run.id


def make_store(tmp_path: Path) -> tuple[EventStore, str]:
    store = EventStore(f"sqlite:///{tmp_path / 'research.db'}")
    return store, prepare_research_run(store, "Synthetic research")


def search_response(*urls: str) -> SearchResponse:
    return SearchResponse(
        hits=[
            SearchHit(url=url, title=f"Title for {index}", snippet="Search excerpt", rank=index)
            for index, url in enumerate(urls, start=1)
        ]
    )


@pytest.mark.asyncio
async def test_orchestrator_research_service_persists_each_evidence_once_for_every_agent(
    tmp_path: Path,
) -> None:
    """Exercise the real service boundary without making any external request."""

    store = EventStore(f"sqlite:///{tmp_path / 'orchestrated-research.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="Offline integrated research",
            agent_count=3,
            turn_count=2,
            research_enabled=True,
        )
    )
    search = MultiEvidenceSearch()
    fetch = MultiEvidenceFetch()
    service = ResearchService(search, fetch, store)

    await DebateOrchestrator(store, step_delay=0, research_service=service).run(run.id)

    summary = store.get_summary(run.id)
    events = store.list_events(run.id, limit=1_000).events
    evidence_events = [event for event in events if event.type == RunEventType.EVIDENCE_CREATED]
    evidence_ids = [item.id for item in summary.evidence]
    persisted_event_ids = [str(event.payload["evidence"]["id"]) for event in evidence_events]
    agent_messages = [item for item in summary.messages if item.author_id.startswith("agent_")]

    assert summary.run.status == "completed"
    assert len(agent_messages) == 6
    assert summary.synthesis is not None
    assert summary.failures == []
    assert search.calls == len(fetch.calls) // 2 == 3
    assert len(summary.evidence) == 6
    assert len(evidence_ids) == len(set(evidence_ids)) == len(persisted_event_ids)
    assert len(persisted_event_ids) == len(set(persisted_event_ids))
    assert persisted_event_ids == evidence_ids
    assert len(evidence_events) == len(summary.evidence)
    assert all(
        len(message.evidence_ids) <= 1
        and set(message.evidence_ids) <= set(evidence_ids)
        for message in agent_messages
    )
    assert not any(
        event.type == RunEventType.AGENT_FAILED
        and "InvariantViolation" in str(event.payload)
        for event in events
    )


@pytest.mark.asyncio
async def test_service_maps_html_and_unavailable_sources_and_replays_provenance(
    tmp_path: Path,
) -> None:
    urls = (
        "https://source.test/html",
        "https://source.test/second",
        "https://source.test/unavailable",
        "https://source.test/missing",
    )
    search = FakeSearch(search_response(*urls))
    fetch = FakeFetch(
        {
            urls[0]: _document(urls[0]),
            urls[1]: _document(urls[1]),
            urls[2]: _document(urls[2], DocumentStatus.UNAVAILABLE, document_type=None, text=None),
        }
    )
    store, run_id = make_store(tmp_path)
    service = ResearchService(search, fetch, store)

    result = await service.search_and_fetch(run_id, "agent_1", "  synthetic topic  ")

    assert result.status == ResearchResultStatus.COMPLETED
    assert len(result.evidence) == 4
    summary = store.get_summary(run_id)
    assert len(summary.evidence) == 4
    by_url = {str(item.canonical_url): item for item in summary.evidence}
    assert by_url[urls[0]].status == EvidenceStatus.AVAILABLE
    assert str(by_url[urls[0]].final_url) == urls[0]
    assert by_url[urls[0]].excerpt == "A bounded synthetic finding."
    assert by_url[urls[1]].document_type == "html"
    assert by_url[urls[1]].extractor_name == "fixture"
    assert by_url[urls[2]].status == EvidenceStatus.UNAVAILABLE
    assert by_url[urls[2]].fetch_status == "unavailable"
    assert by_url[urls[3]].status == EvidenceStatus.UNAVAILABLE
    assert by_url[urls[3]].fetch_status == "unavailable"
    assert store.get_summary(run_id).evidence == summary.evidence


@pytest.mark.asyncio
async def test_server_owned_budget_is_two_per_investigator_plus_one_verification(
    tmp_path: Path,
) -> None:
    url = "https://source.test/one"
    search = FakeSearch(search_response(url))
    fetch = FakeFetch({url: _document(url)})
    store, run_id = make_store(tmp_path)
    service = ResearchService(search, fetch, store)
    recreated_service = ResearchService(search, fetch, store)

    first, second = await asyncio.gather(
        service.search_and_fetch(run_id, "agent_1", "first"),
        recreated_service.search_and_fetch(run_id, "agent_1", "second"),
    )
    exhausted = await ResearchService(search, fetch, store).search_and_fetch(
        run_id, "agent_1", "third"
    )
    verification = await recreated_service.verification_search(run_id, "verify")
    verification_exhausted = await service.verification_search(run_id, "verify again")

    assert first.status == second.status == verification.status == ResearchResultStatus.COMPLETED
    assert (
        exhausted.status == verification_exhausted.status == ResearchResultStatus.BUDGET_EXHAUSTED
    )
    assert len(search.calls) == 3
    assert len(store.get_summary(run_id).evidence) == 2
    assert not store.reserve_research_search(run_id, "agent_not_in_plan")
    assert not store.reserve_research_search(run_id, "other-verification", verification=True)

    second_run_id = prepare_research_run(store, "Second synthetic research")
    second_run_result = await service.search_and_fetch(second_run_id, "agent_1", "new run")
    assert second_run_result.status == ResearchResultStatus.COMPLETED
    assert len(search.calls) == 4


@pytest.mark.asyncio
async def test_research_paths_share_the_persisted_run_retrieval_ceiling(
    tmp_path: Path,
) -> None:
    store, run_id = make_store(tmp_path)
    run = store.get_run(run_id)
    with store._connection() as connection:
        connection.execute(
            "UPDATE runs SET limits_json = ? WHERE id = ?",
            (
                json.dumps(run.limits.model_copy(update={"max_retrieval_calls": 2}).model_dump()),
                run_id,
            ),
        )
    assert store.reserve_research_search(run_id, "agent_1") is True
    assert store.reserve_research_search(run_id, "agent_1") is True
    # A different phase/agent cannot escape the same durable total.
    assert store.reserve_research_search(run_id, "run-verification", verification=True) is False


@pytest.mark.asyncio
async def test_provider_rate_limit_is_normalized_without_service_retry(tmp_path: Path) -> None:
    search = ErrorSearch()
    store, run_id = make_store(tmp_path)
    service = ResearchService(search, FakeFetch({}), store)

    result = await service.search_and_fetch(run_id, "agent_1", "rate limit")

    assert result.status == ResearchResultStatus.SEARCH_FAILED
    assert result.error_category == TavilyErrorCategory.RATE_LIMIT.value
    assert search.calls == 1


@pytest.mark.asyncio
async def test_unsafe_search_candidates_are_rejected_before_fetch(tmp_path: Path) -> None:
    valid_url = "https://source.test/valid"
    search = FakeSearch(
        SearchResponse(
            hits=[
                SearchHit(url="javascript:alert(1)", title="bad", snippet="bad", rank=1),
                SearchHit(url="http://127.0.0.1/private", title="private", snippet="bad", rank=2),
                SearchHit(url=valid_url, title="valid", snippet="ok", rank=3),
            ]
        )
    )
    store, run_id = make_store(tmp_path)
    fetch = FakeFetch({valid_url: _document(valid_url)})
    service = ResearchService(search, fetch, store)

    result = await service.search_and_fetch(run_id, "agent_1", "candidate safety")

    assert result.rejected_hit_count == 2
    assert len(result.warnings) == 2
    assert fetch.calls == [valid_url]
    assert len(store.get_summary(run_id).evidence) == 1


def test_verification_evidence_cannot_impersonate_investigator(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'verification.db'}")
    run, _ = store.create_run(CreateRunRequest(topic="Verification attribution"))
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    briefs = [
        brief.model_copy(
            update={"id": f"brief_{run.id}_{index + 1}", "agent_id": f"agent_{index + 1}"}
        )
        for index, brief in enumerate(TopicOrchestrator.catalog_briefs(run.topic, 3))
    ]
    store.commit_plan(
        run.id,
        PlanningOutcome(
            planner_id="test-planner",
            category=PlanningOutcomeCategory.FALLBACK,
            message="synthetic plan",
            brief_count=3,
        ),
        briefs,
        3,
    )
    evidence = {
        "id": "verification-evidence",
        "canonical_url": "https://source.test/verification",
        "title": "Verification",
        "publisher": "source.test",
        "excerpt": "A bounded verification excerpt.",
    }
    with pytest.raises(InvariantViolation):
        store.append_event(
            run.id,
            RunEventType.EVIDENCE_CREATED,
            RunPhase.RESEARCHING,
            {"evidence": {**evidence, "agent_id": "agent_1"}},
            actor_id="run-verification",
        )
    with pytest.raises(InvariantViolation):
        store.append_event(
            run.id,
            RunEventType.EVIDENCE_CREATED,
            RunPhase.RESEARCHING,
            {"evidence": {**evidence, "id": "unauthorized", "agent_id": "agent_1"}},
            actor_id="not-a-brief-agent",
        )
    store.append_event(
        run.id,
        RunEventType.EVIDENCE_CREATED,
        RunPhase.RESEARCHING,
        {"evidence": {**evidence, "id": "investigator-evidence", "agent_id": "agent_1"}},
        actor_id="agent_1",
    )
    store.append_event(
        run.id,
        RunEventType.EVIDENCE_CREATED,
        RunPhase.RESEARCHING,
        {"evidence": evidence},
        actor_id="run-verification",
    )


def test_evidence_without_authoritative_brief_is_rejected(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'no-brief.db'}")
    run, _ = store.create_run(CreateRunRequest(topic="No brief evidence"))
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    store.append_event(
        run.id,
        RunEventType.PHASE_COMPLETED,
        RunPhase.PLANNING,
        {"phase": RunPhase.PLANNING.value},
    )
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.RESEARCHING, {})
    with pytest.raises(InvariantViolation):
        store.append_event(
            run.id,
            RunEventType.EVIDENCE_CREATED,
            RunPhase.RESEARCHING,
            {
                "evidence": {
                    "id": "no-brief-evidence",
                    "canonical_url": "https://source.test/no-brief",
                    "title": "No brief",
                    "publisher": "source.test",
                    "excerpt": "Untrusted",
                    "agent_id": "agent_1",
                }
            },
            actor_id="agent_1",
        )


def test_search_reservation_requires_research_phase_and_active_run(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'reservation-phase.db'}")
    planning_run, _ = store.create_run(CreateRunRequest(topic="Planning reservation"))
    store.append_event(planning_run.id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    assert not store.reserve_research_search(planning_run.id, "agent_1")

    active_store, active_run_id = make_store(tmp_path / "active")
    active_store.append_event(
        active_run_id,
        RunEventType.RUN_CANCELLED,
        RunPhase.RESEARCHING,
        {"reason": "synthetic completion"},
    )
    assert not active_store.reserve_research_search(active_run_id, "agent_1")


@pytest.mark.asyncio
async def test_duplicate_urls_are_fetched_once_and_fetch_concurrency_is_bounded(
    tmp_path: Path,
) -> None:
    urls = tuple(f"https://source.test/{index}" for index in range(3))
    search = FakeSearch(search_response(urls[0], urls[0], urls[1], urls[2]))
    fetch = FakeFetch({url: _document(url) for url in urls}, delay=0.01)
    store, run_id = make_store(tmp_path)
    service = ResearchService(search, fetch, store, max_concurrency=2)

    result = await service.search_and_fetch(run_id, "agent_1", "duplicates")

    assert result.duplicate_url_count == 1
    assert len(result.evidence) == 3
    assert fetch.calls == list(urls)
    assert fetch.peak == 2
    assert len(store.get_summary(run_id).evidence) == 3


@pytest.mark.asyncio
async def test_reservation_type_error_is_not_silently_retried(tmp_path: Path) -> None:
    del tmp_path
    service = ResearchService(
        FakeSearch(SearchResponse(hits=[])),
        FakeFetch({}),
        TypeErrorReservationStore(),  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="synthetic sink"):
        await service.search_and_fetch("run", "agent_1", "query")


@pytest.mark.asyncio
async def test_cancellation_propagates_and_does_not_persist_partial_fetch(tmp_path: Path) -> None:
    url = "https://source.test/slow"
    search = FakeSearch(search_response(url))
    started = asyncio.Event()
    fetch = FakeFetch({url: _document(url)}, delay=60, started=started)
    store, run_id = make_store(tmp_path)
    service = ResearchService(search, fetch, store, fetch_timeout_seconds=120)

    task = asyncio.create_task(service.search_and_fetch(run_id, "agent_1", "cancel"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.get_summary(run_id).evidence == []


@pytest.mark.asyncio
async def test_inflight_owner_cancel_cancels_waiters_and_allows_later_retry(tmp_path: Path) -> None:
    url = "https://source.test/owner-cancel"
    hit = SearchHit(url=url, title="Source", snippet="Excerpt", rank=1)
    store, run_id = make_store(tmp_path)
    fetch = FakeFetch({url: _document(url)}, delay=60)
    service = ResearchService(
        FakeSearch(search_response(url)), fetch, store, fetch_timeout_seconds=120
    )

    owner = asyncio.create_task(service._fetch_one(run_id, "agent_1", hit))
    await asyncio.sleep(0.02)
    waiter = asyncio.create_task(service._fetch_one(run_id, "agent_1", hit))
    await asyncio.sleep(0.02)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    with pytest.raises(asyncio.CancelledError):
        await waiter

    fetch.delay = 0
    evidence, duplicate = await service._fetch_one(run_id, "agent_1", hit)
    assert evidence is not None
    assert duplicate is False
    assert len(store.get_summary(run_id).evidence) == 1


@pytest.mark.asyncio
async def test_inflight_store_failure_completes_waiter_and_later_retry(tmp_path: Path) -> None:
    url = "https://source.test/store-race"
    hit = SearchHit(url=url, title="Source", snippet="Excerpt", rank=1)
    real_store, run_id = make_store(tmp_path)
    flaky_store = FlakyStore(real_store)
    fetch = FakeFetch({url: _document(url)}, delay=0.03)
    service = ResearchService(FakeSearch(search_response(url)), fetch, flaky_store)

    owner, waiter = await asyncio.gather(
        service._fetch_one(run_id, "agent_1", hit),
        service._fetch_one(run_id, "agent_1", hit),
    )
    assert owner[0] is None
    assert waiter[0] is None
    assert real_store.get_summary(run_id).evidence == []

    retried, duplicate = await service._fetch_one(run_id, "agent_1", hit)
    assert retried is not None and retried.status == EvidenceStatus.AVAILABLE
    assert duplicate is False
    assert len(real_store.get_summary(run_id).evidence) == 1
