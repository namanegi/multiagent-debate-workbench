from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from debate_api.domain.models import (
    CreateRunRequest,
    Evidence,
    EvidenceStatus,
    Run,
    RunEventType,
    RunPhase,
    ToolPermission,
)
from debate_api.orchestration.debate import DebateOrchestrator
from debate_api.orchestration.debater import Debater
from debate_api.orchestration.scheduler import BoundedScheduler
from debate_api.orchestration.topic import TopicOrchestrator
from debate_api.persistence.sqlite import EventStore
from debate_api.research import ResearchResult, ResearchResultStatus, ResearchService


class FakeResearchService:
    def __init__(self, result: ResearchResult) -> None:
        self.result = result

    async def search_and_fetch(
        self, run_id: str, investigator_id: str, query: str
    ) -> ResearchResult:
        del run_id, investigator_id, query
        return self.result


def _run_and_brief(tmp_path: Path) -> tuple[EventStore, Run]:
    store = EventStore(f"sqlite:///{tmp_path / 'debater-research.db'}")
    run, _ = store.create_run(CreateRunRequest(topic="Research availability"))
    return store, run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [ResearchResultStatus.COMPLETED, ResearchResultStatus.BUDGET_EXHAUSTED],
)
async def test_no_available_evidence_is_not_a_debater_failure(
    tmp_path: Path, status: ResearchResultStatus
) -> None:
    store, run = _run_and_brief(tmp_path)
    brief = TopicOrchestrator.catalog_briefs(run.topic, 1)[0]
    result = ResearchResult(
        status=status,
        investigator_id=brief.agent_id,
        query="bounded query",
        evidence=[
            Evidence(
                id="unavailable_source",
                agent_id=brief.agent_id,
                canonical_url="https://source.test/unavailable",
                title="Unavailable source",
                publisher="source.test",
                excerpt="The source could not be used.",
                status=EvidenceStatus.UNAVAILABLE,
            )
        ],
    )
    debater = Debater(
        store,
        brief,
        provider=None,
        research_service=cast(ResearchService, FakeResearchService(result)),
    )

    evidence, failure = await debater.research(
        run, BoundedScheduler(run, store), index=0
    )

    assert evidence is None
    assert failure is None


@pytest.mark.asyncio
async def test_search_failure_and_missing_permission_remain_visible_failures(
    tmp_path: Path,
) -> None:
    store, run = _run_and_brief(tmp_path)
    brief = TopicOrchestrator.catalog_briefs(run.topic, 1)[0]
    search_failed = ResearchResult(
        status=ResearchResultStatus.SEARCH_FAILED,
        investigator_id=brief.agent_id,
        query="bounded query",
        error_category="timeout",
    )
    debater = Debater(
        store,
        brief,
        provider=None,
        research_service=cast(ResearchService, FakeResearchService(search_failed)),
    )

    _, failure = await debater.research(run, BoundedScheduler(run, store), index=0)

    assert failure is not None
    assert failure.category == "agent_search_failed"

    no_search_brief = brief.model_copy(update={"tool_permissions": [ToolPermission.FETCH]})
    no_search_debater = Debater(
        store,
        no_search_brief,
        provider=None,
        research_service=cast(ResearchService, FakeResearchService(search_failed)),
    )
    _, permission_failure = await no_search_debater.research(
        run, BoundedScheduler(run, store), index=0
    )

    assert permission_failure is not None
    assert permission_failure.category == "agent_search_not_permitted"


def available_evidence(agent_id: str) -> Evidence:
    return Evidence(
        id=f"evidence_{agent_id}",
        agent_id=agent_id,
        canonical_url=f"https://source.test/{agent_id}",
        title=f"Evidence for {agent_id}",
        publisher="source.test",
        excerpt="A bounded synthetic source.",
        status=EvidenceStatus.AVAILABLE,
    )


class PersistingFakeResearchService:
    def __init__(self, store: EventStore, results: dict[str, ResearchResult]) -> None:
        self.store = store
        self.results = results

    async def search_and_fetch(
        self, run_id: str, investigator_id: str, query: str
    ) -> ResearchResult:
        del query
        result = self.results[investigator_id]
        for evidence in result.evidence:
            self.store.append_event(
                run_id,
                RunEventType.EVIDENCE_CREATED,
                RunPhase.RESEARCHING,
                {"evidence": evidence.model_dump(mode="json")},
                actor_id=investigator_id,
            )
        return result


def _integrated_results(
    agent_1_status: ResearchResultStatus,
) -> dict[str, ResearchResult]:
    return {
        agent_id: ResearchResult(
            status=agent_1_status if agent_id == "agent_1" else ResearchResultStatus.COMPLETED,
            investigator_id=agent_id,
            query="bounded query",
            evidence=[] if agent_id == "agent_1" else [available_evidence(agent_id)],
            error_category=(
                "timeout" if agent_1_status == ResearchResultStatus.SEARCH_FAILED else None
            ),
        )
        for agent_id in ("agent_1", "agent_2", "agent_3")
    }


@pytest.mark.asyncio
async def test_no_evidence_agent_does_not_make_three_by_two_run_partial(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'no-evidence-integrated.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="No evidence for one investigator",
            agent_count=3,
            turn_count=2,
            research_enabled=True,
        )
    )
    service = PersistingFakeResearchService(
        store, _integrated_results(ResearchResultStatus.COMPLETED)
    )

    await DebateOrchestrator(
        store, step_delay=0, research_service=cast(ResearchService, service)
    ).run(run.id)

    summary = store.get_summary(run.id)
    agent_messages = [item for item in summary.messages if item.author_id.startswith("agent_")]
    assert summary.run.status == "completed"
    assert len(agent_messages) == 6
    assert summary.debate_protocol is not None
    assert summary.debate_protocol.completed_agent_messages == 6
    assert summary.synthesis is not None
    assert summary.failures == []
    assert not any(
        event.type == RunEventType.AGENT_FAILED
        for event in store.list_events(run.id, limit=1_000).events
    )


@pytest.mark.asyncio
async def test_search_failed_agent_remains_visible_and_run_is_partial(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'search-failed-integrated.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="Search fails for one investigator",
            agent_count=3,
            turn_count=2,
            research_enabled=True,
        )
    )
    service = PersistingFakeResearchService(
        store, _integrated_results(ResearchResultStatus.SEARCH_FAILED)
    )

    await DebateOrchestrator(
        store, step_delay=0, research_service=cast(ResearchService, service)
    ).run(run.id)

    summary = store.get_summary(run.id)
    agent_messages = [item for item in summary.messages if item.author_id.startswith("agent_")]
    assert summary.run.status == "partial"
    assert len(agent_messages) == 6
    assert summary.synthesis is not None
    assert any(
        failure.agent_id == "agent_1" and failure.category == "agent_search_failed"
        for failure in summary.failures
    )
    assert any(
        event.type == RunEventType.AGENT_FAILED
        and event.payload["failure"]["category"] == "agent_search_failed"
        for event in store.list_events(run.id, limit=1_000).events
    )
