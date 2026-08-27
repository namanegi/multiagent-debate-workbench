import gc
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory

import pytest

from debate_api.domain.models import (
    AgentBrief,
    CreateRunRequest,
    PlanningOutcome,
    PlanningOutcomeCategory,
    RunEventType,
    RunPhase,
    RunStatus,
)
from debate_api.domain.validation import InvariantViolation
from debate_api.orchestration.topic import TopicOrchestrator
from debate_api.persistence.sqlite import EventStore, IdempotencyConflictError


def make_store(tmp_path) -> EventStore:
    return EventStore(f"sqlite:///{tmp_path / 'debate.db'}")


def commit_test_plan(store: EventStore, run_id: str, topic: str) -> None:
    store.append_event(run_id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    briefs: list[AgentBrief] = [
        brief.model_copy(update={"id": f"brief_{run_id}_{index + 1}"})
        for index, brief in enumerate(TopicOrchestrator.catalog_briefs(topic, 3))
    ]
    store.commit_plan(
        run_id,
        PlanningOutcome(
            planner_id="test-planner",
            category=PlanningOutcomeCategory.FALLBACK,
            message="test plan",
            brief_count=3,
        ),
        briefs,
        3,
    )


def test_create_run_is_idempotent_and_conflicting_payloads_are_rejected(tmp_path) -> None:
    store = make_store(tmp_path)
    request = CreateRunRequest(topic="How do batteries age?", goal="Explain the tradeoffs.")

    first, created = store.create_run(request, "create-key")
    second, created_again = store.create_run(request, "create-key")

    assert created is True
    assert created_again is False
    assert second.id == first.id
    assert store.last_sequence(first.id) == 1

    with pytest.raises(IdempotencyConflictError):
        store.create_run(CreateRunRequest(topic="A different topic"), "create-key")


def test_append_updates_projection_atomically_and_rejects_invalid_references(tmp_path) -> None:
    store = make_store(tmp_path)
    run, _ = store.create_run(CreateRunRequest(topic="A topic"))
    commit_test_plan(store, run.id, run.topic)
    evidence = {
        "evidence": {
            "id": "evidence_1",
            "canonical_url": "https://example.com/report",
            "title": "Report",
            "publisher": "Example",
            "excerpt": "A measured result.",
        }
    }
    store.append_event(
        run.id,
        RunEventType.EVIDENCE_CREATED,
        RunPhase.RESEARCHING,
        evidence,
        actor_id="run-verification",
    )
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.OPENING, {})
    store.append_event(
        run.id,
        RunEventType.MESSAGE_CREATED,
        RunPhase.OPENING,
        {
            "message": {
                "id": "message_1",
                "author_id": "agent_1",
                "author_label": "Investigator",
                "phase": "opening",
                "kind": "opening",
                "content": "A measured result.",
                "claim_ids": ["claim_1"],
                "evidence_ids": ["evidence_1"],
                "turn_index": 1,
            },
            "claims": [
                {
                    "id": "claim_1",
                    "message_id": "message_1",
                    "text": "A measured result.",
                    "claim_type": "fact",
                    "author_id": "agent_1",
                    "evidence_ids": ["evidence_1"],
                }
            ],
        },
        actor_id="agent_1",
    )
    summary = store.get_summary(run.id)
    assert [item.id for item in summary.evidence] == ["evidence_1"]
    assert [item.id for item in summary.messages] == ["message_1"]
    assert [item.id for item in summary.claims] == ["claim_1"]

    attacker_message = {
        "id": "message_attacker",
        "author_id": "attacker",
        "author_label": "Attacker",
        "phase": "opening",
        "kind": "opening",
        "content": "Forged public message.",
        "claim_ids": [],
        "evidence_ids": [],
    }
    with pytest.raises(InvariantViolation, match="authoritative investigator"):
        store.append_event(
            run.id,
            RunEventType.MESSAGE_CREATED,
            RunPhase.OPENING,
            {"message": attacker_message, "claims": []},
            actor_id="attacker",
        )
    with pytest.raises(InvariantViolation, match="actor"):
        store.append_event(
            run.id,
            RunEventType.MESSAGE_CREATED,
            RunPhase.OPENING,
            {
                "message": {**attacker_message, "id": "message_mismatch", "author_id": "agent_1"},
                "claims": [],
            },
            actor_id="attacker",
        )

    with pytest.raises(InvariantViolation, match="support metadata"):
        store.append_event(
            run.id,
            RunEventType.MESSAGE_CREATED,
            RunPhase.OPENING,
            {
                "message": {
                    "id": "message_forged_support",
                    "author_id": "agent_2",
                    "author_label": "Investigator",
                    "phase": "opening",
                    "kind": "opening",
                    "content": "Forged support state.",
                    "claim_ids": ["claim_forged_support"],
                    "evidence_ids": ["evidence_1"],
                    "turn_index": 1,
                },
                "claims": [
                    {
                        "id": "claim_forged_support",
                        "message_id": "message_forged_support",
                        "text": "A measured result.",
                        "claim_type": "fact",
                        "author_id": "agent_2",
                        "evidence_ids": ["evidence_1"],
                        "support_status": "unsupported",
                        "support_warning": (
                            "citation support unavailable: all referenced evidence is unavailable"
                        ),
                    }
                ],
            },
            actor_id="agent_2",
        )

    with pytest.raises(InvariantViolation):
        store.append_event(
            run.id,
            RunEventType.MESSAGE_CREATED,
            RunPhase.DEBATING,
            {
                "message": {
                    "id": "message_2",
                    "author_id": "agent_2",
                    "author_label": "Counter",
                    "phase": "debating",
                    "kind": "reply",
                    "content": "This reply has a missing parent.",
                    "claim_ids": [],
                    "evidence_ids": [],
                    "in_reply_to_message_id": "missing_message",
                },
                "claims": [],
            },
        )
    assert store.last_sequence(run.id) == 11


def test_concurrent_appends_have_unique_increasing_sequences(tmp_path) -> None:
    store = make_store(tmp_path)
    run, _ = store.create_run(CreateRunRequest(topic="Concurrency"))
    commit_test_plan(store, run.id, run.topic)

    def append(index: int) -> int:
        return store.append_event(
            run.id,
            RunEventType.EVIDENCE_CREATED,
            RunPhase.RESEARCHING,
            {
                "evidence": {
                    "id": f"evidence_{index}",
                    "canonical_url": f"https://example.com/source-{index}",
                    "title": f"Source {index}",
                    "publisher": "Example",
                    "excerpt": "A measured result.",
                }
            },
            actor_id="run-verification",
        ).sequence

    with ThreadPoolExecutor(max_workers=8) as executor:
        sequences = list(executor.map(append, range(24)))

    assert sorted(sequences) == list(range(9, 33))
    page = store.list_events(run.id, after_sequence=0, limit=100)
    assert [event.sequence for event in page.events] == list(range(1, 33))


def test_concurrent_cancel_requests_append_one_request_event(tmp_path) -> None:
    store = make_store(tmp_path)
    run, _ = store.create_run(CreateRunRequest(topic="Cancel race"))

    def cancel() -> str:
        return store.request_cancel(run.id).status

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(lambda _: cancel(), range(16)))

    events = store.list_events(run.id, limit=100).events
    cancel_events = [event for event in events if event.type == RunEventType.RUN_CANCEL_REQUESTED]
    assert len(cancel_events) == 1
    assert all(status == RunStatus.QUEUED for status in statuses)


def test_ordered_pagination_by_sequence_and_event_id(tmp_path) -> None:
    store = make_store(tmp_path)
    run, _ = store.create_run(CreateRunRequest(topic="Pagination"))
    for phase in (RunPhase.PLANNING, RunPhase.RESEARCHING, RunPhase.OPENING):
        store.append_event(run.id, RunEventType.PHASE_STARTED, phase, {})

    first_page = store.list_events(run.id, limit=2)
    assert [event.sequence for event in first_page.events] == [1, 2]
    assert first_page.has_more is True

    second_page = store.list_events(run.id, after_event_id=first_page.events[-1].id, limit=10)
    assert [event.sequence for event in second_page.events] == [3, 4]
    assert second_page.has_more is False


def test_replay_survives_a_new_store_instance(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'restart.db'}"
    first_store = EventStore(database_url)
    run, _ = first_store.create_run(CreateRunRequest(topic="Restart me"))
    first_store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    first_store.append_event(
        run.id,
        RunEventType.RUN_PARTIAL,
        RunPhase.PLANNING,
        {"reason": "controlled failure"},
    )

    restarted_store = EventStore(database_url)
    replay = restarted_store.list_events(run.id, limit=20)
    summary = restarted_store.get_summary(run.id)
    assert [event.sequence for event in replay.events] == [1, 2, 3]
    assert summary.run.status == RunStatus.PARTIAL
    assert summary.run.stop_reason == "controlled failure"


def test_repeated_store_reads_release_handles_for_temporary_cleanup(tmp_path) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        store = EventStore(f"sqlite:///{directory}/repeated.db")
        run, _ = store.create_run(CreateRunRequest(topic="Repeated reads"))
        for _ in range(25):
            assert store.get_run(run.id).id == run.id
            assert store.get_summary(run.id).run.id == run.id
            assert store.list_events(run.id, limit=10).events
            assert store.last_sequence(run.id) == 1
        del store
        gc.collect()
