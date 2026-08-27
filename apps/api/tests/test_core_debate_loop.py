from __future__ import annotations

import asyncio
import json
from itertools import product

import pytest

from debate_api.domain.models import (
    CreateRunRequest,
    InvestigatorDirectedUpdateOutput,
    MessageKind,
    RunEventType,
    RunSummary,
)
from debate_api.domain.validation import InvariantViolation
from debate_api.orchestration.debate import DebateOrchestrator
from debate_api.orchestration.debater import Debater
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.model import ModelIdentity, ModelResponse, ModelUsage


@pytest.mark.parametrize("agent_count,turn_count", product(range(1, 8), range(1, 5)))
def test_every_agent_answers_every_turn_before_synthesis(
    tmp_path, agent_count: int, turn_count: int
) -> None:
    store = EventStore(f"sqlite:///{tmp_path / f'grid-{agent_count}-{turn_count}.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="Test the explicit debate grid",
            agent_count=agent_count,
            turn_count=turn_count,
            research_enabled=False,
        )
    )

    asyncio.run(DebateOrchestrator(store, step_delay=0).run(run.id))

    summary = store.get_summary(run.id)
    agent_messages = [
        message for message in summary.messages if message.author_id.startswith("agent_")
    ]
    assert len(agent_messages) == agent_count * turn_count
    assert {(message.author_id, message.turn_index) for message in agent_messages} == {
        (f"agent_{agent_index}", turn_index)
        for turn_index in range(1, turn_count + 1)
        for agent_index in range(1, agent_count + 1)
    }
    assert all(message.kind == MessageKind.OPENING for message in agent_messages[:agent_count])
    assert all(message.kind == MessageKind.UPDATE for message in agent_messages[agent_count:])
    assert all(message.in_reply_to_message_id is None for message in agent_messages[:agent_count])
    assert all(message.target_agent_id is not None for message in agent_messages[agent_count:])
    assert all(message.target_claim_id is not None for message in agent_messages[agent_count:])
    claims = {claim.id: claim for claim in summary.claims}
    messages_by_id = {message.id: message for message in summary.messages}
    for message in agent_messages[agent_count:]:
        parent = messages_by_id[message.in_reply_to_message_id or ""]
        assert parent.turn_index == message.turn_index - 1  # type: ignore[operator]
        assert claims[message.target_claim_id or ""].message_id == parent.id
        assert message.target_agent_id == parent.author_id
        assert message.interaction_kind in {"challenge", "support"}
        if agent_count == 1:
            assert message.target_agent_id == message.author_id
        else:
            assert message.target_agent_id != message.author_id
    assert summary.synthesis is not None
    assert summary.run.status == "completed"
    assert summary.debate_protocol is not None
    assert summary.debate_protocol.completed_agent_messages == agent_count * turn_count

    events = store.list_events(run.id, limit=1_000).events
    message_events = [event for event in events if event.type == RunEventType.MESSAGE_CREATED]
    synthesis_event = next(
        event for event in events if event.type == RunEventType.SYNTHESIS_CREATED
    )
    assert all(event.sequence < synthesis_event.sequence for event in message_events)
    assert [message.turn_index for message in agent_messages] == [
        turn_index for turn_index in range(1, turn_count + 1) for _ in range(agent_count)
    ]


def test_deterministic_research_persists_evidence_exactly_once(tmp_path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'deterministic-research.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="Deterministic research ownership",
            agent_count=3,
            turn_count=2,
            research_enabled=True,
        )
    )

    asyncio.run(DebateOrchestrator(store, step_delay=0).run(run.id))

    summary = store.get_summary(run.id)
    events = store.list_events(run.id, limit=1_000).events
    evidence_events = [event for event in events if event.type == RunEventType.EVIDENCE_CREATED]
    evidence_ids = [item.id for item in summary.evidence]
    event_ids = [str(event.payload["evidence"]["id"]) for event in evidence_events]
    assert summary.run.status == "completed"
    assert len(summary.evidence) == len(evidence_events) == 3
    assert len(evidence_ids) == len(set(evidence_ids))
    assert evidence_ids == event_ids
    assert summary.failures == []


def test_one_orchestrator_keeps_concurrent_runs_isolated(tmp_path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'concurrent-runs.db'}")
    first, _ = store.create_run(
        CreateRunRequest(
            topic="First isolated topic",
            agent_count=3,
            turn_count=2,
            research_enabled=False,
        )
    )
    second, _ = store.create_run(
        CreateRunRequest(
            topic="Second isolated topic",
            agent_count=2,
            turn_count=3,
            research_enabled=False,
        )
    )
    orchestrator = DebateOrchestrator(store, step_delay=0.001)

    async def run_both() -> None:
        await asyncio.gather(orchestrator.run(first.id), orchestrator.run(second.id))

    asyncio.run(run_both())

    for run, agent_count, turn_count in ((first, 3, 2), (second, 2, 3)):
        summary = store.get_summary(run.id)
        messages = [
            message for message in summary.messages if message.author_id.startswith("agent_")
        ]
        assert summary.run.status == "completed"
        assert len(messages) == agent_count * turn_count
        assert {message.author_id for message in messages} == {
            f"agent_{index}" for index in range(1, agent_count + 1)
        }
        assert all(run.topic in message.content for message in messages if message.turn_index == 1)
        assert all(
            claim.author_id in {message.author_id for message in messages}
            for claim in summary.claims
        )
        assert summary.synthesis is not None


@pytest.mark.parametrize(
    "payload",
    [
        {"topic": "x", "agent_count": 0, "turn_count": 2},
        {"topic": "x", "agent_count": 8, "turn_count": 2},
        {"topic": "x", "agent_count": 3, "turn_count": 0},
        {"topic": "x", "agent_count": 3, "turn_count": 5},
    ],
)
def test_public_input_rejects_values_outside_agent_turn_bounds(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CreateRunRequest.model_validate(payload)


def test_forged_directed_target_is_rejected(tmp_path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'forged-target.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="Reject forged targets", agent_count=3, turn_count=2, research_enabled=False
        )
    )
    asyncio.run(DebateOrchestrator(store, step_delay=0).run(run.id))
    completed = store.get_summary(run.id)
    openings = [message for message in completed.messages if message.turn_index == 1]
    opening_claims = [
        claim for claim in completed.claims if claim.message_id in {item.id for item in openings}
    ]
    projection = RunSummary(
        run=completed.run.model_copy(update={"phase": "debating", "status": "running"}),
        briefs=completed.briefs,
        planning_outcome=completed.planning_outcome,
        messages=openings,
        claims=opening_claims,
    )
    valid_update = next(message for message in completed.messages if message.turn_index == 2)
    other_parent = next(
        message for message in openings if message.id != valid_update.in_reply_to_message_id
    )
    forged = valid_update.model_copy(update={"target_claim_id": other_parent.claim_ids[0]})
    with pytest.raises(InvariantViolation, match="does not belong"):
        EventStore._validate_turn(projection.run, projection, forged)

    brief = next(item for item in completed.briefs if item.agent_id == "agent_1")
    known_claims = {claim.id: claim for claim in opening_claims}
    provider_input = json.loads(
        Debater._input_text(
            run.topic,
            brief,
            {},
            turn_index=2,
            previous_answers=openings,
            previous_claims=known_claims,
        )
    )
    assert all(
        record["claims"] and {"claim_id", "text"} <= set(record["claims"][0])
        for record in provider_input["previous_answers"]
    )
    selected_parent = next(item for item in reversed(openings) if item.author_id != brief.agent_id)
    selected_claim_id = selected_parent.claim_ids[0]
    selected_option = next(
        item["target_option"]
        for item in provider_input["eligible_targets"]
        if item["agent_id"] == selected_parent.author_id
    )
    payload = Debater._build(
        run.id,
        brief,
        {},
        "provider_selected_update",
        ModelResponse(
            request_id="provider_selection",
            output=InvestigatorDirectedUpdateOutput.model_validate(
                {
                    "claims": [
                        {
                            "text": "The selected peer claim needs a narrower scope.",
                            "claim_type": "inference",
                            "evidence_ids": [],
                        }
                    ],
                    "target_option": selected_option,
                    "interaction_kind": "challenge",
                }
            ),
            model=ModelIdentity(provider="fake", model="fake"),
            latency_ms=0,
            usage=ModelUsage(),
        ),
        turn_index=2,
        previous_answers=openings,
        messages={item.id: item for item in openings},
        known_claims=known_claims,
    )
    assert payload["message"]["target_claim_id"] == selected_claim_id
    assert payload["message"]["target_agent_id"] == selected_parent.author_id
