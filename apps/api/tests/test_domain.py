from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from debate_api.domain.models import (
    AgentBrief,
    Claim,
    ClaimType,
    CreateRunRequest,
    Evidence,
    EvidenceStatus,
    Message,
    MessageKind,
    RunEvent,
    RunEventType,
    RunPhase,
    Synthesis,
    new_id,
)
from debate_api.domain.validation import (
    InvariantViolation,
    validate_event_sequence,
    validate_message_bundle,
    validate_synthesis,
)


def make_evidence(identifier: str = "evidence_1") -> Evidence:
    return Evidence(
        id=identifier,
        canonical_url="https://example.com/source",
        title="A source",
        publisher="Example",
        excerpt="A useful excerpt.",
    )


def make_message(
    identifier: str = "message_1",
    *,
    kind: MessageKind = MessageKind.OPENING,
    reply_to: str | None = None,
    claim_ids: list[str] | None = None,
) -> Message:
    return Message(
        id=identifier,
        author_id="agent_1",
        author_label="Source investigator",
        phase=RunPhase.OPENING,
        kind=kind,
        content="Adoption increased in the measured period.",
        claim_ids=claim_ids or [],
        in_reply_to_message_id=reply_to,
        turn_index=1,
    )


def test_create_request_rejects_blank_topic_and_invalid_limits() -> None:
    with pytest.raises(ValidationError):
        CreateRunRequest(topic="   ")
    with pytest.raises(ValidationError):
        CreateRunRequest(topic="A topic", agent_count=0)


def test_create_request_defaults_to_closed_book() -> None:
    assert CreateRunRequest(topic="A topic").research_enabled is False


def test_message_bundle_requires_claim_and_evidence_references_to_exist() -> None:
    evidence = {"evidence_1": make_evidence()}
    claim = Claim(
        id="claim_1",
        message_id="message_1",
        text="Adoption increased in the measured period.",
        claim_type=ClaimType.FACT,
        author_id="agent_1",
        evidence_ids=["evidence_1"],
    )
    message = make_message(claim_ids=[claim.id])

    assert validate_message_bundle(message, [claim], {}, {}, evidence) == [claim]

    missing_evidence = claim.model_copy(update={"evidence_ids": ["evidence_missing"]})
    with pytest.raises(InvariantViolation, match="unknown evidence"):
        validate_message_bundle(message, [missing_evidence], {}, {}, evidence)


def test_synthesis_references_only_persisted_artifacts() -> None:
    message = make_message()
    claim = Claim(
        id="claim_1",
        message_id=message.id,
        text=message.content,
        claim_type=ClaimType.FACT,
        author_id=message.author_id,
    )
    evidence = make_evidence()
    synthesis = Synthesis(
        id="synthesis_1",
        message_id=message.id,
        answer="The evidence supports a cautious increase.",
        consensus=["The direction is supported."],
        claim_ids=[claim.id],
        evidence_ids=[evidence.id],
    )

    validate_synthesis(synthesis, {message.id: message}, {claim.id: claim}, {evidence.id: evidence})
    with pytest.raises(InvariantViolation, match="unknown claim"):
        validate_synthesis(
            synthesis.model_copy(update={"claim_ids": ["claim_missing"]}),
            {message.id: message},
            {claim.id: claim},
            {evidence.id: evidence},
        )


def test_event_envelope_round_trips_and_sequences_cannot_move_backward() -> None:
    occurred_at = datetime(2026, 8, 25, 1, 2, 3, tzinfo=UTC)
    first = RunEvent(
        id=new_id("evt"),
        run_id="run_1",
        sequence=1,
        type=RunEventType.RUN_CREATED,
        phase=RunPhase.CREATED,
        occurred_at=occurred_at,
        payload={"topic": "A topic"},
    )
    second = first.model_copy(
        update={
            "id": new_id("evt"),
            "sequence": 2,
            "type": RunEventType.PHASE_STARTED,
            "phase": RunPhase.PLANNING,
        }
    )

    restored = RunEvent.model_validate_json(first.model_dump_json())
    validate_event_sequence([first, second])
    assert restored == first

    with pytest.raises(InvariantViolation, match="moved backward"):
        validate_event_sequence([second, first])
    with pytest.raises(InvariantViolation, match="duplicate"):
        validate_event_sequence([first, first])


def test_agent_brief_and_unavailable_evidence_have_explicit_schema() -> None:
    brief = AgentBrief(
        id="brief_1",
        agent_id="agent_1",
        label="Source investigator",
        focus="Find direct evidence.",
        key_questions=["What changed?"],
        preferred_source_types=["official report"],
        deliverable="A short evidence-backed opening position.",
        tool_permissions=["fetch", "search"],
    )
    evidence = make_evidence().model_copy(
        update={"status": EvidenceStatus.UNAVAILABLE, "unavailable_reason": "timeout"}
    )

    assert brief.tool_permissions == ["fetch", "search"]
    assert evidence.status == EvidenceStatus.UNAVAILABLE
