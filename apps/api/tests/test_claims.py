from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import HttpUrl, ValidationError

from debate_api.domain.claims import (
    claims_from_investigator_output,
    prepare_investigator_message,
    stable_claim_id,
)
from debate_api.domain.models import (
    AgentBrief,
    Claim,
    ClaimSupportStatus,
    ClaimType,
    CreateRunRequest,
    Evidence,
    EvidenceStatus,
    InvestigatorClaimDraft,
    InvestigatorDirectedUpdateOutput,
    InvestigatorOpeningOutput,
    Message,
    MessageKind,
    PlanningOutcome,
    PlanningOutcomeCategory,
    ProviderClaimType,
    RunEventType,
    RunPhase,
)
from debate_api.domain.validation import InvariantViolation, validate_message_bundle
from debate_api.orchestration.model_runner import StructuredGenerationRunner
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.model import (
    FakeModelOutcome,
    FakeModelStep,
    ModelRequest,
    ProgrammableFakeModelProvider,
)


def evidence_item(
    identifier: str = "evidence_1", *, status: EvidenceStatus = EvidenceStatus.AVAILABLE
) -> Evidence:
    return Evidence(
        id=identifier,
        canonical_url=cast(HttpUrl, "https://example.com/source"),
        title="Source",
        publisher="Example",
        excerpt="A bounded excerpt.",
        status=status,
        unavailable_reason="OCR required" if status == EvidenceStatus.UNAVAILABLE else None,
        fetch_status="unavailable" if status == EvidenceStatus.UNAVAILABLE else "extracted",
    )


def message(message_id: str = "message_1") -> Message:
    return Message(
        id=message_id,
        author_id="agent_1",
        author_label="Investigator",
        phase=RunPhase.OPENING,
        kind=MessageKind.OPENING,
        content="Research findings.",
        claim_ids=[],
        turn_index=1,
    )


def test_investigator_schema_is_bounded_and_allowlisted() -> None:
    with pytest.raises(ValidationError):
        InvestigatorClaimDraft(text="x", claim_type=cast(ProviderClaimType, "unknown"))
    with pytest.raises(ValidationError):
        InvestigatorClaimDraft(text="x", claim_type=cast(ProviderClaimType, "made_up"))
    with pytest.raises(ValidationError):
        InvestigatorClaimDraft(text="x", claim_type=ClaimType.FACT, evidence_ids=["e1", "e1"])
    with pytest.raises(ValidationError):
        InvestigatorOpeningOutput.model_validate(
            {"claims": [{"text": "x", "claim_type": "fact", "extra": 1}]}
        )
    with pytest.raises(ValidationError):
        InvestigatorDirectedUpdateOutput.model_validate({"target_option": -1})
    with pytest.raises(ValidationError):
        InvestigatorDirectedUpdateOutput.model_validate(
            {"target_option": 70, "interaction_kind": "support"}
        )
    with pytest.raises(ValidationError):
        InvestigatorDirectedUpdateOutput.model_validate({"target_option": 0})
    with pytest.raises(ValidationError):
        InvestigatorClaimDraft(text="x", claim_type=ClaimType.FACT, evidence_ids=["x" * 121])
    with pytest.raises(ValidationError):
        Message(
            id="message_1",
            author_id="agent_1",
            author_label="Investigator",
            phase=RunPhase.OPENING,
            kind=MessageKind.OPENING,
            content="x",
            claim_ids=["claim_1", "claim_1"],
        )


def test_claims_are_stable_and_mark_unsupported_fact_without_faking_citation() -> None:
    output = InvestigatorOpeningOutput(
        claims=[
            InvestigatorClaimDraft(text="A fact.", claim_type=ClaimType.FACT),
            InvestigatorClaimDraft(
                text="An OCR fact.", claim_type=ClaimType.FACT, evidence_ids=["ocr"]
            ),
            InvestigatorClaimDraft(
                text="A supported fact.", claim_type=ClaimType.FACT, evidence_ids=["available"]
            ),
            InvestigatorClaimDraft(text="A recommendation.", claim_type=ClaimType.RECOMMENDATION),
        ]
    )
    evidence = {
        "ocr": evidence_item("ocr", status=EvidenceStatus.UNAVAILABLE),
        "available": evidence_item("available"),
    }
    first = claims_from_investigator_output(
        output, run_id="run_1", message_id="message_1", author_id="agent_1", evidence=evidence
    )
    second = claims_from_investigator_output(
        output, run_id="run_1", message_id="message_1", author_id="agent_1", evidence=evidence
    )
    assert [claim.id for claim in first] == [claim.id for claim in second]
    assert first[0].support_status == ClaimSupportStatus.UNSUPPORTED
    assert first[0].support_warning and "no evidence" in first[0].support_warning
    assert first[1].support_status == ClaimSupportStatus.UNSUPPORTED
    assert first[2].support_status == ClaimSupportStatus.AVAILABLE
    assert first[3].support_status == ClaimSupportStatus.UNASSESSED
    assert stable_claim_id("run_1", "message_1", "agent_1", 0, output.claims[0]) == first[0].id
    assert first[0].id != stable_claim_id("run_2", "message_1", "agent_1", 0, output.claims[0])
    assert first[0].id != stable_claim_id("run_1", "message_1", "agent_2", 0, output.claims[0])


def test_closed_book_unknown_evidence_is_cleared_before_event_payload() -> None:
    item = message()
    output = InvestigatorOpeningOutput(
        claims=[
            InvestigatorClaimDraft(
                text="A fact.", claim_type=ClaimType.FACT, evidence_ids=["missing"]
            )
        ]
    )
    payload = prepare_investigator_message(
        item.model_copy(update={"evidence_ids": ["missing"]}),
        output,
        run_id="run_1",
        evidence={},
        actor_id="agent_1",
    )
    assert payload["message"]["evidence_ids"] == []
    assert payload["claims"][0]["evidence_ids"] == []


def test_research_enabled_unknown_evidence_is_rejected_before_event_payload() -> None:
    item = message()
    output = InvestigatorOpeningOutput(
        claims=[
            InvestigatorClaimDraft(
                text="A fact.", claim_type=ClaimType.FACT, evidence_ids=["missing"]
            )
        ]
    )
    with pytest.raises(InvariantViolation, match="unknown evidence") as error:
        prepare_investigator_message(
            item,
            output,
            run_id="run_1",
            evidence={"available": evidence_item()},
            actor_id="agent_1",
        )
    assert "missing" not in str(error.value)


def test_persistence_derives_support_and_rejects_forged_metadata() -> None:
    available = {"available": evidence_item("available")}
    output = InvestigatorOpeningOutput(
        claims=[
            InvestigatorClaimDraft(
                text="A supported fact.", claim_type=ClaimType.FACT, evidence_ids=["available"]
            )
        ]
    )
    claims = claims_from_investigator_output(
        output, run_id="run_1", message_id="message_1", author_id="agent_1", evidence=available
    )
    valid_message = message().model_copy(update={"claim_ids": [claims[0].id]})
    assert validate_message_bundle(valid_message, claims, {}, {}, available) == claims
    forged = claims[0].model_copy(update={"support_status": ClaimSupportStatus.UNSUPPORTED})
    with pytest.raises(InvariantViolation, match="support metadata"):
        validate_message_bundle(valid_message, [forged], {}, {}, available)
    unavailable = {"ocr": evidence_item("ocr", status=EvidenceStatus.UNAVAILABLE)}
    unsupported = InvestigatorOpeningOutput(
        claims=[
            InvestigatorClaimDraft(
                text="An OCR fact.", claim_type=ClaimType.FACT, evidence_ids=["ocr"]
            )
        ]
    )
    derived = claims_from_investigator_output(
        unsupported,
        run_id="run_1",
        message_id="message_2",
        author_id="agent_1",
        evidence=unavailable,
    )[0]
    forged_unavailable = derived.model_copy(
        update={"support_status": ClaimSupportStatus.AVAILABLE, "support_warning": None}
    )
    with pytest.raises(InvariantViolation, match="support metadata"):
        validate_message_bundle(
            message("message_2").model_copy(update={"claim_ids": [derived.id]}),
            [forged_unavailable],
            {},
            {},
            unavailable,
        )


def test_old_claim_payload_gets_safe_support_defaults() -> None:
    old_payload = {
        "id": "claim_old",
        "message_id": "message_old",
        "text": "A historical claim.",
        "claim_type": "fact",
        "author_id": "agent_1",
        "evidence_ids": [],
    }
    claim = Claim.model_validate(old_payload)
    assert claim.support_status == ClaimSupportStatus.UNASSESSED
    assert claim.support_warning is None


def test_message_bundle_checks_ownership_and_known_claims() -> None:
    item = message()
    output = InvestigatorOpeningOutput(
        claims=[InvestigatorClaimDraft(text="A fact.", claim_type=ClaimType.FACT)]
    )
    payload = prepare_investigator_message(
        item, output, run_id="run_1", evidence={}, actor_id="agent_1"
    )
    assert payload["claims"][0]["message_id"] == item.id
    with pytest.raises(InvariantViolation, match="actor"):
        prepare_investigator_message(item, output, run_id="run_1", evidence={}, actor_id="agent_2")
    claim_id = payload["claims"][0]["id"]
    item_with_claim = item.model_copy(update={"claim_ids": [claim_id]})
    with pytest.raises(InvariantViolation, match="duplicate claim"):
        prepare_investigator_message(
            item_with_claim,
            output,
            run_id="run_1",
            evidence={},
            known_claims={claim_id: first_claim(item_with_claim, claim_id)},
            actor_id="agent_1",
        )


def first_claim(item: Message, claim_id: str) -> Claim:
    return claims_from_investigator_output(
        InvestigatorOpeningOutput(
            claims=[InvestigatorClaimDraft(text="A fact.", claim_type=ClaimType.FACT)]
        ),
        run_id="run_1",
        message_id=item.id,
        author_id=item.author_id,
        evidence={},
    )[0].model_copy(update={"id": claim_id})


@pytest.mark.asyncio
async def test_fake_provider_supports_one_bounded_schema_repair() -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="investigator.research",
                outcome=FakeModelOutcome.SCHEMA_FAILURE,
                output={"claims": [{"text": "bad", "claim_type": "unknown"}]},
                repair_output={
                    "claims": [{"text": "A fact.", "claim_type": "fact", "evidence_ids": []}]
                },
            )
        ]
    )
    response = await StructuredGenerationRunner(provider).run(
        ModelRequest(
            request_id="request_1",
            operation="investigator.research",
            input_text="bounded internal prompt",
            output_schema_name="placeholder",
            repair_attempts=1,
        ),
        InvestigatorOpeningOutput,
    )
    assert response.output.claims[0].claim_type == ClaimType.FACT
    assert response.repair_attempted is True
    assert provider.calls[0].output_schema_name == "InvestigatorOpeningOutput"


def test_event_store_rejects_unknown_claim_atomically_and_replays_old_claim_payload(
    tmp_path: Path,
) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'claims.db'}")
    run, _ = store.create_run(CreateRunRequest(topic="A topic", agent_count=1))
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    briefs = [
        AgentBrief(
            id=f"brief_{run.id}_1",
            agent_id="agent_1",
            label="Investigator",
            focus="research",
            key_questions=["find evidence"],
            preferred_source_types=["web"],
            deliverable="a bounded finding",
            tool_permissions=[],
        )
    ]
    store.commit_plan(
        run.id,
        PlanningOutcome(
            planner_id="planner",
            category=PlanningOutcomeCategory.FALLBACK,
            message="plan",
            brief_count=1,
        ),
        briefs,
        1,
    )
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.OPENING, {})
    item = message().model_copy(update={"claim_ids": ["claim_unknown"]})
    with pytest.raises(InvariantViolation, match="unknown evidence"):
        store.append_event(
            run.id,
            RunEventType.MESSAGE_CREATED,
            RunPhase.OPENING,
            {
                "message": item.model_dump(mode="json"),
                "claims": [
                    {
                        "id": "claim_unknown",
                        "message_id": item.id,
                        "text": "A fact.",
                        "claim_type": "fact",
                        "author_id": item.author_id,
                        "evidence_ids": ["missing"],
                    }
                ],
            },
            actor_id="agent_1",
        )
    assert store.get_summary(run.id).messages == []
