"""Cross-artifact invariants enforced before an event is committed."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from debate_api.domain.models import (
    Claim,
    ClaimSupportStatus,
    ClaimType,
    Evidence,
    EvidenceStatus,
    Message,
    RunEvent,
    Synthesis,
)


class InvariantViolation(ValueError):
    """Raised when a public artifact would make a run impossible to replay."""


def require_known(ids: Iterable[str], known: Mapping[str, object], label: str) -> None:
    """Require every referenced identifier to belong to the current run."""

    missing_count = sum(identifier not in known for identifier in ids)
    if missing_count:
        raise InvariantViolation(f"unknown {label} reference(s): {missing_count}")


def derive_claim_support(
    claim_type: ClaimType,
    evidence_ids: Iterable[str],
    evidence: Mapping[str, Evidence],
) -> tuple[ClaimSupportStatus, str | None]:
    """Derive citation availability from authoritative evidence, never payload claims."""

    references = list(evidence_ids)
    available = any(
        EvidenceStatus(evidence[identifier].status) == EvidenceStatus.AVAILABLE
        for identifier in references
    )
    if claim_type != ClaimType.FACT:
        return (
            ClaimSupportStatus.AVAILABLE if available else ClaimSupportStatus.UNASSESSED,
            None,
        )
    if available:
        return ClaimSupportStatus.AVAILABLE, None
    if not references:
        return (
            ClaimSupportStatus.UNSUPPORTED,
            "citation support unavailable: factual claim has no evidence references",
        )
    return (
        ClaimSupportStatus.UNSUPPORTED,
        "citation support unavailable: all referenced evidence is unavailable",
    )


def validate_claim_support(claim: Claim, evidence: Mapping[str, Evidence]) -> None:
    """Reject forged support metadata while allowing old events without the fields."""

    if not {"support_status", "support_warning"}.intersection(claim.model_fields_set):
        return
    expected_status, expected_warning = derive_claim_support(
        ClaimType(claim.claim_type), claim.evidence_ids, evidence
    )
    if (
        ClaimSupportStatus(claim.support_status) != expected_status
        or claim.support_warning != expected_warning
    ):
        raise InvariantViolation("claim citation support metadata is inconsistent")


def validate_message_bundle(
    message: Message,
    claims: Iterable[Claim],
    messages: Mapping[str, Message],
    known_claims: Mapping[str, Claim],
    evidence: Mapping[str, Evidence],
) -> list[Claim]:
    """Validate a message and its atomically-created claims."""

    if message.id in messages:
        raise InvariantViolation(f"duplicate message id: {message.id}")
    if message.in_reply_to_message_id is not None:
        parent = messages.get(message.in_reply_to_message_id)
        if parent is None:
            raise InvariantViolation(
                f"reply target does not exist earlier in the run: {message.in_reply_to_message_id}"
            )
        if parent.id == message.id:
            raise InvariantViolation("a message cannot reply to itself")

    require_known(message.evidence_ids, evidence, "evidence")
    claim_list = list(claims)
    claim_ids = [claim.id for claim in claim_list]
    if len(set(claim_ids)) != len(claim_ids):
        raise InvariantViolation("message bundle contains duplicate claim ids")
    if set(message.claim_ids) != set(claim_ids):
        raise InvariantViolation("message claim_ids must match the claims in its event bundle")

    for claim in claim_list:
        if claim.id in known_claims:
            raise InvariantViolation(f"duplicate claim id: {claim.id}")
        if claim.message_id != message.id:
            raise InvariantViolation(f"claim {claim.id} is owned by another message")
        if claim.author_id != message.author_id:
            raise InvariantViolation(f"claim {claim.id} is authored by another agent")
        require_known(claim.evidence_ids, evidence, "evidence")
        validate_claim_support(claim, evidence)
    return claim_list


def validate_synthesis(
    synthesis: Synthesis,
    messages: Mapping[str, Message],
    claims: Mapping[str, Claim],
    evidence: Mapping[str, Evidence],
) -> None:
    """Validate references included in a final synthesis."""

    if synthesis.message_id not in messages:
        raise InvariantViolation(f"synthesis message is unknown: {synthesis.message_id}")
    require_known(synthesis.claim_ids, claims, "claim")
    require_known(synthesis.evidence_ids, evidence, "evidence")


def validate_event_sequence(events: Iterable[RunEvent]) -> None:
    """Require a replay page to be strictly increasing and duplicate-free."""

    previous = 0
    seen: set[int] = set()
    for event in events:
        if event.sequence in seen:
            raise InvariantViolation(f"duplicate event sequence: {event.sequence}")
        if event.sequence <= previous:
            raise InvariantViolation(
                f"event sequence moved backward: {event.sequence} after {previous}"
            )
        seen.add(event.sequence)
        previous = event.sequence
