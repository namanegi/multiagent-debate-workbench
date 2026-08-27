"""Safe conversion of investigator output into persisted claim bundles.

This module is deliberately independent of a model vendor.  The caller supplies
the authoritative evidence projection; model output is never allowed to invent
or resolve evidence identifiers itself.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from debate_api.domain.models import (
    Claim,
    ClaimType,
    Evidence,
    InvestigatorClaimDraft,
    InvestigatorDirectedUpdateOutput,
    InvestigatorOpeningOutput,
    Message,
)
from debate_api.domain.validation import (
    InvariantViolation,
    derive_claim_support,
    validate_message_bundle,
)

_WHITESPACE = re.compile(r"\s+")


def normalize_claim_text(text: str) -> str:
    """Normalize harmless Unicode/spacing/punctuation variance deterministically."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in normalized
    )
    return _WHITESPACE.sub(" ", normalized).strip()


def stable_claim_id(
    run_id: str,
    message_id: str,
    author_id: str,
    ordinal: int,
    draft: InvestigatorClaimDraft,
) -> str:
    """Return an id stable across retries/replay for the same message bundle."""

    if (
        not run_id
        or len(run_id) > 120
        or not message_id
        or len(message_id) > 120
        or not author_id
        or len(author_id) > 120
    ):
        raise InvariantViolation("claim ownership identifiers are not bounded")
    if ordinal < 0 or ordinal >= 10:
        raise InvariantViolation("claim ordinal is outside the bounded output")
    canonical = json.dumps(
        {
            "run_id": run_id,
            "message_id": message_id,
            "author_id": author_id,
            "ordinal": ordinal,
            "text": draft.text,
            "claim_type": str(draft.claim_type),
            "evidence_ids": list(draft.evidence_ids),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"claim_{hashlib.sha256(canonical).hexdigest()[:32]}"


def claims_from_investigator_output(
    output: InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput,
    *,
    run_id: str,
    message_id: str,
    author_id: str,
    evidence: Mapping[str, Evidence],
) -> list[Claim]:
    """Validate references and build bounded, deterministic domain claims.

    Unknown evidence is rejected before a message event can be constructed.  A
    factual claim is retained when citation support is absent, but carries an
    explicit support state/warning rather than being silently removed.
    """

    if not run_id or len(run_id) > 120 or not message_id or len(message_id) > 120:
        raise InvariantViolation("claim ownership identifiers are not bounded")
    if not author_id or len(author_id) > 120:
        raise InvariantViolation("claim ownership identifiers are not bounded")
    claims: list[Claim] = []
    for ordinal, draft in enumerate(output.claims):
        # Closed-book runs have no allowed evidence identifiers. Models can
        # still copy a plausible-looking citation into their JSON, so the
        # server removes those references before constructing the claim or
        # message. When evidence exists, retain strict validation: a mixed
        # known/unknown list must not be silently repaired.
        normalized_draft = draft.model_copy(
            update={"evidence_ids": [] if not evidence else list(draft.evidence_ids)}
        )
        missing = [ref for ref in normalized_draft.evidence_ids if ref not in evidence]
        if missing:
            raise InvariantViolation(f"unknown evidence reference(s): {len(missing)}")
        cross_agent = [
            ref
            for ref in normalized_draft.evidence_ids
            if evidence[ref].agent_id is not None and evidence[ref].agent_id != author_id
        ]
        if cross_agent:
            raise InvariantViolation("evidence reference belongs to another investigator")
        support_status, support_warning = derive_claim_support(
            ClaimType(normalized_draft.claim_type), normalized_draft.evidence_ids, evidence
        )
        claims.append(
            Claim(
                id=stable_claim_id(run_id, message_id, author_id, ordinal, normalized_draft),
                message_id=message_id,
                text=normalized_draft.text,
                claim_type=ClaimType(normalized_draft.claim_type),
                author_id=author_id,
                evidence_ids=list(normalized_draft.evidence_ids),
                support_status=support_status,
                support_warning=support_warning,
            )
        )
    return claims


def prepare_investigator_message(
    message: Message,
    output: InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput,
    *,
    run_id: str,
    evidence: Mapping[str, Evidence],
    messages: Mapping[str, Message] | None = None,
    known_claims: Mapping[str, Claim] | None = None,
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Create an event payload after all ownership and reference checks pass."""

    if actor_id is not None and actor_id != message.author_id:
        raise InvariantViolation("investigator message actor does not own the message")
    claims = claims_from_investigator_output(
        output,
        run_id=run_id,
        message_id=message.id,
        author_id=message.author_id,
        evidence=evidence,
    )
    claim_ids = [claim.id for claim in claims]
    if message.claim_ids and message.claim_ids != claim_ids:
        raise InvariantViolation("message claim_ids do not match investigator output")
    normalized_message = message.model_copy(
        update={
            "claim_ids": claim_ids,
            "evidence_ids": [] if not evidence else list(message.evidence_ids),
        }
    )
    validate_message_bundle(
        normalized_message,
        claims,
        messages or {},
        known_claims or {},
        evidence,
    )
    return {
        "message": normalized_message.model_dump(mode="json"),
        "claims": [claim.model_dump(mode="json") for claim in claims],
    }
