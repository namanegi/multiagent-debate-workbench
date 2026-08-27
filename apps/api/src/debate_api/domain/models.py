"""Small public models for a persisted multi-agent debate."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class RunPhase(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RESEARCHING = "researching"
    OPENING = "opening"
    DEBATING = "debating"
    SYNTHESIZING = "synthesizing"
    FINALIZING = "finalizing"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunStopReason(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EVENT_LIMIT = "event_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    AGENT_FAILURE = "agent_failure"
    SYNTHESIS_FAILURE = "synthesis_failure"
    ORCHESTRATION_ERROR = "orchestration_error"
    STORAGE_ERROR = "storage_error"


class RunEventType(StrEnum):
    RUN_CREATED = "run.created"
    PHASE_STARTED = "phase.started"
    PHASE_COMPLETED = "phase.completed"
    PLANNING_OUTCOME = "planning.outcome"
    BRIEF_CREATED = "brief.created"
    EVIDENCE_CREATED = "evidence.created"
    MESSAGE_CREATED = "message.created"
    DEBATE_PROTOCOL_OUTCOME_CREATED = "debate_protocol.outcome.created"
    SYNTHESIS_CREATED = "synthesis.created"
    AGENT_FAILED = "agent.failed"
    RUN_CANCEL_REQUESTED = "run.cancel_requested"
    RUN_CANCELLED = "run.cancelled"
    RUN_COMPLETED = "run.completed"
    RUN_PARTIAL = "run.partial"
    RUN_FAILED = "run.failed"


class MessageKind(StrEnum):
    OPENING = "opening"
    UPDATE = "update"
    SYNTHESIS = "synthesis"


class InteractionKind(StrEnum):
    CHALLENGE = "challenge"
    SUPPORT = "support"


class ToolPermission(StrEnum):
    SEARCH = "search"
    FETCH = "fetch"


class ClaimType(StrEnum):
    FACT = "fact"
    INFERENCE = "inference"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


PROVIDER_CLAIM_TYPES: tuple[ClaimType, ...] = (
    ClaimType.FACT,
    ClaimType.INFERENCE,
    ClaimType.RECOMMENDATION,
)
ProviderClaimType = Literal[ClaimType.FACT, ClaimType.INFERENCE, ClaimType.RECOMMENDATION]


class ClaimStatus(StrEnum):
    ACTIVE = "active"
    UNSUPPORTED = "unsupported"


class ClaimSupportStatus(StrEnum):
    UNASSESSED = "unassessed"
    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"


class EvidenceStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class DebateProtocolStatus(StrEnum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


DEBATE_PROTOCOL_STOP_REASONS: dict[DebateProtocolStatus, str] = {
    DebateProtocolStatus.COMPLETED: "All configured agent turns completed.",
    DebateProtocolStatus.INCOMPLETE: "One or more configured agent turns did not complete.",
}


class DebateProtocolOutcome(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    configured_turns: int = Field(ge=1, le=4)
    completed_turns: int = Field(ge=0, le=4)
    expected_agent_messages: int = Field(ge=1, le=28)
    completed_agent_messages: int = Field(ge=0, le=28)
    status: DebateProtocolStatus
    stop_reason: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.completed_turns > self.configured_turns:
            raise ValueError("completed turns cannot exceed configured turns")
        if self.completed_agent_messages > self.expected_agent_messages:
            raise ValueError("completed agent messages cannot exceed the configured grid")
        if self.stop_reason != DEBATE_PROTOCOL_STOP_REASONS[self.status]:
            raise ValueError("debate protocol stop reason is not authoritative")
        return self


class RunLimits(StrictModel):
    max_agents: int = Field(default=7, ge=1, le=7)
    max_turns: int = Field(default=4, ge=1, le=4)
    max_events: int = Field(default=200, ge=20, le=2_000)
    max_tool_calls: int = Field(default=80, ge=0, le=100)
    max_retrieval_calls: int = Field(default=7, ge=0, le=7)
    max_context_tokens: int = Field(default=6_144, ge=1_024, le=8_192)


MAX_TOPIC_CHARS = 4_000


class CreateRunRequest(StrictModel):
    topic: str = Field(min_length=1, max_length=MAX_TOPIC_CHARS)
    goal: str | None = Field(default=None, max_length=1_000)
    agent_count: int = Field(default=3, ge=1, le=7)
    turn_count: int = Field(default=2, ge=1, le=4)
    research_enabled: bool = False

    @field_validator("topic", "goal")
    @classmethod
    def trim_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("text must not be blank")
        return trimmed

    def validate_capacity(self) -> None:
        if not 1 <= self.agent_count <= 7 or not 1 <= self.turn_count <= 4:
            raise ValueError("agent-turn configuration is outside hard bounds")


class Run(StrictModel):
    id: str
    topic: str
    goal: str | None = None
    limits: RunLimits
    phase: RunPhase = RunPhase.CREATED
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    stop_reason: str | None = None
    stop_reason_code: RunStopReason | None = None
    agent_count: int = Field(default=3, ge=1, le=7)
    turn_count: int = Field(default=2, ge=1, le=4)
    research_enabled: bool = False
    idempotency_key: str | None = Field(default=None, exclude=True)


class PlanningOutcomeCategory(StrEnum):
    PROVIDER_SUCCESS = "provider_success"
    PROVIDER_REPAIRED = "provider_repaired"
    FALLBACK = "fallback"
    REUSED = "reused"


class SynthesisGeneration(StrEnum):
    DETERMINISTIC = "deterministic"
    PROVIDER_SUCCESS = "provider_success"
    PROVIDER_REPAIRED = "provider_repaired"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class PlanningOutcome(StrictModel):
    planner_id: str = Field(min_length=1, max_length=120)
    category: PlanningOutcomeCategory
    message: str = Field(min_length=1, max_length=300)
    repair_attempted: bool = False
    fallback_used: bool = False
    brief_count: int = Field(ge=1, le=7)
    failure_category: str | None = Field(default=None, max_length=80)


class PlanningOutcomeEventPayload(StrictModel):
    outcome: PlanningOutcome


class AgentBrief(StrictModel):
    id: str
    agent_id: str
    label: str = Field(min_length=1, max_length=80)
    focus: str = Field(min_length=1, max_length=500)
    key_questions: list[str] = Field(min_length=1, max_length=6)
    preferred_source_types: list[str] = Field(min_length=1, max_length=6)
    deliverable: str = Field(min_length=1, max_length=500)
    tool_permissions: list[ToolPermission] = Field(min_length=0, max_length=6)

    @field_validator("key_questions", "preferred_source_types")
    @classmethod
    def reject_blank_items(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("list items must not be blank")
        return normalized


class Evidence(StrictModel):
    id: str
    agent_id: str | None = None
    canonical_url: HttpUrl
    final_url: HttpUrl | None = None
    title: str = Field(min_length=1, max_length=300)
    publisher: str = Field(min_length=1, max_length=160)
    excerpt: str = Field(min_length=1, max_length=2_000)
    source_type: str = Field(default="web", min_length=1, max_length=80)
    status: EvidenceStatus = EvidenceStatus.AVAILABLE
    unavailable_reason: str | None = Field(default=None, max_length=300)
    fetch_status: str | None = Field(default=None, max_length=80)
    document_type: str | None = Field(default=None, max_length=40)
    extraction_warnings: list[str] = Field(default_factory=list, max_length=32)
    extractor_name: str | None = Field(default=None, max_length=120)
    extractor_version: str | None = Field(default=None, max_length=80)
    fetched_at: datetime | None = None
    content_hash: str | None = Field(default=None, max_length=128)

    @field_validator("extraction_warnings")
    @classmethod
    def bound_warnings(cls, values: list[str]) -> list[str]:
        return [value[:300] for value in values if value][:32]


class Claim(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    message_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=1_000)
    claim_type: ClaimType
    author_id: str = Field(min_length=1, max_length=120)
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    status: ClaimStatus = ClaimStatus.ACTIVE
    support_status: ClaimSupportStatus = ClaimSupportStatus.UNASSESSED
    support_warning: str | None = Field(default=None, max_length=300)

    @field_validator("claim_type")
    @classmethod
    def reject_unknown(cls, value: ClaimType) -> ClaimType:
        if value == ClaimType.UNKNOWN:
            raise ValueError("unknown claim types cannot be public")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence(cls, values: list[str]) -> list[str]:
        return _validate_reference_ids(values, "claim evidence references")


class InvestigatorClaimDraft(StrictModel):
    text: str = Field(min_length=1, max_length=1_000)
    claim_type: ProviderClaimType
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence(cls, values: list[str]) -> list[str]:
        return _validate_reference_ids(values, "claim evidence references")


class InvestigatorOpeningOutput(StrictModel):
    """Provider output for an independent first-turn answer."""

    claims: list[InvestigatorClaimDraft] = Field(min_length=1, max_length=10)


class InvestigatorDirectedUpdateOutput(StrictModel):
    """Provider output for a later-turn directed challenge or support."""

    claims: list[InvestigatorClaimDraft] = Field(min_length=1, max_length=10)
    # Provider-only selector. The orchestration layer maps this short option to
    # authoritative persisted claim/message IDs.
    target_option: int = Field(ge=0, le=69)
    interaction_kind: InteractionKind


def _validate_reference_ids(values: list[str], label: str) -> list[str]:
    if any(not item.strip() or len(item) > 120 for item in values):
        raise ValueError(f"{label} must be bounded and nonblank")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


class Message(StrictModel):
    id: str
    author_id: str
    author_label: str = Field(min_length=1, max_length=100)
    phase: RunPhase
    kind: MessageKind
    content: str = Field(min_length=1, max_length=5_000)
    claim_ids: list[str] = Field(default_factory=list, max_length=10)
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    in_reply_to_message_id: str | None = None
    target_claim_id: str | None = Field(default=None, max_length=120)
    target_agent_id: str | None = Field(default=None, max_length=120)
    interaction_kind: InteractionKind | None = None
    turn_index: int | None = Field(default=None, ge=1, le=4)

    @field_validator("claim_ids", "evidence_ids")
    @classmethod
    def unique_references(cls, values: list[str]) -> list[str]:
        return _validate_reference_ids(values, "message references")

    @model_validator(mode="after")
    def validate_interaction_shape(self) -> Self:
        interaction = (
            self.in_reply_to_message_id,
            self.target_claim_id,
            self.target_agent_id,
            self.interaction_kind,
        )
        if self.turn_index == 1 and any(item is not None for item in interaction):
            raise ValueError("turn 1 must be independent")
        if (
            self.turn_index is not None
            and self.turn_index > 1
            and any(item is not None for item in interaction)
            and any(item is None for item in interaction)
        ):
            raise ValueError("directed updates require complete interaction metadata")
        return self


class Failure(StrictModel):
    id: str
    agent_id: str | None = None
    phase: RunPhase
    category: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)
    recoverable: bool = True


class Synthesis(StrictModel):
    id: str
    message_id: str
    answer: str = Field(min_length=1, max_length=5_000)
    is_partial: bool = False
    generation_method: SynthesisGeneration = SynthesisGeneration.DETERMINISTIC
    completed_responsibilities: list[str] = Field(default_factory=list, max_length=20)
    missing_responsibilities: list[str] = Field(default_factory=list, max_length=20)
    stop_reason_code: RunStopReason | None = None
    consensus: list[str] = Field(default_factory=list, max_length=20)
    disagreements: list[str] = Field(default_factory=list, max_length=20)
    changed_positions: list[str] = Field(default_factory=list, max_length=20)
    evidence_gaps: list[str] = Field(default_factory=list, max_length=20)
    follow_up_checks: list[str] = Field(default_factory=list, max_length=20)
    claim_ids: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class RunEvent(StrictModel):
    id: str
    run_id: str
    sequence: int = Field(ge=1)
    type: RunEventType
    phase: RunPhase
    occurred_at: datetime = Field(default_factory=utc_now)
    actor_id: str | None = None
    schema_version: int = Field(default=1, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class RunSummary(StrictModel):
    run: Run
    briefs: list[AgentBrief] = Field(default_factory=list)
    planning_outcome: PlanningOutcome | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    debate_protocol: DebateProtocolOutcome | None = None
    failures: list[Failure] = Field(default_factory=list)
    synthesis: Synthesis | None = None


class EventPage(StrictModel):
    events: list[RunEvent]
    has_more: bool
    next_after_sequence: int | None = None


class CreateRunResponse(StrictModel):
    run: Run
    stream_url: str


class CancelRunResponse(StrictModel):
    run: Run


def run_from_request(request: CreateRunRequest, run_id: str | None = None) -> Run:
    request.validate_capacity()
    now = utc_now()
    return Run(
        id=run_id or new_id("run"),
        topic=request.topic,
        goal=request.goal,
        limits=RunLimits(),
        created_at=now,
        updated_at=now,
        agent_count=request.agent_count,
        turn_count=request.turn_count,
        research_enabled=request.research_enabled,
    )
