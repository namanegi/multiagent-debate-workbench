"""Provider-backed, bounded synthesis over persisted public artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field, field_validator

from debate_api.domain.models import (
    AgentBrief,
    Message,
    MessageKind,
    Run,
    RunPhase,
    RunStatus,
    RunStopReason,
    RunSummary,
    Synthesis,
    SynthesisGeneration,
)
from debate_api.orchestration.model_runner import StructuredGenerationRunner
from debate_api.orchestration.scheduler import (
    BoundedScheduler,
    CooperativeCancellation,
    RunLimitReached,
)
from debate_api.providers.model import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ProviderModel,
)

if TYPE_CHECKING:
    from debate_api.persistence.sqlite import EventStore

POST_PLANNING_PHASES = frozenset(
    {
        RunPhase.RESEARCHING,
        RunPhase.OPENING,
        RunPhase.DEBATING,
        RunPhase.SYNTHESIZING,
        RunPhase.FINALIZING,
    }
)
MAX_SYNTHESIS_INPUT_CHARS = 95_000
MAX_SYNTHESIS_SECTION_ITEM_CHARS = 500
MAX_SYNTHESIS_REFERENCES = 20
SYNTHESIS_REPAIR_SUFFIX = (
    "\nRepair once: return only a specific synthesis grounded in the persisted public artifact IDs."
)
MAX_INITIAL_SYNTHESIS_INPUT_CHARS = MAX_SYNTHESIS_INPUT_CHARS - len(SYNTHESIS_REPAIR_SUFFIX)


class GeneratedSynthesis(ProviderModel):
    """Provider-owned prose and references; server fields remain authoritative."""

    answer: str = Field(min_length=1, max_length=5_000)
    consensus: list[str] = Field(max_length=20)
    disagreements: list[str] = Field(max_length=20)
    changed_positions: list[str] = Field(max_length=20)
    evidence_gaps: list[str] = Field(max_length=20)
    follow_up_checks: list[str] = Field(max_length=20)
    claim_ids: list[str] = Field(max_length=20)
    evidence_ids: list[str] = Field(max_length=20)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("synthesis answer must not be blank")
        return value

    @field_validator(
        "consensus",
        "disagreements",
        "changed_positions",
        "evidence_gaps",
        "follow_up_checks",
    )
    @classmethod
    def trim_items(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > MAX_SYNTHESIS_SECTION_ITEM_CHARS for value in normalized):
            raise ValueError("synthesis list items must not be blank")
        return normalized


@dataclass(frozen=True)
class SynthesisResult:
    message: Message
    synthesis: Synthesis


@dataclass(frozen=True)
class SynthesisPublicInput:
    input_text: str
    claim_ids: frozenset[str]
    evidence_ids: frozenset[str]

    def __len__(self) -> int:
        return len(self.input_text)


class SynthesisOrchestrator:
    """Produce one replayable constrained result, or a deterministic safe fallback."""

    synthesizer_id = "constrained-synthesizer-v1"

    def __init__(
        self,
        provider: ModelProvider | None = None,
        store: EventStore | None = None,
    ) -> None:
        self._runner = StructuredGenerationRunner(provider) if provider is not None else None
        self._store = store

    async def synthesize_normal(
        self,
        run_id: str,
        stop_reason_code: RunStopReason | None = None,
        *,
        scheduler: BoundedScheduler | None = None,
    ) -> SynthesisResult:
        """Synthesize normal output from the store's authoritative projection only."""
        if self._store is None:
            raise ValueError("normal synthesis requires an event store")
        # Read both projections immediately before provider input construction.  The
        # caller cannot smuggle an unpersisted RunSummary into this path.
        run = self._store.get_run(run_id)
        summary = self._store.get_summary(run_id)
        return await self._synthesize_authoritative(
            run,
            summary,
            stop_reason_code,
            scheduler=scheduler,
            partial=False,
        )

    async def synthesize(
        self,
        run: Run,
        summary: RunSummary,
        stop_reason_code: RunStopReason | None,
        *,
        scheduler: BoundedScheduler | None = None,
        partial: bool = True,
    ) -> SynthesisResult:
        if not partial:
            raise ValueError("normal synthesis must use synthesize_normal(run_id)")
        return await self._synthesize_authoritative(
            run, summary, stop_reason_code, scheduler=scheduler, partial=partial
        )

    async def _synthesize_authoritative(
        self,
        run: Run,
        summary: RunSummary,
        stop_reason_code: RunStopReason | None,
        *,
        scheduler: BoundedScheduler | None = None,
        partial: bool = True,
    ) -> SynthesisResult:
        self._validate_context(run, summary, partial=partial)
        completed, missing = self._responsibilities(summary.briefs, summary)
        if summary.synthesis is not None:
            message = next(
                (
                    item
                    for item in summary.messages
                    if item.id == summary.synthesis.message_id
                    and item.kind == MessageKind.SYNTHESIS
                ),
                None,
            )
            if message is None:
                raise ValueError("persisted synthesis is missing its public message")
            return SynthesisResult(message=message, synthesis=summary.synthesis)

        if self._runner is None:
            return self._fallback(
                run, summary, stop_reason_code, completed, missing, partial=partial
            )
        if scheduler is not None:
            try:
                scheduler.check()
            except RunLimitReached:
                return self._fallback(
                    run, summary, stop_reason_code, completed, missing, partial=partial
                )
            if scheduler.remaining_tool_calls <= 0:
                return self._fallback(
                    run, summary, stop_reason_code, completed, missing, partial=partial
                )
            await scheduler.acquire_tool_call()
            remaining = scheduler.remaining_seconds()
        else:
            remaining = None

        public_input = self._public_input(run, summary, stop_reason_code, completed, missing)
        request = ModelRequest(
            request_id=f"synthesis_{run.id}",
            operation="partial_synthesis" if partial else "synthesis",
            input_text=public_input.input_text,
            output_schema_name=GeneratedSynthesis.__name__,
            timeout_seconds=self._request_timeout(remaining),
            max_output_tokens=4_096,
            repair_attempts=1,
        )
        repair_attempted = False
        try:
            response = await self._call(request, scheduler)
            try:
                return self._build(
                    run,
                    stop_reason_code,
                    completed,
                    missing,
                    response,
                    (
                        SynthesisGeneration.PROVIDER_REPAIRED
                        if response.repair_attempted
                        else SynthesisGeneration.PROVIDER_SUCCESS
                    ),
                    partial=partial,
                    allowed_claim_ids=public_input.claim_ids,
                    allowed_evidence_ids=public_input.evidence_ids,
                )
            except ValueError:
                if response.repair_attempted:
                    return self._fallback(
                        run, summary, stop_reason_code, completed, missing, partial=partial
                    )
                repair_attempted = True
                repaired_request = await self._repair_request(request, scheduler)
                repaired = await self._call(repaired_request, scheduler)
                try:
                    return self._build(
                        run,
                        stop_reason_code,
                        completed,
                        missing,
                        repaired,
                        SynthesisGeneration.PROVIDER_REPAIRED,
                        partial=partial,
                        allowed_claim_ids=public_input.claim_ids,
                        allowed_evidence_ids=public_input.evidence_ids,
                    )
                except ValueError:
                    return self._fallback(
                        run, summary, stop_reason_code, completed, missing, partial=partial
                    )
        except ModelProviderError as error:
            if error.category.value == "cancelled":
                raise CooperativeCancellation("provider synthesis was cancelled") from None
            if error.category.value in {"timeout", "provider_error"}:
                return self._fallback(
                    run, summary, stop_reason_code, completed, missing, partial=partial
                )
            if repair_attempted or error.repair_attempted:
                return self._fallback(
                    run, summary, stop_reason_code, completed, missing, partial=partial
                )
            try:
                repair_attempted = True
                repaired_request = await self._repair_request(request, scheduler)
                repaired = await self._call(repaired_request, scheduler)
                return self._build(
                    run,
                    stop_reason_code,
                    completed,
                    missing,
                    repaired,
                    SynthesisGeneration.PROVIDER_REPAIRED,
                    partial=partial,
                    allowed_claim_ids=public_input.claim_ids,
                    allowed_evidence_ids=public_input.evidence_ids,
                )
            except ModelProviderError as repair_error:
                if repair_error.category.value == "cancelled":
                    raise CooperativeCancellation("provider synthesis was cancelled") from None
                return self._fallback(
                    run, summary, stop_reason_code, completed, missing, partial=partial
                )
            except (RunLimitReached, ValueError):
                return self._fallback(
                    run, summary, stop_reason_code, completed, missing, partial=partial
                )
        except (RunLimitReached, ValueError):
            return self._fallback(
                run, summary, stop_reason_code, completed, missing, partial=partial
            )

    async def _call(
        self, request: ModelRequest, scheduler: BoundedScheduler | None
    ) -> ModelResponse[GeneratedSynthesis]:
        if self._runner is None:
            raise RuntimeError("synthesis has no model runner")
        runner = self._runner

        async def operation() -> ModelResponse[GeneratedSynthesis]:
            return await runner.run(request, GeneratedSynthesis)

        if scheduler is None:
            return await operation()
        return await scheduler.run_provider(operation)

    async def _repair_request(
        self, request: ModelRequest, scheduler: BoundedScheduler | None
    ) -> ModelRequest:
        timeout = self._request_timeout()
        if scheduler is not None:
            await scheduler.acquire_tool_call()
            remaining = scheduler.remaining_seconds()
            if remaining is not None:
                timeout = min(timeout, remaining)
        return request.model_copy(
            update={
                "input_text": request.input_text + SYNTHESIS_REPAIR_SUFFIX,
                "timeout_seconds": timeout,
                "repair_attempts": 0,
            }
        )

    def _request_timeout(self, remaining: float | None = None) -> float:
        if self._runner is None:
            raise RuntimeError("synthesis has no model runner")
        timeout = self._runner.request_timeout_seconds
        if remaining is not None:
            timeout = min(timeout, remaining)
        return timeout

    def _build(
        self,
        run: Run,
        stop_reason_code: RunStopReason | None,
        completed: list[str],
        missing: list[str],
        response: ModelResponse[GeneratedSynthesis],
        generation: SynthesisGeneration,
        *,
        partial: bool = True,
        allowed_claim_ids: frozenset[str] | None = None,
        allowed_evidence_ids: frozenset[str] | None = None,
    ) -> SynthesisResult:
        generated = response.output
        if allowed_claim_ids is None:
            allowed_claim_ids = frozenset()
        if allowed_evidence_ids is None:
            allowed_evidence_ids = frozenset()
        if not set(generated.claim_ids) <= allowed_claim_ids:
            raise ValueError("synthesis contains an unknown claim reference")
        if not set(generated.evidence_ids) <= allowed_evidence_ids:
            raise ValueError("synthesis contains an unknown evidence reference")
        if len(generated.claim_ids) != len(set(generated.claim_ids)):
            raise ValueError("synthesis contains duplicate claim references")
        if len(generated.evidence_ids) != len(set(generated.evidence_ids)):
            raise ValueError("synthesis contains duplicate evidence references")
        if not partial and any(
            not section or any(not item.strip() for item in section)
            for section in (
                generated.consensus,
                generated.disagreements,
                generated.changed_positions,
                generated.evidence_gaps,
                generated.follow_up_checks,
            )
        ):
            raise ValueError("normal synthesis sections must all be explicit")
        # The provider response can only be checked against IDs in the persisted
        # context by the caller's projection; the serialized request carries no
        # hidden/provider state, so the server-owned IDs are reconstructed here.
        message = Message(
            id=(
                f"message_{run.id}_partial_synthesis" if partial else f"message_{run.id}_synthesis"
            ),
            author_id="synthesizer",
            author_label="Partial synthesis" if partial else "Synthesis",
            phase=RunPhase(run.phase),
            kind=MessageKind.SYNTHESIS,
            content=generated.answer,
        )
        synthesis = Synthesis(
            id=f"synthesis_{run.id}_partial" if partial else f"synthesis_{run.id}_synthesis",
            message_id=message.id,
            answer=generated.answer,
            is_partial=partial,
            generation_method=generation,
            completed_responsibilities=completed,
            missing_responsibilities=missing,
            stop_reason_code=stop_reason_code if partial else None,
            consensus=generated.consensus,
            disagreements=generated.disagreements,
            changed_positions=generated.changed_positions,
            evidence_gaps=generated.evidence_gaps,
            follow_up_checks=generated.follow_up_checks,
            claim_ids=generated.claim_ids,
            evidence_ids=generated.evidence_ids,
        )
        return SynthesisResult(message=message, synthesis=synthesis)

    def _fallback(
        self,
        run: Run,
        summary: RunSummary,
        stop_reason_code: RunStopReason | None,
        completed: list[str],
        missing: list[str],
        *,
        partial: bool = True,
    ) -> SynthesisResult:
        all_claim_ids = [claim.id for claim in summary.claims]
        all_evidence_ids = [evidence.id for evidence in summary.evidence]
        claim_ids = all_claim_ids[:MAX_SYNTHESIS_REFERENCES]
        evidence_ids = all_evidence_ids[:MAX_SYNTHESIS_REFERENCES]
        omitted_references = (len(all_claim_ids) - len(claim_ids)) + (
            len(all_evidence_ids) - len(evidence_ids)
        )
        bounded_completed = [
            item.strip()[:MAX_SYNTHESIS_SECTION_ITEM_CHARS]
            for item in completed[:MAX_SYNTHESIS_REFERENCES]
            if item.strip()
        ]
        bounded_missing = [
            item.strip()[:MAX_SYNTHESIS_SECTION_ITEM_CHARS]
            for item in missing[:MAX_SYNTHESIS_REFERENCES]
            if item.strip()
        ]
        bounded_failures = [
            failure.message.strip()[:MAX_SYNTHESIS_SECTION_ITEM_CHARS]
            for failure in summary.failures[:MAX_SYNTHESIS_REFERENCES]
            if failure.message.strip()
        ]
        answer = (
            "Deterministic fallback partial synthesis from persisted public artifacts: "
            if partial
            else "Deterministic synthesis from persisted public artifacts: "
        ) + (
            f"{len(all_claim_ids)} claims and {len(all_evidence_ids)} evidence items were "
            "retained. "
            f"{len(completed)} investigator responsibilities produced visible artifacts; "
            f"{len(missing)} remain incomplete."
        )
        message = Message(
            id=(
                f"message_{run.id}_partial_synthesis" if partial else f"message_{run.id}_synthesis"
            ),
            author_id="synthesizer",
            author_label="Partial synthesis" if partial else "Synthesis",
            phase=RunPhase(run.phase),
            kind=MessageKind.SYNTHESIS,
            content=answer,
        )
        synthesis = Synthesis(
            id=f"synthesis_{run.id}_partial" if partial else f"synthesis_{run.id}_synthesis",
            message_id=message.id,
            answer=answer,
            is_partial=partial,
            generation_method=SynthesisGeneration.DETERMINISTIC_FALLBACK,
            completed_responsibilities=bounded_completed,
            missing_responsibilities=bounded_missing,
            stop_reason_code=stop_reason_code if partial else None,
            consensus=(
                []
                if partial
                else ["The constrained synthesis retained only persisted public artifacts."]
            ),
            disagreements=(
                [] if partial else ["No semantic truth determination was made by the fallback."]
            ),
            changed_positions=(
                [] if partial else ["No model-authored position change was persisted."]
            ),
            evidence_gaps=(
                [
                    *bounded_missing,
                    *bounded_failures,
                    *(
                        ["Some references were bounded out of the fallback projection."]
                        if omitted_references
                        else []
                    ),
                ][:MAX_SYNTHESIS_REFERENCES]
                or (["No additional evidence gap was recorded."] if not partial else [])
            ),
            follow_up_checks=(
                bounded_missing
                or (
                    ["Review the persisted evidence and unresolved positions."]
                    if not partial
                    else []
                )
            ),
            claim_ids=claim_ids,
            evidence_ids=evidence_ids,
        )
        return SynthesisResult(message=message, synthesis=synthesis)

    @staticmethod
    def _validate_context(run: Run, summary: RunSummary, *, partial: bool) -> None:
        if summary.run.model_dump(mode="json") != run.model_dump(mode="json"):
            raise ValueError("synthesis run and summary authoritative state do not match")
        if RunPhase(run.phase) not in POST_PLANNING_PHASES:
            raise ValueError("synthesis requires a post-planning active phase")
        if summary.planning_outcome is None:
            raise ValueError("synthesis requires an authoritative plan outcome")
        if len(summary.briefs) != run.agent_count:
            raise ValueError("synthesis requires a complete persisted plan")
        if not partial and summary.debate_protocol is None:
            raise ValueError("normal synthesis requires an authoritative debate outcome")
        if not partial and (
            RunStatus(run.status) != RunStatus.RUNNING
            or RunPhase(run.phase) != RunPhase.SYNTHESIZING
        ):
            raise ValueError("normal synthesis requires an active synthesizing run")

    @staticmethod
    def _responsibilities(
        briefs: list[AgentBrief], summary: RunSummary
    ) -> tuple[list[str], list[str]]:
        completed: list[str] = []
        missing: list[str] = []
        for brief in briefs:
            has_public_work = (
                any(message.author_id == brief.agent_id for message in summary.messages)
                or any(claim.author_id == brief.agent_id for claim in summary.claims)
                or any(evidence.agent_id == brief.agent_id for evidence in summary.evidence)
            )
            responsibility = f"{brief.label}: {brief.deliverable}"
            (completed if has_public_work else missing).append(responsibility)
        return completed, missing

    @staticmethod
    def _public_input(
        run: Run,
        summary: RunSummary,
        stop_reason_code: RunStopReason | None,
        completed: list[str],
        missing: list[str],
    ) -> SynthesisPublicInput:
        assert summary.planning_outcome is not None
        public_artifacts: dict[str, Any] = {
            "topic": run.topic,
            "goal": run.goal,
            "planning_outcome": summary.planning_outcome.model_dump(mode="json"),
            "briefs": [brief.model_dump(mode="json") for brief in summary.briefs],
            "evidence": [
                {
                    **item.model_dump(mode="json"),
                    "excerpt": item.excerpt[:500],
                }
                for item in summary.evidence[:100]
            ],
            "messages": [
                {**item.model_dump(mode="json"), "content": item.content[:800]}
                for item in summary.messages[:100]
            ],
            "claims": [
                {**item.model_dump(mode="json"), "text": item.text[:500]}
                for item in summary.claims[:100]
            ],
            "failures": [
                {**item.model_dump(mode="json"), "message": item.message[:300]}
                for item in summary.failures[:100]
            ],
            "debate_protocol": (
                summary.debate_protocol.model_dump(mode="json")
                if summary.debate_protocol is not None
                else None
            ),
            "completed_responsibilities": completed,
            "missing_responsibilities": missing,
            "stop_reason_code": stop_reason_code.value if stop_reason_code is not None else None,
        }
        serialized = (
            "Synthesize only these persisted public artifacts. Do not infer absent work, "
            "invent consensus, or add references outside the supplied IDs.\n"
            + json.dumps(public_artifacts, ensure_ascii=False, sort_keys=True)
        )
        if len(serialized) <= MAX_INITIAL_SYNTHESIS_INPUT_CHARS:
            return SynthesisPublicInput(
                input_text=serialized,
                claim_ids=frozenset(item["id"] for item in public_artifacts["claims"]),
                evidence_ids=frozenset(item["id"] for item in public_artifacts["evidence"]),
            )
        compact = {
            **public_artifacts,
            "evidence": [
                {"id": item["id"], "agent_id": item.get("agent_id"), "title": item["title"]}
                for item in public_artifacts["evidence"]
            ],
            "messages": [
                {
                    "id": item["id"],
                    "author_id": item["author_id"],
                    "phase": item["phase"],
                    "kind": item["kind"],
                    "claim_ids": item["claim_ids"],
                    "evidence_ids": item["evidence_ids"],
                }
                for item in public_artifacts["messages"]
            ],
            "claims": [
                {
                    "id": item["id"],
                    "message_id": item["message_id"],
                    "author_id": item["author_id"],
                    "evidence_ids": item["evidence_ids"],
                }
                for item in public_artifacts["claims"]
            ],
        }
        serialized = "Synthesize only these persisted public artifacts.\n" + json.dumps(
            compact, ensure_ascii=False, sort_keys=True
        )
        if len(serialized) <= MAX_INITIAL_SYNTHESIS_INPUT_CHARS:
            return SynthesisPublicInput(
                input_text=serialized,
                claim_ids=frozenset(item["id"] for item in compact["claims"]),
                evidence_ids=frozenset(item["id"] for item in compact["evidence"]),
            )
        # The final projection is intentionally ID-only and therefore bounded by
        # the run's event/item limits, independent of model-authored text size.
        minimal = {
            "briefs": [brief.model_dump(mode="json") for brief in summary.briefs],
            "evidence_ids": [item.id for item in summary.evidence],
            "message_ids": [item.id for item in summary.messages],
            "claim_ids": [item.id for item in summary.claims],
            "debate_protocol": (
                summary.debate_protocol.model_dump(mode="json")
                if summary.debate_protocol is not None
                else None
            ),
            "failure_categories": [item.category for item in summary.failures],
            "completed_responsibilities": completed,
            "missing_responsibilities": missing,
            "stop_reason_code": stop_reason_code.value if stop_reason_code is not None else None,
        }
        minimal_text = "Synthesize only these persisted public artifact IDs.\n" + json.dumps(
            minimal, ensure_ascii=False, sort_keys=True
        )
        if len(minimal_text) <= MAX_INITIAL_SYNTHESIS_INPUT_CHARS:
            return SynthesisPublicInput(
                input_text=minimal_text,
                claim_ids=frozenset(cast(list[str], minimal["claim_ids"])),
                evidence_ids=frozenset(cast(list[str], minimal["evidence_ids"])),
            )
        counts = {
            "artifact_counts": {
                "briefs": len(summary.briefs),
                "evidence": len(summary.evidence),
                "messages": len(summary.messages),
                "claims": len(summary.claims),
                "failures": len(summary.failures),
            },
            "completed_responsibilities": completed[:20],
            "missing_responsibilities": missing[:20],
            "stop_reason_code": stop_reason_code.value if stop_reason_code is not None else None,
        }
        return SynthesisPublicInput(
            input_text="Synthesize only these bounded public artifact counts.\n"
            + json.dumps(counts, ensure_ascii=False, sort_keys=True),
            claim_ids=frozenset(),
            evidence_ids=frozenset(),
        )
