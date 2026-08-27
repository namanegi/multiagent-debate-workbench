"""Provider-backed per-agent answer generation for the explicit debate loop.

The provider sees only the bounded public projection needed for one investigator.
Message text is server-derived from the validated claims; it is never a second
provider-owned narrative field.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import HttpUrl

from debate_api.domain.claims import prepare_investigator_message
from debate_api.domain.models import (
    MAX_TOPIC_CHARS,
    PROVIDER_CLAIM_TYPES,
    AgentBrief,
    Claim,
    ClaimSupportStatus,
    ClaimType,
    Evidence,
    EvidenceStatus,
    Failure,
    InteractionKind,
    InvestigatorDirectedUpdateOutput,
    InvestigatorOpeningOutput,
    Message,
    MessageKind,
    Run,
    RunEventType,
    RunPhase,
    RunStatus,
    ToolPermission,
    new_id,
)
from debate_api.domain.validation import InvariantViolation
from debate_api.orchestration.model_runner import StructuredGenerationRunner
from debate_api.orchestration.scheduler import (
    BoundedScheduler,
    CooperativeCancellation,
)
from debate_api.providers.model import (
    ModelChatMessage,
    ModelErrorCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelTextResponse,
)
from debate_api.research import ResearchResultStatus, ResearchService

if TYPE_CHECKING:
    from debate_api.persistence.sqlite import EventStore


MAX_OPENING_INPUT_CHARS = 16_000
MAX_OPENING_EVIDENCE = 20
MAX_OPENING_EXCERPT_CHARS = 500
MAX_OPENING_TITLE_CHARS = 240
MAX_OPENING_MESSAGE_CHARS = 5_000
MAX_OPENING_TARGETS = 70
_PROVIDER_CLAIM_TYPE_TEXT = ", ".join(claim_type.value for claim_type in PROVIDER_CLAIM_TYPES)
_OPENING_CONTRACT_REQUIREMENTS = (
    "Return at least one claim.",
    "Use only the listed evidence IDs in evidence_ids.",
    f"Use claim_type values {_PROVIDER_CLAIM_TYPE_TEXT}; never unknown.",
    "Do not include target_option or interaction_kind.",
)
_DIRECTED_UPDATE_CONTRACT_REQUIREMENTS = (
    "Return at least one claim.",
    "Use only the listed evidence IDs in evidence_ids.",
    f"Use claim_type values {_PROVIDER_CLAIM_TYPE_TEXT}; never unknown.",
    "Select exactly one listed target_option.",
    "Use interaction_kind challenge or support.",
)
_PAPER_FINAL_ANSWER_MARKER = re.compile(r"final answer\s*:", re.IGNORECASE)
_CONCLUSION_MARKER = re.compile(r"(?:final answer|conclusion)\s*:", re.IGNORECASE)
_PAPER_FINAL_ANSWER_END = re.compile(
    r"final answer\s*:\s*[+-]?(?:\d[\d,]*)(?:\.\d+)?\s*\.?\s*$",
    re.IGNORECASE,
)
_PAPER_LOOSE_FINAL_ANSWER = re.compile(
    r"final answer(?:\s+is)?\s*:?\s*[$€£]?\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?)",
    re.IGNORECASE,
)
_PAPER_ANY_NUMBER = re.compile(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?")
_CONCLUSION_TAIL = re.compile(
    r"(?:final answer|conclusion)(?:\s+is)?\s*:?.*$", re.IGNORECASE | re.DOTALL
)
ProtocolMode = Literal["default", "paper_reproduction"]
OutputMode = Literal["structured_json", "plain_text"]
ThinkingMode = Literal["provider_default", "disabled"]


@dataclass(frozen=True)
class AgentTurnResult:
    message: Message
    claims: tuple[Claim, ...]
    repair_attempted: bool = False


class Debater:
    """One assigned agent role that researches and answers within a debate."""

    def __init__(
        self,
        store: EventStore,
        brief: AgentBrief,
        provider: ModelProvider | None,
        research_service: ResearchService | None = None,
        protocol_mode: ProtocolMode = "default",
        output_mode: OutputMode = "structured_json",
        plain_text_max_output_tokens: int = 2_048,
        thinking_mode: ThinkingMode = "provider_default",
    ) -> None:
        self.store = store
        self.brief = brief
        self.provider = provider
        self.research_service = research_service
        if protocol_mode not in {"default", "paper_reproduction"}:
            raise ValueError("unsupported protocol_mode")
        if output_mode not in {"structured_json", "plain_text"}:
            raise ValueError("unsupported output_mode")
        if output_mode == "plain_text" and protocol_mode != "paper_reproduction":
            raise ValueError("plain_text output requires paper_reproduction protocol_mode")
        if (
            isinstance(plain_text_max_output_tokens, bool)
            or not 1 <= plain_text_max_output_tokens <= 32_768
        ):
            raise ValueError("plain_text_max_output_tokens must be between 1 and 32768")
        if thinking_mode not in {"provider_default", "disabled"}:
            raise ValueError("unsupported thinking_mode")
        if thinking_mode != "provider_default" and output_mode != "plain_text":
            raise ValueError("thinking_mode overrides require plain_text output")
        self.protocol_mode = protocol_mode
        self.output_mode = output_mode
        self.plain_text_max_output_tokens = plain_text_max_output_tokens
        self.thinking_mode = thinking_mode
        self.runner = StructuredGenerationRunner(provider) if provider is not None else None

    async def initial_answer(
        self,
        run_id: str,
        *,
        evidence: Evidence | None = None,
        scheduler: BoundedScheduler | None = None,
    ) -> AgentTurnResult:
        """Produce this role's independent first-turn answer."""

        if self.provider is None:
            return await self._deterministic_answer(run_id, 1, [], evidence, scheduler)
        if self.output_mode == "plain_text":
            return await self._generate_plain_turn(run_id, 1, [], scheduler=scheduler)
        return await self._generate_turn(run_id, 1, [], scheduler=scheduler)

    async def directed_update(
        self,
        run_id: str,
        turn_index: int,
        previous_answers: list[Message],
        *,
        evidence: Evidence | None = None,
        scheduler: BoundedScheduler | None = None,
    ) -> AgentTurnResult:
        """Review every prior answer and explicitly challenge or support one claim."""

        if turn_index <= 1:
            raise InvariantViolation("directed updates begin after the initial answer")
        if self.provider is None:
            return await self._deterministic_answer(
                run_id, turn_index, previous_answers, evidence, scheduler
            )
        if self.output_mode == "plain_text":
            return await self._generate_plain_turn(
                run_id, turn_index, previous_answers, scheduler=scheduler
            )
        return await self._generate_turn(
            run_id, turn_index, previous_answers, scheduler=scheduler
        )

    async def research(
        self,
        run: Run,
        scheduler: BoundedScheduler,
        index: int,
    ) -> tuple[Evidence | None, Failure | None]:
        """Run this role's permitted research, or produce deterministic evidence."""

        await scheduler.acquire_tool_call()
        scheduler.check()
        if self.research_service is not None:
            if ToolPermission.SEARCH not in self.brief.tool_permissions:
                return None, self._research_failure("agent_search_not_permitted")
            result = await self.research_service.search_and_fetch(
                run.id, self.brief.agent_id, self._research_query(run)
            )
            if result.status == ResearchResultStatus.SEARCH_FAILED:
                return None, self._research_failure("agent_search_failed")
            available = [
                item
                for item in result.evidence
                if item.status == EvidenceStatus.AVAILABLE
                and item.agent_id == self.brief.agent_id
            ]
            if available:
                return available[0], None
            # Research is optional grounding. A completed search with no usable
            # sources, or an exhausted budget, still leaves the debater able to
            # produce an ungrounded answer. Only provider/search failure is a
            # research failure visible to orchestration.
            return None, None
        evidence = self._evidence(run.topic, index)
        # Deterministic research has no ResearchService to own persistence.
        # Keep this boundary symmetric with the provider-backed path: the
        # debater persists the evidence exactly once and returns it only as
        # answer context.  The orchestrator must not emit it again.
        self.store.append_event(
            run.id,
            RunEventType.EVIDENCE_CREATED,
            RunPhase.RESEARCHING,
            {"evidence": evidence.model_dump(mode="json")},
            actor_id=self.brief.agent_id,
        )
        return evidence, None

    async def _deterministic_answer(
        self,
        run_id: str,
        turn_index: int,
        previous_answers: list[Message],
        evidence: Evidence | None,
        scheduler: BoundedScheduler | None = None,
    ) -> AgentTurnResult:
        if scheduler is not None:
            scheduler.check()
        if self.store.is_cancel_requested(run_id):
            raise CooperativeCancellation("debater answer was cancelled")
        run = self.store.get_run(run_id)
        target = next(
            (item for item in previous_answers if item.author_id != self.brief.agent_id),
            None,
        )
        if target is None and previous_answers:
            target = previous_answers[0]
        message = Message(
            id=f"message_{run.id}_{self.brief.agent_id}_turn_{turn_index}",
            author_id=self.brief.agent_id,
            author_label=self.brief.label,
            phase=RunPhase.OPENING if turn_index == 1 else RunPhase.DEBATING,
            kind=MessageKind.OPENING if turn_index == 1 else MessageKind.UPDATE,
            content=(
                f"{self.brief.label} initial answer for {run.topic}."
                if turn_index == 1
                else (
                    f"{self.brief.label} turn {turn_index} update after reviewing "
                    "all prior answers."
                )
            ),
            claim_ids=[f"claim_{run.id}_{self.brief.agent_id}_turn_{turn_index}"],
            evidence_ids=[evidence.id] if evidence is not None else [],
            in_reply_to_message_id=target.id if target is not None else None,
            target_agent_id=target.author_id if target is not None else None,
            target_claim_id=(
                target.claim_ids[0] if target is not None and target.claim_ids else None
            ),
            interaction_kind=(InteractionKind.CHALLENGE if target is not None else None),
            turn_index=turn_index,
        )
        claim = Claim(
            id=message.claim_ids[0],
            message_id=message.id,
            text=(
                "The initial answer follows the assigned role."
                if turn_index == 1
                else "The updated answer accounts for all previous-turn outputs."
            ),
            claim_type=ClaimType.INFERENCE,
            author_id=self.brief.agent_id,
            evidence_ids=message.evidence_ids,
            support_status=(
                ClaimSupportStatus.AVAILABLE
                if evidence is not None
                else ClaimSupportStatus.UNASSESSED
            ),
        )
        self.store.append_event(
            run_id,
            RunEventType.MESSAGE_CREATED,
            message.phase,
            {
                "message": message.model_dump(mode="json"),
                "claims": [claim.model_dump(mode="json")],
            },
            actor_id=self.brief.agent_id,
        )
        return AgentTurnResult(message, (claim,))

    def _research_query(self, run: Run) -> str:
        question = self.brief.key_questions[0] if self.brief.key_questions else self.brief.focus
        return " ".join(f"{run.topic} {self.brief.focus} {question}".split())[:500]

    def _research_failure(self, category: str) -> Failure:
        return Failure(
            id=new_id("failure"),
            agent_id=self.brief.agent_id,
            phase=RunPhase.RESEARCHING,
            category=category,
            message=f"{self.brief.label} did not produce usable research evidence.",
        )

    def _evidence(self, topic: str, index: int) -> Evidence:
        return Evidence(
            id=new_id("evidence"),
            agent_id=self.brief.agent_id,
            canonical_url=cast(HttpUrl, f"https://open-debate.local/source-{index + 1}"),
            title=f"Deterministic source {index + 1} for {topic}",
            publisher="Open Debate corpus",
            excerpt="A bounded source supports an inspectable answer.",
            source_type=self.brief.preferred_source_types[0],
        )

    async def _generate_plain_turn(
        self,
        run_id: str,
        turn_index: int,
        previous_answers: list[Message],
        *,
        scheduler: BoundedScheduler | None = None,
    ) -> AgentTurnResult:
        """Generate a paper-style answer without JSON, claims, or directed targets."""

        run = self.store.get_run(run_id)
        summary = self.store.get_summary(run_id)
        existing = next(
            (
                message
                for message in summary.messages
                if message.author_id == self.brief.agent_id
                and message.turn_index == turn_index
            ),
            None,
        )
        if existing is not None:
            claims = tuple(
                claim for claim in summary.claims if claim.message_id == existing.id
            )
            return AgentTurnResult(message=existing, claims=claims)

        expected_phase = RunPhase.OPENING if turn_index == 1 else RunPhase.DEBATING
        if not 1 <= turn_index <= run.turn_count:
            raise InvariantViolation("agent turn is outside the configured turn grid")
        if RunStatus(run.status) != RunStatus.RUNNING or RunPhase(run.phase) != expected_phase:
            raise InvariantViolation("agent answer requires its active turn phase")
        if turn_index == 1 and previous_answers:
            raise InvariantViolation("turn 1 must not receive peer answers")
        if turn_index > 1:
            expected_authors = {item.agent_id for item in summary.briefs}
            previous_authors = {item.author_id for item in previous_answers}
            if previous_authors != expected_authors or any(
                item.turn_index != turn_index - 1 for item in previous_answers
            ):
                raise InvariantViolation("later turns require every previous-turn answer")

        bounded_scheduler = scheduler or BoundedScheduler(run, self.store)
        conversation = self._paper_conversation(
            run.topic,
            self.brief,
            turn_index=turn_index,
            messages=summary.messages,
            goal=run.goal,
            thinking_mode=self.thinking_mode,
        )
        input_text = conversation[-1].content
        request = ModelRequest(
            request_id=(
                f"agent_turn_{run_id}_{self.brief.agent_id}_{turn_index}_plain"
            ),
            operation=(
                "agent.answer.initial" if turn_index == 1 else "agent.answer.update"
            ),
            input_text=input_text,
            conversation=conversation,
            output_schema_name=None,
            timeout_seconds=self._timeout(bounded_scheduler),
            max_output_tokens=self.plain_text_max_output_tokens,
            repair_attempts=0,
        )
        try:
            response = await self._call_text(request, bounded_scheduler)
        except ModelProviderError as error:
            if error.category == ModelErrorCategory.CANCELLED:
                raise CooperativeCancellation(
                    "provider investigator answer was cancelled"
                ) from None
            raise
        message_id = self._stable_message_id(run_id, self.brief.agent_id, turn_index)
        content = response.output.strip()
        if len(content) > MAX_OPENING_MESSAGE_CHARS:
            content = content[: MAX_OPENING_MESSAGE_CHARS - 1].rstrip() + "…"
        message = Message(
            id=message_id,
            author_id=self.brief.agent_id,
            author_label=self.brief.label,
            phase=expected_phase,
            kind=MessageKind.OPENING if turn_index == 1 else MessageKind.UPDATE,
            content=content,
            turn_index=turn_index,
        )
        payload: dict[str, object] = {
            "message": message.model_dump(mode="json"),
            "claims": [],
        }
        try:
            self.store.append_event(
                run_id,
                RunEventType.MESSAGE_CREATED,
                expected_phase,
                payload,
                actor_id=self.brief.agent_id,
            )
        except InvariantViolation:
            replayed = self.store.get_summary(run_id)
            persisted = next(
                (item for item in replayed.messages if item.id == message_id), None
            )
            if persisted is None:
                raise
            return AgentTurnResult(persisted, ())
        return AgentTurnResult(message, ())

    @staticmethod
    def _paper_conversation(
        topic: str,
        brief: AgentBrief,
        *,
        turn_index: int,
        messages: list[Message],
        goal: str | None,
        thinking_mode: ThinkingMode = "provider_default",
    ) -> tuple[ModelChatMessage, ...]:
        """Rebuild one agent's role-preserving paper conversation."""

        conversation: list[ModelChatMessage] = []
        for current_turn in range(1, turn_index + 1):
            peer_answers = sorted(
                (
                    message
                    for message in messages
                    if current_turn > 1
                    and message.turn_index == current_turn - 1
                    and message.author_id.startswith("agent_")
                    and message.author_id != brief.agent_id
                ),
                key=lambda message: message.author_id,
            )
            conversation.append(
                ModelChatMessage(
                    role="user",
                    content=Debater._paper_prompt(
                        topic,
                        brief,
                        goal=goal,
                        turn_index=current_turn,
                        peer_answers=peer_answers,
                        thinking_mode=thinking_mode,
                    ),
                )
            )
            if current_turn == turn_index:
                continue
            own_answer = next(
                (
                    message
                    for message in messages
                    if message.turn_index == current_turn
                    and message.author_id == brief.agent_id
                ),
                None,
            )
            if own_answer is None:
                raise InvariantViolation("paper conversation is missing the agent's prior answer")
            conversation.append(
                ModelChatMessage(role="assistant", content=own_answer.content)
            )
        return tuple(conversation)

    @staticmethod
    def _paper_prompt(
        topic: str,
        brief: AgentBrief,
        *,
        goal: str | None,
        turn_index: int,
        peer_answers: list[Message],
        thinking_mode: ThinkingMode = "provider_default",
    ) -> str:
        """Render one phase-specific natural-language paper prompt."""

        control_suffix = "\n\n/no_think" if thinking_mode == "disabled" else ""
        role_context = (
            f"Your role is {brief.label}.\n"
            f"Your focus is: {brief.focus}\n"
            f"Your deliverable is: {brief.deliverable}\n"
        )
        if goal:
            role_context += f"The run goal is: {goal}\n"
        final_contract = (
            "Explain your reasoning and check every arithmetic operation. Give exactly one "
            "conclusion, ending your response with 'Final answer: <number>'. Put no number, "
            "unit, or text after that final answer."
        )
        if turn_index == 1:
            return (
                f"{role_context}\n"
                f"Can you solve the following math problem independently?\n\n{topic}\n\n"
                f"{final_contract}{control_suffix}"
            )
        peer_sections = "\n\n".join(
            f"One other agent solution:\n```\n{message.content}\n```"
            for message in peer_answers
        )
        return (
            f"{role_context}\n"
            f"The original math problem is:\n\n{topic}\n\n"
            f"These are the recent solutions from other agents:\n\n{peer_sections}\n\n"
            "Use these solutions carefully as additional advice. Respond to any arithmetic "
            "error you find, but do not copy an answer or follow a majority without checking "
            "the reasoning yourself. Change your answer only when verification warrants it.\n\n"
            f"{final_contract}{control_suffix}"
        )

    async def _generate_turn(
        self,
        run_id: str,
        turn_index: int,
        previous_answers: list[Message],
        *,
        scheduler: BoundedScheduler | None = None,
    ) -> AgentTurnResult:
        """Generate one replayable answer after validating the turn boundary."""

        run = self.store.get_run(run_id)
        agent_id = self.brief.agent_id
        summary = self.store.get_summary(run_id)
        existing = next(
            (
                message
                for message in summary.messages
                if message.author_id == agent_id and message.turn_index == turn_index
            ),
            None,
        )
        if existing is not None:
            claims = tuple(claim for claim in summary.claims if claim.message_id == existing.id)
            if not claims:
                raise InvariantViolation("persisted investigator opening has no claims")
            return AgentTurnResult(message=existing, claims=claims)

        expected_phase = RunPhase.OPENING if turn_index == 1 else RunPhase.DEBATING
        if not 1 <= turn_index <= run.turn_count:
            raise InvariantViolation("agent turn is outside the configured turn grid")
        if RunStatus(run.status) != RunStatus.RUNNING or RunPhase(run.phase) != expected_phase:
            raise InvariantViolation("agent answer requires its active turn phase")
        if turn_index == 1 and previous_answers:
            raise InvariantViolation("turn 1 must not receive peer answers")
        if turn_index > 1:
            expected_authors = {item.agent_id for item in summary.briefs}
            previous_authors = {item.author_id for item in previous_answers}
            if previous_authors != expected_authors or any(
                item.turn_index != turn_index - 1 for item in previous_answers
            ):
                raise InvariantViolation("later turns require every previous-turn answer")
        bounded_scheduler = scheduler or BoundedScheduler(run, self.store)
        brief = self.brief
        # Evidence attributed to another investigator is deliberately not passed
        # to the model or the domain conversion path.
        own_evidence = dict(
            sorted(
                (
                    item.id,
                    item,
                )
                for item in summary.evidence
                if item.agent_id == agent_id
            )[:MAX_OPENING_EVIDENCE]
        )
        input_text = self._input_text(
            run.topic,
            brief,
            own_evidence,
            turn_index=turn_index,
            previous_answers=previous_answers,
            previous_claims={item.id: item for item in summary.claims},
            goal=run.goal,
            protocol_mode=self.protocol_mode,
        )
        message_id = self._stable_message_id(run_id, agent_id, turn_index)
        messages = {item.id: item for item in summary.messages}
        known_claims = {item.id: item for item in summary.claims}
        output_schema = (
            InvestigatorOpeningOutput if turn_index == 1 else InvestigatorDirectedUpdateOutput
        )
        request = ModelRequest(
            request_id=f"agent_turn_{run_id}_{agent_id}_{turn_index}",
            operation="agent.answer.initial" if turn_index == 1 else "agent.answer.update",
            input_text=input_text,
            output_schema_name=output_schema.__name__,
            timeout_seconds=self._timeout(bounded_scheduler),
            max_output_tokens=2_048,
            repair_attempts=1,
        )

        repair_attempted = False
        try:
            try:
                response = await self._call(request, output_schema, bounded_scheduler)
                repair_attempted = response.repair_attempted
            except ModelProviderError as error:
                recoverable_provider_error = (
                    error.category != ModelErrorCategory.CANCELLED
                    and (
                        error.category
                        in {
                            ModelErrorCategory.MALFORMED_OUTPUT,
                            ModelErrorCategory.SCHEMA_VALIDATION,
                        }
                        or error.retryable
                    )
                )
                if not recoverable_provider_error or error.repair_attempted:
                    raise
                repair_attempted = True
                response = await self._call(
                    request.model_copy(
                        update={
                            "input_text": input_text
                            + "\n"
                            + self._repair_instruction(
                                turn_index,
                                brief,
                                previous_answers,
                                {item.id: item for item in summary.claims},
                                self.protocol_mode,
                            ),
                            "repair_attempts": 0,
                            "timeout_seconds": self._timeout(bounded_scheduler),
                        }
                    ),
                    output_schema,
                    bounded_scheduler,
                )
            try:
                payload = self._build(
                    run_id,
                    brief,
                    own_evidence,
                    message_id,
                    response,
                    turn_index=turn_index,
                    previous_answers=previous_answers,
                    messages=messages,
                    known_claims=known_claims,
                    protocol_mode=self.protocol_mode,
                )
            except (InvariantViolation, ValueError):
                repair_attempted = True
                try:
                    repaired = await self._call(
                        request.model_copy(
                            update={
                                "input_text": input_text
                                + "\n"
                                + self._repair_instruction(
                                    turn_index,
                                    brief,
                                    previous_answers,
                                    {item.id: item for item in summary.claims},
                                    self.protocol_mode,
                                ),
                                "repair_attempts": 0,
                                "timeout_seconds": self._timeout(bounded_scheduler),
                            }
                        ),
                        output_schema,
                        bounded_scheduler,
                    )
                except ModelProviderError as repair_error:
                    if repair_error.category == ModelErrorCategory.CANCELLED:
                        raise
                    # The first response is already schema-valid. A failed
                    # best-effort semantic retry must not discard that usable
                    # turn; normalize only its conclusion expression instead.
                    repaired = response
                payload = self._build(
                    run_id,
                    brief,
                    own_evidence,
                    message_id,
                    repaired,
                    turn_index=turn_index,
                    previous_answers=previous_answers,
                    messages=messages,
                    known_claims=known_claims,
                    protocol_mode=self.protocol_mode,
                    allow_semantic_fallback=True,
                )
        except ModelProviderError as error:
            if error.category == ModelErrorCategory.CANCELLED:
                raise CooperativeCancellation(
                    "provider investigator opening was cancelled"
                ) from None
            raise

        message = Message.model_validate(payload["message"])
        built_claims = tuple(Claim.model_validate(item) for item in payload["claims"])
        try:
            self.store.append_event(
                run_id,
                # The ordinary artifact event is atomic for its message+claims.
                event_type=RunEventType.MESSAGE_CREATED,
                phase=expected_phase,
                payload=payload,
                actor_id=agent_id,
            )
        except InvariantViolation:
            # A concurrent caller may have won the stable message ID race.
            replayed = self.store.get_summary(run_id)
            existing = next(
                (
                    item
                    for item in replayed.messages
                    if item.id == message_id and item.author_id == agent_id
                ),
                None,
            )
            if existing is None:
                raise
            replayed_claims = tuple(
                item for item in replayed.claims if item.message_id == existing.id
            )
            if not replayed_claims:
                raise
            return AgentTurnResult(existing, replayed_claims, repair_attempted)
        return AgentTurnResult(message, built_claims, repair_attempted)

    async def _call(
        self,
        request: ModelRequest,
        output_schema: type[InvestigatorOpeningOutput] | type[InvestigatorDirectedUpdateOutput],
        scheduler: BoundedScheduler,
    ) -> ModelResponse[InvestigatorOpeningOutput] | ModelResponse[InvestigatorDirectedUpdateOutput]:
        if self.runner is None:
            raise RuntimeError("provider-backed answer generation is not configured")
        runner = self.runner
        await scheduler.acquire_tool_call()

        async def generate() -> (
            ModelResponse[InvestigatorOpeningOutput]
            | ModelResponse[InvestigatorDirectedUpdateOutput]
        ):
            if output_schema is InvestigatorOpeningOutput:
                return await runner.run(request, InvestigatorOpeningOutput)
            return await runner.run(request, InvestigatorDirectedUpdateOutput)

        return await scheduler.run_provider(generate)

    async def _call_text(
        self,
        request: ModelRequest,
        scheduler: BoundedScheduler,
    ) -> ModelTextResponse:
        if self.runner is None:
            raise RuntimeError("provider-backed answer generation is not configured")
        runner = self.runner
        await scheduler.acquire_tool_call()
        return await scheduler.run_provider(lambda: runner.run_text(request))

    @staticmethod
    def _stable_message_id(run_id: str, agent_id: str, turn_index: int) -> str:
        digest = hashlib.sha256(f"{run_id}\x1f{agent_id}\x1f{turn_index}".encode()).hexdigest()[:32]
        return f"agent_turn_{digest}"

    def _timeout(self, scheduler: BoundedScheduler) -> float:
        if self.runner is None:
            raise RuntimeError("provider-backed answer generation is not configured")
        timeout = self.runner.request_timeout_seconds
        remaining = scheduler.remaining_seconds()
        if remaining is not None:
            timeout = min(timeout, remaining)
        return timeout

    @staticmethod
    def _input_text(
        topic: str,
        brief: AgentBrief,
        evidence: dict[str, Evidence],
        *,
        turn_index: int,
        previous_answers: list[Message],
        previous_claims: Mapping[str, Claim],
        goal: str | None = None,
        protocol_mode: ProtocolMode = "default",
        output_mode: OutputMode = "structured_json",
    ) -> str:
        records = [
            {
                "id": item.id,
                "title": item.title[:MAX_OPENING_TITLE_CHARS],
                "excerpt": item.excerpt[:MAX_OPENING_EXCERPT_CHARS],
                "status": str(item.status),
            }
            for item in sorted(evidence.values(), key=lambda item: item.id)
        ]
        peer_records: list[dict[str, object]] = []
        for item in previous_answers:
            peer_record: dict[str, object] = {
                "agent_id": item.author_id,
                "agent_label": item.author_label,
                "content": item.content[:1_200],
            }
            if protocol_mode != "paper_reproduction":
                peer_record["claims"] = [
                    {"claim_id": claim_id, "text": previous_claims[claim_id].text[:500]}
                    for claim_id in item.claim_ids
                    if claim_id in previous_claims
                ]
            peer_records.append(peer_record)
        eligible_targets = (
            [
                {
                    "target_option": option,
                    "agent_id": item.author_id,
                    "agent_label": item.author_label,
                    "text": previous_claims[claim_id].text[:500],
                }
                for option, item, claim_id in Debater._eligible_targets(
                    previous_answers, previous_claims, brief.agent_id
                )
            ]
            if output_mode == "structured_json"
            else []
        )
        payload: dict[str, object] = {
            "topic": topic[:MAX_TOPIC_CHARS],
            "goal": goal[:1_000] if goal is not None else None,
            "brief": brief.model_dump(mode="json"),
            "evidence": records,
            "turn_index": turn_index,
            "previous_answers": peer_records,
            "instruction": Debater._turn_instruction(
                turn_index, protocol_mode, output_mode
            ),
        }
        if output_mode == "structured_json":
            payload["eligible_targets"] = eligible_targets
        if output_mode == "structured_json":
            output_schema_name = (
                InvestigatorOpeningOutput.__name__
                if turn_index == 1
                else InvestigatorDirectedUpdateOutput.__name__
            )
            payload["contract"] = {
                "output": output_schema_name,
                "requirements": list(
                    _OPENING_CONTRACT_REQUIREMENTS
                    if turn_index == 1
                    else _DIRECTED_UPDATE_CONTRACT_REQUIREMENTS
                ),
            }
        rendered = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if len(rendered) > MAX_OPENING_INPUT_CHARS:
            payload["brief"] = {
                "id": brief.id,
                "agent_id": brief.agent_id,
                "label": brief.label,
                "focus": brief.focus[:240],
                "key_questions": [item[:180] for item in brief.key_questions],
                "deliverable": brief.deliverable[:240],
            }
            for record in records:
                record["title"] = record["title"][:120]
                record["excerpt"] = record["excerpt"][:160]
            for peer_record in peer_records:
                content = peer_record.get("content")
                if isinstance(content, str):
                    peer_record["content"] = content[:600]
                claims = peer_record.get("claims")
                if isinstance(claims, list):
                    for claim in claims:
                        if isinstance(claim, dict):
                            claim["text"] = str(claim.get("text", ""))[:180]
            for target in eligible_targets:
                target["text"] = str(target["text"])[:180]
            rendered = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if len(rendered) > MAX_OPENING_INPUT_CHARS:
            raise ValueError("bounded investigator opening input cannot fit provider contract")
        return rendered

    @staticmethod
    def _turn_instruction(
        turn_index: int,
        protocol_mode: ProtocolMode,
        output_mode: OutputMode = "structured_json",
    ) -> str:
        if output_mode == "plain_text":
            if turn_index == 1:
                return (
                    "Solve independently without using peer answers. Recompute every quantity "
                    "and check each arithmetic operation. Return plain text, not JSON. Give one "
                    "concise calculation and end with exactly 'Final answer: <number>'. Put no "
                    "number, unit, or text after it."
                )
            return (
                "Read every previous-turn solution, then solve the problem yourself from the "
                "original question. Verify arithmetic instead of copying a peer or choosing by "
                "majority. Change your answer only when your recomputation warrants it. Return "
                "plain text, not JSON, with one concise calculation and exactly one terminal "
                "'Final answer: <number>'. Do not select a target or emit claim metadata."
            )
        final_answer_contract = (
            "Return exactly one claim. Put a concise step-by-step calculation in that claim, "
            "then end the same claim with exactly 'Final answer: <number>'. Use the phrase "
            "'Final answer:' exactly once, give only one candidate answer, and put no number "
            "or unit after it."
        )
        if turn_index == 1:
            if protocol_mode == "paper_reproduction":
                return (
                    "Solve independently without using peer answers. Check every arithmetic "
                    f"operation. {final_answer_contract}"
                )
            return (
                "Produce an independent initial answer without using peer answers. State one "
                "unambiguous conclusion and never present competing final answers."
            )
        if protocol_mode == "paper_reproduction":
            return (
                "Treat every previous-turn peer solution as additional information. Recompute "
                "the problem yourself, verify each peer step and your own arithmetic, and change "
                "your answer only when verification warrants it; otherwise retain it. "
                f"{final_answer_contract} Select exactly one eligible target_option and label "
                "the checked peer claim as challenge or support; never target your own claim."
            )
        return (
            "Review every previous-turn answer, challenge or support one explicit peer claim "
            "from eligible_targets, and return your updated answer with one unambiguous "
            "conclusion. Never present competing final answers. Select exactly one eligible "
            "target_option and never target your own claim."
        )

    @staticmethod
    def _repair_instruction(
        turn_index: int,
        brief: AgentBrief,
        previous_answers: list[Message],
        previous_claims: Mapping[str, Claim],
        protocol_mode: ProtocolMode = "default",
    ) -> str:
        output_schema_name = (
            InvestigatorOpeningOutput.__name__
            if turn_index == 1
            else InvestigatorDirectedUpdateOutput.__name__
        )
        if turn_index == 1:
            requirements = (
                "Turn 1 requirements: return at least one claim; use only listed evidence IDs; "
                "and omit all target fields, including target_option and interaction_kind."
            )
        else:
            requirements = (
                "Later-turn requirements: return at least one claim; use only listed evidence IDs; "
                "select exactly one target_option from the eligible options below; "
                "use interaction_kind "
                "challenge or support; and never target the current agent's own claim."
            )
        if protocol_mode == "paper_reproduction":
            requirements += (
                " Paper safeguard: return exactly one claim whose text contains the complete "
                "reasoning and ends with exactly one 'Final answer: <number>'. Do not put "
                "another candidate answer, number, unit, or text after that marker."
            )
        else:
            requirements += (
                " Across all claims, state one internally consistent conclusion. Never include "
                "more than one Final answer: or Conclusion: marker, and never present competing "
                "candidate conclusions."
            )
        return (
            f"Repair once. Return exactly one valid {output_schema_name} JSON object. "
            + requirements
            + " Eligible target options: "
            + json.dumps(
                [
                    option
                    for option, _, _ in Debater._eligible_targets(
                        previous_answers, previous_claims, brief.agent_id
                    )
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "."
        )

    @staticmethod
    def _validate_unambiguous_output(
        output: InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput,
    ) -> None:
        marker_count = sum(
            len(_CONCLUSION_MARKER.findall(claim.text)) for claim in output.claims
        )
        if marker_count > 1:
            raise InvariantViolation("agent output contains competing conclusion markers")

    @staticmethod
    def _validate_paper_output(
        output: InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput,
    ) -> None:
        if len(output.claims) != 1:
            raise InvariantViolation("paper output must contain exactly one claim")
        text = output.claims[0].text.strip()
        if len(_PAPER_FINAL_ANSWER_MARKER.findall(text)) != 1:
            raise InvariantViolation("paper output must contain one final answer marker")
        if _PAPER_FINAL_ANSWER_END.search(text) is None:
            raise InvariantViolation("paper final answer must be the terminal numeric text")

    @staticmethod
    def _canonicalize_default_output(
        output: InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput,
    ) -> InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput:
        """Keep multiple evidence claims while making one conclusion authoritative."""

        marker_locations = [
            (claim_index, match.start(), match.end())
            for claim_index, claim in enumerate(output.claims)
            for match in _CONCLUSION_MARKER.finditer(claim.text)
        ]
        if len(marker_locations) <= 1:
            return output
        final_location = marker_locations[-1]
        normalized_claims = []
        for claim_index, claim in enumerate(output.claims):
            matches = list(_CONCLUSION_MARKER.finditer(claim.text))
            if not matches:
                normalized_claims.append(claim)
                continue
            pieces: list[str] = []
            cursor = 0
            for match in matches:
                pieces.append(claim.text[cursor : match.start()])
                location = (claim_index, match.start(), match.end())
                pieces.append(
                    match.group(0)
                    if location == final_location
                    else "Discarded candidate:"
                )
                cursor = match.end()
            pieces.append(claim.text[cursor:])
            normalized_claims.append(
                claim.model_copy(update={"text": "".join(pieces).strip()})
            )
        return output.model_copy(update={"claims": normalized_claims})

    @staticmethod
    def _canonicalize_paper_output(
        output: InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput,
    ) -> InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput:
        """Collapse a repaired math response into one parser-safe final claim."""

        joined = "\n".join(claim.text for claim in output.claims)
        explicit_answers = list(_PAPER_LOOSE_FINAL_ANSWER.finditer(joined))
        if explicit_answers:
            answer = explicit_answers[-1].group(1)
        else:
            numbers = _PAPER_ANY_NUMBER.findall(joined)
            if not numbers:
                raise InvariantViolation("paper output has no numeric conclusion to canonicalize")
            answer = numbers[-1]
        answer = answer.replace(",", "")

        reasoning_parts = [
            cleaned
            for claim in output.claims
            if (cleaned := _CONCLUSION_TAIL.sub("", claim.text).strip(" -\n"))
        ]
        suffix = f"Final answer: {answer}"
        max_reasoning_chars = 1_000 - len(suffix) - 1
        reasoning = " ".join(reasoning_parts).strip()
        if reasoning:
            reasoning = reasoning[:max_reasoning_chars].rstrip()
            text = f"{reasoning} {suffix}"
        else:
            text = suffix

        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for claim in output.claims
                for evidence_id in claim.evidence_ids
            )
        )[:10]
        canonical_claim = output.claims[0].model_copy(
            update={"text": text, "evidence_ids": evidence_ids}
        )
        return output.model_copy(update={"claims": [canonical_claim]})

    @staticmethod
    def _canonicalize_output(
        output: InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput,
        protocol_mode: ProtocolMode,
    ) -> InvestigatorOpeningOutput | InvestigatorDirectedUpdateOutput:
        if protocol_mode == "paper_reproduction":
            return Debater._canonicalize_paper_output(output)
        return Debater._canonicalize_default_output(output)

    @staticmethod
    def _build(
        run_id: str,
        brief: AgentBrief,
        evidence: Mapping[str, Evidence],
        message_id: str,
        response: ModelResponse[InvestigatorOpeningOutput]
        | ModelResponse[InvestigatorDirectedUpdateOutput],
        *,
        turn_index: int,
        previous_answers: list[Message],
        messages: Mapping[str, Message],
        known_claims: Mapping[str, Claim],
        protocol_mode: ProtocolMode = "default",
        allow_semantic_fallback: bool = False,
    ) -> dict[str, Any]:
        if not response.output.claims:
            raise InvariantViolation("investigator opening must contain at least one claim")
        output = response.output
        try:
            Debater._validate_unambiguous_output(output)
            if protocol_mode == "paper_reproduction":
                Debater._validate_paper_output(output)
        except InvariantViolation:
            if not allow_semantic_fallback:
                raise
            output = Debater._canonicalize_output(output, protocol_mode)
            Debater._validate_unambiguous_output(output)
            if protocol_mode == "paper_reproduction":
                Debater._validate_paper_output(output)
        target_message: Message | None = None
        target_claim_id: str | None = None
        interaction_kind: InteractionKind | None = None
        if turn_index == 1:
            if not isinstance(output, InvestigatorOpeningOutput):
                raise InvariantViolation("turn 1 provider output must be independent")
        else:
            if not isinstance(output, InvestigatorDirectedUpdateOutput):
                raise InvariantViolation("later-turn provider output must select an interaction")
            target_option = output.target_option
            eligible_targets = Debater._eligible_targets(
                previous_answers, known_claims, brief.agent_id
            )
            selected_target = next(
                (target for target in eligible_targets if target[0] == target_option), None
            )
            if selected_target is None:
                raise InvariantViolation(
                    "provider target option is not an eligible previous-turn target"
                )
            _, target_message, target_claim_id = selected_target
            interaction_kind = output.interaction_kind
        content_parts: list[str] = []
        for draft in output.claims:
            part = f"- {draft.text}"
            content_parts.append(part)
        content = f"{brief.label}: " + " ".join(content_parts)
        if len(content) > MAX_OPENING_MESSAGE_CHARS:
            content = content[: MAX_OPENING_MESSAGE_CHARS - 1].rstrip() + "…"
        draft_message = Message(
            id=message_id,
            author_id=brief.agent_id,
            author_label=brief.label,
            phase=RunPhase.OPENING if turn_index == 1 else RunPhase.DEBATING,
            kind=MessageKind.OPENING if turn_index == 1 else MessageKind.UPDATE,
            content=content,
            evidence_ids=list(
                dict.fromkeys(
                    evidence_id
                    for draft in output.claims
                    for evidence_id in draft.evidence_ids
                )
            )[:10],
            turn_index=turn_index,
            in_reply_to_message_id=target_message.id if target_message is not None else None,
            target_agent_id=target_message.author_id if target_message is not None else None,
            target_claim_id=target_claim_id,
            interaction_kind=interaction_kind,
        )
        payload = prepare_investigator_message(
            draft_message,
            output,
            run_id=run_id,
            evidence=evidence,
            messages=messages,
            known_claims=known_claims,
            actor_id=brief.agent_id,
        )
        return payload

    @staticmethod
    def _eligible_targets(
        previous_answers: list[Message],
        previous_claims: Mapping[str, Claim],
        agent_id: str,
    ) -> list[tuple[int, Message, str]]:
        """Return deterministic short options and their authoritative targets."""

        targets: list[tuple[int, Message, str]] = []
        for item in previous_answers:
            if item.author_id == agent_id:
                continue
            for claim_id in item.claim_ids:
                if claim_id in previous_claims:
                    targets.append((len(targets), item, claim_id))
                if len(targets) >= MAX_OPENING_TARGETS:
                    return targets
        return targets


__all__ = [
    "AgentTurnResult",
    "Debater",
    "OutputMode",
    "ProtocolMode",
]
