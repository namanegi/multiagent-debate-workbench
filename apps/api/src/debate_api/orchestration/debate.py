"""Core orchestration for one persisted multi-agent debate."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from debate_api.domain.models import (
    DEBATE_PROTOCOL_STOP_REASONS,
    AgentBrief,
    DebateProtocolOutcome,
    DebateProtocolStatus,
    Evidence,
    Failure,
    Message,
    PlanningOutcome,
    Run,
    RunEventType,
    RunPhase,
    RunStatus,
    RunStopReason,
    new_id,
)
from debate_api.orchestration.debater import (
    AgentTurnResult,
    Debater,
    OutputMode,
    ProtocolMode,
    ThinkingMode,
)
from debate_api.orchestration.scheduler import (
    BoundedScheduler,
    CooperativeCancellation,
    RunLimitReached,
)
from debate_api.orchestration.synthesis import SynthesisOrchestrator
from debate_api.orchestration.topic import PaperRoleProfile, TopicOrchestrator
from debate_api.persistence.sqlite import EventLimitReached, EventStore
from debate_api.providers.model import ModelProvider, ModelProviderError
from debate_api.research import ResearchService


class DebateOrchestrator:
    """Run the explicit agent-by-turn grid and persist every public artifact."""

    def __init__(
        self,
        store: EventStore,
        step_delay: float = 0.01,
        *,
        provider: ModelProvider | None = None,
        research_service: ResearchService | None = None,
        protocol_mode: ProtocolMode = "default",
        paper_role_profile: PaperRoleProfile = "homogeneous",
        output_mode: OutputMode = "structured_json",
        plain_text_max_output_tokens: int = 2_048,
        thinking_mode: ThinkingMode = "provider_default",
    ) -> None:
        self.store = store
        self.step_delay = max(0.0, step_delay)
        self._provider = provider
        self.research_service = research_service
        if protocol_mode not in {"default", "paper_reproduction"}:
            raise ValueError("unsupported protocol_mode")
        if paper_role_profile not in {
            "homogeneous",
            "checker",
            "checker_semantic",
        }:
            raise ValueError("unsupported paper_role_profile")
        if protocol_mode != "paper_reproduction" and paper_role_profile != "homogeneous":
            raise ValueError("paper_role_profile requires paper_reproduction protocol_mode")
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
        self.paper_role_profile = paper_role_profile
        self.output_mode = output_mode
        self.plain_text_max_output_tokens = plain_text_max_output_tokens
        self.thinking_mode = thinking_mode
        self._topic = TopicOrchestrator(provider)
        synthesis_provider = None if protocol_mode == "paper_reproduction" else provider
        self._synthesis = SynthesisOrchestrator(synthesis_provider, store=store)
        self._schedulers: dict[str, BoundedScheduler] = {}

    async def run(self, run_id: str) -> None:
        """Execute planning, research, all configured turns, and synthesis."""

        if run_id in self._schedulers:
            raise RuntimeError(f"run is already executing: {run_id}")
        run = self.store.get_run(run_id)
        if RunStatus(run.status) != RunStatus.QUEUED:
            raise RuntimeError(f"run is not queued: {run_id}")
        scheduler = BoundedScheduler(run, self.store)
        self._schedulers[run_id] = scheduler
        failures: list[Failure] = []
        try:
            await self._phase(run_id, RunPhase.PLANNING)
            if self.protocol_mode == "paper_reproduction":
                plan = self._topic.paper_reproduction_plan(run, self.paper_role_profile)
            else:
                plan = await self._topic.plan(run, [], scheduler=scheduler)
            outcome = PlanningOutcome(
                planner_id=plan.planner_id,
                category=plan.category,
                message="Task-specific agent roles were assigned.",
                repair_attempted=plan.repair_attempted,
                fallback_used=plan.fallback_used,
                brief_count=len(plan.briefs),
                failure_category=plan.failure_category,
            )
            _, briefs, _ = self.store.commit_plan(run_id, outcome, plan.briefs, run.agent_count)
            debaters = {
                brief.agent_id: Debater(
                    self.store,
                    brief,
                    self._provider,
                    self.research_service,
                    protocol_mode=self.protocol_mode,
                    output_mode=self.output_mode,
                    plain_text_max_output_tokens=self.plain_text_max_output_tokens,
                    thinking_mode=self.thinking_mode,
                )
                for brief in briefs
            }
            evidence = await self._run_research(run, briefs, debaters, scheduler, failures)
            await self._phase_done(run_id, RunPhase.RESEARCHING)

            await self._phase(run_id, RunPhase.OPENING)
            previous = await self._run_turn(
                run,
                briefs,
                debaters,
                evidence,
                turn_index=1,
                previous=[],
                scheduler=scheduler,
                failures=failures,
            )
            await self._phase_done(run_id, RunPhase.OPENING)

            await self._phase(run_id, RunPhase.DEBATING)
            completed_turns = 1 if len(previous) == run.agent_count else 0
            for turn_index in range(2, run.turn_count + 1):
                if len(previous) != run.agent_count:
                    break
                previous = await self._run_turn(
                    run,
                    briefs,
                    debaters,
                    evidence,
                    turn_index=turn_index,
                    previous=previous,
                    scheduler=scheduler,
                    failures=failures,
                )
                if len(previous) == run.agent_count:
                    completed_turns = turn_index
            completed_messages = sum(
                message.author_id.startswith("agent_") and message.turn_index is not None
                for message in self.store.get_summary(run_id).messages
            )
            expected_messages = run.agent_count * run.turn_count
            protocol_status = (
                DebateProtocolStatus.COMPLETED
                if completed_messages == expected_messages
                else DebateProtocolStatus.INCOMPLETE
            )
            self.store.append_event(
                run_id,
                RunEventType.DEBATE_PROTOCOL_OUTCOME_CREATED,
                RunPhase.DEBATING,
                {
                    "outcome": DebateProtocolOutcome(
                        id=new_id("debate_protocol"),
                        configured_turns=run.turn_count,
                        completed_turns=completed_turns,
                        expected_agent_messages=expected_messages,
                        completed_agent_messages=completed_messages,
                        status=protocol_status,
                        stop_reason=DEBATE_PROTOCOL_STOP_REASONS[protocol_status],
                    ).model_dump(mode="json")
                },
                actor_id="orchestrator",
            )
            await self._phase_done(run_id, RunPhase.DEBATING)

            await self._phase(run_id, RunPhase.SYNTHESIZING)
            if protocol_status != DebateProtocolStatus.COMPLETED or failures:
                await self._commit_partial_synthesis(run_id, scheduler)
                return
            synthesis = await self._synthesis.synthesize_normal(run_id, scheduler=scheduler)
            self.store.commit_synthesis(run_id, synthesis.message, synthesis.synthesis)
            await self._phase_done(run_id, RunPhase.SYNTHESIZING)
            await self._phase(run_id, RunPhase.FINALIZING)
            await self._emit(
                run_id,
                RunEventType.RUN_COMPLETED,
                RunPhase.FINALIZING,
                {
                    "reason": "All configured agent turns and synthesis completed.",
                    "reason_code": RunStopReason.COMPLETED,
                },
            )
        except (CooperativeCancellation, _RunCancelled):
            self._cancel(run_id)
        except asyncio.CancelledError:
            self._cancel(run_id)
            raise
        except (EventLimitReached, RunLimitReached) as error:
            await self._fail_partially(run_id, scheduler, error)
        except Exception as error:
            await self._fail_partially(run_id, scheduler, error, failed=True)
        finally:
            self._schedulers.pop(run_id, None)

    async def _run_research(
        self,
        run: Run,
        briefs: list[AgentBrief],
        debaters: dict[str, Debater],
        scheduler: BoundedScheduler,
        failures: list[Failure],
    ) -> dict[str, Evidence]:
        if not run.research_enabled:
            return {}
        operations = [
            self._research_operation(run, debaters, scheduler, index, brief)
            for index, brief in enumerate(briefs)
        ]
        results = await scheduler.run_concurrent(operations)
        evidence: dict[str, Evidence] = {}
        for brief, item, failure in results:
            if item is not None:
                evidence[brief.agent_id] = item
            if failure is not None:
                failures.append(failure)
        return evidence

    async def _run_turn(
        self,
        run: Run,
        briefs: list[AgentBrief],
        debaters: dict[str, Debater],
        evidence: dict[str, Evidence],
        *,
        turn_index: int,
        previous: list[Message],
        scheduler: BoundedScheduler,
        failures: list[Failure],
    ) -> list[Message]:
        async def one(brief: AgentBrief) -> AgentTurnResult:
            debater = debaters[brief.agent_id]
            if turn_index == 1:
                return await debater.initial_answer(
                    run.id, evidence=evidence.get(brief.agent_id), scheduler=scheduler
                )
            return await debater.directed_update(
                run.id,
                turn_index,
                previous,
                evidence=evidence.get(brief.agent_id),
                scheduler=scheduler,
            )

        results = await asyncio.gather(
            *(one(brief) for brief in briefs),
            return_exceptions=True,
        )
        answers: list[Message] = []
        for brief, result in zip(briefs, results, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, (CooperativeCancellation, RunLimitReached)):
                    raise result
                failure = Failure(
                    id=new_id("failure"),
                    agent_id=brief.agent_id,
                    phase=RunPhase.OPENING if turn_index == 1 else RunPhase.DEBATING,
                    category=(
                        "agent_provider_failure"
                        if isinstance(result, ModelProviderError)
                        else "agent_turn_failure"
                    ),
                    message=f"{brief.label} did not complete turn {turn_index}.",
                )
                failures.append(failure)
                await self._emit(
                    run.id,
                    RunEventType.AGENT_FAILED,
                    failure.phase,
                    {"failure": failure.model_dump(mode="json")},
                    actor_id=brief.agent_id,
                )
            else:
                answers.append(result.message)
        return answers

    def _research_operation(
        self,
        run: Run,
        debaters: dict[str, Debater],
        scheduler: BoundedScheduler,
        index: int,
        brief: AgentBrief,
    ) -> Callable[[], Awaitable[tuple[AgentBrief, Evidence | None, Failure | None]]]:
        async def operation() -> tuple[AgentBrief, Evidence | None, Failure | None]:
            evidence, failure = await debaters[brief.agent_id].research(
                run, scheduler, index
            )
            if failure is not None:
                await self._emit(
                    run.id,
                    RunEventType.AGENT_FAILED,
                    RunPhase.RESEARCHING,
                    {"failure": failure.model_dump(mode="json")},
                    actor_id=brief.agent_id,
                )
            return brief, evidence, failure

        return operation

    async def _commit_partial_synthesis(self, run_id: str, scheduler: BoundedScheduler) -> None:
        run = self.store.get_run(run_id)
        result = await self._synthesis.synthesize(
            run, self.store.get_summary(run_id), RunStopReason.PARTIAL, scheduler=scheduler
        )
        self.store.commit_partial_synthesis(
            run_id,
            result.message,
            result.synthesis,
            RunStopReason.PARTIAL,
            reason="The persisted agent-turn grid is incomplete.",
        )

    async def _fail_partially(
        self,
        run_id: str,
        scheduler: BoundedScheduler,
        error: BaseException,
        *,
        failed: bool = False,
    ) -> None:
        run = self.store.get_run(run_id)
        if RunStatus(run.status) in {
            RunStatus.COMPLETED,
            RunStatus.PARTIAL,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }:
            return
        if RunPhase(run.phase) not in {RunPhase.CREATED, RunPhase.PLANNING}:
            try:
                result = await self._synthesis.synthesize(
                    run, self.store.get_summary(run_id), RunStopReason.PARTIAL, scheduler=scheduler
                )
                self.store.commit_partial_synthesis(
                    run_id,
                    result.message,
                    result.synthesis,
                    RunStopReason.PARTIAL,
                    reason=f"The run stopped with a visible {type(error).__name__}.",
                )
                return
            except Exception:
                pass
        self.store.append_event(
            run_id,
            RunEventType.RUN_FAILED if failed else RunEventType.RUN_PARTIAL,
            run.phase,
            {
                "reason": f"The run stopped with a visible {type(error).__name__}.",
                "reason_code": RunStopReason.ORCHESTRATION_ERROR,
            },
        )

    async def _phase(self, run_id: str, phase: RunPhase) -> None:
        await self._emit(run_id, RunEventType.PHASE_STARTED, phase, {"phase": phase.value})

    async def _phase_done(self, run_id: str, phase: RunPhase) -> None:
        await self._emit(run_id, RunEventType.PHASE_COMPLETED, phase, {"phase": phase.value})

    async def _emit(
        self,
        run_id: str,
        event_type: RunEventType,
        phase: RunPhase,
        payload: dict[str, object],
        actor_id: str | None = None,
    ) -> None:
        scheduler = self._schedulers.get(run_id)
        if scheduler is not None:
            scheduler.check()
        if self.store.is_cancel_requested(run_id):
            raise _RunCancelled
        self.store.append_event(run_id, event_type, phase, payload, actor_id=actor_id)
        if self.step_delay:
            await asyncio.sleep(self.step_delay)

    def _cancel(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if RunStatus(run.status) not in {
            RunStatus.COMPLETED,
            RunStatus.PARTIAL,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }:
            self.store.append_event(
                run_id,
                RunEventType.RUN_CANCELLED,
                run.phase,
                {"reason": "cancelled by user", "reason_code": RunStopReason.CANCELLED},
            )



class _RunCancelled(Exception):
    """Internal cooperative-cancellation marker."""
