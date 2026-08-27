"""One-time, bounded topic planning behind the provider-neutral model contract."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar as ClassVar
from typing import Literal

from pydantic import Field, field_validator

from debate_api.domain.models import (
    AgentBrief,
    PlanningOutcome,
    PlanningOutcomeCategory,
    Run,
    ToolPermission,
)
from debate_api.orchestration.model_runner import StructuredGenerationRunner
from debate_api.orchestration.scheduler import (
    BoundedScheduler,
    CooperativeCancellation,
)
from debate_api.providers.model import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ProviderModel,
)

PaperRoleProfile = Literal["homogeneous", "checker", "checker_semantic"]


class GeneratedBrief(ProviderModel):
    """Provider-owned brief content; IDs and permissions remain server-owned."""

    focus: str = Field(min_length=1, max_length=500)
    key_questions: list[str] = Field(min_length=1, max_length=6)
    preferred_source_types: list[str] = Field(min_length=1, max_length=6)
    deliverable: str = Field(min_length=1, max_length=500)

    @field_validator("focus", "deliverable")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("key_questions", "preferred_source_types")
    @classmethod
    def trim_items(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("list items must not be blank")
        return normalized


class GeneratedTopicPlan(ProviderModel):
    briefs: list[GeneratedBrief] = Field(min_length=1, max_length=7)


@dataclass(frozen=True)
class TopicPlanResult:
    briefs: list[AgentBrief]
    category: PlanningOutcomeCategory
    repair_attempted: bool
    fallback_used: bool
    planner_id: str
    failure_category: str | None = None


class TopicOrchestrator:
    """Generate one public plan, or use a stable server-owned fallback catalog."""

    planner_id = "topic-planner-v1"
    allowed_tool_permissions: ClassVar[frozenset[ToolPermission]] = frozenset(
        {ToolPermission.SEARCH, ToolPermission.FETCH}
    )
    _catalog = (
        (
            "Source investigator",
            "Establish the strongest direct evidence and what is actually known about the topic.",
            ("What is the strongest direct measurement?", "Which source publishes or measures it?"),
            ("official report", "primary dataset"),
            "One concise public position with claim-level citations.",
        ),
        (
            "Counter-evidence investigator",
            "Test limits, alternative explanations, and credible disagreement around the topic.",
            ("What could make the main claim misleading?", "Which caveat matters most?"),
            ("independent analysis", "methodology review"),
            "A public counter-position identifying limits and competing explanations.",
        ),
        (
            "Practical investigator",
            "Check real-world constraints, applicability, and failure modes for the topic.",
            ("When does the finding transfer to practice?", "What should a user verify next?"),
            ("field study", "implementation guide"),
            "A practical decision brief with explicit constraints and next checks.",
        ),
        (
            "Assumption auditor",
            "Identify hidden assumptions and test how conclusions change when they fail.",
            ("Which assumption drives the answer?", "What evidence would falsify it?"),
            ("methodology review", "sensitivity analysis"),
            "An assumption audit with concrete falsification checks.",
        ),
        (
            "Stakeholder analyst",
            "Compare impacts across affected groups and surface distributional trade-offs.",
            ("Who benefits or bears costs?", "Which impact is missing?"),
            ("impact assessment", "stakeholder report"),
            "A stakeholder impact position with explicit trade-offs.",
        ),
        (
            "Historical comparator",
            "Find relevant precedents and distinguish transferable lessons from context.",
            ("Which precedent is closest?", "Where does the analogy break?"),
            ("case study", "historical dataset"),
            "A precedent-based position with limits on transferability.",
        ),
        (
            "Decision critic",
            "Stress-test the proposed decision against risks and reversible alternatives.",
            ("What is the worst credible failure?", "What can be tested reversibly?"),
            ("risk assessment", "decision framework"),
            "A risk-focused recommendation with a reversible next step.",
        ),
    )

    def __init__(self, provider: ModelProvider | None = None) -> None:
        self._runner = StructuredGenerationRunner(provider) if provider is not None else None

    @classmethod
    def catalog_briefs(cls, topic: str, count: int) -> list[AgentBrief]:
        """Build the zero-secret catalog used by deterministic planning."""

        catalog_key = sha256(topic.encode("utf-8")).hexdigest()[:16]
        return cls()._catalog_briefs(
            topic,
            count,
            tool_permissions=(
                sorted(cls.allowed_tool_permissions, key=lambda value: value.value)
                if count <= 2
                else []
            ),
            run_id=f"catalog_{catalog_key}",
        )

    @classmethod
    def paper_reproduction_plan(
        cls, run: Run, role_profile: PaperRoleProfile = "homogeneous"
    ) -> TopicPlanResult:
        """Build deterministic, tool-free math roles for paper runs and ablations."""
        if role_profile not in {"homogeneous", "checker", "checker_semantic"}:
            raise ValueError("unsupported paper_role_profile")
        checker_index = run.agent_count - 1
        briefs = [
            AgentBrief(
                id=_brief_id(run.id, index),
                agent_id=f"agent_{index + 1}",
                label=(
                    "Semantic and arithmetic verifier"
                    if role_profile == "checker_semantic" and index == checker_index
                    else "Arithmetic verifier"
                    if role_profile == "checker" and index == checker_index
                    else "Independent math solver"
                ),
                focus=(
                    "Treat every peer conclusion as untrusted. Identify the exact quantity and "
                    "unit requested, solve independently from the original facts, then locate "
                    "the earliest semantic or arithmetic error in each competing solution. Do "
                    "not follow a majority or a more verbose answer without recomputation."
                    if role_profile == "checker_semantic" and index == checker_index
                    else
                    "Independently recompute the expression, check operator precedence, signs, "
                    "and every peer calculation, then explicitly correct any arithmetic error."
                    if role_profile == "checker" and index == checker_index
                    else "Solve the math problem independently and verify every calculation."
                ),
                key_questions=[
                    *(
                        [
                            "What exact quantity and unit does the question ask for, and does "
                            "each candidate answer that quantity?",
                            "What is the earliest unsupported semantic or arithmetic step in "
                            "each competing solution?",
                        ]
                        if role_profile == "checker_semantic" and index == checker_index
                        else
                        [
                            "Did every solution apply multiplication before addition "
                            "and subtraction?",
                            "Which exact intermediate operation, if any, is incorrect?",
                        ]
                        if role_profile == "checker" and index == checker_index
                        else [
                            "What arithmetic operations are required?",
                            "What is the final number?",
                        ]
                    ),
                ],
                preferred_source_types=["none"],
                deliverable=(
                    "One independent recomputation, an explicit verdict on competing answers, "
                    "and one final numerical answer for the requested quantity."
                    if role_profile == "checker_semantic" and index == checker_index
                    else
                    "An independently verified solution that identifies any correction and ends "
                    "in one explicit numerical answer."
                    if role_profile == "checker" and index == checker_index
                    else "A checked solution ending in one explicit numerical answer."
                ),
                tool_permissions=[],
            )
            for index in range(run.agent_count)
        ]
        return TopicPlanResult(
            briefs=briefs,
            category=PlanningOutcomeCategory.FALLBACK,
            repair_attempted=False,
            fallback_used=True,
            planner_id=(
                "math-paper-checker-semantic-v1"
                if role_profile == "checker_semantic"
                else "math-paper-checker-v1"
                if role_profile == "checker"
                else "math-paper-v1"
            ),
        )

    async def plan(
        self,
        run: Run,
        existing_briefs: list[AgentBrief] | None = None,
        scheduler: BoundedScheduler | None = None,
        existing_outcome: PlanningOutcome | None = None,
    ) -> TopicPlanResult:
        requested = run.agent_count
        persisted = list(existing_briefs or [])
        if persisted:
            # Persisted briefs are authoritative after a restart; never mint a second team.
            if existing_outcome is None or len(persisted) != requested:
                raise ValueError("persisted plan count does not match run configuration")
            return TopicPlanResult(
                briefs=persisted,
                category=PlanningOutcomeCategory.REUSED,
                repair_attempted=False,
                fallback_used=False,
                planner_id=self.planner_id,
            )

        if self._runner is None or run.limits.max_tool_calls == 0:
            return self._fallback(run, requested, scheduler=scheduler)

        if scheduler is not None:
            scheduler.check()
            await scheduler.acquire_tool_call()
        timeout_seconds = self._request_timeout(scheduler)

        request = ModelRequest(
            request_id=f"plan_{run.id}",
            operation="topic_planning",
            input_text=self._planning_input(run),
            output_schema_name=GeneratedTopicPlan.__name__,
            timeout_seconds=timeout_seconds,
            max_output_tokens=2_048,
            repair_attempts=1,
        )
        repair_attempted = False
        try:
            response = await self._call(request, scheduler, GeneratedTopicPlan)
            try:
                briefs = self._build_briefs(
                    run,
                    response.output.briefs,
                    requested,
                    self._tool_permissions(
                        run, requested, self._remaining_tool_calls(scheduler, run)
                    ),
                )
            except ValueError:
                if response.repair_attempted:
                    return self._fallback(
                        run,
                        requested,
                        scheduler=scheduler,
                        repair_attempted=True,
                        failure_category="invalid_plan",
                    )
                repair_attempted = True
                repaired_request = await self._repair_request(
                    request,
                    scheduler,
                    "Repair the plan once: return exactly the requested count of distinct "
                    "briefs with distinct evidence coverage.",
                )
                try:
                    repaired = await self._call(repaired_request, scheduler, GeneratedTopicPlan)
                except ModelProviderError as repair_error:
                    if repair_error.category.value == "cancelled":
                        raise CooperativeCancellation("provider planning was cancelled") from None
                    return self._fallback(
                        run,
                        requested,
                        scheduler=scheduler,
                        repair_attempted=True,
                        failure_category=repair_error.category.value,
                    )
                try:
                    briefs = self._build_briefs(
                        run,
                        repaired.output.briefs,
                        requested,
                        self._tool_permissions(
                            run, requested, self._remaining_tool_calls(scheduler, run)
                        ),
                    )
                except ValueError:
                    return self._fallback(
                        run,
                        requested,
                        scheduler=scheduler,
                        repair_attempted=True,
                        failure_category="invalid_plan",
                    )
                return TopicPlanResult(
                    briefs=briefs,
                    category=PlanningOutcomeCategory.PROVIDER_REPAIRED,
                    repair_attempted=True,
                    fallback_used=False,
                    planner_id=self.planner_id,
                )
        except ModelProviderError as error:
            if error.category.value == "cancelled":
                raise CooperativeCancellation("provider planning was cancelled") from None
            if repair_attempted or error.repair_attempted:
                return self._fallback(
                    run,
                    requested,
                    scheduler=scheduler,
                    repair_attempted=True,
                    failure_category=error.category.value,
                )
            if error.category.value in {"timeout", "provider_error"}:
                return self._fallback(
                    run,
                    requested,
                    scheduler=scheduler,
                    repair_attempted=False,
                    failure_category=error.category.value,
                )
            try:
                repair_attempted = True
                repaired_request = await self._repair_request(
                    request,
                    scheduler,
                    "Repair the structured plan once without repeating private input.",
                )
                repaired = await self._call(repaired_request, scheduler, GeneratedTopicPlan)
                briefs = self._build_briefs(
                    run,
                    repaired.output.briefs,
                    requested,
                    self._tool_permissions(
                        run, requested, self._remaining_tool_calls(scheduler, run)
                    ),
                )
            except ModelProviderError as repair_error:
                if repair_error.category.value == "cancelled":
                    raise CooperativeCancellation("provider planning was cancelled") from None
                return self._fallback(
                    run,
                    requested,
                    scheduler=scheduler,
                    repair_attempted=True,
                    failure_category=repair_error.category.value,
                )
            except ValueError:
                return self._fallback(
                    run,
                    requested,
                    scheduler=scheduler,
                    repair_attempted=True,
                    failure_category="invalid_plan",
                )
            return TopicPlanResult(
                briefs=briefs,
                category=PlanningOutcomeCategory.PROVIDER_REPAIRED,
                repair_attempted=True,
                fallback_used=False,
                planner_id=self.planner_id,
            )
        except ValueError:
            return self._fallback(
                run,
                requested,
                scheduler=scheduler,
                repair_attempted=False,
                failure_category="invalid_plan",
            )

        return TopicPlanResult(
            briefs=briefs,
            category=(
                PlanningOutcomeCategory.PROVIDER_REPAIRED
                if response.repair_attempted
                else PlanningOutcomeCategory.PROVIDER_SUCCESS
            ),
            repair_attempted=response.repair_attempted,
            fallback_used=False,
            planner_id=self.planner_id,
        )

    async def _call(
        self,
        request: ModelRequest,
        scheduler: BoundedScheduler | None,
        output_schema: type[GeneratedTopicPlan],
    ) -> ModelResponse[GeneratedTopicPlan]:
        if self._runner is None:
            raise RuntimeError("topic planner has no model runner")
        runner = self._runner

        async def operation() -> ModelResponse[GeneratedTopicPlan]:
            return await runner.run(request, output_schema)

        if scheduler is None:
            return await operation()
        return await scheduler.run_provider(operation)

    async def _repair_request(
        self,
        request: ModelRequest,
        scheduler: BoundedScheduler | None,
        instruction: str,
    ) -> ModelRequest:
        if scheduler is not None:
            await scheduler.acquire_tool_call()
        timeout = self._request_timeout(scheduler, request.timeout_seconds)
        return request.model_copy(
            update={
                "input_text": f"{request.input_text}\n{instruction}",
                "timeout_seconds": timeout,
                "repair_attempts": 0,
            }
        )

    def _request_timeout(
        self, scheduler: BoundedScheduler | None, requested: float | None = None
    ) -> float:
        if self._runner is None:
            raise RuntimeError("topic planner has no model runner")
        timeout = self._runner.request_timeout_seconds
        if requested is not None:
            timeout = min(timeout, requested)
        if scheduler is not None:
            remaining = scheduler.remaining_seconds()
            if remaining is not None:
                timeout = min(timeout, remaining)
        return timeout

    @staticmethod
    def _remaining_tool_calls(scheduler: BoundedScheduler | None, run: Run) -> int:
        return (
            scheduler.remaining_tool_calls if scheduler is not None else run.limits.max_tool_calls
        )

    def _planning_input(self, run: Run) -> str:
        goal = run.goal or "Produce a useful, inspectable answer."
        return (
            "Generate a bounded evidence coverage plan.\n"
            f"Topic: {run.topic}\nGoal: {goal}\n"
            f"Agent count: {run.agent_count}\n"
            f"Maximum agent tool calls: {run.limits.max_tool_calls}\n"
            "Return only the requested structured plan."
        )

    def _build_briefs(
        self,
        run: Run,
        generated: list[GeneratedBrief],
        requested: int,
        tool_permissions: list[ToolPermission],
    ) -> list[AgentBrief]:
        if len(generated) != requested:
            raise ValueError("planner returned an unexpected brief count")
        if not self._distinct(generated):
            raise ValueError("planner returned insufficiently distinct briefs")
        return [
            AgentBrief(
                id=_brief_id(run.id, index),
                agent_id=f"agent_{index + 1}",
                label=self._catalog[index][0],
                focus=brief.focus,
                key_questions=brief.key_questions,
                preferred_source_types=brief.preferred_source_types,
                deliverable=brief.deliverable,
                tool_permissions=tool_permissions,
            )
            for index, brief in enumerate(generated)
        ]

    def _fallback(
        self,
        run: Run,
        requested: int,
        scheduler: BoundedScheduler | None = None,
        repair_attempted: bool = False,
        failure_category: str | None = "no_provider",
    ) -> TopicPlanResult:
        briefs = self._catalog_briefs(
            run.topic,
            requested,
            tool_permissions=self._tool_permissions(
                run, requested, self._remaining_tool_calls(scheduler, run)
            ),
            run_id=run.id,
        )
        return TopicPlanResult(
            briefs=briefs,
            category=PlanningOutcomeCategory.FALLBACK,
            repair_attempted=repair_attempted,
            fallback_used=True,
            planner_id=self.planner_id,
            failure_category=failure_category,
        )

    def _tool_permissions(
        self, run: Run, requested: int, remaining_tool_calls: int
    ) -> list[ToolPermission]:
        if (
            remaining_tool_calls < requested
            or not run.research_enabled
            or run.limits.max_retrieval_calls == 0
        ):
            return []
        return sorted(self.allowed_tool_permissions, key=lambda value: value.value)

    def _catalog_briefs(
        self,
        topic: str,
        requested: int,
        *,
        tool_permissions: list[ToolPermission],
        run_id: str,
    ) -> list[AgentBrief]:
        return [
            AgentBrief(
                id=_brief_id(run_id, index),
                agent_id=f"agent_{index + 1}",
                label=label,
                focus=f"{focus} Topic: {topic}"[:500],
                key_questions=list(questions),
                preferred_source_types=list(source_types),
                deliverable=deliverable,
                tool_permissions=tool_permissions,
            )
            for index, (label, focus, questions, source_types, deliverable) in enumerate(
                self._catalog[:requested]
            )
        ]

    @staticmethod
    def _distinct(generated: list[GeneratedBrief]) -> bool:
        signatures = [
            _tokens(f"{brief.focus} {' '.join(brief.key_questions)}") for brief in generated
        ]
        for index, left in enumerate(signatures):
            for right in signatures[index + 1 :]:
                if not left or not right:
                    return False
                overlap = len(left & right) / min(len(left), len(right))
                jaccard = len(left & right) / len(left | right)
                if overlap >= 0.8 or jaccard >= 0.7:
                    return False
        return True


def _tokens(value: str) -> set[str]:
    ignored = {"agent", "investigator", "researcher", "focus", "topic"}
    synonyms = {
        "research": "study",
        "researching": "study",
        "researched": "study",
        "studies": "study",
        "studied": "study",
        "effect": "impact",
        "effects": "impact",
        "impacts": "impact",
    }
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = [
        synonyms.get(token, token)
        for token in re.findall(r"\w+", normalized, flags=re.UNICODE)
        if token not in ignored
    ]
    tokens = {f"w:{token}" for token in words}
    compact = "".join(words)
    if any(ord(character) > 127 for character in normalized):
        tokens.update(
            f"g:{compact[index : index + 2]}" for index in range(max(0, len(compact) - 1))
        )
    return tokens


def _brief_id(run_id: str, index: int) -> str:
    return f"brief_{run_id}_{index + 1}"
