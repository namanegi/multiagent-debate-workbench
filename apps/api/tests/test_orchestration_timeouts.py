from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from debate_api.domain.models import CreateRunRequest, InvestigatorOpeningOutput
from debate_api.orchestration.debate import DebateOrchestrator
from debate_api.orchestration.synthesis import GeneratedSynthesis, SynthesisOrchestrator
from debate_api.orchestration.topic import TopicOrchestrator
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.model import (
    FakeModelOutcome,
    FakeModelStep,
    ModelRequest,
    ProgrammableFakeModelProvider,
)


def _plan_output() -> dict[str, object]:
    return {
        "briefs": [
            {
                "focus": "Measure the strongest available evidence.",
                "key_questions": ["What is the strongest direct measurement?"],
                "preferred_source_types": ["official report"],
                "deliverable": "One concise evidence-grounded position.",
            }
        ]
    }


def _opening_output() -> dict[str, object]:
    return InvestigatorOpeningOutput.model_validate(
        {
            "claims": [
                {
                    "text": "The opening identifies the main evidence.",
                    "claim_type": "inference",
                    "evidence_ids": [],
                }
            ]
        }
    ).model_dump(mode="json")


def _synthesis_output() -> dict[str, object]:
    return GeneratedSynthesis.model_validate(
        {
            "answer": "The persisted artifacts support a bounded conclusion.",
            "consensus": ["The main point is inspectable."],
            "disagreements": ["No disagreement was recorded."],
            "changed_positions": ["No position change was recorded."],
            "evidence_gaps": ["Additional evidence could strengthen the result."],
            "follow_up_checks": ["Review the cited artifacts."],
            "claim_ids": [],
            "evidence_ids": [],
        }
    ).model_dump(mode="json")


def test_synthesis_accepts_concise_answer_but_rejects_blank() -> None:
    payload = {
        "answer": "42",
        "consensus": [],
        "disagreements": [],
        "changed_positions": [],
        "evidence_gaps": [],
        "follow_up_checks": [],
        "claim_ids": [],
        "evidence_ids": [],
    }
    assert GeneratedSynthesis.model_validate(payload).answer == "42"
    with pytest.raises(ValidationError, match="must not be blank"):
        GeneratedSynthesis.model_validate({**payload, "answer": "   "})


def test_topic_permissions_require_budget_for_every_agent(tmp_path: Path) -> None:
    store = EventStore(f"sqlite:///{tmp_path / 'topic-budget.db'}")
    run, _ = store.create_run(
        CreateRunRequest(topic="budget boundary", agent_count=3, research_enabled=True)
    )
    orchestrator = TopicOrchestrator()

    assert orchestrator._tool_permissions(run, 3, 2) == []
    assert orchestrator._tool_permissions(run, 3, 3) == ["fetch", "search"]


@pytest.mark.asyncio
async def test_topic_repair_uses_provider_timeout(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="topic_planning",
                outcome=FakeModelOutcome.MALFORMED_OUTPUT,
                output=["malformed"],
            ),
            FakeModelStep(operation="topic_planning", output=_plan_output()),
        ],
        request_timeout_seconds=90,
    )
    store = EventStore(f"sqlite:///{tmp_path / 'topic-timeout.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="provider timeout",
            agent_count=1,
            turn_count=1,
            research_enabled=True,
        )
    )

    result = await TopicOrchestrator(provider).plan(run)

    assert result.repair_attempted is True
    assert result.briefs[0].tool_permissions == ["fetch", "search"]
    assert [request.timeout_seconds for request in provider.calls] == [90, 90]


@pytest.mark.asyncio
async def test_debate_turns_and_synthesis_use_provider_timeout(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(operation="topic_planning", output=_plan_output()),
            FakeModelStep(operation="agent.answer.initial", output=_opening_output()),
            FakeModelStep(operation="synthesis", output=_synthesis_output()),
        ],
        request_timeout_seconds=90,
    )
    store = EventStore(f"sqlite:///{tmp_path / 'debate-timeout.db'}")
    run, _ = store.create_run(
        CreateRunRequest(
            topic="provider timeout",
            agent_count=1,
            turn_count=1,
            research_enabled=False,
        )
    )

    await DebateOrchestrator(store, step_delay=0, provider=provider).run(run.id)

    summary = store.get_summary(run.id)
    assert summary.run.status == "completed"
    assert summary.briefs[0].tool_permissions == []
    assert [request.timeout_seconds for request in provider.calls] == [90, 90, 90]


@pytest.mark.asyncio
async def test_synthesis_repair_uses_provider_timeout() -> None:
    provider = ProgrammableFakeModelProvider(
        [FakeModelStep(operation="synthesis", output=_synthesis_output())],
        request_timeout_seconds=90,
    )
    orchestrator = SynthesisOrchestrator(provider)
    request = ModelRequest(
        request_id="synthesis-repair-timeout",
        operation="synthesis",
        input_text="public artifacts",
        output_schema_name=GeneratedSynthesis.__name__,
        timeout_seconds=30,
    )

    repaired = await orchestrator._repair_request(request, None)
    await orchestrator._call(repaired, None)

    assert repaired.timeout_seconds == 90
    assert provider.calls[0].timeout_seconds == 90
