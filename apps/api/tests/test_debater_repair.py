from __future__ import annotations

import json
from pathlib import Path

import pytest

from debate_api.domain.models import (
    AgentBrief,
    Claim,
    ClaimType,
    CreateRunRequest,
    InvestigatorDirectedUpdateOutput,
    InvestigatorOpeningOutput,
    Message,
    MessageKind,
    PlanningOutcome,
    PlanningOutcomeCategory,
    Run,
    RunEventType,
    RunPhase,
    ToolPermission,
)
from debate_api.domain.validation import InvariantViolation
from debate_api.orchestration.debater import Debater
from debate_api.orchestration.scheduler import BoundedScheduler
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.model import (
    FakeModelOutcome,
    FakeModelStep,
    ModelProviderError,
    ProgrammableFakeModelProvider,
)


def _brief(agent_id: str, label: str) -> AgentBrief:
    return AgentBrief(
        id=f"brief_{agent_id}",
        agent_id=agent_id,
        label=label,
        focus="focus",
        key_questions=["question"],
        preferred_source_types=["web"],
        deliverable="deliverable",
        tool_permissions=[ToolPermission.SEARCH],
    )


def _store_with_opening_phase(
    tmp_path: Path, provider: ProgrammableFakeModelProvider
) -> tuple[EventStore, Run, list[AgentBrief], BoundedScheduler]:
    store = EventStore(f"sqlite:///{tmp_path / 'debater-repair.db'}")
    run, _ = store.create_run(
        CreateRunRequest(topic="repair target selection", agent_count=2, turn_count=2)
    )
    briefs = [_brief("agent_1", "One"), _brief("agent_2", "Two")]
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.PLANNING, {})
    store.commit_plan(
        run.id,
        PlanningOutcome(
            planner_id="test",
            category=PlanningOutcomeCategory.FALLBACK,
            message="test plan",
            brief_count=2,
        ),
        briefs,
        2,
    )
    store.append_event(run.id, RunEventType.PHASE_COMPLETED, RunPhase.RESEARCHING, {})
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.OPENING, {})
    return store, run, briefs, BoundedScheduler(store.get_run(run.id), store)


def _output(
    text: str, *, target_option: int | None = None, interaction: str | None = None
) -> dict[str, object]:
    payload = {"claims": [{"text": text, "claim_type": "inference", "evidence_ids": []}]}
    if target_option is None and interaction is None:
        return InvestigatorOpeningOutput.model_validate(payload).model_dump(mode="json")
    if target_option is None or interaction is None:
        raise ValueError("directed update test output requires both interaction fields")
    return InvestigatorDirectedUpdateOutput.model_validate(
        {**payload, "target_option": target_option, "interaction_kind": interaction}
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_invalid_self_target_is_repaired_once_to_model_selected_peer(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(operation="agent.answer.initial", output=_output("opening one")),
            FakeModelStep(operation="agent.answer.initial", output=_output("opening two")),
            FakeModelStep(
                operation="agent.answer.update",
                output=_output("bad self target", target_option=1, interaction="challenge"),
            ),
            FakeModelStep(
                operation="agent.answer.update",
                output=_output("repaired peer target", target_option=0, interaction="support"),
            ),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)
    first = await Debater(store, briefs[0], provider).initial_answer(run.id, scheduler=scheduler)
    second = await Debater(store, briefs[1], provider).initial_answer(run.id, scheduler=scheduler)
    peer_claim = second.claims[0].id
    store.append_event(run.id, RunEventType.PHASE_COMPLETED, RunPhase.OPENING, {})
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.DEBATING, {})

    result = await Debater(store, briefs[0], provider).directed_update(
        run.id, 2, [first.message, second.message], scheduler=scheduler
    )

    assert result.repair_attempted is True
    assert result.message.target_claim_id == peer_claim
    assert len(provider.calls) == 4
    assert [call.output_schema_name for call in provider.calls] == [
        "InvestigatorOpeningOutput",
        "InvestigatorOpeningOutput",
        "InvestigatorDirectedUpdateOutput",
        "InvestigatorDirectedUpdateOutput",
    ]
    assert provider.calls[-1].repair_attempts == 0
    assert "Eligible target options: [0]" in provider.calls[-1].input_text


@pytest.mark.asyncio
async def test_malformed_output_uses_one_explicit_repair(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                outcome=FakeModelOutcome.MALFORMED_OUTPUT,
                output=["malformed"],
            ),
            FakeModelStep(operation="agent.answer.initial", output=_output("repaired opening")),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)
    result = await Debater(store, briefs[0], provider).initial_answer(run.id, scheduler=scheduler)

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert all(call.output_schema_name == "InvestigatorOpeningOutput" for call in provider.calls)
    assert provider.calls[-1].repair_attempts == 0
    assert "omit all target fields" in provider.calls[-1].input_text


@pytest.mark.asyncio
async def test_malformed_then_invalid_repair_stops_after_two_calls(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                outcome=FakeModelOutcome.MALFORMED_OUTPUT,
                output=["malformed"],
            ),
            FakeModelStep(
                operation="agent.answer.initial",
                output=_output(
                    "invalid repaired opening", target_option=0, interaction="challenge"
                ),
            ),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    with pytest.raises(ModelProviderError):
        await Debater(store, briefs[0], provider).initial_answer(run.id, scheduler=scheduler)

    assert len(provider.calls) == 2
    assert not store.get_summary(run.id).messages


@pytest.mark.asyncio
async def test_schema_validation_uses_one_explicit_repair(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                outcome=FakeModelOutcome.SCHEMA_FAILURE,
                output={"claims": []},
            ),
            FakeModelStep(operation="agent.answer.initial", output=_output("repaired schema")),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    result = await Debater(store, briefs[0], provider).initial_answer(run.id, scheduler=scheduler)

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert provider.calls[-1].repair_attempts == 0
    assert "omit all target fields" in provider.calls[-1].input_text


@pytest.mark.asyncio
async def test_paper_semantic_safeguard_repairs_multiple_final_answers(
    tmp_path: Path,
) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                output=InvestigatorOpeningOutput.model_validate(
                    {
                        "claims": [
                            {
                                "text": "Candidate. Final answer: 3",
                                "claim_type": "fact",
                                "evidence_ids": [],
                            },
                            {
                                "text": "Other candidate. Final answer: 4",
                                "claim_type": "fact",
                                "evidence_ids": [],
                            },
                        ]
                    }
                ).model_dump(mode="json"),
            ),
            FakeModelStep(
                operation="agent.answer.initial",
                output=_output("Compute 2 + 2 = 4. Final answer: 4"),
            ),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    result = await Debater(
        store, briefs[0], provider, protocol_mode="paper_reproduction"
    ).initial_answer(run.id, scheduler=scheduler)

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert "Paper safeguard" in provider.calls[-1].input_text
    assert result.message.content.count("Final answer:") == 1
    assert result.message.content.endswith("Final answer: 4")
    assert "citation support unavailable" not in result.message.content


@pytest.mark.asyncio
async def test_paper_semantic_safeguard_canonicalizes_second_ambiguous_output(
    tmp_path: Path,
) -> None:
    ambiguous = InvestigatorOpeningOutput.model_validate(
        {
            "claims": [
                {
                    "text": "First calculation. Final answer: 3",
                    "claim_type": "inference",
                    "evidence_ids": [],
                },
                {
                    "text": "Correction gives 4. Final answer: 4 units",
                    "claim_type": "inference",
                    "evidence_ids": [],
                },
            ]
        }
    ).model_dump(mode="json")
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(operation="agent.answer.initial", output=ambiguous),
            FakeModelStep(operation="agent.answer.initial", output=ambiguous),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    result = await Debater(
        store, briefs[0], provider, protocol_mode="paper_reproduction"
    ).initial_answer(run.id, scheduler=scheduler)

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert len(result.claims) == 1
    assert result.message.content.count("Final answer:") == 1
    assert result.message.content.endswith("Final answer: 4")
    assert "Final answer: 3" not in result.message.content


@pytest.mark.asyncio
async def test_paper_semantic_safeguard_uses_first_output_if_retry_is_malformed(
    tmp_path: Path,
) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                output=InvestigatorOpeningOutput.model_validate(
                    {
                        "claims": [
                            {
                                "text": "Candidate. Final answer: 3",
                                "claim_type": "inference",
                                "evidence_ids": [],
                            },
                            {
                                "text": "Correction. Final answer: 4",
                                "claim_type": "inference",
                                "evidence_ids": [],
                            },
                        ]
                    }
                ).model_dump(mode="json"),
            ),
            FakeModelStep(
                operation="agent.answer.initial",
                outcome=FakeModelOutcome.MALFORMED_OUTPUT,
                output=["malformed repair"],
            ),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    result = await Debater(
        store, briefs[0], provider, protocol_mode="paper_reproduction"
    ).initial_answer(run.id, scheduler=scheduler)

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert len(result.claims) == 1
    assert result.message.content.endswith("Final answer: 4")


@pytest.mark.asyncio
async def test_default_semantic_safeguard_repairs_competing_conclusions(
    tmp_path: Path,
) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                output=InvestigatorOpeningOutput.model_validate(
                    {
                        "claims": [
                            {
                                "text": "Final answer: first",
                                "claim_type": "inference",
                                "evidence_ids": [],
                            },
                            {
                                "text": "Conclusion: second",
                                "claim_type": "inference",
                                "evidence_ids": [],
                            },
                        ]
                    }
                ).model_dump(mode="json"),
            ),
            FakeModelStep(
                operation="agent.answer.initial",
                output=InvestigatorOpeningOutput.model_validate(
                    {
                        "claims": [
                            {
                                "text": "One internally consistent conclusion.",
                                "claim_type": "fact",
                                "evidence_ids": [],
                            }
                        ]
                    }
                ).model_dump(mode="json"),
            ),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    result = await Debater(store, briefs[0], provider).initial_answer(
        run.id, scheduler=scheduler
    )

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert "internally consistent conclusion" in provider.calls[-1].input_text
    assert "citation support unavailable" not in result.message.content
    assert result.claims[0].support_warning is not None


@pytest.mark.asyncio
async def test_default_semantic_safeguard_canonicalizes_second_ambiguous_output(
    tmp_path: Path,
) -> None:
    ambiguous = InvestigatorOpeningOutput.model_validate(
        {
            "claims": [
                {
                    "text": "Final answer: first candidate",
                    "claim_type": "inference",
                    "evidence_ids": [],
                },
                {
                    "text": "Conclusion: retained conclusion",
                    "claim_type": "inference",
                    "evidence_ids": [],
                },
            ]
        }
    ).model_dump(mode="json")
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(operation="agent.answer.initial", output=ambiguous),
            FakeModelStep(operation="agent.answer.initial", output=ambiguous),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    result = await Debater(store, briefs[0], provider).initial_answer(
        run.id, scheduler=scheduler
    )

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert "Discarded candidate: first candidate" in result.message.content
    assert result.message.content.count("Final answer:") == 0
    assert result.message.content.count("Conclusion:") == 1


@pytest.mark.asyncio
async def test_plain_paper_mode_uses_no_schema_claims_or_directed_target(
    tmp_path: Path,
) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                output="Compute independently. Final answer: 20",
            ),
            FakeModelStep(
                operation="agent.answer.initial",
                output="Compute independently. Final answer: 19",
            ),
            FakeModelStep(
                operation="agent.answer.update",
                output=(
                    "Earlier candidate. Final answer: 19. Recompute from the question. "
                    "Final answer: 20"
                ),
            ),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)
    first = await Debater(
        store,
        briefs[0],
        provider,
        protocol_mode="paper_reproduction",
        output_mode="plain_text",
    ).initial_answer(run.id, scheduler=scheduler)
    second = await Debater(
        store,
        briefs[1],
        provider,
        protocol_mode="paper_reproduction",
        output_mode="plain_text",
    ).initial_answer(run.id, scheduler=scheduler)
    store.append_event(run.id, RunEventType.PHASE_COMPLETED, RunPhase.OPENING, {})
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.DEBATING, {})

    result = await Debater(
        store,
        briefs[0],
        provider,
        protocol_mode="paper_reproduction",
        output_mode="plain_text",
        plain_text_max_output_tokens=8_192,
    ).directed_update(
        run.id, 2, [first.message, second.message], scheduler=scheduler
    )

    assert len(provider.calls) == 3
    assert all(call.output_schema_name is None for call in provider.calls)
    update_request = provider.calls[-1]
    assert update_request.conversation is not None
    assert update_request.max_output_tokens == 8_192
    assert [message.role for message in update_request.conversation] == [
        "user",
        "assistant",
        "user",
    ]
    assert update_request.conversation[1].content == (
        "Compute independently. Final answer: 20"
    )
    update_prompt = update_request.conversation[-1].content
    assert not update_prompt.lstrip().startswith("{")
    assert "These are the recent solutions from other agents" in update_prompt
    assert "Compute independently. Final answer: 19" in update_prompt
    assert "Compute independently. Final answer: 20" not in update_prompt
    assert "eligible_targets" not in update_prompt
    assert result.claims == ()
    assert result.message.claim_ids == []
    assert result.message.in_reply_to_message_id is None
    assert result.message.target_claim_id is None
    assert result.message.interaction_kind is None
    assert result.message.content == (
        "Earlier candidate. Final answer: 19. Recompute from the question. "
        "Final answer: 20"
    )


@pytest.mark.asyncio
async def test_paper_disabled_thinking_adds_qwen_control_to_user_prompt(
    tmp_path: Path,
) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                output="Direct solution. Final answer: 4",
            )
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    await Debater(
        store,
        briefs[0],
        provider,
        protocol_mode="paper_reproduction",
        output_mode="plain_text",
        thinking_mode="disabled",
    ).initial_answer(run.id, scheduler=scheduler)

    conversation = provider.calls[0].conversation
    assert conversation is not None
    assert conversation[0].content.endswith("\n\n/no_think")


@pytest.mark.asyncio
async def test_retryable_provider_error_uses_one_explicit_repair(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                outcome=FakeModelOutcome.PROVIDER_ERROR,
            ),
            FakeModelStep(operation="agent.answer.initial", output=_output("repaired provider")),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    result = await Debater(store, briefs[0], provider).initial_answer(run.id, scheduler=scheduler)

    assert result.repair_attempted is True
    assert len(provider.calls) == 2
    assert provider.calls[-1].repair_attempts == 0
    assert "omit all target fields" in provider.calls[-1].input_text


@pytest.mark.asyncio
async def test_nonretryable_provider_error_is_not_retried(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(
                operation="agent.answer.initial",
                outcome=FakeModelOutcome.PROVIDER_ERROR,
                retryable=False,
            )
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)

    with pytest.raises(ModelProviderError):
        await Debater(store, briefs[0], provider).initial_answer(run.id, scheduler=scheduler)

    assert len(provider.calls) == 1
    assert not store.get_summary(run.id).messages


@pytest.mark.asyncio
async def test_second_invalid_target_fails_without_server_fallback(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider(
        [
            FakeModelStep(operation="agent.answer.initial", output=_output("opening one")),
            FakeModelStep(operation="agent.answer.initial", output=_output("opening two")),
            FakeModelStep(
                operation="agent.answer.update",
                output=_output("bad one", target_option=9, interaction="challenge"),
            ),
            FakeModelStep(
                operation="agent.answer.update",
                output=_output("bad two", target_option=9, interaction="support"),
            ),
        ]
    )
    store, run, briefs, scheduler = _store_with_opening_phase(tmp_path, provider)
    first = await Debater(store, briefs[0], provider).initial_answer(run.id, scheduler=scheduler)
    second = await Debater(store, briefs[1], provider).initial_answer(run.id, scheduler=scheduler)
    store.append_event(run.id, RunEventType.PHASE_COMPLETED, RunPhase.OPENING, {})
    store.append_event(run.id, RunEventType.PHASE_STARTED, RunPhase.DEBATING, {})

    with pytest.raises(InvariantViolation):
        await Debater(store, briefs[0], provider).directed_update(
            run.id, 2, [first.message, second.message], scheduler=scheduler
        )
    assert len(provider.calls) == 4
    assert not [item for item in store.get_summary(run.id).messages if item.turn_index == 2]


def test_later_turn_input_lists_only_other_agents_claims(tmp_path: Path) -> None:
    provider = ProgrammableFakeModelProvider([])
    store, run, briefs, _ = _store_with_opening_phase(tmp_path, provider)
    del store
    first = Debater._stable_message_id(run.id, "agent_1", 1)
    second = Debater._stable_message_id(run.id, "agent_2", 1)
    first_message = Message(
        id=first,
        author_id="agent_1",
        author_label="One",
        phase=RunPhase.OPENING,
        kind=MessageKind.OPENING,
        content="self answer",
        claim_ids=["claim_self"],
        turn_index=1,
    )
    second_message = first_message.model_copy(
        update={
            "id": second,
            "author_id": "agent_2",
            "author_label": "Two",
            "claim_ids": ["claim_peer"],
        }
    )
    claims = {
        "claim_self": Claim(
            id="claim_self",
            message_id=first,
            text="self claim",
            claim_type=ClaimType.INFERENCE,
            author_id="agent_1",
        ),
        "claim_peer": Claim(
            id="claim_peer",
            message_id=second,
            text="peer claim",
            claim_type=ClaimType.INFERENCE,
            author_id="agent_2",
        ),
    }
    payload = json.loads(
        Debater._input_text(
            run.topic,
            briefs[0],
            {},
            turn_index=2,
            previous_answers=[first_message, second_message],
            previous_claims=claims,
        )
    )
    assert [item["target_option"] for item in payload["eligible_targets"]] == [0]
