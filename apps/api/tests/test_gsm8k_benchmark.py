from __future__ import annotations

from types import SimpleNamespace

import pytest

from debate_api.benchmark.arithmetic import ArithmeticAdapter
from debate_api.benchmark.gsm8k import (
    BenchmarkSample,
    GSM8KAdapter,
    extract_final_answer,
    select_samples,
)
from debate_api.benchmark.runner import (
    MatrixCell,
    SampleResult,
    _paper_plurality_answer,
    _per_turn_majority_accuracy,
    _strict_majority_answer,
    aggregate_results,
    build_matrix,
    complete_matrix,
    run_sample,
)
from debate_api.benchmark.svamp import SVAMPAdapter
from debate_api.providers.model import FakeModelStep, ProgrammableFakeModelProvider


def test_final_round_strict_majority_is_correct() -> None:
    assert _strict_majority_answer(["7", "7", "3"], expected_agent_count=3) == "7"


@pytest.mark.asyncio
async def test_synthesis_numbers_do_not_affect_final_round_score(monkeypatch) -> None:
    import debate_api.benchmark.runner as runner
    monkeypatch.setattr(
        runner,
        "extract_final_answer",
        lambda text: "999" if "synthesis" in text.lower() else "7",
    )

    result = await runner.run_sample(
        BenchmarkSample("fixture-0", "What is 2 + 2?", "#### 7", "7"),
        MatrixCell(3, 1),
    )
    assert result.prediction == "7"


def test_final_round_without_strict_majority_is_incorrect() -> None:
    assert _strict_majority_answer(["7", "3", "5"], expected_agent_count=3) is None


def test_paper_plurality_matches_first_agent_tie_breaking() -> None:
    assert _paper_plurality_answer(["7", "3", "5"]) == "7"
    assert _paper_plurality_answer(["3", "7", "7"]) == "7"
    assert _paper_plurality_answer([None, None]) is None


@pytest.mark.asyncio
async def test_partial_paper_turn_is_not_scored_as_a_vote(monkeypatch) -> None:
    import debate_api.benchmark.runner as runner

    original_latest_messages = runner._latest_messages

    def drop_one_agent(messages, turn: int, count: int):
        selected = original_latest_messages(messages, turn, count)
        return selected[:-1] if turn == 1 else selected

    monkeypatch.setattr(runner, "_latest_messages", drop_one_agent)
    result = await runner.run_sample(
        BenchmarkSample("fixture-partial", "What is 2 + 2?", "#### 4", "4"),
        MatrixCell(3, 1),
        protocol_mode="paper_reproduction",
    )

    assert result.per_turn_predictions["1"] is None


def test_extract_final_answer_is_deterministic() -> None:
    assert extract_final_answer("steps ... #### 1,200") == "1200"
    assert extract_final_answer(r"\boxed{3.50}") == "3.5"
    assert extract_final_answer("no number") is None


def test_adapter_selection_is_revision_pinned_and_deterministic() -> None:
    rows = [
        {"question": "one", "answer": "work\n#### 1"},
        {"question": "two", "answer": "work\n#### 2"},
        {"question": "three", "answer": "work\n#### 3"},
    ]

    def loader(*args: object, **kwargs: object) -> list[dict[str, str]]:
        assert args == ("openai/gsm8k", "main")
        assert kwargs["revision"] == GSM8KAdapter.revision
        assert kwargs["split"] == "test"
        return rows

    adapter = GSM8KAdapter(loader=loader)
    assert [item.sample_id for item in adapter.load("test")] == [
        "test-0",
        "test-1",
        "test-2",
    ]


def test_arithmetic_adapter_reproduces_authors_numpy_sequence() -> None:
    samples = ArithmeticAdapter(count=3).load()
    actual = [
        (sample.sample_id, sample.question, sample.reference_final_answer)
        for sample in samples
    ]
    suffix = " Make sure to state your answer at the end of the response."
    assert actual == [
        (
            "arithmetic-0",
            "What is the result of 12+15*21+0-3*27?" + suffix,
            "246",
        ),
        (
            "arithmetic-1",
            "What is the result of 3+7*9+19-21*18?" + suffix,
            "-293",
        ),
        (
            "arithmetic-2",
            "What is the result of 4+23*6+24-24*12?" + suffix,
            "-122",
        ),
    ]
    assert all(sample.benchmark == "arithmetic" for sample in samples)


def test_svamp_adapter_is_revision_pinned_and_normalizes_results() -> None:
    rows = [
        {
            "id": "svamp__chal-1",
            "question": "What is one thousand?",
            "chain": "<result>1000</result>",
            "result": "1_000.0",
        }
    ]

    def loader(*args: object, **kwargs: object) -> list[dict[str, str]]:
        assert args == ("MU-NLPC/Calc-svamp",)
        assert kwargs["revision"] == SVAMPAdapter.revision
        assert kwargs["split"] == "test"
        return rows

    samples = SVAMPAdapter(loader=loader).load()
    assert samples == [
        BenchmarkSample(
            sample_id="svamp__chal-1",
            question="What is one thousand?",
            reference_answer="<result>1000</result>",
            reference_final_answer="1000",
            benchmark="svamp",
        )
    ]


def test_matrix_covers_all_agent_turn_cells_and_explicit_deduplicates() -> None:
    cells = complete_matrix()
    assert len(cells) == 28
    assert {cell.agent_count for cell in cells} == set(range(1, 8))
    assert {cell.turn_count for cell in cells} == set(range(1, 5))
    assert build_matrix(explicit=[(3, 2), MatrixCell(3, 2), (1, 1)]) == [
        MatrixCell(1, 1),
        MatrixCell(3, 2),
    ]


def test_seed_selection_is_reproducible_and_id_selection_is_exact() -> None:
    samples = [BenchmarkSample(f"test-{index}", "q", "#### 1", "1") for index in range(5)]
    first = [sample.sample_id for sample in select_samples(samples, seed=7, limit=3)]
    second = [sample.sample_id for sample in select_samples(samples, seed=7, limit=3)]
    assert first == second
    assert [
        sample.sample_id for sample in select_samples(samples, sample_ids=["test-3", "test-1"])
    ] == [
        "test-3",
        "test-1",
    ]


def test_failed_sample_stays_in_accuracy_and_completion_denominators() -> None:
    good = SampleResult(
        sample_id="test-0",
        cell="1x1",
        agent_count=1,
        turn_count=1,
        prediction="4",
        reference_final_answer="4",
        status="completed",
        completed=True,
        error=None,
        completed_agent_messages=1,
        expected_agent_messages=1,
        elapsed_ms=1,
        calls=1,
        input_tokens=1,
        output_tokens=1,
        per_turn={"1": {"agent_1": True}},
        answer_changes=0,
        incorrect_to_correct=0,
        correct_to_incorrect=0,
        consensus=True,
        false_consensus=False,
        directed_target_validity=None,
        challenge_count=0,
        support_count=0,
    )
    failed = SampleResult(
        sample_id="test-1",
        cell="1x1",
        agent_count=1,
        turn_count=1,
        prediction=None,
        reference_final_answer="4",
        status="error",
        completed=False,
        error="ProviderError",
        completed_agent_messages=0,
        expected_agent_messages=1,
        elapsed_ms=1,
        calls=0,
        input_tokens=None,
        output_tokens=None,
        per_turn={"1": {"agent_1": None}},
        answer_changes=0,
        incorrect_to_correct=0,
        correct_to_incorrect=0,
        consensus=None,
        false_consensus=None,
        directed_target_validity=None,
        challenge_count=0,
        support_count=0,
    )
    aggregate = aggregate_results([good, failed])[0]
    assert aggregate["sample_count"] == 2
    assert aggregate["accuracy"] == 0.5
    assert aggregate["completion_rate"] == 0.5
    assert aggregate["per_turn_accuracy"] == {"1": 0.5}
    assert aggregate["per_turn_majority_accuracy"] == {"1": 0.5}


def test_nested_turn_curve_scores_one_shared_trajectory() -> None:
    row = SimpleNamespace(
        turn_count=4,
        agent_count=3,
        per_turn={
            "1": {"agent_1": True, "agent_2": True, "agent_3": False},
            "2": {"agent_1": True, "agent_2": False, "agent_3": False},
            "3": {"agent_1": True, "agent_2": True, "agent_3": True},
            "4": {},
        },
    )

    assert _per_turn_majority_accuracy([row]) == {
        "1": 1.0,
        "2": 0.0,
        "3": 1.0,
        "4": 0.0,
    }


def test_paper_reproduction_plan_and_prompt_are_isolated_from_default() -> None:
    from debate_api.orchestration.debater import Debater
    from debate_api.orchestration.topic import TopicOrchestrator

    run = SimpleNamespace(id="run-paper", topic="What is 2 + 2?", agent_count=3)
    plan = TopicOrchestrator.paper_reproduction_plan(run)

    assert len(plan.briefs) == 3
    assert {brief.label for brief in plan.briefs} == {"Independent math solver"}
    assert len({brief.focus for brief in plan.briefs}) == 1
    assert all(not brief.tool_permissions for brief in plan.briefs)
    default_prompt = Debater._turn_instruction(2, "default")
    assert "challenge or support one explicit peer claim" in default_prompt
    assert "one unambiguous conclusion" in default_prompt
    assert "Never present competing final answers" in default_prompt
    paper_prompt = Debater._turn_instruction(2, "paper_reproduction")
    assert "Recompute the problem yourself" in paper_prompt
    assert "change your answer only when verification warrants it" in paper_prompt
    assert "Final answer: <number>" in paper_prompt
    assert "Return exactly one claim" in paper_prompt
    assert "exactly once" in paper_prompt


def test_checker_role_profile_is_deterministic_and_tool_free() -> None:
    from debate_api.orchestration.topic import TopicOrchestrator

    run = SimpleNamespace(id="run-checker", topic="What is 2 + 2?", agent_count=3)
    plan = TopicOrchestrator.paper_reproduction_plan(run, "checker")

    assert [brief.label for brief in plan.briefs] == [
        "Independent math solver",
        "Independent math solver",
        "Arithmetic verifier",
    ]
    assert "operator precedence" in plan.briefs[-1].focus
    assert all(not brief.tool_permissions for brief in plan.briefs)
    assert plan.planner_id == "math-paper-checker-v1"


def test_semantic_checker_role_rejects_majority_without_recomputation() -> None:
    from debate_api.orchestration.topic import TopicOrchestrator

    run = SimpleNamespace(id="run-semantic-checker", topic="word problem", agent_count=4)
    plan = TopicOrchestrator.paper_reproduction_plan(run, "checker_semantic")

    assert [brief.label for brief in plan.briefs] == [
        "Independent math solver",
        "Independent math solver",
        "Independent math solver",
        "Semantic and arithmetic verifier",
    ]
    checker = plan.briefs[-1]
    assert "exact quantity and unit" in checker.focus
    assert "Do not follow a majority" in checker.focus
    assert "earliest unsupported semantic" in checker.key_questions[1]
    assert plan.planner_id == "math-paper-checker-semantic-v1"


def test_invalid_protocol_mode_fails_before_execution() -> None:
    from debate_api.orchestration.debate import DebateOrchestrator

    with pytest.raises(ValueError, match="unsupported protocol_mode"):
        DebateOrchestrator(object(), protocol_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="paper_role_profile requires"):
        DebateOrchestrator(object(), paper_role_profile="checker")  # type: ignore[arg-type]


def test_false_consensus_uses_last_agent_answers_not_synthesis() -> None:
    result = SampleResult(
        sample_id="test-0",
        cell="2x2",
        agent_count=2,
        turn_count=2,
        prediction="4",
        reference_final_answer="5",
        status="completed",
        completed=True,
        error=None,
        completed_agent_messages=4,
        expected_agent_messages=4,
        elapsed_ms=1,
        calls=1,
        input_tokens=1,
        output_tokens=1,
        per_turn={},
        answer_changes=0,
        incorrect_to_correct=0,
        correct_to_incorrect=0,
        consensus=True,
        false_consensus=True,
        directed_target_validity=1,
        challenge_count=1,
        support_count=1,
    )
    assert aggregate_results([result])[0]["false_consensus_rate"] == 1.0


@pytest.mark.asyncio
async def test_run_sample_reuses_production_orchestrator_grid() -> None:
    from debate_api.benchmark.runner import run_sample

    result = await run_sample(
        BenchmarkSample("fixture-0", "What is 2 + 2?", "#### 4", "4"),
        MatrixCell(3, 2),
        provider=None,
    )
    assert result.status == "completed"
    assert result.completed is True
    assert result.completed_agent_messages == 6
    assert result.expected_agent_messages == 6


@pytest.mark.asyncio
async def test_paper_mode_preserves_long_gsm8k_questions() -> None:
    from debate_api.benchmark.runner import run_sample

    result = await run_sample(
        BenchmarkSample("fixture-long", "word " * 140, "#### 4", "4"),
        MatrixCell(3, 2),
        provider=None,
        protocol_mode="paper_reproduction",
    )

    assert result.status == "completed"
    assert result.completed is True
    assert result.completed_agent_messages == 6


@pytest.mark.asyncio
async def test_plain_paper_benchmark_uses_one_text_call_per_agent_turn() -> None:
    provider = ProgrammableFakeModelProvider(
        [
            *[
                FakeModelStep(
                    operation="agent.answer.initial",
                    output="Compute 2 + 2 = 4. Final answer: 4",
                )
                for _ in range(3)
            ],
            *[
                FakeModelStep(
                    operation="agent.answer.update",
                    output="Recompute 2 + 2 = 4. Final answer: 4",
                )
                for _ in range(3)
            ],
        ]
    )

    result = await run_sample(
        BenchmarkSample("fixture-plain", "What is 2 + 2?", "#### 4", "4"),
        MatrixCell(3, 2),
        provider=provider,
        protocol_mode="paper_reproduction",
        output_mode="plain_text",
    )

    assert result.completed is True
    assert result.completed_agent_messages == 6
    assert result.calls == 6
    assert result.per_turn_predictions == {"1": "4", "2": "4"}
    assert result.directed_target_validity is None
    assert result.challenge_count == 0
    assert result.support_count == 0


def test_cli_requires_live_and_real_provider_and_rejects_matrix_sweep() -> None:
    from scripts.gsm8k_benchmark import parse_args

    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--provider", "openai"])
    with pytest.raises(SystemExit):
        parse_args(["--provider", "fake", "--live"])
    with pytest.raises(SystemExit):
        parse_args(["--provider", "ollama", "--live", "--matrix", "1x1", "--agent-sweep"])
    with pytest.raises(SystemExit):
        parse_args(["--provider", "ollama", "--live", "--paper-role-profile", "checker"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--provider",
                "ollama",
                "--live",
                "--model-output-mode",
                "plain_text",
            ]
        )

    args = parse_args(
        [
            "--provider",
            "ollama",
            "--live",
            "--benchmark",
            "arithmetic",
            "--matrix",
            "1x1",
            "--reasoning-effort",
            "medium",
            "--temperature",
            "0.7",
            "--max-output-tokens",
            "32768",
            "--request-timeout-seconds",
            "180",
        ]
    )
    assert args.reasoning_effort == "medium"
    assert args.temperature == 0.7
    assert args.max_output_tokens == 32_768
    assert args.request_timeout_seconds == 180
    assert args.benchmark == "arithmetic"

    for invalid_temperature in ("-0.1", "2.1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--provider",
                    "ollama",
                    "--live",
                    "--temperature",
                    invalid_temperature,
                ]
            )

    for invalid_budget in ("0", "32769"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--provider",
                    "ollama",
                    "--live",
                    "--max-output-tokens",
                    invalid_budget,
                ]
            )


@pytest.mark.asyncio
async def test_cli_records_reasoning_effort_in_benchmark_metadata(monkeypatch, tmp_path) -> None:
    import scripts.gsm8k_benchmark as benchmark

    from debate_api.providers.runtime import ModelProviderRuntime

    class FixtureAdapter:
        dataset_name = "fixture/dataset"
        revision = "fixture-revision"

        def __init__(self, revision: str) -> None:
            self.revision = revision

        def load(self, split: str) -> list[BenchmarkSample]:
            assert split == "test"
            return [BenchmarkSample("test-0", "question", "#### 1", "1")]

    captured: dict[str, object] = {}

    def build_runtime(settings: object) -> ModelProviderRuntime:
        captured["settings"] = settings
        return ModelProviderRuntime(provider=None, mode="ollama", identity="ollama/fixture")

    async def run_fixture(
        sample: BenchmarkSample,
        cell: MatrixCell,
        *,
        provider: object,
        database_path: object = None,
        protocol_mode: str = "default",
        paper_role_profile: str = "homogeneous",
        output_mode: str = "structured_json",
        plain_text_max_output_tokens: int = 2_048,
        thinking_mode: str = "provider_default",
    ) -> object:
        del sample, cell, provider, database_path
        captured["protocol_mode"] = protocol_mode
        captured["paper_role_profile"] = paper_role_profile
        captured["output_mode"] = output_mode
        captured["max_output_tokens"] = plain_text_max_output_tokens
        captured["thinking_mode"] = thinking_mode
        return object()

    def write_fixture(results: object, output_dir, metadata: dict[str, object]) -> None:
        del results, output_dir
        captured["metadata"] = metadata

    monkeypatch.setattr(benchmark, "GSM8KAdapter", FixtureAdapter)
    monkeypatch.setattr(benchmark, "build_model_provider", build_runtime)
    monkeypatch.setattr(benchmark, "run_sample", run_fixture)
    monkeypatch.setattr(benchmark, "write_reports", write_fixture)

    args = benchmark.parse_args(
        [
            "--provider",
            "ollama",
            "--live",
            "--matrix",
            "1x1",
            "--reasoning-effort",
            "medium",
            "--temperature",
            "0.7",
            "--max-output-tokens",
            "32768",
            "--request-timeout-seconds",
            "180",
            "--thinking-mode",
            "disabled",
            "--protocol-mode",
            "paper_reproduction",
            "--paper-role-profile",
            "checker",
            "--output",
            str(tmp_path / "report"),
        ]
    )
    await benchmark._main(args)

    assert captured["metadata"]["reasoning_effort"] == "medium"
    assert captured["metadata"]["temperature"] == 0.7
    assert captured["metadata"]["max_output_tokens"] == 32_768
    assert captured["metadata"]["request_timeout_seconds"] == 180
    assert captured["metadata"]["thinking_mode"] == "disabled"
    assert captured["settings"].model_temperature == 0.7
    assert captured["settings"].ollama_request_timeout_seconds == 180
    assert captured["metadata"]["protocol_mode"] == "paper_reproduction"
    assert captured["metadata"]["paper_role_profile"] == "checker"
    assert captured["metadata"]["model_output_mode"] == "plain_text"
    assert captured["metadata"]["nested_turn_curve_field"] == "per_turn_vote_accuracy"
    assert captured["protocol_mode"] == "paper_reproduction"
    assert captured["paper_role_profile"] == "checker"
    assert captured["output_mode"] == "plain_text"
    assert captured["max_output_tokens"] == 32_768
    assert captured["thinking_mode"] == "disabled"
