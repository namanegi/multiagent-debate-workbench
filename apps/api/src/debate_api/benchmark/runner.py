"""Matrix execution and academic benchmark metrics."""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from debate_api.benchmark.gsm8k import BenchmarkSample, extract_final_answer
from debate_api.domain.models import CreateRunRequest, Message, RunStatus
from debate_api.orchestration.debate import DebateOrchestrator
from debate_api.orchestration.debater import OutputMode, ProtocolMode, ThinkingMode
from debate_api.orchestration.topic import PaperRoleProfile
from debate_api.persistence.sqlite import EventStore
from debate_api.providers.model import (
    ModelProvider,
    ModelResponse,
    ModelTextResponse,
    TextModelProvider,
)


@dataclass(frozen=True, slots=True, order=True)
class MatrixCell:
    agent_count: int
    turn_count: int

    def __post_init__(self) -> None:
        if not 1 <= self.agent_count <= 7 or not 1 <= self.turn_count <= 4:
            raise ValueError("agent_count must be 1..7 and turn_count must be 1..4")

    @property
    def label(self) -> str:
        return f"{self.agent_count}x{self.turn_count}"


def build_matrix(
    *,
    agent_counts: Sequence[int] | None = None,
    turn_counts: Sequence[int] | None = None,
    explicit: Sequence[tuple[int, int] | MatrixCell] | None = None,
) -> list[MatrixCell]:
    """Build a de-duplicated, deterministic matrix (Cartesian sweeps included)."""

    if explicit is not None:
        cells = [item if isinstance(item, MatrixCell) else MatrixCell(*item) for item in explicit]
    else:
        agents = list(agent_counts or [3])
        turns = list(turn_counts or [2])
        cells = [MatrixCell(agent, turn) for agent in agents for turn in turns]
    return sorted(set(cells))


def complete_matrix() -> list[MatrixCell]:
    return build_matrix(agent_counts=range(1, 8), turn_counts=range(1, 5))


@dataclass(slots=True)
class _UsageCounter:
    calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0


class _CountingProvider:
    """Keep provider usage visible to the report while preserving production calls."""

    def __init__(self, provider: ModelProvider, counter: _UsageCounter) -> None:
        self.provider = provider
        self.counter = counter

    @property
    def request_timeout_seconds(self) -> float:
        return self.provider.request_timeout_seconds

    async def generate_structured(self, request: Any, output_schema: Any) -> ModelResponse[Any]:
        response = await self.provider.generate_structured(request, output_schema)
        self._record(response)
        return response

    async def generate_text(self, request: Any) -> ModelTextResponse:
        response = await cast(TextModelProvider, self.provider).generate_text(request)
        self._record(response)
        return response

    def _record(self, response: ModelResponse[Any] | ModelTextResponse) -> None:
        self.counter.calls += 1
        self.counter.latency_ms += response.latency_ms
        if response.usage.prompt_tokens is not None:
            self.counter.input_tokens = (
                self.counter.input_tokens or 0
            ) + response.usage.prompt_tokens
        if response.usage.completion_tokens is not None:
            self.counter.output_tokens = (
                self.counter.output_tokens or 0
            ) + response.usage.completion_tokens


@dataclass(slots=True)
class SampleResult:
    sample_id: str
    cell: str
    agent_count: int
    turn_count: int
    prediction: str | None
    reference_final_answer: str
    status: str
    completed: bool
    error: str | None
    completed_agent_messages: int
    expected_agent_messages: int
    elapsed_ms: float
    calls: int
    input_tokens: int | None
    output_tokens: int | None
    per_turn: dict[str, dict[str, bool | None]]
    answer_changes: int
    incorrect_to_correct: int
    correct_to_incorrect: int
    consensus: bool | None
    false_consensus: bool | None
    directed_target_validity: float | None
    challenge_count: int
    support_count: int
    per_turn_predictions: dict[str, str | None] = field(default_factory=dict)
    vote_mode: str = "strict_majority"

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def _latest_messages(messages: Iterable[Message], turn: int, count: int) -> list[Message]:
    return [
        message
        for message in messages
        if message.turn_index == turn and message.author_id.startswith("agent_")
    ][:count]


def _strict_majority_answer(
    answers: Sequence[str | None], expected_agent_count: int
) -> str | None:
    """Return an answer only when all final agent answers parse and one wins strictly."""

    if len(answers) != expected_agent_count or any(answer is None for answer in answers):
        return None
    counts = Counter(answers)
    majority = [
        answer
        for answer, count in counts.items()
        if count * 2 > expected_agent_count
    ]
    return majority[0] if len(majority) == 1 else None


def _paper_plurality_answer(answers: Sequence[str | None]) -> str | None:
    """Match the authors' mode vote, including first-agent tie breaking."""

    parsed = [answer for answer in answers if answer is not None]
    if not parsed:
        return None
    counts = Counter(parsed)
    highest = max(counts.values())
    return next(answer for answer in parsed if counts[answer] == highest)


async def run_sample(
    sample: BenchmarkSample,
    cell: MatrixCell,
    *,
    provider: ModelProvider | None = None,
    database_path: Path | None = None,
    protocol_mode: ProtocolMode = "default",
    paper_role_profile: PaperRoleProfile = "homogeneous",
    output_mode: OutputMode = "structured_json",
    plain_text_max_output_tokens: int = 32_768,
    thinking_mode: ThinkingMode = "provider_default",
) -> SampleResult:
    """Run one sample through ``DebateOrchestrator`` and retain failures."""

    counter = _UsageCounter()
    counting_provider = _CountingProvider(provider, counter) if provider is not None else None
    temp_context = None
    if database_path is None:
        temp_context = tempfile.TemporaryDirectory(prefix="gsm8k-")
        path = Path(temp_context.name) / "run.db"
    else:
        path = database_path
    started = perf_counter()
    error: str | None = None
    try:
        store = EventStore(f"sqlite:///{path}")
        request = CreateRunRequest(
            topic=sample.question,
            goal=(
                "Solve the synthetic arithmetic expression and give a final number."
                if sample.benchmark == "arithmetic"
                else (
                    f"Solve the {sample.benchmark.upper()} math word problem. "
                    "Give a concise public answer with a final number."
                )
            ),
            agent_count=cell.agent_count,
            turn_count=cell.turn_count,
            research_enabled=False,
        )
        run, _ = store.create_run(request)
        await DebateOrchestrator(
            store,
            step_delay=0,
            provider=counting_provider,
            protocol_mode=protocol_mode,
            paper_role_profile=paper_role_profile,
            output_mode=output_mode,
            plain_text_max_output_tokens=plain_text_max_output_tokens,
            thinking_mode=thinking_mode,
        ).run(run.id)
        summary = store.get_summary(run.id)
        completed = (
            summary.run.status == RunStatus.COMPLETED
            and summary.debate_protocol is not None
            and summary.debate_protocol.completed_agent_messages
            == cell.agent_count * cell.turn_count
            and summary.synthesis is not None
            and not summary.synthesis.is_partial
        )
        completed_agent_messages = (
            summary.debate_protocol.completed_agent_messages
            if summary.debate_protocol is not None
            else 0
        )
        expected_agent_messages = (
            summary.debate_protocol.expected_agent_messages
            if summary.debate_protocol is not None
            else cell.agent_count * cell.turn_count
        )
        status = RunStatus(summary.run.status).value
        previous: dict[str, str | None] = {}
        per_turn: dict[str, dict[str, bool | None]] = {}
        per_turn_predictions: dict[str, str | None] = {}
        changes = incorrect_to_correct = correct_to_incorrect = 0
        for turn in range(1, cell.turn_count + 1):
            entries: dict[str, bool | None] = {}
            turn_messages = _latest_messages(summary.messages, turn, cell.agent_count)
            turn_answers: list[str | None] = []
            for message in turn_messages:
                answer = extract_final_answer(message.content)
                turn_answers.append(answer)
                correct = answer == sample.reference_final_answer if answer is not None else None
                entries[message.author_id] = correct
                if message.author_id in previous and answer != previous[message.author_id]:
                    changes += 1
                    if (
                        previous[message.author_id] == sample.reference_final_answer
                        and correct is not True
                    ):
                        correct_to_incorrect += 1
                    if (
                        previous[message.author_id] != sample.reference_final_answer
                        and correct is True
                    ):
                        incorrect_to_correct += 1
                previous[message.author_id] = answer
            # A missing cell breaks adjacency; do not compare turn N+1 to N-1.
            previous = {
                message.author_id: extract_final_answer(message.content)
                for message in turn_messages
            }
            per_turn[str(turn)] = entries
            per_turn_predictions[str(turn)] = (
                None
                if len(turn_answers) != cell.agent_count
                else _paper_plurality_answer(turn_answers)
                if protocol_mode == "paper_reproduction"
                else _strict_majority_answer(turn_answers, cell.agent_count)
            )
        latest = [
            extract_final_answer(message.content)
            for message in _latest_messages(summary.messages, cell.turn_count, cell.agent_count)
        ]
        # Synthesis remains a UI artifact; benchmark correctness uses final agent
        # answers only, so unrelated numbers in synthesis cannot affect scoring.
        prediction = per_turn_predictions.get(str(cell.turn_count))
        valid_updates = [
            message
            for message in summary.messages
            if message.author_id.startswith("agent_")
            and message.turn_index
            and message.turn_index > 1
        ]
        messages_by_id = {item.id: item for item in summary.messages}
        claims_by_id = {item.id: item for item in summary.claims}
        valid_target_count = 0
        for message in valid_updates:
            parent = messages_by_id.get(message.in_reply_to_message_id or "")
            if (
                parent is not None
                and parent.turn_index == (message.turn_index or 0) - 1
                and message.target_agent_id == parent.author_id
                and message.target_claim_id in parent.claim_ids
                and message.target_claim_id in claims_by_id
            ):
                valid_target_count += 1
        consensus = (
            len(latest) == cell.agent_count and len(set(latest)) == 1 and latest[0] is not None
        )
        failure_categories = sorted({failure.category for failure in summary.failures})
        return SampleResult(
            sample_id=sample.sample_id,
            cell=cell.label,
            agent_count=cell.agent_count,
            turn_count=cell.turn_count,
            prediction=prediction,
            reference_final_answer=sample.reference_final_answer,
            status=status,
            completed=completed,
            error=",".join(failure_categories) or None,
            completed_agent_messages=completed_agent_messages,
            expected_agent_messages=expected_agent_messages,
            elapsed_ms=(perf_counter() - started) * 1000,
            calls=counter.calls,
            input_tokens=counter.input_tokens,
            output_tokens=counter.output_tokens,
            per_turn=per_turn,
            answer_changes=changes,
            incorrect_to_correct=incorrect_to_correct,
            correct_to_incorrect=correct_to_incorrect,
            consensus=consensus if latest else None,
            false_consensus=(consensus and latest[0] != sample.reference_final_answer)
            if latest
            else None,
            directed_target_validity=(
                valid_target_count / len(valid_updates)
                if valid_updates and output_mode == "structured_json"
                else None
            ),
            challenge_count=sum(
                message.interaction_kind == "challenge" for message in valid_updates
            ),
            support_count=sum(message.interaction_kind == "support" for message in valid_updates),
            per_turn_predictions=per_turn_predictions,
            vote_mode=(
                "paper_plurality"
                if protocol_mode == "paper_reproduction"
                else "strict_majority"
            ),
        )
    except Exception as exc:
        error = type(exc).__name__
        return SampleResult(
            sample_id=sample.sample_id,
            cell=cell.label,
            agent_count=cell.agent_count,
            turn_count=cell.turn_count,
            prediction=None,
            reference_final_answer=sample.reference_final_answer,
            status="error",
            completed=False,
            error=error,
            completed_agent_messages=0,
            expected_agent_messages=cell.agent_count * cell.turn_count,
            elapsed_ms=(perf_counter() - started) * 1000,
            calls=counter.calls,
            input_tokens=counter.input_tokens,
            output_tokens=counter.output_tokens,
            per_turn={},
            answer_changes=0,
            incorrect_to_correct=0,
            correct_to_incorrect=0,
            consensus=None,
            false_consensus=None,
            directed_target_validity=None,
            challenge_count=0,
            support_count=0,
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def aggregate_results(results: Sequence[SampleResult]) -> list[dict[str, Any]]:
    """Aggregate by matrix cell; failed/incomplete rows remain in every denominator."""

    grouped: dict[str, list[SampleResult]] = {}
    for result in results:
        grouped.setdefault(result.cell, []).append(result)
    aggregates: list[dict[str, Any]] = []
    for cell, rows in sorted(grouped.items()):
        total = len(rows)
        aggregates.append(
            {
                "cell": cell,
                "sample_count": total,
                "accuracy": sum(row.prediction == row.reference_final_answer for row in rows)
                / total
                if total
                else 0.0,
                "completion_rate": sum(row.completed for row in rows) / total if total else 0.0,
                "consensus_rate": sum(row.consensus is True for row in rows) / total
                if total
                else 0.0,
                "false_consensus_rate": sum(row.false_consensus is True for row in rows) / total
                if total
                else 0.0,
                "answer_changes": sum(row.answer_changes for row in rows),
                "incorrect_to_correct": sum(row.incorrect_to_correct for row in rows),
                "correct_to_incorrect": sum(row.correct_to_incorrect for row in rows),
                "completed_agent_messages": sum(row.completed_agent_messages for row in rows),
                "expected_agent_messages": sum(row.expected_agent_messages for row in rows),
                "per_turn_accuracy": _per_turn_accuracy(rows),
                "per_turn_majority_accuracy": _per_turn_majority_accuracy(rows),
                "per_turn_vote_accuracy": _per_turn_vote_accuracy(rows),
                "directed_target_validity": _mean([row.directed_target_validity for row in rows]),
                "challenge_count": sum(row.challenge_count for row in rows),
                "support_count": sum(row.support_count for row in rows),
                "latency_ms": _mean([row.elapsed_ms for row in rows]),
                "calls": sum(row.calls for row in rows),
                "input_tokens": _sum_optional([row.input_tokens for row in rows]),
                "output_tokens": _sum_optional([row.output_tokens for row in rows]),
            }
        )
    return aggregates


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _per_turn_accuracy(rows: Sequence[SampleResult]) -> dict[str, float | None]:
    values: dict[str, tuple[int, int]] = {}
    for row in rows:
        for turn in range(1, row.turn_count + 1):
            answers = row.per_turn.get(str(turn), {})
            correct, denominator = values.get(str(turn), (0, 0))
            values[str(turn)] = (
                correct + sum(answer is True for answer in answers.values()),
                denominator + row.agent_count,
            )
    return {
        turn: (correct / denominator if denominator else None)
        for turn, (correct, denominator) in sorted(values.items())
    }


def _per_turn_majority_accuracy(rows: Sequence[SampleResult]) -> dict[str, float | None]:
    """Score every turn snapshot from the same completed trajectory."""

    if not rows:
        return {}
    maximum_turn = max(row.turn_count for row in rows)
    return {
        str(turn): sum(
            sum(value is True for value in row.per_turn.get(str(turn), {}).values())
            * 2
            > row.agent_count
            for row in rows
        )
        / len(rows)
        for turn in range(1, maximum_turn + 1)
    }


def _per_turn_vote_accuracy(rows: Sequence[SampleResult]) -> dict[str, float | None]:
    """Score each shared-trajectory snapshot with its configured vote rule."""

    if not rows:
        return {}
    maximum_turn = max(row.turn_count for row in rows)
    return {
        str(turn): sum(
            (
                row.per_turn_predictions.get(str(turn)) == row.reference_final_answer
                if row.per_turn_predictions
                else sum(
                    value is True
                    for value in row.per_turn.get(str(turn), {}).values()
                )
                * 2
                > row.agent_count
            )
            for row in rows
        )
        / len(rows)
        for turn in range(1, maximum_turn + 1)
    }


def write_reports(
    results: Sequence[SampleResult], output_dir: Path, metadata: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_record(), ensure_ascii=False, sort_keys=True) + "\n")
    aggregates = aggregate_results(results)
    (output_dir / "aggregates.json").write_text(
        json.dumps({"metadata": metadata, "aggregates": aggregates}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if aggregates:
        with (output_dir / "aggregates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
