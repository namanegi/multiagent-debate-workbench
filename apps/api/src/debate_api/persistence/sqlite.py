"""Lean SQLite event store for the replayable agent-by-turn debate."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from debate_api.domain.models import (
    AgentBrief,
    Claim,
    CreateRunRequest,
    DebateProtocolOutcome,
    EventPage,
    Evidence,
    Failure,
    Message,
    MessageKind,
    PlanningOutcome,
    PlanningOutcomeEventPayload,
    Run,
    RunEvent,
    RunEventType,
    RunPhase,
    RunStatus,
    RunStopReason,
    RunSummary,
    Synthesis,
    new_id,
    run_from_request,
    utc_now,
)
from debate_api.domain.state_machine import (
    require_phase_complete,
    require_phase_start,
    require_terminal,
)
from debate_api.domain.validation import (
    InvariantViolation,
    validate_message_bundle,
    validate_synthesis,
)


class EventStoreError(RuntimeError):
    pass


class EventLimitReached(EventStoreError):
    pass


class RunNotFoundError(EventStoreError):
    pass


class IdempotencyConflictError(EventStoreError):
    pass


_TERMINAL_EVENTS = {
    RunEventType.RUN_COMPLETED: RunStatus.COMPLETED,
    RunEventType.RUN_PARTIAL: RunStatus.PARTIAL,
    RunEventType.RUN_CANCELLED: RunStatus.CANCELLED,
    RunEventType.RUN_FAILED: RunStatus.FAILED,
}


class EventStore:
    """Persist commands as ordered events and derive the public summary by replay."""

    def __init__(self, database_url: str) -> None:
        prefix = "sqlite:///"
        if not database_url.startswith(prefix):
            raise ValueError("only sqlite:/// database URLs are supported")
        raw_path = database_url[len(prefix) :]
        self.path = Path(raw_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    goal TEXT,
                    limits_json TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    stop_reason TEXT,
                    stop_reason_code TEXT,
                    agent_count INTEGER NOT NULL,
                    turn_count INTEGER NOT NULL,
                    research_enabled INTEGER NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    request_json TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    sequence INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    actor_id TEXT,
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS events_run_sequence ON events(run_id, sequence);
                CREATE TABLE IF NOT EXISTS run_tool_usage (
                    run_id TEXT PRIMARY KEY REFERENCES runs(id),
                    tool_calls INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS research_search_reservations (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    agent_id TEXT NOT NULL,
                    reservation_index INTEGER NOT NULL,
                    verification INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(run_id, agent_id, reservation_index)
                );
                """
            )

    def create_run(
        self, request: CreateRunRequest, idempotency_key: str | None = None
    ) -> tuple[Run, bool]:
        request_json = json.dumps(request.model_dump(mode="json"), sort_keys=True)
        run = run_from_request(request).model_copy(update={"idempotency_key": idempotency_key})
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if idempotency_key:
                    row = connection.execute(
                        "SELECT * FROM runs WHERE idempotency_key = ?", (idempotency_key,)
                    ).fetchone()
                    if row is not None:
                        if row["request_json"] != request_json:
                            raise IdempotencyConflictError("idempotency key payload conflicts")
                        connection.commit()
                        return self._run_from_row(row), False
                self._insert_run(connection, run, request_json)
                self._insert_event(
                    connection,
                    run,
                    RunEventType.RUN_CREATED,
                    RunPhase.CREATED,
                    {"run": run.model_dump(mode="json", exclude={"idempotency_key"})},
                    None,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return run, True

    def append_event(
        self,
        run_id: str,
        event_type: RunEventType,
        phase: RunPhase,
        payload: dict[str, object],
        actor_id: str | None = None,
    ) -> RunEvent:
        event_type, phase = RunEventType(event_type), RunPhase(phase)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._load_run(connection, run_id)
                summary = self._summary(connection, run)
                self._validate_event(run, summary, event_type, phase, payload, actor_id)
                event = self._insert_event(connection, run, event_type, phase, payload, actor_id)
                self._apply_run_state(connection, run, event)
                connection.commit()
                return event
            except Exception:
                connection.rollback()
                raise

    def commit_plan(
        self,
        run_id: str,
        outcome: PlanningOutcome,
        briefs: list[AgentBrief],
        requested_count: int,
    ) -> tuple[PlanningOutcome, list[AgentBrief], bool]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._load_run(connection, run_id)
                existing = self._summary(connection, run)
                if existing.planning_outcome is not None:
                    connection.commit()
                    return existing.planning_outcome, existing.briefs, False
                if RunPhase(run.phase) != RunPhase.PLANNING:
                    raise InvariantViolation("plan requires planning phase")
                if requested_count != run.agent_count or len(briefs) != run.agent_count:
                    raise InvariantViolation("plan must contain exactly the configured agents")
                expected_agents = {f"agent_{index}" for index in range(1, run.agent_count + 1)}
                if {brief.agent_id for brief in briefs} != expected_agents:
                    raise InvariantViolation("plan agent ids do not match the configured grid")
                if len({brief.id for brief in briefs}) != len(briefs):
                    raise InvariantViolation("plan brief ids must be unique")
                if outcome.brief_count != len(briefs):
                    raise InvariantViolation("planning outcome brief count is inconsistent")
                self._insert_event(
                    connection,
                    run,
                    RunEventType.PLANNING_OUTCOME,
                    RunPhase.PLANNING,
                    PlanningOutcomeEventPayload(outcome=outcome).model_dump(mode="json"),
                    outcome.planner_id,
                )
                for brief in briefs:
                    self._insert_event(
                        connection,
                        run,
                        RunEventType.BRIEF_CREATED,
                        RunPhase.PLANNING,
                        {"brief": brief.model_dump(mode="json")},
                        outcome.planner_id,
                    )
                self._insert_event(
                    connection,
                    run,
                    RunEventType.PHASE_COMPLETED,
                    RunPhase.PLANNING,
                    {"phase": "planning"},
                    None,
                )
                require_phase_start(run, RunPhase.RESEARCHING)
                start = self._insert_event(
                    connection,
                    run,
                    RunEventType.PHASE_STARTED,
                    RunPhase.RESEARCHING,
                    {"phase": "researching"},
                    None,
                )
                self._apply_run_state(connection, run, start)
                connection.commit()
                return outcome, briefs, True
            except Exception:
                connection.rollback()
                raise

    def commit_synthesis(
        self, run_id: str, message: Message, synthesis: Synthesis
    ) -> tuple[Message, Synthesis, bool]:
        return self._commit_synthesis(run_id, message, synthesis, partial=False)

    def commit_partial_synthesis(
        self,
        run_id: str,
        message: Message,
        synthesis: Synthesis,
        reason_code: RunStopReason,
        reason: str = "A partial synthesis preserved the available public artifacts.",
    ) -> tuple[Message, Synthesis, bool]:
        return self._commit_synthesis(
            run_id, message, synthesis, partial=True, reason_code=reason_code, reason=reason
        )

    def _commit_synthesis(
        self,
        run_id: str,
        message: Message,
        synthesis: Synthesis,
        *,
        partial: bool,
        reason_code: RunStopReason = RunStopReason.PARTIAL,
        reason: str = "partial synthesis",
    ) -> tuple[Message, Synthesis, bool]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._load_run(connection, run_id)
                summary = self._summary(connection, run)
                if summary.synthesis is not None:
                    persisted = next(
                        item for item in summary.messages if item.id == summary.synthesis.message_id
                    )
                    connection.commit()
                    return persisted, summary.synthesis, False
                expected_phase = RunPhase(run.phase) if partial else RunPhase.SYNTHESIZING
                if (
                    MessageKind(message.kind) != MessageKind.SYNTHESIS
                    or message.phase != expected_phase
                ):
                    raise InvariantViolation("synthesis must use a synthesis message")
                if message.claim_ids:
                    raise InvariantViolation("synthesis message does not create claims")
                validate_message_bundle(
                    message,
                    [],
                    {item.id: item for item in summary.messages},
                    {item.id: item for item in summary.claims},
                    {item.id: item for item in summary.evidence},
                )
                messages = {item.id: item for item in summary.messages}
                messages[message.id] = message
                validate_synthesis(
                    synthesis,
                    messages,
                    {item.id: item for item in summary.claims},
                    {item.id: item for item in summary.evidence},
                )
                self._insert_event(
                    connection,
                    run,
                    RunEventType.MESSAGE_CREATED,
                    expected_phase,
                    {"message": message.model_dump(mode="json"), "claims": []},
                    message.author_id,
                )
                self._insert_event(
                    connection,
                    run,
                    RunEventType.SYNTHESIS_CREATED,
                    expected_phase,
                    {"synthesis": synthesis.model_dump(mode="json")},
                    message.author_id,
                )
                if partial:
                    terminal = self._insert_event(
                        connection,
                        run,
                        RunEventType.RUN_PARTIAL,
                        RunPhase(run.phase),
                        {"reason": reason, "reason_code": reason_code.value},
                        None,
                    )
                    self._apply_run_state(connection, run, terminal)
                connection.commit()
                return message, synthesis, True
            except Exception:
                connection.rollback()
                raise

    def get_run(self, run_id: str) -> Run:
        with self._connection() as connection:
            return self._load_run(connection, run_id)

    def get_summary(self, run_id: str) -> RunSummary:
        with self._connection() as connection:
            run = self._load_run(connection, run_id)
            return self._summary(connection, run)

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        after_event_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connection() as connection:
            self._load_run(connection, run_id)
            if after_event_id is not None:
                row = connection.execute(
                    "SELECT sequence FROM events WHERE run_id = ? AND id = ?",
                    (run_id, after_event_id),
                ).fetchone()
                if row is None:
                    raise EventStoreError("event cursor is not part of this run")
                after_sequence = int(row["sequence"])
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (run_id, after_sequence, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        events = [self._event_from_row(row) for row in rows[:limit]]
        return EventPage(
            events=events,
            has_more=has_more,
            next_after_sequence=events[-1].sequence if has_more and events else None,
        )

    def last_sequence(self, run_id: str) -> int:
        with self._connection() as connection:
            self._load_run(connection, run_id)
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS value FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()
            return int(row["value"])

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            return bool(row["cancel_requested"])

    def request_cancel(self, run_id: str) -> Run:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._load_run(connection, run_id)
                if not connection.execute(
                    "SELECT cancel_requested FROM runs WHERE id = ?", (run_id,)
                ).fetchone()[0]:
                    connection.execute(
                        "UPDATE runs SET cancel_requested = 1 WHERE id = ?", (run_id,)
                    )
                    self._insert_event(
                        connection,
                        run,
                        RunEventType.RUN_CANCEL_REQUESTED,
                        RunPhase(run.phase),
                        {},
                        None,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_run(run_id)

    def tool_calls(self, run_id: str) -> int:
        with self._connection() as connection:
            self._load_run(connection, run_id)
            row = connection.execute(
                "SELECT tool_calls FROM run_tool_usage WHERE run_id = ?", (run_id,)
            ).fetchone()
            return int(row["tool_calls"]) if row else 0

    def reserve_tool_call(self, run_id: str, max_tool_calls: int | None = None) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._load_run(connection, run_id)
                if RunStatus(run.status) not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                    connection.rollback()
                    return False
                limit = min(run.limits.max_tool_calls, max_tool_calls or run.limits.max_tool_calls)
                row = connection.execute(
                    "SELECT tool_calls FROM run_tool_usage WHERE run_id = ?", (run_id,)
                ).fetchone()
                current = int(row["tool_calls"]) if row else 0
                if current >= limit:
                    connection.rollback()
                    return False
                connection.execute(
                    "INSERT INTO run_tool_usage(run_id, tool_calls) VALUES (?, 1) "
                    "ON CONFLICT(run_id) DO UPDATE SET tool_calls = tool_calls + 1",
                    (run_id,),
                )
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    def reserve_research_search(
        self,
        run_id: str,
        agent_id: str,
        *,
        verification: bool = False,
        max_searches: int = 2,
    ) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._load_run(connection, run_id)
                if (
                    RunStatus(run.status) != RunStatus.RUNNING
                    or RunPhase(run.phase) != RunPhase.RESEARCHING
                ):
                    connection.rollback()
                    return False
                summary = self._summary(connection, run)
                if verification:
                    if agent_id != "run-verification":
                        connection.rollback()
                        return False
                    per_agent_limit = 1
                else:
                    if agent_id not in {item.agent_id for item in summary.briefs}:
                        connection.rollback()
                        return False
                    per_agent_limit = max_searches
                total = connection.execute(
                    "SELECT COUNT(*) AS count FROM research_search_reservations WHERE run_id = ?",
                    (run_id,),
                ).fetchone()["count"]
                used = connection.execute(
                    "SELECT COUNT(*) AS count FROM research_search_reservations "
                    "WHERE run_id = ? AND agent_id = ?",
                    (run_id, agent_id),
                ).fetchone()["count"]
                if total >= run.limits.max_retrieval_calls or used >= per_agent_limit:
                    connection.rollback()
                    return False
                connection.execute(
                    "INSERT INTO research_search_reservations"
                    "(run_id, agent_id, reservation_index, verification) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, agent_id, used + 1, int(verification)),
                )
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    def _validate_event(
        self,
        run: Run,
        summary: RunSummary,
        event_type: RunEventType,
        phase: RunPhase,
        payload: dict[str, object],
        actor_id: str | None,
    ) -> None:
        if event_type == RunEventType.PHASE_STARTED:
            require_phase_start(run, phase)
            return
        if event_type == RunEventType.PHASE_COMPLETED:
            require_phase_complete(run, phase)
            return
        if event_type in _TERMINAL_EVENTS:
            require_terminal(run, _TERMINAL_EVENTS[event_type], phase)
            return
        if RunStatus(run.status) not in {RunStatus.QUEUED, RunStatus.RUNNING}:
            raise InvariantViolation("terminal runs cannot accept artifacts")
        if phase != RunPhase(run.phase) and event_type != RunEventType.RUN_CANCEL_REQUESTED:
            raise InvariantViolation("artifact phase does not match the active run phase")
        briefs = {item.agent_id: item for item in summary.briefs}
        if event_type == RunEventType.EVIDENCE_CREATED:
            evidence = Evidence.model_validate(payload.get("evidence"))
            if phase != RunPhase.RESEARCHING:
                raise InvariantViolation("evidence is only produced during initial research")
            if actor_id == "run-verification":
                if evidence.agent_id is not None:
                    raise InvariantViolation("verification evidence cannot impersonate an agent")
            elif actor_id not in briefs or evidence.agent_id != actor_id:
                raise InvariantViolation("evidence actor is not an authoritative investigator")
            if any(item.id == evidence.id for item in summary.evidence):
                raise InvariantViolation("duplicate evidence id")
        elif event_type == RunEventType.MESSAGE_CREATED:
            message = Message.model_validate(payload.get("message"))
            raw_claims = payload.get("claims", [])
            if not isinstance(raw_claims, list):
                raise InvariantViolation("message claims payload must be a list")
            claims = [Claim.model_validate(item) for item in raw_claims]
            if actor_id != message.author_id:
                raise InvariantViolation("message actor does not own the message")
            if (
                MessageKind(message.kind) != MessageKind.SYNTHESIS
                and message.author_id not in briefs
            ):
                raise InvariantViolation("message author is not an authoritative investigator")
            if message.phase != phase:
                raise InvariantViolation("message phase is inconsistent")
            self._validate_turn(run, summary, message)
            validate_message_bundle(
                message,
                claims,
                {item.id: item for item in summary.messages},
                {item.id: item for item in summary.claims},
                {item.id: item for item in summary.evidence},
            )
        elif event_type == RunEventType.DEBATE_PROTOCOL_OUTCOME_CREATED:
            if phase != RunPhase.DEBATING:
                raise InvariantViolation("debate outcome belongs to the debate phase")
            outcome = DebateProtocolOutcome.model_validate(payload.get("outcome"))
            if outcome.expected_agent_messages != run.agent_count * run.turn_count:
                raise InvariantViolation("debate outcome grid is inconsistent")
        elif event_type == RunEventType.SYNTHESIS_CREATED:
            synthesis = Synthesis.model_validate(payload.get("synthesis"))
            validate_synthesis(
                synthesis,
                {item.id: item for item in summary.messages},
                {item.id: item for item in summary.claims},
                {item.id: item for item in summary.evidence},
            )
        elif event_type == RunEventType.AGENT_FAILED:
            failure = Failure.model_validate(payload.get("failure"))
            if failure.phase != phase or (
                failure.agent_id is not None and failure.agent_id not in briefs
            ):
                raise InvariantViolation("failure attribution is inconsistent")

    @staticmethod
    def _validate_turn(run: Run, summary: RunSummary, message: Message) -> None:
        if MessageKind(message.kind) == MessageKind.SYNTHESIS:
            return
        if message.turn_index is None or not 1 <= message.turn_index <= run.turn_count:
            raise InvariantViolation("message is outside the configured turn grid")
        if any(
            item.author_id == message.author_id and item.turn_index == message.turn_index
            for item in summary.messages
        ):
            raise InvariantViolation("agent already answered this turn")
        if message.turn_index == 1:
            if message.kind != MessageKind.OPENING or message.phase != RunPhase.OPENING:
                raise InvariantViolation("turn 1 must be an opening")
            return
        if message.kind != MessageKind.UPDATE or message.phase != RunPhase.DEBATING:
            raise InvariantViolation("later turns must be debate updates")
        previous = [item for item in summary.messages if item.turn_index == message.turn_index - 1]
        if {item.author_id for item in previous} != {item.agent_id for item in summary.briefs}:
            raise InvariantViolation("later turns require every previous-turn answer")
        interaction = (
            message.in_reply_to_message_id,
            message.target_claim_id,
            message.target_agent_id,
            message.interaction_kind,
        )
        if all(item is None for item in interaction):
            if message.claim_ids:
                raise InvariantViolation("undirected updates cannot create directed claims")
            return
        parent = next(
            (item for item in previous if item.id == message.in_reply_to_message_id), None
        )
        if parent is None:
            raise InvariantViolation("directed target parent is not in the previous turn")
        if message.target_agent_id != parent.author_id:
            raise InvariantViolation("directed target agent does not own the parent")
        target = next(
            (claim for claim in summary.claims if claim.id == message.target_claim_id), None
        )
        if target is None or target.message_id != parent.id:
            raise InvariantViolation("directed target claim does not belong to the parent")
        if run.agent_count > 1 and message.target_agent_id == message.author_id:
            raise InvariantViolation("multi-agent updates cannot target self")
        if run.agent_count == 1 and message.target_agent_id != message.author_id:
            raise InvariantViolation("single-agent updates target their own prior answer")

    def _summary(self, connection: sqlite3.Connection, run: Run) -> RunSummary:
        briefs: list[AgentBrief] = []
        evidence: list[Evidence] = []
        messages: list[Message] = []
        claims: list[Claim] = []
        failures: list[Failure] = []
        planning: PlanningOutcome | None = None
        protocol: DebateProtocolOutcome | None = None
        synthesis: Synthesis | None = None
        rows = connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY sequence", (run.id,)
        ).fetchall()
        for row in rows:
            event_type = RunEventType(row["type"])
            payload = json.loads(row["payload_json"])
            if event_type == RunEventType.PLANNING_OUTCOME:
                planning = PlanningOutcomeEventPayload.model_validate(payload).outcome
            elif event_type == RunEventType.BRIEF_CREATED:
                briefs.append(AgentBrief.model_validate(payload["brief"]))
            elif event_type == RunEventType.EVIDENCE_CREATED:
                evidence.append(Evidence.model_validate(payload["evidence"]))
            elif event_type == RunEventType.MESSAGE_CREATED:
                messages.append(Message.model_validate(payload["message"]))
                claims.extend(Claim.model_validate(item) for item in payload.get("claims", []))
            elif event_type == RunEventType.DEBATE_PROTOCOL_OUTCOME_CREATED:
                protocol = DebateProtocolOutcome.model_validate(payload["outcome"])
            elif event_type == RunEventType.AGENT_FAILED:
                failures.append(Failure.model_validate(payload["failure"]))
            elif event_type == RunEventType.SYNTHESIS_CREATED:
                synthesis = Synthesis.model_validate(payload["synthesis"])
        return RunSummary(
            run=run,
            briefs=briefs,
            planning_outcome=planning,
            evidence=evidence,
            messages=messages,
            claims=claims,
            debate_protocol=protocol,
            failures=failures,
            synthesis=synthesis,
        )

    def _insert_run(self, connection: sqlite3.Connection, run: Run, request_json: str) -> None:
        connection.execute(
            "INSERT INTO runs "
            "(id, topic, goal, limits_json, phase, status, created_at, updated_at, "
            "completed_at, stop_reason, stop_reason_code, agent_count, turn_count, "
            "research_enabled, idempotency_key, request_json, cancel_requested) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.id,
                run.topic,
                run.goal,
                json.dumps(run.limits.model_dump(mode="json")),
                run.phase,
                run.status,
                run.created_at.isoformat(),
                run.updated_at.isoformat(),
                None,
                None,
                None,
                run.agent_count,
                run.turn_count,
                int(run.research_enabled),
                run.idempotency_key,
                request_json,
                0,
            ),
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        run: Run,
        event_type: RunEventType,
        phase: RunPhase,
        payload: dict[str, object],
        actor_id: str | None,
    ) -> RunEvent:
        current = connection.execute(
            "SELECT COUNT(*) AS count, COALESCE(MAX(sequence), 0) AS maximum "
            "FROM events WHERE run_id = ?",
            (run.id,),
        ).fetchone()
        if current["count"] >= run.limits.max_events:
            raise EventLimitReached("run event limit reached")
        event = RunEvent(
            id=new_id("event"),
            run_id=run.id,
            sequence=int(current["maximum"]) + 1,
            type=event_type,
            phase=phase,
            actor_id=actor_id,
            payload=payload,
        )
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.run_id,
                event.sequence,
                event.type,
                event.phase,
                event.occurred_at.isoformat(),
                event.actor_id,
                event.schema_version,
                json.dumps(event.payload, ensure_ascii=True),
            ),
        )
        return event

    def _apply_run_state(self, connection: sqlite3.Connection, run: Run, event: RunEvent) -> None:
        now = utc_now().isoformat()
        if event.type == RunEventType.PHASE_STARTED:
            connection.execute(
                "UPDATE runs SET phase = ?, status = ?, updated_at = ? WHERE id = ?",
                (event.phase, RunStatus.RUNNING, now, run.id),
            )
        elif event.type in _TERMINAL_EVENTS:
            reason = str(event.payload.get("reason") or event.type.value)
            code = (
                event.payload.get("reason_code")
                or RunStopReason(_TERMINAL_EVENTS[event.type].value).value
            )
            connection.execute(
                "UPDATE runs SET status = ?, stop_reason = ?, stop_reason_code = ?, "
                "completed_at = ?, updated_at = ? WHERE id = ?",
                (_TERMINAL_EVENTS[event.type], reason, str(code), now, now, run.id),
            )

    def _load_run(self, connection: sqlite3.Connection, run_id: str) -> Run:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return self._run_from_row(row)

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> Run:
        return Run.model_validate(
            {
                "id": row["id"],
                "topic": row["topic"],
                "goal": row["goal"],
                "limits": json.loads(row["limits_json"]),
                "phase": row["phase"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "completed_at": row["completed_at"],
                "stop_reason": row["stop_reason"],
                "stop_reason_code": row["stop_reason_code"],
                "agent_count": row["agent_count"],
                "turn_count": row["turn_count"],
                "research_enabled": bool(row["research_enabled"]),
            }
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> RunEvent:
        return RunEvent.model_validate(
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "sequence": row["sequence"],
                "type": row["type"],
                "phase": row["phase"],
                "occurred_at": row["occurred_at"],
                "actor_id": row["actor_id"],
                "schema_version": row["schema_version"],
                "payload": json.loads(row["payload_json"]),
            }
        )


__all__ = [
    "EventLimitReached",
    "EventStore",
    "EventStoreError",
    "IdempotencyConflictError",
    "RunNotFoundError",
]
