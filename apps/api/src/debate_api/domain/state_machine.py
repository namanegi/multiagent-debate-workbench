"""Run phase transitions and preconditions owned by the domain layer."""

from __future__ import annotations

from collections.abc import Mapping

from debate_api.domain.models import Run, RunPhase, RunStatus


class InvalidTransition(ValueError):
    """Raised when a command would move a run outside its protocol."""


PHASE_ORDER: tuple[RunPhase, ...] = (
    RunPhase.CREATED,
    RunPhase.PLANNING,
    RunPhase.RESEARCHING,
    RunPhase.OPENING,
    RunPhase.DEBATING,
    RunPhase.SYNTHESIZING,
    RunPhase.FINALIZING,
)

NEXT_PHASE: Mapping[RunPhase, RunPhase] = {
    left: right for left, right in zip(PHASE_ORDER[:-1], PHASE_ORDER[1:], strict=True)
}

TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.PARTIAL,
    RunStatus.CANCELLED,
    RunStatus.FAILED,
}


def require_active(run: Run) -> None:
    status = RunStatus(run.status)
    if status in TERMINAL_STATUSES:
        raise InvalidTransition(f"run {run.id} is already terminal: {status.value}")


def require_phase_start(run: Run, target: RunPhase) -> None:
    """Validate the one legal next phase."""

    require_active(run)
    current = RunPhase(run.phase)
    target = RunPhase(target)
    if target == RunPhase.CREATED:
        raise InvalidTransition("created is not an executable phase")
    expected = NEXT_PHASE.get(current)
    if target != expected:
        raise InvalidTransition(
            f"illegal phase transition: {current.value} -> {target.value}; "
            f"expected {expected.value if expected else 'terminal'}"
        )


def require_phase_complete(run: Run, phase: RunPhase) -> None:
    require_active(run)
    current = RunPhase(run.phase)
    phase = RunPhase(phase)
    if current != phase:
        raise InvalidTransition(f"cannot complete {phase.value} while run is in {current.value}")


def require_terminal(run: Run, status: RunStatus, phase: RunPhase) -> None:
    status = RunStatus(status)
    phase = RunPhase(phase)
    current = RunPhase(run.phase)
    if status not in TERMINAL_STATUSES:
        raise InvalidTransition(f"{status.value} is not a terminal status")
    if RunStatus(run.status) in TERMINAL_STATUSES:
        raise InvalidTransition(f"run {run.id} is already terminal: {RunStatus(run.status).value}")
    if status == RunStatus.COMPLETED:
        if current != RunPhase.FINALIZING or phase != RunPhase.FINALIZING:
            raise InvalidTransition("completed runs must terminate from finalizing")
    elif phase != current:
        raise InvalidTransition(
            f"terminal event phase {phase.value} does not match {current.value}"
        )
