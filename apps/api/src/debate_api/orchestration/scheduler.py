"""In-process scheduling primitives for one run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from debate_api.domain.models import Run, RunStopReason
from debate_api.persistence.sqlite import EventStore

T = TypeVar("T")


class RunLimitReached(RuntimeError):
    """Raised when a run cannot safely schedule more work."""

    def __init__(self, reason: RunStopReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class CooperativeCancellation(RuntimeError):
    """Raised at an await boundary after a user cancellation request."""


@dataclass
class BoundedScheduler:
    """Enforce concurrency, call budgets, and cancellation."""

    run: Run
    store: EventStore | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    def __post_init__(self) -> None:
        # In-flight agent work is capped by the configured hard agent bound.
        self.semaphore = asyncio.Semaphore(self.run.limits.max_agents)
        self._tool_calls = 0
        self._lock = asyncio.Lock()

    @property
    def tool_calls(self) -> int:
        if self.store is not None:
            return self.store.tool_calls(self.run.id)
        return self._tool_calls

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.run.limits.max_tool_calls - self.tool_calls)

    def check(self) -> None:
        if self.store is not None and self.store.is_cancel_requested(self.run.id):
            raise CooperativeCancellation("run cancellation requested")

    def remaining_seconds(self) -> float | None:
        return None

    async def acquire_tool_call(self) -> None:
        self.check()
        async with self._lock:
            if self.store is not None:
                if not self.store.reserve_tool_call(self.run.id, self.run.limits.max_tool_calls):
                    raise RunLimitReached(
                        RunStopReason.TOOL_CALL_LIMIT,
                        "The per-run tool-call limit was reached.",
                    )
                return
            if self._tool_calls >= self.run.limits.max_tool_calls:
                raise RunLimitReached(
                    RunStopReason.TOOL_CALL_LIMIT,
                    "The per-run tool-call limit was reached.",
                )
            self._tool_calls += 1

    async def run_concurrent(
        self,
        operations: Iterable[Callable[[], Awaitable[T]]],
    ) -> list[T]:
        """Run independent operations with the run semaphore and cancellation checks."""

        async def bounded(operation: Callable[[], Awaitable[T]]) -> T:
            self.check()
            async with self.semaphore:
                self.check()
                return await operation()

        tasks = [asyncio.create_task(bounded(operation)) for operation in operations]
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=0.05,
                )
                for task in done:
                    task.result()
                self.check()
            return [task.result() for task in tasks]
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def run_provider(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one cancellable provider operation."""

        self.check()

        async def invoke() -> T:
            return await operation()

        task: asyncio.Task[T] = asyncio.create_task(invoke())
        try:
            while not task.done():
                await asyncio.wait(
                    {task},
                    timeout=0.05,
                )
                self.check()
            return task.result()
        except BaseException:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
