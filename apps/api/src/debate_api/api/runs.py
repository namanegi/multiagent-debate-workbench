"""Run commands, replay, cancellation, and Server-Sent Events."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Protocol, cast

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from debate_api.domain.models import (
    CancelRunResponse,
    CreateRunRequest,
    CreateRunResponse,
    EventPage,
    RunEvent,
    RunStatus,
    RunSummary,
)
from debate_api.persistence.sqlite import (
    EventStore,
    EventStoreError,
    IdempotencyConflictError,
    RunNotFoundError,
)


class RunLauncher(Protocol):
    async def run(self, run_id: str) -> None:
        """Execute a newly-created run in the background."""


router = APIRouter(prefix="/runs", tags=["runs"])


def _store(request: Request) -> EventStore:
    return cast(EventStore, request.app.state.event_store)


def _not_found(error: RunNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


@router.post("", response_model=CreateRunResponse, status_code=202)
async def create_run(
    body: CreateRunRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CreateRunResponse:
    store = _store(request)
    try:
        run, created = store.create_run(body, idempotency_key)
    except IdempotencyConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (ValueError, EventStoreError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    if created:
        launcher: RunLauncher | None = request.app.state.run_launcher
        if launcher is not None:
            task = asyncio.create_task(launcher.run(run.id))
            request.app.state.run_tasks[run.id] = task
            task.add_done_callback(lambda _: request.app.state.run_tasks.pop(run.id, None))
    prefix = request.app.state.settings.api_prefix
    return CreateRunResponse(run=run, stream_url=f"{prefix}/runs/{run.id}/stream")


@router.get("/{run_id}", response_model=RunSummary)
def get_run(run_id: str, request: Request) -> RunSummary:
    try:
        return _store(request).get_summary(run_id)
    except RunNotFoundError as error:
        raise _not_found(error) from error


@router.get("/{run_id}/events", response_model=EventPage)
def list_events(
    run_id: str,
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
    after_event_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> EventPage:
    try:
        return _store(request).list_events(
            run_id,
            after_sequence=after_sequence,
            after_event_id=after_event_id,
            limit=limit,
        )
    except RunNotFoundError as error:
        raise _not_found(error) from error
    except EventStoreError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get(
    "/{run_id}/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_events(
    run_id: str,
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    store = _store(request)
    try:
        store.get_run(run_id)
        if last_event_id:
            initial_page = store.list_events(
                run_id,
                after_sequence=after_sequence,
                after_event_id=last_event_id,
                limit=100,
            )
            if initial_page.events:
                after_sequence = max(after_sequence, initial_page.events[0].sequence - 1)
            else:
                after_sequence = max(after_sequence, store.last_sequence(run_id))
            # The cursor was validated and resolved successfully before this log.
    except RunNotFoundError as error:
        raise _not_found(error) from error
    except EventStoreError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    async def event_stream() -> AsyncIterator[str]:
        cursor = after_sequence
        last_keepalive = time.monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    return
                page = store.list_events(run_id, after_sequence=cursor, limit=100)
                for event in page.events:
                    cursor = event.sequence
                    yield _format_sse(event)
                run = store.get_run(run_id)
                if run.status in {
                    RunStatus.COMPLETED,
                    RunStatus.PARTIAL,
                    RunStatus.CANCELLED,
                    RunStatus.FAILED,
                } and cursor >= store.last_sequence(run_id):
                    return
                if time.monotonic() - last_keepalive >= 10:
                    yield ": keepalive\n\n"
                    last_keepalive = time.monotonic()
                await asyncio.sleep(0.05)
        finally:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{run_id}/cancel", response_model=CancelRunResponse)
def cancel_run(run_id: str, request: Request) -> CancelRunResponse:
    try:
        result = CancelRunResponse(run=_store(request).request_cancel(run_id))
        return result
    except RunNotFoundError as error:
        raise _not_found(error) from error


def _format_sse(event: RunEvent) -> str:
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.id}\nevent: {event.type}\ndata: {data}\n\n"
