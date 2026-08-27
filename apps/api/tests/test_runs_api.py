import json

from fastapi.testclient import TestClient

from debate_api.domain.models import RunEventType, RunPhase


def create_run(client: TestClient, topic: str = "A topic") -> str:
    response = client.post("/v1/runs", json={"topic": topic})
    assert response.status_code == 202
    return response.json()["run"]["id"]


def append_phase(client: TestClient, run_id: str, phase: RunPhase) -> dict:
    event = client.app.state.event_store.append_event(
        run_id, RunEventType.PHASE_STARTED, phase, {"source": "test"}
    )
    return event.model_dump(mode="json")


def test_create_read_and_idempotency_use_the_real_event_store(client: TestClient) -> None:
    payload = {
        "topic": "How should a team compare evidence?",
        "goal": "Make uncertainty explicit.",
    }
    first = client.post("/v1/runs", json=payload, headers={"Idempotency-Key": "same-command"})
    second = client.post("/v1/runs", json=payload, headers={"Idempotency-Key": "same-command"})

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["run"]["id"] == first.json()["run"]["id"]
    run_id = first.json()["run"]["id"]
    assert first.json()["stream_url"] == f"/v1/runs/{run_id}/stream"

    summary = client.get(f"/v1/runs/{run_id}")
    assert summary.status_code == 200
    assert summary.json()["run"]["topic"] == payload["topic"]
    assert summary.json()["run"]["research_enabled"] is False
    events = client.get(f"/v1/runs/{run_id}/events").json()["events"]
    assert [event["sequence"] for event in events] == [1]

    conflict = client.post(
        "/v1/runs",
        json={"topic": "A different topic"},
        headers={"Idempotency-Key": "same-command"},
    )
    assert conflict.status_code == 409


def test_create_run_exposes_agent_turn_bounds(client: TestClient) -> None:
    accepted = client.post(
        "/v1/runs",
        json={
            "topic": "Maximum grid",
            "agent_count": 7,
            "turn_count": 4,
            "research_enabled": False,
        },
    )
    assert accepted.status_code == 202
    returned = accepted.json()["run"]
    assert returned["agent_count"] == 7
    assert returned["turn_count"] == 4
    assert returned["research_enabled"] is False
    for payload in (
        {"topic": "bad", "agent_count": 0, "turn_count": 2},
        {"topic": "bad", "agent_count": 8, "turn_count": 2},
        {"topic": "bad", "agent_count": 3, "turn_count": 0},
        {"topic": "bad", "agent_count": 3, "turn_count": 5},
    ):
        assert client.post("/v1/runs", json=payload).status_code == 422


def test_event_replay_supports_sequence_and_last_event_id_resume(client: TestClient) -> None:
    run_id = create_run(client, "Replayable")
    first = append_phase(client, run_id, RunPhase.PLANNING)
    append_phase(client, run_id, RunPhase.RESEARCHING)
    append_phase(client, run_id, RunPhase.OPENING)

    page = client.get(f"/v1/runs/{run_id}/events?after_sequence=1&limit=2")
    assert page.status_code == 200
    assert [event["sequence"] for event in page.json()["events"]] == [2, 3]
    assert page.json()["has_more"] is True

    resumed = client.get(
        f"/v1/runs/{run_id}/events?after_event_id={first['id']}",
    )
    assert resumed.status_code == 200
    assert [event["sequence"] for event in resumed.json()["events"]] == [3, 4]


def test_sse_replays_persisted_events_and_ends_at_terminal_state(client: TestClient) -> None:
    run_id = create_run(client, "SSE")
    first = append_phase(client, run_id, RunPhase.PLANNING)
    append_phase(client, run_id, RunPhase.RESEARCHING)
    for phase in (
        RunPhase.OPENING,
        RunPhase.DEBATING,
        RunPhase.SYNTHESIZING,
        RunPhase.FINALIZING,
    ):
        append_phase(client, run_id, phase)
    client.app.state.event_store.append_event(
        run_id,
        RunEventType.RUN_COMPLETED,
        RunPhase.FINALIZING,
        {"reason": "test complete"},
    )

    with client.stream("GET", f"/v1/runs/{run_id}/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert body.count("event: ") == 8
    assert f"id: {first['id']}" in body
    assert "event: run.completed" in body
    data_lines = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    assert [json.loads(line)["sequence"] for line in data_lines] == list(range(1, 9))


def test_sse_last_event_id_resumes_without_duplicate_events(client: TestClient) -> None:
    run_id = create_run(client, "SSE resume")
    first = append_phase(client, run_id, RunPhase.PLANNING)
    append_phase(client, run_id, RunPhase.RESEARCHING)
    client.app.state.event_store.append_event(
        run_id, RunEventType.RUN_PARTIAL, RunPhase.RESEARCHING, {"reason": "partial"}
    )

    with client.stream(
        "GET", f"/v1/runs/{run_id}/stream", headers={"Last-Event-ID": first["id"]}
    ) as response:
        body = "".join(response.iter_text())

    data_lines = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    assert [json.loads(line)["sequence"] for line in data_lines] == [3, 4]


def test_cancel_is_cooperative_and_idempotent(client: TestClient) -> None:
    run_id = create_run(client, "Cancel me")
    first = client.post(f"/v1/runs/{run_id}/cancel")
    second = client.post(f"/v1/runs/{run_id}/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["run"]["status"] == "queued"
    events = client.get(f"/v1/runs/{run_id}/events").json()["events"]
    assert [event["type"] for event in events] == ["run.created", "run.cancel_requested"]
