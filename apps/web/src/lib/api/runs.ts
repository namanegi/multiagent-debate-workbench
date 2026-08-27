import type { components } from "@open-debate/contracts/generated/api";

import { apiBaseUrl } from "../config";
import type { RunEvent, RunSummary } from "../../features/run/runState";

type CreateRunResponse = components["schemas"]["CreateRunResponse"];
type CancelRunResponse = components["schemas"]["CancelRunResponse"];
type EventPage = components["schemas"]["EventPage"];

export interface CreateRunInput {
  topic: string;
  goal: string;
  agentCount?: number;
  turnCount?: number;
  researchEnabled?: boolean;
  idempotencyKey?: string;
}

export type StreamStatus = "connecting" | "connected" | "reconnecting" | "closed" | "error";

export interface RunStreamSubscription {
  close: () => void;
}

export async function createRun(
  input: CreateRunInput,
  signal?: AbortSignal,
  baseUrl: string = apiBaseUrl
): Promise<CreateRunResponse> {
  const body = {
    topic: input.topic,
    goal: input.goal || null,
    agent_count: input.agentCount ?? 3,
    turn_count: input.turnCount ?? 2,
    research_enabled: input.researchEnabled ?? false
  };
  const response = await fetch(`${baseUrl}/v1/runs`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "Idempotency-Key": input.idempotencyKey ?? createIdempotencyKey()
    },
    body: JSON.stringify(body),
    signal
  });
  return readJson<CreateRunResponse>(response);
}

export async function fetchRunSummary(runId: string, baseUrl: string = apiBaseUrl): Promise<RunSummary> {
  const response = await fetch(`${baseUrl}/v1/runs/${encodeURIComponent(runId)}`, {
    headers: { Accept: "application/json" }
  });
  return readJson<RunSummary>(response);
}

export async function fetchRunEvents(runId: string, baseUrl: string = apiBaseUrl): Promise<RunEvent[]> {
  const events: RunEvent[] = [];
  let afterSequence = 0;
  let hasMore = true;
  while (hasMore) {
    const query = new URLSearchParams({ after_sequence: String(afterSequence), limit: "1000" });
    const response = await fetch(
      `${baseUrl}/v1/runs/${encodeURIComponent(runId)}/events?${query.toString()}`,
      { headers: { Accept: "application/json" } }
    );
    const page = await readJson<EventPage>(response);
    events.push(...page.events);
    hasMore = page.has_more;
    const lastEvent = page.events[page.events.length - 1];
    afterSequence = page.next_after_sequence ?? (lastEvent?.sequence ?? afterSequence);
  }
  return events;
}

export async function cancelRun(runId: string, baseUrl: string = apiBaseUrl): Promise<CancelRunResponse> {
  const response = await fetch(`${baseUrl}/v1/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
    headers: { Accept: "application/json" }
  });
  return readJson<CancelRunResponse>(response);
}

export function subscribeToRunStream(
  runId: string,
  afterSequence: number,
  onEvent: (event: RunEvent) => void,
  onStatus: (status: StreamStatus) => void,
  baseUrl: string = apiBaseUrl
): RunStreamSubscription {
  const source = new EventSource(
    `${baseUrl}/v1/runs/${encodeURIComponent(runId)}/stream?after_sequence=${afterSequence}`
  );
  const eventTypes: components["schemas"]["RunEventType"][] = [
    "run.created",
    "phase.started",
    "phase.completed",
    "planning.outcome",
    "brief.created",
    "evidence.created",
    "message.created",
    "debate_protocol.outcome.created",
    "synthesis.created",
    "agent.failed",
    "run.cancel_requested",
    "run.cancelled",
    "run.completed",
    "run.partial",
    "run.failed"
  ];

  const handleEvent = (event: Event): void => {
    try {
      const parsed = JSON.parse((event as MessageEvent<string>).data) as RunEvent;
      onEvent(parsed);
      if (["run.completed", "run.partial", "run.cancelled", "run.failed"].includes(parsed.type)) {
        source.close();
        onStatus("closed");
      }
    } catch {
      onStatus("error");
    }
  };

  eventTypes.forEach((eventType) => source.addEventListener(eventType, handleEvent));
  source.onopen = () => onStatus("connected");
  source.onerror = () => onStatus("reconnecting");

  return {
    close: () => {
      source.close();
      onStatus("closed");
    }
  };
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `API request failed with status ${response.status}.`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Keep the status-based error when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function createIdempotencyKey(): string {
  return globalThis.crypto?.randomUUID?.() ?? `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
