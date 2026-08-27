import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { App } from "./App";
import type { Run, RunSummary } from "./features/run/runState";

const run: Run = {
  id: "run_demo",
  topic: "How can teams compare evidence?",
  goal: "Make the evidence, challenge, and unresolved uncertainty visible.",
  limits: {
    max_agents: 7,
    max_turns: 4,
    max_events: 200,
    max_tool_calls: 12,
    max_retrieval_calls: 7,
    max_context_tokens: 6144
  },
  phase: "finalizing",
  status: "completed",
  created_at: "2026-08-25T00:00:00Z",
  updated_at: "2026-08-25T00:00:06Z",
  completed_at: "2026-08-25T00:00:06Z",
  stop_reason: "complete",
  agent_count: 3,
  turn_count: 2,
  research_enabled: true
};

const summary: RunSummary = {
  run,
  briefs: [],
  evidence: [],
  messages: [],
  claims: [],
  failures: [],
  synthesis: null
};

function response(payload: unknown): Response {
  return {
    ok: true,
    json: async () => payload
  } as Response;
}

describe("homepage demo flow", () => {
  afterEach(() => vi.restoreAllMocks());

  it("creates the demo run from the CTA and reaches its terminal workspace", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/health/ready")) {
        return response({
          status: "ready",
          service: "open-debate-api",
          version: "0.1.0",
          environment: "test",
          checks: {
            configuration: { status: "ok", detail: "Configuration loaded." },
            database: { status: "ok", detail: "SQLite is available." }
          }
        });
      }
      if (init?.method === "POST" && url.endsWith("/v1/runs")) {
        return response({ run, stream_url: `/v1/runs/${run.id}/stream` });
      }
      if (url.includes("/events?")) {
        return response({ events: [], has_more: false, next_after_sequence: null });
      }
      if (url.endsWith(`/v1/runs/${run.id}`)) {
        return response(summary);
      }
      throw new Error(`Unexpected request: ${url}`);
    });

    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>
    );

    fireEvent.click(await screen.findByRole("link", { name: "Open example workspace" }));

    expect(await screen.findByRole("heading", { name: run.topic })).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
    const createCall = fetchMock.mock.calls.find(([, request]) => request?.method === "POST");
    expect(createCall?.[1]?.headers).toMatchObject({ "Idempotency-Key": "homepage-demo" });
    expect(JSON.parse(String(createCall?.[1]?.body))).toEqual({
      topic: run.topic,
      goal: run.goal,
      agent_count: 3,
      turn_count: 2,
      research_enabled: true
    });
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });
});
