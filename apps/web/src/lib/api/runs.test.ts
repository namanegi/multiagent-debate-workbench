import { createRun, subscribeToRunStream, type StreamStatus } from "./runs";
import type { RunEvent } from "../../features/run/runState";
import type { CreateRunInput } from "./runs";

type EventListener = (event: Event) => void;

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readonly listeners: Record<string, EventListener[]> = {};
  closed = false;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListener): void {
    this.listeners[type] = [...(this.listeners[type] ?? []), listener];
  }

  close(): void {
    this.closed = true;
  }

  emitOpen(): void {
    this.onopen?.();
  }

  emitError(): void {
    this.onerror?.();
  }

  emit(type: string, event: RunEvent): void {
    const message = new MessageEvent(type, { data: JSON.stringify(event) });
    this.listeners[type]?.forEach((listener) => listener(message));
  }
}

const terminalEvent: RunEvent = {
  id: "event_5",
  run_id: "run_1",
  sequence: 5,
  type: "run.completed",
  phase: "finalizing",
  occurred_at: "2026-08-25T00:00:05Z",
  actor_id: null,
  schema_version: 1,
  payload: { reason: "complete" }
};

const debateProtocolEvent: RunEvent = {
  id: "event_6",
  run_id: "run_1",
  sequence: 6,
  type: "debate_protocol.outcome.created",
  phase: "debating",
  occurred_at: "2026-08-25T00:00:06Z",
  actor_id: "orchestrator",
  schema_version: 1,
  payload: {
    outcome: {
      id: "debate_protocol_run_1",
      configured_turns: 1,
      completed_turns: 1,
      expected_agent_messages: 3,
      completed_agent_messages: 3,
      status: "completed",
      stop_reason: "All configured agent turns completed."
    }
  }
};

describe("run event stream client", () => {
  afterEach(() => {
    FakeEventSource.instances = [];
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("reports connection recovery, delivers events, and closes on terminal state", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onEvent = vi.fn<(event: RunEvent) => void>();
    const onStatus = vi.fn<(status: StreamStatus) => void>();

    subscribeToRunStream("run_1", 4, onEvent, onStatus, "http://api.test");

    const source = FakeEventSource.instances[0]!;
    expect(source.url).toBe("http://api.test/v1/runs/run_1/stream?after_sequence=4");

    source.emitOpen();
    source.emitError();
    source.emitOpen();
    source.emit("debate_protocol.outcome.created", debateProtocolEvent);
    source.emit("run.completed", terminalEvent);

    expect(onStatus).toHaveBeenNthCalledWith(1, "connected");
    expect(onStatus).toHaveBeenNthCalledWith(2, "reconnecting");
    expect(onStatus).toHaveBeenNthCalledWith(3, "connected");
    expect(onEvent).toHaveBeenCalledWith(terminalEvent);
    expect(onEvent).toHaveBeenCalledWith(debateProtocolEvent);
    expect(source.closed).toBe(true);
    expect(onStatus).toHaveBeenLastCalledWith("closed");
  });
});

describe("createRun configuration contract", () => {
  afterEach(() => vi.restoreAllMocks());
  const cases: Array<[CreateRunInput, number, number]> = [
    [{ topic: "defaults", goal: "" }, 3, 2],
    [{ topic: "single", goal: "", agentCount: 1, turnCount: 4 }, 1, 4],
    [{ topic: "maximum", goal: "", agentCount: 7, turnCount: 4 }, 7, 4]
  ];
  it.each(cases)("maps configuration to snake_case (%j)", async (input, investigators, rounds) => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => ({}) } as Response);
    await createRun(input);
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(body.agent_count).toBe(investigators);
    expect(body.turn_count).toBe(rounds);
    expect(body.research_enabled).toBe(input.researchEnabled ?? false);
    expect(Number.isNaN(body.agent_count)).toBe(false);
    expect(Number.isNaN(body.turn_count)).toBe(false);
  });

  it("preserves an explicit research opt-in", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => ({}) } as Response);
    await createRun({ topic: "research", goal: "", researchEnabled: true });
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(body.research_enabled).toBe(true);
  });
});
