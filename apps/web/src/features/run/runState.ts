import type { components } from "@open-debate/contracts/generated/api";

export type Run = components["schemas"]["Run"];
export type RunEvent = components["schemas"]["RunEvent"];
export type RunSummary = components["schemas"]["RunSummary"];
export type AgentBrief = components["schemas"]["AgentBrief"];
export type PlanningOutcome = components["schemas"]["PlanningOutcome"];
export type Evidence = components["schemas"]["Evidence"];
export type Message = components["schemas"]["Message"];
export type Claim = components["schemas"]["Claim"];
export type DebateProtocolOutcome = components["schemas"]["DebateProtocolOutcome"];
export type Failure = components["schemas"]["Failure"];
export type Synthesis = components["schemas"]["Synthesis"];

export interface RunViewState {
  run: Run | null;
  events: RunEvent[];
  lastSequence: number;
  lastEventId: string | null;
  cancelRequested: boolean;
  briefs: Record<string, AgentBrief>;
  planningOutcome: PlanningOutcome | null;
  evidence: Record<string, Evidence>;
  messages: Record<string, Message>;
  claims: Record<string, Claim>;
  debateProtocol: DebateProtocolOutcome | null;
  failures: Record<string, Failure>;
  synthesis: Synthesis | null;
}

export const emptyRunViewState: RunViewState = {
  run: null,
  events: [],
  lastSequence: 0,
  lastEventId: null,
  cancelRequested: false,
  briefs: {},
  planningOutcome: null,
  evidence: {},
  messages: {},
  claims: {},
  debateProtocol: null,
  failures: {},
  synthesis: null
};

export function createRunViewState(run: Run | null = null): RunViewState {
  return { ...emptyRunViewState, run };
}

export function hydrateRunSummary(summary: RunSummary): RunViewState {
  return {
    ...createRunViewState(summary.run),
    briefs: indexById(summary.briefs ?? []),
    planningOutcome: summary.planning_outcome ?? null,
    evidence: indexById(summary.evidence ?? []),
    messages: indexById(summary.messages ?? []),
    claims: indexById(summary.claims ?? []),
    debateProtocol: summary.debate_protocol ?? null,
    failures: indexById(summary.failures ?? []),
    synthesis: summary.synthesis ?? null
  };
}

export function reduceEvents(initial: RunViewState, events: RunEvent[]): RunViewState {
  return events.reduce(reduceRunEvent, initial);
}

export function reduceRunEvent(state: RunViewState, event: RunEvent): RunViewState {
  if (state.events.some((existing) => existing.id === event.id) || event.sequence <= state.lastSequence) {
    return state;
  }

  const next: RunViewState = {
    ...state,
    events: [...state.events, event],
    lastSequence: event.sequence,
    lastEventId: event.id
  };
  const payload = event.payload ?? {};

  if (event.type === "run.created") {
    const run = readPayload<Run>(payload, "run");
    if (run) {
      next.run = run;
    }
    return next;
  }

  if (event.type === "phase.started") {
    next.run = updateRun(next.run, { phase: event.phase, status: "running" });
    return next;
  }

  if (event.type === "planning.outcome") {
    const outcome = readPayload<PlanningOutcome>(payload, "outcome");
    if (outcome) next.planningOutcome = outcome;
    return next;
  }

  if (event.type === "run.cancel_requested") {
    next.cancelRequested = true;
    return next;
  }

  if (event.type === "run.completed" || event.type === "run.partial" || event.type === "run.cancelled" || event.type === "run.failed") {
    const status = terminalStatus(event.type);
    next.cancelRequested = false;
    next.run = updateRun(next.run, {
      phase: event.phase,
      status,
      completed_at: event.occurred_at,
      stop_reason: readPayload<string>(payload, "reason") ?? null,
      stop_reason_code: readPayload<Run["stop_reason_code"]>(payload, "reason_code") ?? null
    });
    return next;
  }

  if (event.type === "brief.created") {
    const brief = readPayload<AgentBrief>(payload, "brief");
    if (brief) {
      next.briefs = { ...next.briefs, [brief.id]: brief };
    }
  } else if (event.type === "evidence.created") {
    const evidence = readPayload<Evidence>(payload, "evidence");
    if (evidence) {
      next.evidence = { ...next.evidence, [evidence.id]: evidence };
    }
  } else if (event.type === "message.created") {
    const message = readPayload<Message>(payload, "message");
    const claims = readPayload<Claim[]>(payload, "claims") ?? [];
    if (message) {
      next.messages = { ...next.messages, [message.id]: message };
    }
    if (claims.length > 0) {
      next.claims = { ...next.claims, ...indexById(claims) };
    }
  } else if (event.type === "debate_protocol.outcome.created") {
    next.debateProtocol = readPayload<DebateProtocolOutcome>(payload, "outcome") ?? next.debateProtocol;
  } else if (event.type === "agent.failed") {
    const failure = readPayload<Failure>(payload, "failure");
    if (failure) {
      next.failures = { ...next.failures, [failure.id]: failure };
    }
  } else if (event.type === "synthesis.created") {
    next.synthesis = readPayload<Synthesis>(payload, "synthesis") ?? next.synthesis;
  }

  return next;
}

function updateRun(run: Run | null, update: Partial<Run>): Run | null {
  return run ? { ...run, ...update } : run;
}

function terminalStatus(type: RunEvent["type"]): Run["status"] {
  if (type === "run.completed") return "completed";
  if (type === "run.partial") return "partial";
  if (type === "run.cancelled") return "cancelled";
  return "failed";
}

function readPayload<T>(payload: Record<string, unknown>, key: string): T | undefined {
  const value = payload[key];
  return value === undefined ? undefined : (value as T);
}

function indexById<T extends { id: string }>(items: T[]): Record<string, T> {
  return items.reduce<Record<string, T>>((indexed, item) => {
    indexed[item.id] = item;
    return indexed;
  }, {});
}
