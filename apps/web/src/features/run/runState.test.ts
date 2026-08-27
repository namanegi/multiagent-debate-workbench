import { hydrateRunSummary, reduceRunEvent, type RunEvent, type RunSummary } from "./runState";

const summary: RunSummary = {
  run: {
    id: "run_1",
    topic: "Replay the grid",
    goal: null,
    limits: { max_agents: 7, max_turns: 4, max_events: 200, max_tool_calls: 80, max_retrieval_calls: 7, max_context_tokens: 6144 },
    phase: "debating",
    status: "running",
    created_at: "2026-08-26T00:00:00Z",
    updated_at: "2026-08-26T00:00:00Z",
    completed_at: null,
    stop_reason: null,
    stop_reason_code: null,
    agent_count: 3,
    turn_count: 2,
    research_enabled: false
  },
  briefs: [], evidence: [], messages: [], claims: [], failures: [], synthesis: null,
  planning_outcome: null, debate_protocol: null
};

it("replays a directed update and its stable target claim", () => {
  const event: RunEvent = {
    id: "event_2", run_id: "run_1", sequence: 2, type: "message.created", phase: "debating",
    occurred_at: "2026-08-26T00:00:02Z", actor_id: "agent_1", schema_version: 1,
    payload: {
      message: {
        id: "update_1", author_id: "agent_1", author_label: "Agent 1", phase: "debating",
        kind: "update", content: "A directed update.", claim_ids: ["claim_new"], evidence_ids: [],
        in_reply_to_message_id: "opening_2", target_claim_id: "claim_target",
        target_agent_id: "agent_2", interaction_kind: "challenge", turn_index: 2
      },
      claims: [{ id: "claim_new", message_id: "update_1", text: "Updated claim", claim_type: "inference", author_id: "agent_1", evidence_ids: [], status: "active", support_status: "unassessed", support_warning: null }]
    }
  };
  const state = reduceRunEvent(hydrateRunSummary(summary), event);
  expect(state.messages.update_1.target_claim_id).toBe("claim_target");
  expect(state.messages.update_1.interaction_kind).toBe("challenge");
  expect(state.claims.claim_new.text).toBe("Updated claim");
});
