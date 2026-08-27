import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { ClaimInspector, MessageCard } from "./RunWorkspace";
import type { Claim, Message } from "./runState";

const targetClaim: Claim = {
  id: "claim_target", message_id: "opening_2", text: "Targeted prior claim", claim_type: "inference",
  author_id: "agent_2", evidence_ids: [], status: "active", support_status: "unassessed", support_warning: null
};
const update: Message = {
  id: "update_1", author_id: "agent_1", author_label: "Agent 1", phase: "debating", kind: "update",
  content: "I challenge the prior claim.", claim_ids: [], evidence_ids: [], in_reply_to_message_id: "opening_2",
  target_claim_id: targetClaim.id, target_agent_id: "agent_2", interaction_kind: "challenge", turn_index: 2
};

it("shows and navigates a directed challenge update", () => {
  const onClaim = vi.fn();
  render(<MessageCard message={update} claims={[]} messages={{}} highlighted={false} onTarget={vi.fn()} onClaim={onClaim} />);
  expect(screen.getByText(/challenge response to agent_2/i)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /inspect targeted claim/i }));
  expect(onClaim).toHaveBeenCalledWith(targetClaim.id);
});

it("shows persisted directed update history in the claim inspector", () => {
  render(<ClaimInspector claim={targetClaim} evidence={{}} claims={{ [targetClaim.id]: targetClaim }} messages={{ [update.id]: update }} onClaim={vi.fn()} />);
  expect(screen.getByText("Directed update history")).toBeInTheDocument();
  expect(screen.getByText(/Agent 1, turn 2/i)).toBeInTheDocument();
});
