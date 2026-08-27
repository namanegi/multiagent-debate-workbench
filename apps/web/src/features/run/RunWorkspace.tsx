import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { apiBaseUrl } from "../../lib/config";
import {
  cancelRun,
  fetchRunEvents,
  fetchRunSummary,
  subscribeToRunStream,
  type StreamStatus
} from "../../lib/api/runs";
import {
  createRunViewState,
  hydrateRunSummary,
  reduceEvents,
  reduceRunEvent,
  type AgentBrief,
  type Claim,
  type Evidence,
  type Failure,
  type Message,
  type Run,
  type RunViewState
} from "./runState";
import { safeSourceUrl } from "./inspectorUtils";

interface RunWorkspaceProps {
  runId: string;
  baseUrl?: string;
}

const terminalStatuses = new Set(["completed", "partial", "cancelled", "failed"]);

export function RunWorkspace({ runId, baseUrl = apiBaseUrl }: RunWorkspaceProps): JSX.Element {
  const [state, setState] = useState<RunViewState>(createRunViewState());
  const [loadError, setLoadError] = useState<string | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("connecting");
  const [phaseFilter, setPhaseFilter] = useState("all");
  const [agentFilter, setAgentFilter] = useState("all");
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(null);
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null);
  const [claimFocusRequest, setClaimFocusRequest] = useState(0);

  useEffect(() => {
    if (claimFocusRequest > 0) focusElement("claim-inspector");
  }, [claimFocusRequest]);

  useEffect(() => {
    let active = true;
    let subscription: { close: () => void } | null = null;
    const load = async (): Promise<void> => {
      try {
        const [summary, events] = await Promise.all([
          fetchRunSummary(runId, baseUrl),
          fetchRunEvents(runId, baseUrl)
        ]);
        if (!active) return;
        const replayed = reduceEvents(hydrateRunSummary(summary), events);
        const initial = replayed.run ? replayed : { ...replayed, run: summary.run };
        setState(initial);
        if (initial.run && !terminalStatuses.has(initial.run.status)) {
          subscription = subscribeToRunStream(
            runId,
            initial.lastSequence,
            (event) => setState((current) => reduceRunEvent(current, event)),
            setStreamStatus,
            baseUrl
          );
        } else {
          setStreamStatus("closed");
        }
      } catch (error: unknown) {
        if (!active) return;
        setLoadError(error instanceof Error ? error.message : "The run could not be loaded.");
        setStreamStatus("error");
      }
    };
    void load();
    return () => {
      active = false;
      subscription?.close();
    };
  }, [baseUrl, runId]);

  const messages = useMemo(() => {
    return Object.values(state.messages).filter((message) => {
      const phaseMatches = phaseFilter === "all" || message.phase === phaseFilter;
      const agentMatches = agentFilter === "all" || message.author_id === agentFilter;
      return phaseMatches && agentMatches;
    });
  }, [agentFilter, phaseFilter, state.messages]);
  const agents = useMemo(
    () => Array.from(new Map(Object.values(state.messages).map((message) => [message.author_id, message.author_label])).entries()),
    [state.messages]
  );
  const selectedClaim = selectedClaimId ? state.claims[selectedClaimId] : null;
  const isActive = state.run ? !terminalStatuses.has(state.run.status) : false;

  const selectClaim = (claimId: string): void => {
    setSelectedClaimId(claimId);
    setClaimFocusRequest((request) => request + 1);
  };

  const focusTarget = (message: Message): void => {
    if (!message.in_reply_to_message_id) return;
    setHighlightedMessageId(message.in_reply_to_message_id);
    setPhaseFilter("all");
    setAgentFilter("all");
    const parentId = message.in_reply_to_message_id;
    if (document.getElementById(parentId)) {
      focusElement(parentId);
    } else {
      window.setTimeout(() => focusElement(parentId), 0);
    }
  };

  const handleCancel = async (): Promise<void> => {
    try {
      const response = await cancelRun(runId, baseUrl);
      setState((current) => ({ ...current, run: response.run, cancelRequested: true }));
    } catch (error: unknown) {
      setLoadError(error instanceof Error ? error.message : "The run could not be cancelled.");
    }
  };

  if (loadError && !state.run) {
    return (
      <main className="placeholder-page" role="alert">
        <p className="eyebrow">Run unavailable</p>
        <h1>We could not reconstruct this run.</h1>
        <p>{loadError}</p>
        <Link className="secondary-button" to="/">
          Back to setup
        </Link>
      </main>
    );
  }

  return (
    <main className="run-page">
      <header className="run-header">
        <div>
          <p className="eyebrow">Event-first workspace</p>
          <h1>{state.run?.topic ?? "Loading the run…"}</h1>
          <p className="run-goal">{state.run?.goal || "A traceable multi-agent debate with evidence and directed updates."}</p>
        </div>
        <div className="run-controls">
          <div className="run-status" role="status" aria-live="polite">
            <span className={`status-pill status-pill--${state.run?.status ?? "queued"}`}>
              {state.run?.status ?? "loading"}
            </span>
            <span>{state.run?.phase ?? "created"}</span>
            <span>{streamStatusLabel(streamStatus)}</span>
          </div>
          {isActive && (
            <button className="secondary-button" type="button" onClick={() => void handleCancel()} disabled={state.cancelRequested}>
              {state.cancelRequested ? "Cancellation requested" : "Cancel run"}
            </button>
          )}
          <Link className="secondary-button" to="/">
            New run
          </Link>
        </div>
      </header>

      <section className="run-summary-strip" aria-label="Run progress">
        {state.run && <RunConfiguration run={state.run} />}
        <span>{state.events.length} persisted events</span>
        <span>{Object.keys(state.briefs).length} agent roles</span>
        {state.planningOutcome && (
          <span role="status">
            Planning: {state.planningOutcome.category.split("_").join(" ")}
          </span>
        )}
        <span>{Object.keys(state.evidence).length} evidence items</span>
        <span>{Object.keys(state.failures).length} visible failures</span>
        {state.debateProtocol && (
          <span role="status">
            Turns: {state.debateProtocol.completed_turns}/{state.debateProtocol.configured_turns} · agent answers {state.debateProtocol.completed_agent_messages}/{state.debateProtocol.expected_agent_messages} · {state.debateProtocol.status.replace(/_/g, " ")}
          </span>
        )}
      </section>

      <div className="workspace-grid">
        <section className="timeline-panel" aria-labelledby="timeline-heading">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Chronological artifacts</p>
              <h2 id="timeline-heading">Debate timeline</h2>
            </div>
            <div className="filter-row" aria-label="Timeline filters">
              <label>
                Phase
                <select value={phaseFilter} onChange={(event) => setPhaseFilter(event.target.value)}>
                  <option value="all">All phases</option>
                  <option value="opening">Opening</option>
                  <option value="debating">Debating</option>
                  <option value="synthesizing">Synthesis</option>
                </select>
              </label>
              <label>
                Agent
                <select value={agentFilter} onChange={(event) => setAgentFilter(event.target.value)}>
                  <option value="all">All agents</option>
                  {agents.map(([id, label]) => (
                    <option key={id} value={id}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
          <div className="role-legend" aria-label="Role color legend">
            <span><i className="role-swatch role-swatch--source" aria-hidden="true" />Source</span>
            <span><i className="role-swatch role-swatch--counter" aria-hidden="true" />Counter-evidence</span>
            <span><i className="role-swatch role-swatch--practical" aria-hidden="true" />Practical</span>
            <span><i className="role-swatch role-swatch--synthesis" aria-hidden="true" />Synthesis</span>
          </div>
          <div className="timeline" aria-live="polite">
            {messages.length === 0 && <p className="empty-state">Waiting for the first public message…</p>}
            {messages.map((message) => (
              <MessageCard
                key={message.id}
                message={message}
                claims={Object.values(state.claims).filter((claim) => claim.message_id === message.id)}
                messages={state.messages}
                selectedClaimId={selectedClaimId}
                highlighted={highlightedMessageId === message.id}
                onTarget={() => focusTarget(message)}
                onClaim={selectClaim}
              />
            ))}
          </div>
        </section>

        <aside className="inspector-column" aria-label="Run inspector">
          <BriefInspector briefs={Object.values(state.briefs)} />
          <ClaimInspector
            claim={selectedClaim}
            evidence={state.evidence}
            claims={state.claims}
            messages={state.messages}
            onClaim={selectClaim}
          />
          <SynthesisInspector
            synthesis={state.synthesis}
            failures={Object.values(state.failures)}
            claims={state.claims}
            onClaim={selectClaim}
          />
        </aside>
      </div>
    </main>
  );
}

type RunConfigurationData = Pick<Run, "agent_count" | "turn_count" | "research_enabled"> &
  Partial<Record<string, unknown>>;

export function RunConfiguration({ run }: { run: RunConfigurationData }): JSX.Element {
  return (
    <span role="status" aria-label="Run configuration">
      Configuration: {run.agent_count} agents × {run.turn_count} total turns; research {run.research_enabled ? "enabled" : "disabled"}
    </span>
  );
}

interface MessageCardProps {
  message: Message;
  claims: Claim[];
  messages?: RunViewState["messages"];
  selectedClaimId?: string | null;
  highlighted: boolean;
  onTarget: () => void;
  onClaim: (claimId: string) => void;
}

export function MessageCard({
  message,
  claims,
  highlighted,
  onTarget,
  onClaim,
  messages = {},
  selectedClaimId = null
}: MessageCardProps): JSX.Element {
  const roleTone = messageRoleTone(message);
  const cardClassName = [
    "message-card",
    `message-card--role-${roleTone}`,
    `message-card--kind-${message.kind}`,
    highlighted ? "message-card--highlighted" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article id={message.id} className={cardClassName} tabIndex={-1}>
      <div className="message-meta">
        <span className="message-kind">{message.kind}</span>
        <span>{message.author_label}</span>
        <span>{message.turn_index ? `turn ${message.turn_index}` : message.phase}</span>
        {message.target_agent_id && <span>targets {message.target_agent_id}</span>}
      </div>
      <p className="message-content">{message.content}</p>
      {message.in_reply_to_message_id && message.target_claim_id && (
        <button className="reply-preview" type="button" onClick={onTarget}>
          ↳ {message.interaction_kind} response to {message.target_agent_id}'s prior claim
        </button>
      )}
      {message.kind === "update" && message.in_reply_to_message_id && (
        <p className="challenge-parent-preview">
          Previous-turn target: {messages[message.in_reply_to_message_id]?.content?.slice(0, 240) ?? "unavailable"}
        </p>
      )}
      {message.kind === "update" && message.target_claim_id && (
        <button className="reply-target-link" type="button" onClick={() => onClaim(message.target_claim_id ?? "")}>
          Inspect targeted claim
        </button>
      )}
      {claims.length > 0 && (
        <div className="claim-list" aria-label="Claims in this message">
          {claims.map((claim) => (
            <button
              id={`claim-chip-${claim.id}`}
              key={claim.id}
              className={`claim-chip${selectedClaimId === claim.id ? " claim-chip--selected" : ""}`}
              type="button"
              aria-current={selectedClaimId === claim.id ? "true" : undefined}
              onClick={() => onClaim(claim.id)}
            >
              <span aria-hidden="true">◇</span> {claim.text}
              {selectedClaimId === claim.id && <span className="claim-chip-selection">Selected target claim</span>}
            </button>
          ))}
        </div>
      )}
    </article>
  );
}

export function BriefInspector({ briefs }: { briefs: AgentBrief[] }): JSX.Element {
  return (
    <section className="inspector-card" aria-labelledby="briefs-heading">
      <p className="eyebrow">Why these agents?</p>
      <h2 id="briefs-heading">Investigation briefs</h2>
      {briefs.length === 0 && <p className="muted-copy">Briefs will appear during planning.</p>}
      {briefs.map((brief) => (
        <details key={brief.id} className="brief-detail">
          <summary>{brief.label}</summary>
          <p>{brief.focus}</p>
          <strong>Questions</strong>
          <ul>
            {brief.key_questions.map((question, index) => (
              <li key={`${brief.id}-question-${index}`}>{question}</li>
            ))}
          </ul>
          <strong>Deliverable</strong>
          <p>{brief.deliverable}</p>
        </details>
      ))}
    </section>
  );
}

export function ClaimInspector({
  claim,
  evidence,
  claims,
  messages = {},
  onClaim
}: {
  claim: Claim | null | undefined;
  evidence: RunViewState["evidence"];
  claims: RunViewState["claims"];
  messages?: RunViewState["messages"];
  onClaim: (claimId: string) => void;
}): JSX.Element {
  const evidenceIds = claim?.evidence_ids ?? [];
  const directedUpdates = Object.values(messages).filter(
    (message) => message.kind === "update" && message.target_claim_id === claim?.id
  );
  return (
    <section
      id="claim-inspector"
      className="inspector-card"
      aria-labelledby="claim-heading"
      aria-describedby={claim ? "claim-support-semantics" : undefined}
      aria-live="polite"
      tabIndex={-1}
    >
      <p className="eyebrow">Traceable provenance</p>
      <h2 id="claim-heading">Claim inspector</h2>
      {!claim && <p className="muted-copy">Select a claim in the timeline to inspect its evidence.</p>}
      {claim && (
        <>
          <p className="model-authored-label">Model-authored claim</p>
          <p className="inspector-claim">{claim.text}</p>
          <dl className="inspector-facts">
            <div><dt>Claim type</dt><dd>{claim.claim_type}</dd></div>
            <div><dt>Lifecycle status</dt><dd>{claim.status}</dd></div>
            <div><dt>Citation support</dt><dd>{claim.support_status}</dd></div>
          </dl>
          <p id="claim-support-semantics" className="citation-semantics" role="note">
            Citation availability only; this does not verify whether the claim is true.
          </p>
          {claim.support_warning && (
            <p className="support-warning" role="status">
              Citation support warning (not a fact-verification conclusion): {claim.support_warning}
            </p>
          )}
          <div className="evidence-list">
            <h3>Linked evidence</h3>
            {evidenceIds.length === 0 && (
              <p className="muted-copy">This claim has no evidence references.</p>
            )}
            {evidenceIds.map((evidenceId) => {
              const item = evidence[evidenceId];
              return item ? (
                <EvidenceCard key={item.id} item={item} claims={claims} onClaim={onClaim} />
              ) : (
                <p key={evidenceId} className="missing-record" role="status">
                  Evidence record unavailable for this reference.
                </p>
              );
            })}
          </div>
          {directedUpdates.length > 0 && (
            <div className="revision-history" aria-label="Directed update history">
              <strong>Directed update history</strong>
              {directedUpdates.map((update) => (
                <article key={update.id} className="revision-entry">
                  <span className="status-pill status-pill--neutral">
                    {update.interaction_kind}
                  </span>
                  <p>{update.author_label}, turn {update.turn_index}: {update.content}</p>
                </article>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}

export function EvidenceCard({
  item,
  claims,
  onClaim
}: {
  item: Evidence;
  claims: RunViewState["claims"];
  onClaim: (claimId: string) => void;
}): JSX.Element {
  const linkedClaims = Object.values(claims).filter((claim) => claim.evidence_ids?.includes(item.id));
  const sourceUrl = safeSourceUrl(item);
  return (
    <article className="evidence-card">
      <h3>{item.title}</h3>
      <p className="source-label">Persisted evidence excerpt</p>
      <p className="excerpt-provenance">
        Source extraction or search-provider snippet; separate from the model-authored claim.
      </p>
      <p className="evidence-excerpt">{item.excerpt}</p>
      <dl className="evidence-facts">
        <div><dt>Publisher</dt><dd>{item.publisher}</dd></div>
        <div><dt>Source type</dt><dd>{item.source_type}</dd></div>
        <div><dt>Availability</dt><dd>{item.status}</dd></div>
        <div><dt>Fetch status</dt><dd>{item.fetch_status ?? "not reported"}</dd></div>
        {item.unavailable_reason && <div><dt>Unavailable reason</dt><dd>{item.unavailable_reason}</dd></div>}
      </dl>
      {item.extraction_warnings && item.extraction_warnings.length > 0 && (
        <div className="evidence-warnings" role="status">
          <strong>Extraction warnings</strong>
          <ul>{item.extraction_warnings.map((warning, index) => <li key={`${item.id}-warning-${index}`}>{warning}</li>)}</ul>
        </div>
      )}
      {sourceUrl ? (
        <a
          className="source-link"
          href={sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
          referrerPolicy="no-referrer"
        >
          Open validated source in a new tab
        </a>
      ) : (
        <span className="muted-copy">Source link unavailable: URL is not a validated HTTP(S) address.</span>
      )}
      <div className="linked-claims">
        <strong>Linked claims</strong>
        {linkedClaims.length === 0 ? (
          <p className="muted-copy">No claim record links to this evidence.</p>
        ) : (
          <ul>
            {linkedClaims.map((linkedClaim) => (
              <li key={linkedClaim.id}>
                <button type="button" className="linked-claim-button" onClick={() => onClaim(linkedClaim.id)}>
                  {linkedClaim.text}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </article>
  );
}

export function SynthesisInspector({
  synthesis,
  failures,
  claims,
  onClaim
}: {
  synthesis: RunViewState["synthesis"];
  failures: Failure[];
  claims: RunViewState["claims"];
  onClaim: (claimId: string) => void;
}): JSX.Element {
  return (
    <section className="inspector-card result-card" aria-labelledby="result-heading">
      <p className="eyebrow">Persisted result</p>
      <h2 id="result-heading">Synthesis</h2>
      {!synthesis && <p className="muted-copy">The synthesis will be assembled from persisted artifacts.</p>}
      {synthesis && (
        <>
          <p className="eyebrow" role="status">
            {synthesis.is_partial ? "Partial synthesis" : "Complete synthesis"}
            {synthesis.stop_reason_code ? ` · ${synthesis.stop_reason_code.replace(/_/g, " ")}` : ""}
          </p>
          <p className="result-answer">{synthesis.answer}</p>
          <div className="result-claims">
            <strong>Cited claims</strong>
            {(synthesis.claim_ids ?? []).length === 0 ? (
              <p className="muted-copy">No cited claim references were persisted.</p>
            ) : (
              <ul>
                {(synthesis.claim_ids ?? []).map((claimId, index) => {
                  const claim = claims[claimId];
                  return (
                    <li key={`${claimId}-${index}`}>
                      {claim ? (
                        <button type="button" className="linked-claim-button" aria-label={`Inspect cited claim: ${claim.text}`} onClick={() => onClaim(claimId)}>
                          {claim.text}
                        </button>
                      ) : (
                        <span className="missing-record" role="status">Cited claim record unavailable.</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
          <ResultList title="Completed responsibilities" items={synthesis.completed_responsibilities ?? []} />
          <ResultList title="Missing responsibilities" items={synthesis.missing_responsibilities ?? []} />
          <ResultList title="Consensus" items={synthesis.consensus ?? []} />
          <ResultList title="Unresolved" items={synthesis.disagreements ?? []} />
          <ResultList title="Changed positions" items={synthesis.changed_positions ?? []} />
          <ResultList title="Follow-up checks" items={synthesis.follow_up_checks ?? []} />
        </>
      )}
      {failures.length > 0 && (
        <div className="failure-list" role="status">
          <strong>Partial progress</strong>
          {failures.map((failure) => (
            <p key={failure.id}>{failure.message}</p>
          ))}
        </div>
      )}
    </section>
  );
}

function ResultList({ title, items }: { title: string; items: string[] }): JSX.Element | null {
  if (items.length === 0) return null;
  return (
    <div className="result-list">
      <strong>{title}</strong>
      <ul>
        {items.map((item, index) => (
          <li key={`${title}-${index}-${item}`}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function focusElement(id: string): void {
  const element = document.getElementById(id);
  element?.scrollIntoView({ behavior: "smooth", block: "center" });
  element?.focus({ preventScroll: true });
}
function streamStatusLabel(status: StreamStatus): string {
  if (status === "connected") return "live stream connected";
  if (status === "reconnecting") return "reconnecting…";
  if (status === "closed") return "replay complete";
  if (status === "error") return "stream unavailable";
  return "connecting…";
}

function messageRoleTone(message: Message): "source" | "counter" | "practical" | "synthesis" | "system" {
  if (message.kind === "synthesis" || message.author_id === "synthesizer") return "synthesis";
  if (message.author_id === "agent_1") return "source";
  if (message.author_id === "agent_2") return "counter";
  if (message.author_id === "agent_3") return "practical";
  return "system";
}
