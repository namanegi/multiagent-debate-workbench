import { type FormEvent, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useNavigate, useParams } from "react-router-dom";

import { HealthState } from "./features/health/HealthState";
import { RunWorkspace } from "./features/run/RunWorkspace";
import { createRun } from "./lib/api/runs";

function WorkspacePage(): JSX.Element {
  const navigate = useNavigate();
  const [topic, setTopic] = useState("");
  const [goal, setGoal] = useState("");
  const [agentCount, setAgentCount] = useState(3);
  const [turnCount, setTurnCount] = useState(2);
  const [researchEnabled, setResearchEnabled] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!topic.trim()) {
      setError("Add a question before starting the run.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await createRun({ topic, goal, agentCount, turnCount, researchEnabled });
      navigate(`/runs/${response.run.id}`);
    } catch (submitError: unknown) {
      setError(submitError instanceof Error ? submitError.message : "The run could not be started.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="page-grid">
      <section className="hero-panel">
        <p className="eyebrow">Inspectable AI debate</p>
        <h1>Make disagreement useful.</h1>
        <p className="hero-copy">
          Open Debate Workbench turns an open question into a traceable investigation with visible
          evidence, challenges, and uncertainty.
        </p>
        <Link className="primary-button" to="/runs/demo">
          Open example workspace
        </Link>
      </section>

      <section className="setup-panel" aria-labelledby="setup-heading">
        <div>
          <p className="eyebrow">Inspectable local run</p>
          <h2 id="setup-heading">Start an inspectable run</h2>
        </div>
        <form className="run-form" onSubmit={(event) => void handleSubmit(event)}>
          <label htmlFor="topic">Question to investigate</label>
          <textarea
            id="topic"
            name="topic"
            value={topic}
            onChange={(event) => setTopic(event.target.value)}
            placeholder="What would you like the investigators to examine?"
            rows={5}
            required
          />
          <label htmlFor="goal">Goal or constraint <span>(optional)</span></label>
          <textarea
            id="goal"
            name="goal"
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            placeholder="What should the final synthesis make clear?"
            rows={3}
          />
          <div className="form-row" aria-label="Debate configuration">
            <label htmlFor="agent-count">
              Agents
              <select id="agent-count" name="agent_count" value={agentCount} onChange={(event) => {
                const value = Number(event.target.value);
                setAgentCount(Number.isInteger(value) && value >= 1 && value <= 7 ? value : 3);
              }}>
                {[1, 2, 3, 4, 5, 6, 7].map((value) => <option key={value} value={value}>{value} agent{value === 1 ? "" : "s"}</option>)}
              </select>
            </label>
            <label htmlFor="turn-count">
              Total turns
              <select id="turn-count" name="turn_count" aria-describedby="turn-count-help" value={turnCount} onChange={(event) => {
                const value = Number(event.target.value);
                setTurnCount(Number.isInteger(value) && value >= 1 && value <= 4 ? value : 2);
              }}>
                <option value="1">1 turn</option><option value="2">2 turns</option><option value="3">3 turns</option><option value="4">4 turns</option>
              </select>
            </label>
          </div>
          <p className="muted-copy" id="turn-count-help">
            Every agent answers once per turn. Turn 1 is independent; later turns review the complete previous turn.
          </p>
          <label className="checkbox-row">
            <input type="checkbox" checked={researchEnabled} onChange={(event) => setResearchEnabled(event.target.checked)} />
            Enable web research for this run
          </label>
          <button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? "Starting…" : "Start run"}
          </button>
          {error && <p className="form-error" role="alert">{error}</p>}
        </form>
        <HealthState />
      </section>
    </main>
  );
}

function DemoRunPage(): JSX.Element {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (runId !== "demo") return;

    let active = true;
    void createRun({
      topic: "How can teams compare evidence?",
      goal: "Make the evidence, challenge, and unresolved uncertainty visible.",
      agentCount: 3,
      turnCount: 2,
      researchEnabled: true,
      idempotencyKey: "homepage-demo"
    })
      .then((response) => {
        if (active) navigate(`/runs/${response.run.id}`, { replace: true });
      })
      .catch((loadError: unknown) => {
        if (active) {
          setError(loadError instanceof Error ? loadError.message : "The demo could not be started.");
        }
      });

    return () => {
      active = false;
    };
  }, [navigate, runId]);

  if (!runId) {
    return <NotFoundPage />;
  }
  if (runId === "demo") {
    return (
      <main className="placeholder-page" role={error ? "alert" : "status"}>
        <p className="eyebrow">Example workspace</p>
        <h1>{error ? "The example workspace could not be opened." : "Opening the example workspace…"}</h1>
        {error ? <p>{error}</p> : <p>Creating the persisted run and reconstructing its timeline.</p>}
        {error && (
          <Link className="secondary-button" to="/">
            Back to setup
          </Link>
        )}
      </main>
    );
  }
  return (
    <RunWorkspace runId={runId} />
  );
}

function NotFoundPage(): JSX.Element {
  return (
    <main className="placeholder-page">
      <p className="eyebrow">404</p>
      <h1>That workspace does not exist.</h1>
      <Link className="secondary-button" to="/">
        Return home
      </Link>
    </main>
  );
}

export function App(): JSX.Element {
  return (
    <div className="app-shell">
      <header className="topbar">
        <Link className="brand" to="/" aria-label="Open Debate Workbench home">
          <span className="brand-mark" aria-hidden="true">
            ◇
          </span>
          <span>Open Debate Workbench</span>
        </Link>
        <nav aria-label="Primary navigation">
          <NavLink className={({ isActive }) => (isActive ? "nav-link nav-link--active" : "nav-link")} to="/">
            Workspace
          </NavLink>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<WorkspacePage />} />
        <Route path="/runs/:runId" element={<DemoRunPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
      <footer className="footer">
        <span>Local-first · evidence before eloquence</span>
        <span>v0.1.0</span>
      </footer>
    </div>
  );
}
