import { useEffect, useState } from "react";

import { apiBaseUrl } from "../../lib/config";
import { fetchReadyHealth, type HealthResponse } from "../../lib/api/health";

type HealthStateValue =
  | { status: "loading" }
  | { status: "ready"; data: HealthResponse }
  | { status: "error"; message: string };

interface HealthStateProps {
  baseUrl?: string;
}

export function HealthState({ baseUrl = apiBaseUrl }: HealthStateProps): JSX.Element {
  const [state, setState] = useState<HealthStateValue>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    void fetchReadyHealth(baseUrl, controller.signal)
      .then((data) => setState({ status: "ready", data }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }

        const message = error instanceof Error ? error.message : "The API could not be reached.";
        setState({ status: "error", message });
      });

    return () => controller.abort();
  }, [baseUrl]);

  if (state.status === "loading") {
    return (
      <section className="health-card" aria-live="polite" aria-busy="true">
        <span className="status-dot status-dot--pending" aria-hidden="true" />
        <div>
          <p className="card-label">Local API</p>
          <p>Checking readiness…</p>
        </div>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="health-card health-card--error" role="status" aria-live="polite">
        <span className="status-dot status-dot--error" aria-hidden="true" />
        <div>
          <p className="card-label">Local API · unavailable</p>
          <p>{state.message}</p>
        </div>
      </section>
    );
  }

  return (
    <section className="health-card health-card--ready" role="status" aria-live="polite">
      <span className="status-dot status-dot--ready" aria-hidden="true" />
      <div>
        <p className="card-label">Local API · ready</p>
        <p>
          {state.data.service} is running in {state.data.environment} mode.
        </p>
      </div>
    </section>
  );
}
