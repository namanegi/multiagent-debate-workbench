# Web application

This directory contains the React and TypeScript client.

Source layout:

```text
src/
  app/             Application shell and routing
  features/run/    Topic form, status, cancel, and retry
  features/debate/ Timeline, agent lanes, and annotations
  features/evidence/
                   Evidence and claim inspector
  features/result/ Final synthesis and unresolved questions
  lib/api/         Generated client wrapper and SSE subscription
```

The UI renders both live and replayed runs by reducing the same ordered event stream into normalized
view state. Native EventSource reconnection resumes from the server's `Last-Event-ID`; duplicate or
stale events are ignored by the pure reducer.

## Local development

From the repository root:

```powershell
npm --prefix apps/web install
npm --prefix apps/web run dev
```

The Vite development server runs at `http://localhost:5173` and expects the API at
`http://localhost:8000`. Set `VITE_API_BASE_URL` in `apps/web/.env.local` to override that URL.

Available checks:

```powershell
npm --prefix apps/web run typecheck
npm --prefix apps/web run lint
npm --prefix apps/web run test
npm --prefix apps/web run build
```
