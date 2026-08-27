# Open Debate Workbench

Open Debate Workbench is an open-source reproduction and visualization of the
multiagent debate protocol studied by Du et al. in *Improving Factuality and
Reasoning in Language Models through Multiagent Debate* (ICML 2024).

The repository has two connected goals:

1. make every agent response and revision visible in a small Web application;
2. test when additional agents, debate turns, or verification roles improve a
   model's answers on ground-truth reasoning benchmarks.

Our initial experiments with a quantized Qwen2.5-1.5B model are deliberately
exploratory. A dedicated checker changed the direction of error correction on
small GSM8K and SVAMP samples, but did not establish a statistically reliable
accuracy gain. Increasing the group from one to seven agents also failed to
produce a monotonic improvement while latency continued to rise. These results
suggest that verification ability and interaction structure matter more than
agent count alone.

## Experiment report

- [English visual report](docs/reports/small-model-debate-visual-en.html)
- [Chinese visual report](docs/reports/small-model-debate-visual.html)
- [Methods, tables, and limitations](docs/reports/small-model-debate-failure-analysis.md)

The report contains aggregate measurements and selected qualitative examples.
Raw model transcripts and downloaded benchmark data are intentionally excluded
from the repository.

## What is implemented

- A FastAPI debate service with SQLite persistence and server-sent events.
- A React interface for live and replayed agent-turn timelines.
- One OpenAI-compatible model adapter for hosted OpenAI and local Ollama.
- Configurable debates with one to seven agents and one to four turns.
- Paper-style plain-text debate and a structured, inspectable Web protocol.
- Reproducible GSM8K, SVAMP, and arithmetic benchmark adapters.
- Optional checker-role and agent-count ablations.

See [System architecture](docs/architecture.md) for the runtime design and
[Paper notes](docs/references/du-et-al-2024-multiagent-debate-notes.md) for the
research context.

## Repository layout

```text
apps/api/             FastAPI service, debate runtime, and benchmark adapters
apps/web/             React visualization
packages/contracts/   Generated OpenAPI TypeScript contracts
docs/                 Architecture, paper notes, and experiment reports
scripts/              Development and experiment entry points
```

## Run locally

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), Node.js 20 or
newer, and npm.

```powershell
uv sync --directory apps/api --dev
npm --prefix apps/web ci
npm --prefix packages/contracts ci
npm run dev
```

The API starts at `http://localhost:8000`; the Web application starts at
`http://localhost:5173`. The default fake providers make the application and
test suite deterministic and require no network access or credentials.

Run the full offline verification suite with:

```powershell
npm run check
```

Live model and benchmark commands are documented in
[apps/api/README.md](apps/api/README.md).

## Configuration and data policy

Copy `.env.example` to an ignored `.env` or `.env.local` file when a live
provider is needed. Never commit credentials.

The following remain local and ignored:

- provider credentials and machine-specific settings;
- model weights and dataset caches;
- raw prompts, model transcripts, and downloaded source documents;
- SQLite databases, JSONL samples, aggregates, and other run artifacts.

Only source code, deterministic test fixtures, generated API contracts,
aggregate findings, and publication-ready visualizations belong in the public
repository.

## Research scope and limitations

The baseline follows the public protocol and reported parameter sweeps from Du
et al. Our implementation also supports structured claims, directed updates,
optional retrieval, and specialized checker roles so that these choices can be
studied as explicit ablations.

The current findings use small samples and independently sampled runs. They do
not show that checker roles always help, that three agents are optimal, or that
multiagent debate generalizes uniformly across models and tasks. Reproduction
details and confounders are recorded alongside every reported result.

## Citation

This project is based on:

> Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch.
> “Improving Factuality and Reasoning in Language Models through Multiagent
> Debate.” ICML 2024. [OpenReview](https://openreview.net/forum?id=zj7YuTE4t8)

GSM8K and SVAMP remain subject to their respective upstream terms. Local model
weights are not distributed by this repository.

## License

Source code and original documentation are available under the [MIT License](LICENSE).
