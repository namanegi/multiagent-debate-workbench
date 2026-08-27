# System architecture

Open Debate Workbench uses one debate runtime for the interactive application
and the benchmark runner. This keeps the visible transcript and the measured
protocol on the same implementation path.

```text
React Web client --HTTP/SSE--> FastAPI service
                                  |
Benchmark CLI ----------------> DebateOrchestrator
                                  |
                    +-------------+-------------+
                    |             |             |
                 Debaters    Research tools   SQLite
                    |
             OpenAI-compatible client
                 /             \
          OpenAI API        Ollama /v1
```

## Runtime components

### Model provider

`OpenAICompatibleClient` wraps the official asynchronous OpenAI SDK and targets
`/v1/chat/completions`. Changing the configured base URL, key, and model selects
hosted OpenAI or a local model served by Ollama without changing orchestration
code.

The adapter returns parsed output, model identity, reported token usage,
latency, and a compact error category. Provider concurrency and request timeout
are connection settings. Structured Web runs use validated JSON; paper-style
benchmarks can select plain text through the same adapter.

### Debate runtime

Each `Debater` owns a role and its public answer history. A
`DebateOrchestrator` assigns roles, optionally gathers research, requests the
independent first-turn answers, runs the configured peer-review turns, and
finally synthesizes the result.

For `N` agents and `T` turns, a complete run contains one explicit answer or
failure record for every one of the `N × T` cells. Independent calls within a
phase run concurrently up to the provider limit. Cancellation, per-request
timeouts, and configured call or token budgets remain explicit terminal
conditions.

### Research boundary

Optional Web research is split into three narrow adapters:

- `SearchClient` normalizes Tavily search results;
- `WebFetcher` performs bounded HTTP requests with redirect and address checks;
- `TextExtractor` converts supported HTML into bounded excerpts.

Fetched content is untrusted input. It cannot alter tool permissions or runtime
policy. Closed-book math benchmarks disable research entirely.

### Persistence and streaming

SQLite stores run configuration, roles, research evidence, turn answers,
directed targets, synthesis, and an ordered public event stream. Events are
persisted before publication. HTTP endpoints provide summaries and replay;
server-sent events provide live updates and resumable delivery.

The React client reduces live and replayed events through the same state logic.
Its primary view is an agent-by-turn workspace with linked evidence, target
claims, failures, and the final synthesis.

## Supported protocols

### Paper-style reproduction

The reproduction profile follows the core protocol from Du et al.:

1. agents answer independently on the first turn;
2. each later turn receives the other agents' latest full answers;
3. every agent recomputes and may revise its answer;
4. the group prediction uses plurality vote, including first-agent tie
   breaking used by the reference implementation.

This profile uses phase-specific natural-language messages, retains each
agent's own assistant history, and performs one model call per agent turn. It
does not send response schemas, developer messages, claims, tools, or directed
target metadata unless an ablation explicitly enables them.

### Inspectable Web protocol

The Web protocol adds structured answers, task-specific roles, optional source
research, stable claims, and directed challenge or support links. These
features are treated as observable protocol choices rather than as part of the
paper reproduction.

## Evaluation path

```text
Dataset adapter -> BenchmarkSample
                         |
Matrix runner -> DebateOrchestrator -> SampleResult
                         |
Metrics -> aggregate JSON/CSV and publication plots
```

The benchmark runner supports:

- the pinned `openai/gsm8k` test split;
- the pinned SVAMP test set;
- the arithmetic expression generator used in the paper's agent/round sweep.

Scoring records plurality accuracy per turn, strict-majority accuracy,
individual-agent accuracy, answer changes, consensus, failures, latency, and
provider metadata. Provider comparisons use the same sample identifiers and
remain separate in aggregates.

## Public data boundary

Credentials, model weights, dataset caches, raw provider responses, full
transcripts, SQLite databases, and intermediate JSONL/aggregate files stay in
ignored local directories. The public repository contains source code,
deterministic test fixtures, generated API contracts, aggregate findings, and
publication-ready reports only.
