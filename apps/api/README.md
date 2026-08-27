# API application

This directory contains the FastAPI application, orchestration domain, provider adapters, persistence adapters, and Python tests.

Internal packages:

```text
src/debate_api/
  api/             HTTP and SSE transport
  domain/          State machine and domain models
  orchestration/   Run planning and phase execution
  providers/       Model and research adapters
  persistence/     Event and projection repositories
  settings.py
tests/
```

The API must remain runnable with deterministic fake providers so local development and CI do not require model downloads, network access, or secrets.

## Local development

From the repository root:

```powershell
uv sync --directory apps/api --dev
uv run --directory apps/api uvicorn debate_api.main:app --app-dir src --reload
```

The service exposes `/health/live`, `/health/ready`, and FastAPI's `/docs`. Run the API tests with:

```powershell
uv run --directory apps/api pytest tests
```

The default application launcher runs the deterministic debate orchestrator in the background after
`POST /v1/runs`. It persists every public artifact before it can be replayed or streamed. Tests inject
the launcher and use isolated SQLite files, so provider secrets and network access are not required.

`uv.lock` pins the resolved dependency set and `apps/api/.venv` is managed by uv. Configuration is loaded from optional `.env` or `.env.local` files. Variables use the `DEBATE_API_` prefix; no variable is required for the scaffold.

## Math benchmarks

Install the pinned benchmark extra and explicitly opt into a live provider run:

```powershell
uv run --extra benchmark --directory apps/api python ../../scripts/gsm8k_benchmark.py --live --provider ollama --limit 10 --seed 7 --matrix 1x1,3x2
```

The command uses the pinned Parquet-backed `main` configuration of
`openai/gsm8k`, disables research, and writes ignored JSONL/JSON/CSV reports
under `artifacts/`. `--provider openai`
uses the configured hosted model. The CLI has no fake-provider or non-live
benchmark mode; offline behavior is covered by unit tests.

For a paper-style homogeneous math debate, run the maximum turn count once and
score every turn from that shared trajectory:

```powershell
uv run --extra benchmark --directory apps/api python ../../scripts/gsm8k_benchmark.py --live --provider openai --model gpt-5.6-luna --reasoning-effort low --limit 100 --seed 7 --matrix 3x4 --protocol-mode paper_reproduction
```

The `per_turn_vote_accuracy` aggregate is the paper-compatible nested
1-4-turn curve. Paper reproduction uses the authors' plurality vote (including
first-agent tie breaking); the stricter majority metric is retained alongside
it. Paper reproduction defaults to `--model-output-mode plain_text`: it uses the
same core solve/recompute instructions, sends phase-specific natural-language
prompts with each agent's role-preserving conversation history, and sends no
developer message or response schema. It creates no claims or directed targets.
Use `--model-output-mode structured_json` only as an explicit product-protocol
ablation. The Web/default protocol remains the
product's structured JSON, dynamic-role, directed-interaction workflow. Partial
reports are checkpointed every 10 results by default; use
`--checkpoint-every` to change that interval, and rerun with `--resume` plus the
same arguments to continue completed cells from a compatible output directory.
Sampling temperature remains provider-default unless `--temperature 0..2` is
supplied. The selected value is forwarded to both OpenAI and Ollama and recorded
in benchmark metadata, so temperature can be swept later without changing the
prompt or provider code.
Benchmark requests default to the provider's 32,768-token per-agent maximum
rather than the Web path's conservative 2,048-token budget. Use
`--max-output-tokens` and
`--request-timeout-seconds` for local thinking-model runs; both values are
recorded in metadata and checked when resuming.
For Qwen3 ablations, `--thinking-mode disabled` places the model's `/no_think`
soft control in every paper user turn. The default is `provider_default`, so
other models and ordinary Web runs receive no model-specific instruction.
Du et al. Figure 9(b) reports the
paper's Arithmetic task, not GSM8K, so a GSM8K curve should not be described as
an exact Figure 9 reproduction or assumed to be monotonic.

For qualitative error analysis, add `--database-dir <path>` to retain the full
messages, claims, planning outcome, failures, and synthesis in `runs.db`. The
normal JSONL report remains compact and suitable for aggregate analysis.

Run the exact 100-expression Arithmetic generator from the authors' Figure 9
setup with NumPy seed 0:

```powershell
uv run --extra benchmark --directory apps/api python ../../scripts/gsm8k_benchmark.py --live --benchmark arithmetic --provider openai --model gpt-5.6-luna --reasoning-effort low --matrix 3x4 --protocol-mode paper_reproduction
```

The paper profile defaults to homogeneous independent solvers. To run a
non-paper ablation with the last agent assigned as a dedicated arithmetic
verifier, add `--paper-role-profile checker`. The selected profile is recorded
in report metadata. With `--matrix 4x4`, this profile represents three ordinary
solvers plus one additional checker; compare it with homogeneous `4x4` to
separate the role effect from the extra vote and compute.

`--paper-role-profile checker_semantic` is the stronger verification ablation:
the final agent must identify the requested quantity and unit, recompute from
the original facts, locate the earliest semantic or arithmetic divergence, and
ignore majority or verbosity unless its own recomputation supports them.

The generator samples six integers from `[0, 29]` and evaluates
`a + b*c + d - e*f`. It is offline and pinned to the authors' source revision;
only model execution requires a live provider. Use `--agent-sweep --turns 4`
for the 1-7-agent axis, or one `3x4` trajectory for the 1-4-turn axis.

For a robustness-oriented word-problem extension, select the pinned 1,000-item
SVAMP challenge set with `--benchmark svamp`. SVAMP uses its official test-only
split and scalar results, so it shares the same answer extraction and debate
metrics as GSM8K:

```powershell
uv run --extra benchmark --directory apps/api python ../../scripts/gsm8k_benchmark.py --live --benchmark svamp --provider ollama --limit 20 --seed 19 --matrix 3x4 --protocol-mode paper_reproduction --paper-role-profile checker_semantic
```
