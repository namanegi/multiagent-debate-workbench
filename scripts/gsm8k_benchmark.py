"""Opt-in math benchmark command; no provider call occurs without execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
from pathlib import Path
from typing import cast

from debate_api.benchmark.arithmetic import (
    ARITHMETIC_DEFAULT_COUNT,
    ARITHMETIC_DEFAULT_SEED,
    ArithmeticAdapter,
)
from debate_api.benchmark.gsm8k import GSM8KAdapter, select_samples
from debate_api.benchmark.runner import (
    MatrixCell,
    SampleResult,
    build_matrix,
    run_sample,
    write_reports,
)
from debate_api.benchmark.svamp import SVAMPAdapter
from debate_api.orchestration.debater import OutputMode
from debate_api.providers.runtime import build_model_provider
from debate_api.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
_RESUME_METADATA_KEYS = (
    "benchmark",
    "dataset",
    "revision",
    "split",
    "sample_ids",
    "generator_seed",
    "model_provider",
    "model",
    "reasoning_effort",
    "temperature",
    "max_output_tokens",
    "request_timeout_seconds",
    "thinking_mode",
    "protocol_mode",
    "paper_role_profile",
    "model_output_mode",
    "matrix",
)


def _cells(value: str) -> list[MatrixCell]:
    try:
        return [
            MatrixCell(*(int(part) for part in item.strip().lower().split("x", 1)))
            for item in value.split(",")
            if item.strip()
        ]
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("matrix must look like 1x1,3x2") from error


def _load_resume_results(
    output_dir: Path, metadata: dict[str, object]
) -> list[SampleResult]:
    samples_path = output_dir / "samples.jsonl"
    aggregates_path = output_dir / "aggregates.json"
    if not samples_path.exists() and not aggregates_path.exists():
        return []
    if not samples_path.is_file() or not aggregates_path.is_file():
        raise SystemExit("resume requires both samples.jsonl and aggregates.json")
    try:
        existing_report = json.loads(aggregates_path.read_text(encoding="utf-8"))
        existing_metadata = existing_report["metadata"]
        mismatches = [
            key
            for key in _RESUME_METADATA_KEYS
            if existing_metadata.get(key) != metadata.get(key)
        ]
        rows = [
            SampleResult(**json.loads(line))
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read resume checkpoint: {error}") from error
    if mismatches:
        raise SystemExit(f"resume metadata mismatch: {', '.join(mismatches)}")
    completed = [row for row in rows if row.completed]
    keys = [(row.cell, row.sample_id) for row in completed]
    if len(keys) != len(set(keys)):
        raise SystemExit("resume checkpoint contains duplicate completed results")
    return completed


async def _main(args: argparse.Namespace) -> None:
    if not args.live:
        raise SystemExit("benchmark execution requires explicit --live")
    model_output_mode = cast(
        OutputMode,
        args.model_output_mode
        or (
            "plain_text"
            if args.protocol_mode == "paper_reproduction"
            else "structured_json"
        ),
    )
    if args.benchmark == "arithmetic":
        arithmetic = ArithmeticAdapter(
            seed=args.generator_seed, count=args.arithmetic_count
        )
        samples = arithmetic.load()
        dataset_name = arithmetic.dataset_name
        revision = arithmetic.revision
        split = "generated"
    elif args.benchmark == "gsm8k":
        gsm8k = GSM8KAdapter(revision=args.revision or GSM8KAdapter.revision)
        samples = gsm8k.load(args.split)
        dataset_name = gsm8k.dataset_name
        revision = gsm8k.revision
        split = args.split
    else:
        if args.split != "test":
            raise SystemExit("SVAMP only provides the test split")
        svamp = SVAMPAdapter(revision=args.revision or SVAMPAdapter.revision)
        samples = svamp.load(args.split)
        dataset_name = svamp.dataset_name
        revision = svamp.revision
        split = args.split
    try:
        samples = select_samples(
            samples,
            sample_ids=(item.strip() for item in args.sample_ids.split(","))
            if args.sample_ids
            else None,
            seed=args.seed,
            limit=args.limit,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    explicit_cells = [cell for group in (args.matrix or []) for cell in group]
    cells = (
        build_matrix(explicit=explicit_cells)
        if explicit_cells
        else build_matrix(
            agent_counts=range(1, 8) if args.agent_sweep else [args.agents],
            turn_counts=range(1, 5) if args.turn_sweep else [args.turns],
        )
    )
    settings = Settings(
        model_provider=args.provider,
        **(
            {"openai_model": args.model}
            if args.provider == "openai" and args.model
            else {"ollama_model": args.model}
            if args.provider == "ollama" and args.model
            else {}
        ),
        **(
            {"model_reasoning_effort": args.reasoning_effort}
            if args.reasoning_effort is not None
            else {}
        ),
        **(
            {"model_temperature": args.temperature}
            if args.temperature is not None
            else {}
        ),
        **(
            {"openai_request_timeout_seconds": args.request_timeout_seconds}
            if args.provider == "openai" and args.request_timeout_seconds is not None
            else {"ollama_request_timeout_seconds": args.request_timeout_seconds}
            if args.provider == "ollama" and args.request_timeout_seconds is not None
            else {}
        ),
    )
    runtime = build_model_provider(settings)
    provider = runtime.provider
    output_dir = (
        Path(args.output)
        if args.output is not None
        else REPO_ROOT / "artifacts" / args.benchmark
    )
    database_path = None
    if args.database_dir is not None:
        database_dir = Path(args.database_dir)
        database_dir.mkdir(parents=True, exist_ok=True)
        database_path = database_dir / "runs.db"
    metadata = {
        "benchmark": args.benchmark,
        "dataset": dataset_name,
        "revision": revision,
        "split": split,
        "sample_ids": [sample.sample_id for sample in samples],
        "seed": args.seed,
        "generator_seed": args.generator_seed
        if args.benchmark == "arithmetic"
        else None,
        "model_provider": runtime.mode,
        "model": runtime.identity,
        "reasoning_effort": settings.model_reasoning_effort,
        "temperature": settings.model_temperature,
        "max_output_tokens": args.max_output_tokens,
        "request_timeout_seconds": (
            settings.openai_request_timeout_seconds
            if args.provider == "openai"
            else settings.ollama_request_timeout_seconds
        ),
        "thinking_mode": args.thinking_mode,
        "research_enabled": False,
        "protocol_mode": args.protocol_mode,
        "paper_role_profile": args.paper_role_profile,
        "model_output_mode": model_output_mode,
        "cell_execution_mode": "independent_cells",
        "turn_curve_mode": "nested_snapshots",
        "nested_turn_curve_field": (
            "per_turn_vote_accuracy"
            if args.protocol_mode == "paper_reproduction"
            else "per_turn_majority_accuracy"
        ),
        "vote_mode": (
            "paper_plurality"
            if args.protocol_mode == "paper_reproduction"
            else "strict_majority"
        ),
        "benchmark_note": (
            "Non-paper Arithmetic ablation with one dedicated verifier role."
            if args.benchmark == "arithmetic"
            and args.paper_role_profile != "homogeneous"
            else "Du et al. synthetic Arithmetic reproduction profile."
            if args.benchmark == "arithmetic"
            else "SVAMP robustness challenge profile; no monotonicity claim across turns."
            if args.benchmark == "svamp"
            else "GSM8K protocol profile; no monotonicity claim across turns."
        ),
        "matrix": [cell.label for cell in cells],
        "source_commit": _source_commit(),
        "source_dirty": _source_dirty(),
    }
    results = _load_resume_results(output_dir, metadata) if args.resume else []
    completed_keys = {(result.cell, result.sample_id) for result in results}
    expected_results = len(cells) * len(samples)
    requested_keys = {
        (cell.label, sample.sample_id) for cell in cells for sample in samples
    }
    if not completed_keys.issubset(requested_keys):
        raise SystemExit("resume checkpoint contains results outside the requested matrix")
    if results:
        print(f"benchmark resumed: {len(results)}/{expected_results}", flush=True)
    if provider is not None:
        await provider.__aenter__()
    try:
        for cell in cells:
            for sample in samples:
                if (cell.label, sample.sample_id) in completed_keys:
                    continue
                results.append(
                    await run_sample(
                        sample,
                        cell,
                        provider=provider,
                        database_path=database_path,
                        **(
                            {"protocol_mode": args.protocol_mode}
                            if args.protocol_mode != "default"
                            else {}
                        ),
                        **(
                            {"paper_role_profile": args.paper_role_profile}
                            if args.paper_role_profile != "homogeneous"
                            else {}
                        ),
                        output_mode=model_output_mode,
                        plain_text_max_output_tokens=args.max_output_tokens,
                        thinking_mode=args.thinking_mode,
                    )
                )
                if len(results) % args.checkpoint_every == 0:
                    write_reports(
                        results,
                        output_dir,
                        {
                            **metadata,
                            "partial": len(results) < expected_results,
                            "completed_results": len(results),
                            "expected_results": expected_results,
                        },
                    )
                    print(f"benchmark progress: {len(results)}/{expected_results}", flush=True)
    finally:
        if provider is not None:
            await provider.close()
    write_reports(
        results,
        output_dir,
        {
            **metadata,
            "partial": False,
            "completed_results": len(results),
            "expected_results": expected_results,
        },
    )


def _source_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value if value else None


def _source_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("gsm8k", "arithmetic", "svamp"),
        default="gsm8k",
        help="Dataset profile; arithmetic reproduces Du et al. and SVAMP tests robustness",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama"),
        required=True,
        help="Live provider configuration; benchmark runs never use deterministic fakes",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Confirm that this command may download data and call the provider",
    )
    parser.add_argument("--model", help="Provider model override")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        help="Optional shared Chat Completions reasoning effort",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional shared sampling temperature (0 through 2)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32_768,
        help="Per-agent output budget; benchmark runs default to the provider maximum",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        help="Optional provider request timeout override",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("provider_default", "disabled"),
        default="provider_default",
        help="Paper prompt control; disabled injects Qwen3 /no_think into each user turn",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--revision",
        help="Immutable dataset revision; defaults to the selected adapter's pinned commit",
    )
    parser.add_argument("--generator-seed", type=int, default=ARITHMETIC_DEFAULT_SEED)
    parser.add_argument("--arithmetic-count", type=int, default=ARITHMETIC_DEFAULT_COUNT)
    parser.add_argument(
        "--sample-ids",
        help="Comma-separated IDs such as test-0, arithmetic-3, or svamp__chal-1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument(
        "--protocol-mode", choices=("default", "paper_reproduction"), default="default",
        help="Use homogeneous independent math solvers and paper-style review prompts",
    )
    parser.add_argument(
        "--paper-role-profile",
        choices=("homogeneous", "checker", "checker_semantic"),
        default="homogeneous",
        help=(
            "Deterministic paper roles; checker_semantic makes the final agent verify "
            "the requested quantity, units, semantics, and arithmetic"
        ),
    )
    parser.add_argument(
        "--model-output-mode",
        choices=("structured_json", "plain_text"),
        help=(
            "Provider output contract; defaults to plain_text for paper reproduction and "
            "structured_json otherwise"
        ),
    )
    parser.add_argument("--agent-sweep", action="store_true")
    parser.add_argument("--turn-sweep", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Rewrite partial reports and print progress after this many results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed cells from a compatible output checkpoint",
    )
    parser.add_argument(
        "--matrix", type=_cells, action="append", help="Explicit cell(s), e.g. 1x1,3x2"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--database-dir",
        type=Path,
        help="Optional persistent SQLite directory for inspecting full run transcripts",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_ids and args.limit is not None:
        parser.error("choose --sample-ids or --limit, not both")
    if args.matrix and (args.agent_sweep or args.turn_sweep):
        parser.error("--matrix cannot be combined with sweep flags")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.arithmetic_count < 1:
        parser.error("--arithmetic-count must be at least 1")
    if args.temperature is not None and (
        not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2
    ):
        parser.error("--temperature must be a finite value between 0 and 2")
    if not 1 <= args.max_output_tokens <= 32_768:
        parser.error("--max-output-tokens must be between 1 and 32768")
    if args.request_timeout_seconds is not None and (
        not math.isfinite(args.request_timeout_seconds)
        or not 0 < args.request_timeout_seconds <= 900
    ):
        parser.error("--request-timeout-seconds must be between 0 and 900")
    if args.paper_role_profile != "homogeneous" and args.protocol_mode != "paper_reproduction":
        parser.error("--paper-role-profile requires --protocol-mode paper_reproduction")
    if args.model_output_mode == "plain_text" and args.protocol_mode != "paper_reproduction":
        parser.error("--model-output-mode plain_text requires paper_reproduction")
    if args.thinking_mode != "provider_default" and args.protocol_mode != "paper_reproduction":
        parser.error("--thinking-mode overrides require paper_reproduction")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
