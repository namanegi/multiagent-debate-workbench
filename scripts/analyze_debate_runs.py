"""Summarize benchmark trajectories and optional persisted transcripts.

Usage:
    python scripts/analyze_debate_runs.py \
      --run label=artifacts/gsm8k/example \
      --output docs/reports/data/example.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("run must look like label=path")
    return label.strip(), Path(raw_path)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _curve(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {
        str(turn): sum(
            row[field].get(str(turn)) == row["reference_final_answer"] for row in rows
        )
        / len(rows)
        for turn in range(1, 5)
    }


def _agent_curve(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for turn in range(1, 5):
        values = [
            value
            for row in rows
            for value in row.get("per_turn", {}).get(str(turn), {}).values()
            if value is not None
        ]
        result[str(turn)] = sum(values) / len(values) if values else None
    return result


def _per_agent_curve(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    agent_ids = sorted(
        {
            agent_id
            for row in rows
            for values in row.get("per_turn", {}).values()
            for agent_id in values
        }
    )
    return {
        agent_id: {
            str(turn): (
                sum(value is True for value in values) / len(values) if values else None
            )
            for turn in range(1, 5)
            for values in [
                [
                    row.get("per_turn", {}).get(str(turn), {}).get(agent_id)
                    for row in rows
                    if row.get("per_turn", {}).get(str(turn), {}).get(agent_id)
                    is not None
                ]
            ]
        }
        for agent_id in agent_ids
    }


def _transitions(rows: list[dict[str, Any]], *, vote: bool) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for turn in range(2, 5):
        counts: Counter[str] = Counter()
        for row in rows:
            reference = row["reference_final_answer"]
            if vote:
                previous = row.get("per_turn_predictions", {}).get(str(turn - 1))
                current = row.get("per_turn_predictions", {}).get(str(turn))
                pairs = [(previous == reference, current == reference)]
            else:
                previous_agents = row.get("per_turn", {}).get(str(turn - 1), {})
                current_agents = row.get("per_turn", {}).get(str(turn), {})
                pairs = [
                    (previous_agents[agent], current_agents[agent])
                    for agent in previous_agents.keys() & current_agents.keys()
                    if previous_agents[agent] is not None and current_agents[agent] is not None
                ]
            for previous, current in pairs:
                counts[
                    "correct_to_correct"
                    if previous and current
                    else "correct_to_incorrect"
                    if previous
                    else "incorrect_to_correct"
                    if current
                    else "incorrect_to_incorrect"
                ] += 1
        result[f"{turn - 1}->{turn}"] = dict(counts)
    return result


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    trajectories: Counter[str] = Counter()
    for row in rows:
        reference = row["reference_final_answer"]
        trajectory = "".join(
            "C"
            if row.get("per_turn_predictions", {}).get(str(turn)) == reference
            else "W"
            for turn in range(1, 5)
        )
        trajectories[trajectory] += 1
    return dict(sorted(trajectories.items()))


def _transcript_metrics(paths: list[Path]) -> dict[str, Any] | None:
    payloads: list[tuple[str]] = []
    for path in paths:
        database = path.parent / f"{path.name}-transcripts" / "runs.db"
        if not database.is_file():
            continue
        connection = sqlite3.connect(database)
        try:
            payloads.extend(
                connection.execute(
                    "SELECT payload_json FROM events WHERE type = 'message.created'"
                ).fetchall()
            )
        finally:
            connection.close()
    if not payloads:
        return None
    by_turn: dict[int, list[str]] = {turn: [] for turn in range(1, 5)}
    for (raw_payload,) in payloads:
        message = json.loads(raw_payload)["message"]
        turn = message.get("turn_index")
        if message["author_id"].startswith("agent_") and turn in by_turn:
            by_turn[turn].append(message["content"])
    return {
        "message_count": sum(map(len, by_turn.values())),
        "mean_chars_by_turn": {
            str(turn): mean(map(len, messages)) if messages else None
            for turn, messages in by_turn.items()
        },
        "short_under_120_by_turn": {
            str(turn): sum(len(message) < 120 for message in messages)
            for turn, messages in by_turn.items()
        },
        "missing_final_marker_by_turn": {
            str(turn): sum(
                not re.search(r"final answer\s*:", message, flags=re.IGNORECASE)
                for message in messages
            )
            for turn, messages in by_turn.items()
        },
    }


def summarize(label: str, paths: list[Path]) -> dict[str, Any]:
    rows = [row for path in paths for row in _rows(path)]
    reports = [
        json.loads((path / "aggregates.json").read_text(encoding="utf-8"))
        for path in paths
    ]
    completed = [row for row in rows if row["completed"]]
    t1_correct = [
        row
        for row in rows
        if row.get("per_turn_predictions", {}).get("1") == row["reference_final_answer"]
    ]
    t1_wrong = [row for row in rows if row not in t1_correct]
    return {
        "label": label,
        "paths": [str(path) for path in paths],
        "metadata": [report["metadata"] for report in reports],
        "sample_count": len(rows),
        "completed_count": len(completed),
        "completion_rate": len(completed) / len(rows),
        "failure_categories": dict(Counter(row["error"] or "none" for row in rows)),
        "vote_accuracy": _curve(rows, "per_turn_predictions"),
        "agent_accuracy_among_parsed": _agent_curve(rows),
        "per_agent_accuracy": _per_agent_curve(rows),
        "agent_transitions": _transitions(rows, vote=False),
        "vote_transitions": _transitions(rows, vote=True),
        "trajectories": _trajectory_summary(rows),
        "t1_correct_retained_at_t4": sum(
            row.get("per_turn_predictions", {}).get("4") == row["reference_final_answer"]
            for row in t1_correct
        ),
        "t1_correct_count": len(t1_correct),
        "t1_wrong_corrected_at_t4": sum(
            row.get("per_turn_predictions", {}).get("4") == row["reference_final_answer"]
            for row in t1_wrong
        ),
        "t1_wrong_count": len(t1_wrong),
        "answer_changes": sum(row["answer_changes"] for row in rows),
        "incorrect_to_correct": sum(row["incorrect_to_correct"] for row in rows),
        "correct_to_incorrect": sum(row["correct_to_incorrect"] for row in rows),
        "consensus_rate": sum(row["consensus"] is True for row in rows) / len(rows),
        "false_consensus_rate": sum(row["false_consensus"] is True for row in rows)
        / len(rows),
        "mean_latency_ms": mean(row["elapsed_ms"] for row in rows),
        "calls": sum(row["calls"] for row in rows),
        "input_tokens": sum(row["input_tokens"] or 0 for row in rows),
        "output_tokens": sum(row["output_tokens"] or 0 for row in rows),
        "transcripts": _transcript_metrics(paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_run, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups: dict[str, list[Path]] = {}
    for label, path in args.run:
        groups.setdefault(label, []).append(path)
    payload = {label: summarize(label, paths) for label, paths in groups.items()}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
