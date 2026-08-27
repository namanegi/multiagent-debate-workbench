"""Small benchmark primitives for reproducible debate experiments."""

from typing import Any

from debate_api.benchmark.arithmetic import ArithmeticAdapter
from debate_api.benchmark.gsm8k import (
    BenchmarkSample,
    GSM8KAdapter,
    extract_final_answer,
    select_samples,
)
from debate_api.benchmark.svamp import SVAMPAdapter

__all__ = [
    "ArithmeticAdapter",
    "BenchmarkSample",
    "GSM8KAdapter",
    "MatrixCell",
    "SampleResult",
    "SVAMPAdapter",
    "aggregate_results",
    "build_matrix",
    "complete_matrix",
    "extract_final_answer",
    "run_sample",
    "select_samples",
]


def __getattr__(name: str) -> Any:
    """Keep dataset parsing usable in a minimal offline Python environment."""
    if name in {
        "MatrixCell",
        "SampleResult",
        "aggregate_results",
        "build_matrix",
        "complete_matrix",
        "run_sample",
    }:
        from debate_api.benchmark import runner

        return getattr(runner, name)
    raise AttributeError(name)
