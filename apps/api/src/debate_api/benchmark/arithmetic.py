"""Deterministic reproduction of Du et al.'s synthetic Arithmetic task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from debate_api.benchmark.gsm8k import BenchmarkSample

ARITHMETIC_DATASET = "composable-models/llm_multiagent_debate/math"
ARITHMETIC_REVISION = "9846749350eb917ae5bfaaff4c645fc705b8d3af"
ARITHMETIC_DEFAULT_SEED = 0
ARITHMETIC_DEFAULT_COUNT = 100


@dataclass(frozen=True, slots=True)
class ArithmeticAdapter:
    """Generate the exact expression family used by the authors' ``gen_math.py``."""

    seed: int = ARITHMETIC_DEFAULT_SEED
    count: int = ARITHMETIC_DEFAULT_COUNT
    dataset_name: str = ARITHMETIC_DATASET
    revision: str = ARITHMETIC_REVISION

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("Arithmetic sample count must be positive")

    def load(self) -> list[BenchmarkSample]:
        # RandomState intentionally reproduces np.random.seed(...)+randint from
        # the authors' legacy NumPy code. default_rng produces a different set.
        values = np.random.RandomState(self.seed).randint(0, 30, size=(self.count, 6))
        samples: list[BenchmarkSample] = []
        for index, row in enumerate(values.tolist()):
            a, b, c, d, e, f = (int(value) for value in row)
            answer = a + b * c + d - e * f
            question = (
                f"What is the result of {a}+{b}*{c}+{d}-{e}*{f}? "
                "Make sure to state your answer at the end of the response."
            )
            samples.append(
                BenchmarkSample(
                    sample_id=f"arithmetic-{index}",
                    question=question,
                    reference_answer=str(answer),
                    reference_final_answer=str(answer),
                    benchmark="arithmetic",
                )
            )
        return samples
