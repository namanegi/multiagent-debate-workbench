"""Pinned SVAMP challenge-set loading and numeric answer normalization."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

from debate_api.benchmark.gsm8k import BenchmarkSample, extract_final_answer

SVAMP_DATASET = "MU-NLPC/Calc-svamp"
SVAMP_REVISION = "1895931ce53425d46db1ffeb055d74ca1d912f17"
SVAMP_SPLITS = ("test",)


class SVAMPAdapter:
    """Load the 1,000-example SVAMP challenge set at an immutable revision."""

    dataset_name = SVAMP_DATASET
    revision = SVAMP_REVISION

    def __init__(
        self,
        *,
        revision: str = SVAMP_REVISION,
        loader: Callable[..., Iterable[dict[str, Any]]] | None = None,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("SVAMP revision must be an immutable 40-character commit")
        self.revision = revision
        self._loader = loader

    def load(self, split: str = "test") -> list[BenchmarkSample]:
        if split not in SVAMP_SPLITS:
            raise ValueError(f"split must be one of {SVAMP_SPLITS}")
        if self._loader is None:
            try:
                datasets = import_module("datasets")
                load_dataset = datasets.load_dataset
            except (ImportError, AttributeError) as error:  # pragma: no cover
                raise RuntimeError(
                    "Install the benchmark extra (datasets) to download SVAMP."
                ) from error
            rows = load_dataset(
                self.dataset_name,
                revision=self.revision,
                split=split,
            )
        else:
            rows = self._loader(
                self.dataset_name,
                revision=self.revision,
                split=split,
            )

        samples: list[BenchmarkSample] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise ValueError(f"invalid SVAMP row at index {index}")
            raw_result = str(row.get("result") or "")
            final = extract_final_answer(raw_result.replace("_", ""))
            if final is None:
                raise ValueError(f"SVAMP row has no extractable result at index {index}")
            raw_chain = row.get("chain")
            reference_answer = (
                str(raw_chain) if isinstance(raw_chain, str) and raw_chain else raw_result
            )
            source_id = row.get("id")
            sample_id = (
                str(source_id)
                if isinstance(source_id, str) and source_id
                else f"svamp-{index}"
            )
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    question=row["question"],
                    reference_answer=reference_answer,
                    reference_final_answer=final,
                    benchmark="svamp",
                )
            )
        return samples
