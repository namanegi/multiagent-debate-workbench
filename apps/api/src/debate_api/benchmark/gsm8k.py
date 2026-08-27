"""GSM8K loading and deterministic answer normalization.

The adapter deliberately imports ``datasets`` only when a live load is
requested. Tests can inject a tiny loader and never download benchmark data.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib import import_module
from typing import Any

GSM8K_DATASET = "openai/gsm8k"
# Immutable Hub commit for the Parquet-backed official dataset.
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_SPLITS = ("train", "test")


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    sample_id: str
    question: str
    reference_answer: str
    reference_final_answer: str
    benchmark: str = "gsm8k"


def _canonical_number(value: str) -> str | None:
    value = value.strip().replace(",", "").replace("$", "")
    if value.endswith("."):
        value = value[:-1]
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0", "+0"} else normalized


def extract_final_answer(text: str | None) -> str | None:
    """Extract a GSM8K scalar answer without depending on model formatting.

    GSM8K references use ``#### number``. Provider answers are accepted with
    that marker, a final boxed value, or a final-answer phrase. If none is
    present, the last numeric scalar is used as a conservative fallback.
    """

    if not text or not text.strip():
        return None
    number = r"[-+]?\d[\d,]*(?:\.\d+)?"
    patterns = (
        rf"####\s*({number})",
        rf"\\boxed\{{\s*({number})\s*\}}",
        rf"(?:final answer|answer)\s*(?:is|=|:)\s*\$?({number})",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            parsed = _canonical_number(matches[-1])
            if parsed is not None:
                return parsed
    values = re.findall(rf"(?<![\w.])\$?({number})(?![\w.])", text)
    for value in reversed(values):
        parsed = _canonical_number(value)
        if parsed is not None:
            return parsed
    return None


def _reference_answer(value: Any) -> tuple[str, str]:
    answer = str(value or "")
    final = extract_final_answer(answer)
    if final is None:
        raise ValueError("GSM8K row has no extractable reference answer")
    return answer, final


def select_samples(
    samples: list[BenchmarkSample],
    *,
    sample_ids: Iterable[str] | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> list[BenchmarkSample]:
    """Select exact IDs or a reproducible subset, never both."""

    if sample_ids is not None and limit is not None:
        raise ValueError("sample_ids and limit are mutually exclusive")
    if sample_ids is not None:
        requested = list(dict.fromkeys(sample_ids))
        by_id = {sample.sample_id: sample for sample in samples}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"unknown sample IDs: {', '.join(missing)}")
        return [by_id[sample_id] for sample_id in requested]
    if limit is None:
        return list(samples)
    if limit < 0 or limit > len(samples):
        raise ValueError("limit must be between zero and the split size")
    return random.Random(seed).sample(samples, limit)


class GSM8KAdapter:
    """Load official GSM8K rows at one pinned Hub revision."""

    dataset_name = GSM8K_DATASET
    revision = GSM8K_REVISION

    def __init__(
        self,
        *,
        revision: str = GSM8K_REVISION,
        loader: Callable[..., Iterable[dict[str, Any]]] | None = None,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("GSM8K revision must be an immutable 40-character commit")
        self.revision = revision
        self._loader = loader

    def load(self, split: str = "test") -> list[BenchmarkSample]:
        if split not in GSM8K_SPLITS:
            raise ValueError(f"split must be one of {GSM8K_SPLITS}")
        if self._loader is None:
            try:
                datasets = import_module("datasets")
                load_dataset = datasets.load_dataset
            except (
                ImportError,
                AttributeError,
            ) as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Install the benchmark extra (datasets) to download GSM8K."
                ) from error
            rows = load_dataset(
                self.dataset_name,
                "main",
                revision=self.revision,
                split=split,
            )
        else:
            rows = self._loader(
                self.dataset_name,
                "main",
                revision=self.revision,
                split=split,
            )
        samples: list[BenchmarkSample] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise ValueError(f"invalid GSM8K row at index {index}")
            raw_reference, final = _reference_answer(row.get("answer"))
            samples.append(
                BenchmarkSample(
                    sample_id=f"{split}-{index}",
                    question=row["question"],
                    reference_answer=raw_reference,
                    reference_final_answer=final,
                )
            )
        return samples
