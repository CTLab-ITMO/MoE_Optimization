from __future__ import annotations

import random
from typing import Mapping, Sequence, Tuple


TRACE_SPLITS = ("calibration", "validation", "test")


def split_text_rows(
    rows: Sequence[str],
    *,
    fractions: Tuple[float, float, float],
    seed: int,
) -> Mapping[str, Tuple[str, ...]]:
    if len(rows) < 3:
        raise ValueError("At least three non-empty text rows are required")
    if any(value <= 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to 1")

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n = len(order)
    calibration_end = max(1, int(n * fractions[0]))
    validation_end = max(calibration_end + 1, calibration_end + int(n * fractions[1]))
    validation_end = min(validation_end, n - 1)

    indices = {
        "calibration": order[:calibration_end],
        "validation": order[calibration_end:validation_end],
        "test": order[validation_end:],
    }
    return {
        split: tuple(str(rows[idx]) for idx in indices[split])
        for split in TRACE_SPLITS
    }


def resolve_text_column(column_names: Sequence[str], requested: str = "") -> str:
    names = tuple(str(name) for name in column_names)
    if requested:
        if requested not in names:
            raise ValueError(f"Text column '{requested}' not found; available columns: {list(names)}")
        return requested
    for candidate in ("text", "prompt", "question", "input", "content"):
        if candidate in names:
            return candidate
    if not names:
        raise ValueError("Dataset has no columns")
    return names[0]
