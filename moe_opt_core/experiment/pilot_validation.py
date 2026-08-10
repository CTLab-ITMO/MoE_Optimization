from __future__ import annotations

import math
from typing import Sequence, TypeVar

from moe_opt_core.experiment.contracts import (
    EvaluationMetricsContract,
    ValidationRobustnessContract,
)


BatchT = TypeVar("BatchT")


def partition_validation_batches(
    batches: Sequence[BatchT],
    slice_count: int,
) -> tuple[tuple[BatchT, ...], ...]:
    if slice_count < 1:
        raise ValueError("slice_count must be positive")
    if slice_count > len(batches):
        raise ValueError("slice_count must not exceed the number of validation batches")
    return tuple(tuple(batches[index::slice_count]) for index in range(slice_count))


def aggregate_evaluation_metrics(
    metrics: Sequence[EvaluationMetricsContract],
) -> EvaluationMetricsContract:
    if not metrics:
        raise ValueError("metrics must not be empty")
    total_tokens = sum(item.evaluated_tokens for item in metrics)
    if total_tokens > 0:
        nll = sum(item.nll * item.evaluated_tokens for item in metrics) / total_tokens
    else:
        nll = sum(item.nll for item in metrics) / len(metrics)
    elapsed = sum(item.elapsed_seconds for item in metrics)
    return EvaluationMetricsContract(
        nll=float(nll),
        perplexity=float(math.exp(min(nll, 50.0))),
        tokens_per_second=float(total_tokens / max(elapsed, 1e-12)),
        peak_memory_mb=max(item.peak_memory_mb for item in metrics),
        model_parameter_mb=max(item.model_parameter_mb for item in metrics),
        evaluated_tokens=total_tokens,
        elapsed_seconds=float(elapsed),
    )


def summarize_validation_degradations(
    degradations: Sequence[float],
    *,
    std_penalty: float,
    max_slice_degradation: float | None,
) -> ValidationRobustnessContract:
    if not degradations:
        raise ValueError("degradations must not be empty")
    if std_penalty < 0.0:
        raise ValueError("std_penalty must be non-negative")
    if max_slice_degradation is not None and max_slice_degradation < 0.0:
        raise ValueError("max_slice_degradation must be non-negative")

    values = tuple(float(value) for value in degradations)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    standard_deviation = math.sqrt(variance)
    worst = max(values)
    return ValidationRobustnessContract(
        slice_count=len(values),
        relative_perplexity_degradations=values,
        mean_relative_perplexity_degradation=float(mean),
        std_relative_perplexity_degradation=float(standard_deviation),
        worst_relative_perplexity_degradation=float(worst),
        std_penalty=float(std_penalty),
        robust_score=float(mean + std_penalty * standard_deviation),
        max_slice_degradation=(
            float(max_slice_degradation) if max_slice_degradation is not None else None
        ),
        passes_max_slice_degradation=(
            max_slice_degradation is None or worst <= max_slice_degradation
        ),
    )
