from __future__ import annotations

import math
from statistics import median, pstdev
from typing import Sequence, TypeVar

from moe_opt_core.experiment.contracts import (
    EvaluationMetricsContract,
    EvaluationRepeatSummaryContract,
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


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_evaluation_repeats(
    metrics: Sequence[EvaluationMetricsContract],
    *,
    aggregation: str = "median",
) -> tuple[EvaluationMetricsContract, EvaluationRepeatSummaryContract]:
    if not metrics:
        raise ValueError("metrics must not be empty")
    if aggregation != "median":
        raise ValueError("aggregation must be median")
    token_counts = {item.evaluated_tokens for item in metrics}
    parameter_sizes = {round(item.model_parameter_mb, 9) for item in metrics}
    if len(token_counts) != 1:
        raise ValueError("evaluation repeats must use the same token count")
    if len(parameter_sizes) != 1:
        raise ValueError("evaluation repeats must use the same model parameters")

    nll_values = tuple(item.nll for item in metrics)
    perplexities = tuple(item.perplexity for item in metrics)
    throughputs = tuple(item.tokens_per_second for item in metrics)
    elapsed_values = tuple(item.elapsed_seconds for item in metrics)
    nll = float(median(nll_values))
    aggregate = EvaluationMetricsContract(
        nll=nll,
        perplexity=float(math.exp(min(nll, 50.0))),
        tokens_per_second=float(median(throughputs)),
        peak_memory_mb=max(item.peak_memory_mb for item in metrics),
        model_parameter_mb=float(median(item.model_parameter_mb for item in metrics)),
        evaluated_tokens=next(iter(token_counts)),
        elapsed_seconds=float(median(elapsed_values)),
    )
    summary = EvaluationRepeatSummaryContract(
        repeat_count=len(metrics),
        aggregation=aggregation,
        perplexity_std=float(pstdev(perplexities)),
        tokens_per_second_median=float(median(throughputs)),
        tokens_per_second_p05=float(_linear_quantile(throughputs, 0.05)),
        tokens_per_second_p95=float(_linear_quantile(throughputs, 0.95)),
    )
    return aggregate, summary


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
