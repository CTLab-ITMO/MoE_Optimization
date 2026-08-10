import math

import pytest

from moe_opt_core.experiment import EvaluationMetricsContract
from moe_opt_core.experiment.pilot_validation import (
    aggregate_evaluation_metrics,
    partition_validation_batches,
    summarize_validation_degradations,
)


def _metrics(*, nll: float, tokens: int, elapsed: float) -> EvaluationMetricsContract:
    return EvaluationMetricsContract(
        nll=nll,
        perplexity=math.exp(nll),
        tokens_per_second=tokens / elapsed,
        peak_memory_mb=100.0 + nll,
        model_parameter_mb=200.0,
        evaluated_tokens=tokens,
        elapsed_seconds=elapsed,
    )


@pytest.mark.parametrize("batch_count,slice_count", [(3, 1), (10, 3), (32, 3), (32, 8)])
def test_validation_slices_are_balanced_disjoint_and_complete(
    batch_count: int,
    slice_count: int,
) -> None:
    batches = tuple(range(batch_count))
    slices = partition_validation_batches(batches, slice_count)

    flattened = tuple(item for batch_slice in slices for item in batch_slice)
    sizes = tuple(len(batch_slice) for batch_slice in slices)
    assert len(slices) == slice_count
    assert sorted(flattened) == list(batches)
    assert len(flattened) == len(set(flattened))
    assert max(sizes) - min(sizes) <= 1
    assert slices == partition_validation_batches(batches, slice_count)


def test_validation_slices_reject_empty_partitions() -> None:
    with pytest.raises(ValueError, match="positive"):
        partition_validation_batches((1, 2), 0)
    with pytest.raises(ValueError, match="number of validation batches"):
        partition_validation_batches((1, 2), 3)


def test_metric_aggregation_preserves_weighted_nll_and_tokens() -> None:
    aggregated = aggregate_evaluation_metrics(
        (
            _metrics(nll=1.0, tokens=10, elapsed=2.0),
            _metrics(nll=2.0, tokens=30, elapsed=3.0),
        )
    )
    assert aggregated.nll == pytest.approx(1.75)
    assert aggregated.perplexity == pytest.approx(math.exp(1.75))
    assert aggregated.evaluated_tokens == 40
    assert aggregated.elapsed_seconds == pytest.approx(5.0)
    assert aggregated.tokens_per_second == pytest.approx(8.0)
    assert aggregated.peak_memory_mb == pytest.approx(102.0)


def test_robust_score_penalizes_dispersion_and_applies_hard_limit() -> None:
    stable = summarize_validation_degradations(
        (0.01, 0.02, 0.03),
        std_penalty=1.0,
        max_slice_degradation=0.05,
    )
    unstable = summarize_validation_degradations(
        (-0.01, 0.02, 0.06),
        std_penalty=1.0,
        max_slice_degradation=0.05,
    )
    assert stable.mean_relative_perplexity_degradation == pytest.approx(0.02)
    assert stable.robust_score > stable.mean_relative_perplexity_degradation
    assert stable.passes_max_slice_degradation is True
    assert unstable.robust_score > stable.robust_score
    assert unstable.passes_max_slice_degradation is False


def test_single_slice_summary_matches_legacy_degradation() -> None:
    summary = summarize_validation_degradations(
        (0.025,),
        std_penalty=1.0,
        max_slice_degradation=None,
    )
    assert summary.std_relative_perplexity_degradation == 0.0
    assert summary.robust_score == pytest.approx(0.025)
    assert summary.passes_max_slice_degradation is True
