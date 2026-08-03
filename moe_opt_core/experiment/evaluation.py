from __future__ import annotations

from moe_opt_core.experiment.contracts import EvaluationComparisonContract, EvaluationMetricsContract


def compare_evaluations(
    *,
    candidate_id: str,
    baseline: EvaluationMetricsContract,
    candidate: EvaluationMetricsContract,
    quality_budgets: tuple[float, ...],
) -> EvaluationComparisonContract:
    relative_perplexity_degradation = (
        candidate.perplexity / max(baseline.perplexity, 1e-12) - 1.0
    )
    throughput_gain = candidate.tokens_per_second / max(baseline.tokens_per_second, 1e-12) - 1.0
    memory_reduction = 1.0 - candidate.model_parameter_mb / max(baseline.model_parameter_mb, 1e-12)
    accepted = {
        f"{budget:.6g}": relative_perplexity_degradation <= budget
        for budget in quality_budgets
    }
    return EvaluationComparisonContract(
        candidate_id=candidate_id,
        baseline=baseline,
        candidate=candidate,
        relative_perplexity_degradation=relative_perplexity_degradation,
        throughput_gain=throughput_gain,
        memory_reduction=memory_reduction,
        accepted_quality_budgets=accepted,
    )
