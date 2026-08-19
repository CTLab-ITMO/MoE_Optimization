from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from moe_opt_core.config import ExperimentConfig
from moe_opt_core.experiment.candidates import build_candidate_plans
from moe_opt_core.pipelines.rmt_diagnostics import (
    resolve_spectral_rank_grid,
    spectral_embeddings_at_ranks,
)


def build_layer_candidate_plans(
    config: ExperimentConfig,
    *,
    layer: int,
    num_experts: int,
    trace: Mapping[str, object],
    embedding: Mapping[str, object],
    methods: Sequence[str] | None = None,
    compression_fractions: Sequence[float] | None = None,
    namespace_layer: bool = False,
):
    """Materialize candidate plans from typed config and layer artifacts."""
    selected_methods = (
        tuple(methods) if methods is not None else config.pilot.methods
    )
    merge_weighting_methods = tuple(
        method
        for method in config.pilot.merge_weighting.methods
        if method in selected_methods
    )
    diagnostics = embedding.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    useful_rank = int(
        diagnostics.get("r_useful", embedding["spectral_embedding"].shape[1])
    )
    mp_rank = diagnostics.get("mp_rank")
    policy = config.pilot.spectral_rank_policy
    effective_ranks = resolve_spectral_rank_grid(
        embedding["raw_embedding"],
        fixed_ranks=config.pilot.spectral_pairing_ranks,
        useful_rank=useful_rank,
        mp_rank=(int(mp_rank) if mp_rank is not None else None),
        include_useful_rank=policy.include_useful_rank,
        include_mp_rank=policy.include_mp_rank,
        explained_variance_thresholds=policy.explained_variance_thresholds,
    )
    spectral_embeddings = spectral_embeddings_at_ranks(
        embedding["raw_embedding"],
        effective_ranks,
    )
    plans = build_candidate_plans(
        num_experts=num_experts,
        layer=layer,
        methods=selected_methods,
        compression_fractions=(
            tuple(compression_fractions)
            if compression_fractions is not None
            else config.pilot.compression_fractions
        ),
        seeds=config.pilot.seeds,
        rmt_pairs=embedding["rmt_pairs"],
        spectral_embedding=embedding["spectral_embedding"],
        functional_embedding=embedding["raw_embedding"],
        routing_probs=trace["routing_probs"],
        hybrid_weights=(
            config.pilot.hybrid_pairing.rmt_weight,
            config.pilot.hybrid_pairing.functional_weight,
            config.pilot.hybrid_pairing.usage_weight,
        ),
        spectral_embeddings=spectral_embeddings,
        hybrid_weight_variants=tuple(
            (
                variant.name,
                (
                    variant.rmt_weight,
                    variant.functional_weight,
                    variant.usage_weight,
                ),
            )
            for variant in config.pilot.hybrid_pairing.variants
        ),
        spectral_rank_schedule=tuple(
            (
                assignment.method,
                assignment.compression_fraction,
                assignment.rank,
            )
            for assignment in config.pilot.spectral_rank_schedule
            if assignment.method in selected_methods
        ),
        merge_weighting_methods=merge_weighting_methods,
        merge_weighting_variants=(
            tuple(
                (
                    variant.name,
                    variant.policy,
                    variant.minimum_weight,
                    variant.ridge_lambda,
                )
                for variant in config.pilot.merge_weighting.variants
            )
            if merge_weighting_methods
            else ()
        ),
    )
    if not namespace_layer:
        return plans
    return tuple(
        replace(plan, candidate_id=f"l{int(layer)}-{plan.candidate_id}")
        for plan in plans
    )
