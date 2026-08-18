from __future__ import annotations

import torch

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import CandidateMethod
from moe_opt_runner.candidate_factory import build_layer_candidate_plans


def _layer_artifacts() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    routing_probs = torch.full((8, 60), 1.0 / 60.0)
    raw_embedding = torch.arange(60 * 4, dtype=torch.float32).reshape(60, 4)
    trace = {"routing_probs": routing_probs}
    embedding = {
        "raw_embedding": raw_embedding,
        "spectral_embedding": raw_embedding[:, :1],
        "rmt_pairs": tuple((index, index + 1) for index in range(0, 60, 2)),
    }
    return trace, embedding


def test_non_weighted_method_ignores_unrelated_merge_weight_variants() -> None:
    config = load_experiment_config(
        "configs/experiments/h1_rmt_multilayer_representative_pilot.json"
    )
    trace, embedding = _layer_artifacts()

    plans = build_layer_candidate_plans(
        config,
        layer=0,
        num_experts=60,
        trace=trace,
        embedding=embedding,
        methods=("functional_cosine",),
        compression_fractions=(0.1,),
    )

    assert len(plans) == 1
    assert plans[0].method == CandidateMethod.FUNCTIONAL_COSINE


def test_weighted_method_retains_configured_merge_weight_variants() -> None:
    config = load_experiment_config(
        "configs/experiments/h1_rmt_multilayer_representative_pilot.json"
    )
    trace, embedding = _layer_artifacts()

    plans = build_layer_candidate_plans(
        config,
        layer=0,
        num_experts=60,
        trace=trace,
        embedding=embedding,
        methods=("rmt_functional_usage",),
        compression_fractions=(0.1,),
    )

    assert plans
    assert all(plan.method == CandidateMethod.RMT_FUNCTIONAL_USAGE for plan in plans)
    assert {plan.merge_weighting.value for plan in plans} == {
        variant.policy for variant in config.pilot.merge_weighting.variants
    }
