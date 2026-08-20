import pytest
import torch

from moe_opt_core.experiment import CandidateMethod, MergeWeightingPolicy
from moe_opt_core.experiment.candidates import build_candidate_plans
from moe_opt_core.experiment.merge_weighting import build_group_merge_weights
from moe_opt_core.pipelines.h1_merge import merge_weight_matrices_baseline


@pytest.mark.parametrize(
    ("policy", "minimum_weight", "expected_pair"),
    (
        (MergeWeightingPolicy.UNIFORM, 0.0, (0.5, 0.5)),
        (MergeWeightingPolicy.USAGE_PROPORTIONAL, 0.0, (0.25, 0.75)),
        (MergeWeightingPolicy.USAGE_BOUNDED, 0.2, (0.35, 0.65)),
    ),
)
def test_group_merge_weights_follow_the_selected_policy(
    policy: MergeWeightingPolicy,
    minimum_weight: float,
    expected_pair: tuple[float, float],
) -> None:
    weights = build_group_merge_weights(
        ((0, 1), (2,)),
        (0.1, 0.3, 0.6),
        policy=policy,
        minimum_weight=minimum_weight,
    )
    assert weights[0] == pytest.approx(expected_pair)
    assert weights[1] == (1.0,)
    assert all(sum(group) == pytest.approx(1.0) for group in weights)


def test_zero_usage_falls_back_to_uniform_weights() -> None:
    weights = build_group_merge_weights(
        ((0, 1),),
        (0.0, 0.0),
        policy=MergeWeightingPolicy.USAGE_PROPORTIONAL,
    )
    assert weights == ((0.5, 0.5),)


def test_bounded_usage_weights_limit_expert_dominance() -> None:
    weights = build_group_merge_weights(
        ((0, 1),),
        (0.0, 1.0),
        policy=MergeWeightingPolicy.USAGE_BOUNDED,
        minimum_weight=0.2,
    )
    assert weights[0] == pytest.approx((0.2, 0.8))


def test_candidate_weight_variants_keep_identical_pairs_and_fixed_rank() -> None:
    plans = build_candidate_plans(
        num_experts=4,
        layer=0,
        methods=("rmt_geometry",),
        compression_fractions=(0.25,),
        seeds=(7,),
        rmt_pairs=((0, 1), (2, 3)),
        spectral_embedding=torch.ones((4, 1)),
        spectral_embeddings={
            2: torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
            )
        },
        spectral_rank_schedule=(("rmt_geometry", 0.25, 2),),
        functional_embedding=torch.ones((4, 2)),
        routing_probs=torch.tensor([[0.1, 0.3, 0.2, 0.4]]),
        merge_weighting_methods=("rmt_geometry",),
        merge_weighting_variants=(
            ("equal", "uniform", 0.0),
            ("proportional", "usage_proportional", 0.0),
            ("bounded_20", "usage_bounded", 0.2),
        ),
    )
    assert len(plans) == 3
    assert all(plan.method == CandidateMethod.RMT_GEOMETRY for plan in plans)
    assert len({plan.groups for plan in plans}) == 1
    assert all(dict(plan.diagnostics)["spectral_pairing_rank"] == 2.0 for plan in plans)
    assert {plan.merge_weighting for plan in plans} == {
        MergeWeightingPolicy.UNIFORM,
        MergeWeightingPolicy.USAGE_PROPORTIONAL,
        MergeWeightingPolicy.USAGE_BOUNDED,
    }
    assert {plan.candidate_id for plan in plans} == {
        "rmt_geometry-r2-mwequal-f0.250-s7",
        "rmt_geometry-r2-mwproportional-f0.250-s7",
        "rmt_geometry-r2-mwbounded_20-f0.250-s7",
    }


def test_matrix_merge_normalizes_explicit_coefficients() -> None:
    first = torch.tensor([[1.0, 1.0]])
    second = torch.tensor([[5.0, 5.0]])
    merged = merge_weight_matrices_baseline((first, second), usage=(1.0, 3.0))
    assert torch.equal(merged, torch.tensor([[4.0, 4.0]]))


def test_explicit_uniform_merge_is_equivalent_to_legacy_default() -> None:
    first = torch.tensor([[1.0, 3.0]])
    second = torch.tensor([[5.0, 7.0]])
    legacy = merge_weight_matrices_baseline((first, second))
    explicit = merge_weight_matrices_baseline((first, second), usage=(0.5, 0.5))
    assert torch.equal(explicit, legacy)


def test_matrix_merge_rejects_misaligned_coefficients() -> None:
    with pytest.raises(ValueError, match="align"):
        merge_weight_matrices_baseline((torch.ones(1), torch.ones(1)), usage=(1.0,))


def test_functional_ridge_weights_follow_token_conditioned_expert_outputs() -> None:
    weights = build_group_merge_weights(
        ((0, 1),),
        (1.0, 0.0),
        policy=MergeWeightingPolicy.FUNCTIONAL_RIDGE,
        ridge_lambda=1e-3,
        functional_embedding=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        routing_probs=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert weights[0][0] > 0.99
    assert weights[0][1] < 0.01
    assert sum(weights[0]) == pytest.approx(1.0)


def test_functional_ridge_supports_non_token_aligned_weight_sketches() -> None:
    weights = build_group_merge_weights(
        ((0, 1),),
        (0.75, 0.25),
        policy=MergeWeightingPolicy.FUNCTIONAL_RIDGE,
        ridge_lambda=1e-3,
        functional_embedding=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
        ),
        routing_probs=torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]]
        ),
    )
    assert weights[0][0] == pytest.approx(2.5 / 3.0, abs=1e-3)
    assert weights[0][1] == pytest.approx(0.5 / 3.0, abs=1e-3)
    assert sum(weights[0]) == pytest.approx(1.0)


def test_candidate_plan_records_functional_ridge_sketch_mode() -> None:
    plans = build_candidate_plans(
        num_experts=4,
        layer=11,
        methods=("rmt_geometry",),
        compression_fractions=(0.25,),
        seeds=(2026,),
        rmt_pairs=((0, 1), (2, 3)),
        spectral_embedding=torch.ones((4, 1)),
        functional_embedding=torch.eye(4, 5),
        routing_probs=torch.tensor(
            [
                [0.7, 0.2, 0.1, 0.0],
                [0.6, 0.3, 0.1, 0.0],
                [0.5, 0.2, 0.2, 0.1],
            ]
        ),
        merge_weighting_methods=("rmt_geometry",),
        merge_weighting_variants=(
            ("functional_ridge", "functional_ridge", 0.0, 1e-3),
        ),
    )
    assert len(plans) == 1
    diagnostics = dict(plans[0].diagnostics)
    assert diagnostics["merge_ridge_token_aligned"] == 0.0
    assert diagnostics["merge_ridge_routing_rows"] == 3.0
    assert diagnostics["merge_ridge_functional_features"] == 5.0
    assert sum(plans[0].group_merge_weights[0]) == pytest.approx(1.0)
