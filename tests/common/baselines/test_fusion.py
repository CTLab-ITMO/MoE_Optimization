"""Tests for the HC-SMoE FrequencyWeightedAverage merge (k-agnostic, ctx variant A).

The merge weights a cluster's experts by their raw top-k selection counts (the
train CoActivationMatrix diagonal), read from the artifact store via ``ctx``;
absent context or all-zero counts fall back to equal weights.
"""

import os

import pytest
import torch

import common.baselines  # noqa: F401  ensure FrequencyWeightedAverage is registered
from common import layout
from common.artifacts import CoActivationMatrix, Group, Partition
from common.baselines.hcsmoe import FrequencyWeightedAverage
from common.context import StageContext
from common.fusion import build_fusion_methods


@pytest.fixture
def pair(make_advanced_mlp):
    a = make_advanced_mlp(4, 6, 4, seed=1)
    b = make_advanced_mlp(4, 6, 4, seed=2)
    return a, b


# --------------------------------------------------------------------------- #
# FrequencyWeightedAverage (HC-SMoE merge, k-agnostic, ctx variant A).
# --------------------------------------------------------------------------- #
def _write_freq_artifacts(data_dir, layer, counts, group_members):
    """Lay out the two stage-1/2 artifacts the method reads (variant A): a train
    CoActivationMatrix whose diagonal holds the selection counts, and a Partition."""
    matrix = torch.diag(torch.tensor(counts, dtype=torch.float32))
    os.makedirs(layout.activations_dir(data_dir), exist_ok=True)
    CoActivationMatrix(
        matrix, split=layout.TRAIN_SPLIT, sample_size=int(matrix.sum())
    ).save(layout.activation_matrix_path(data_dir, layout.TRAIN_SPLIT, layer))
    os.makedirs(layout.groups_dir(data_dir), exist_ok=True)
    Partition(
        groups=tuple(Group(members=tuple(m)) for m in group_members)
    ).save(layout.group_path(data_dir, layer))


def _weighted_state(mlps, weights):
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    states = [mlp.state_dict() for mlp in mlps]
    return {
        key: sum(states[i][key].float() * weights[i] for i in range(len(mlps)))
        for key in states[0]
    }


def test_freq_weighted_is_registered_k_agnostic_without_ls_wrapper():
    methods = build_fusion_methods()
    method = methods["FrequencyWeightedAverage"]
    assert method.supported_arity is None  # k-agnostic
    assert method.needs_data is False  # counts, not train_data
    assert method.wrappable is False
    assert "LS_FrequencyWeightedAverage" not in methods


def test_freq_weighted_k1_is_identity_copy(pair):
    a, _ = pair
    merged = FrequencyWeightedAverage()(a, train_data=[])
    assert merged is not a
    for key, value in merged.state_dict().items():
        assert torch.equal(value, a.state_dict()[key])


def test_freq_weighted_no_ctx_falls_back_to_equal_weights(pair):
    # Without an artifact context the merge degrades to a plain average.
    a, b = pair
    merged = FrequencyWeightedAverage()(a, b, train_data=[])
    expected = (a.gate.weight + b.gate.weight) / 2
    assert torch.allclose(merged.gate.weight, expected)


def test_freq_weighted_weights_by_selection_counts(pair, tmp_path):
    a, b = pair
    data_dir = str(tmp_path)
    _write_freq_artifacts(data_dir, layer=5, counts=[3.0, 1.0], group_members=[(0, 1)])
    ctx = StageContext(data_dir=data_dir, layer_index=5, group_index=0)

    merged = FrequencyWeightedAverage()(a, b, train_data=[], ctx=ctx)

    expected = _weighted_state((a, b), [3.0, 1.0])
    for key, value in merged.state_dict().items():
        assert torch.allclose(value, expected[key], atol=1e-6)
    # Distinct from a plain average (proves counts actually drive the weighting).
    assert not torch.allclose(merged.gate.weight, (a.gate.weight + b.gate.weight) / 2)


def test_freq_weighted_empty_train_data_still_uses_counts(pair, tmp_path):
    # needs_data=False: an empty train_data must NOT swap in the Average fallback
    # inside __call__ -- the frequency weights still apply.
    a, b = pair
    data_dir = str(tmp_path)
    _write_freq_artifacts(data_dir, layer=0, counts=[4.0, 1.0], group_members=[(0, 1)])
    ctx = StageContext(data_dir=data_dir, layer_index=0, group_index=0)

    merged = FrequencyWeightedAverage()(a, b, train_data=[], ctx=ctx)
    expected = _weighted_state((a, b), [4.0, 1.0])
    assert torch.allclose(merged.gate.weight, expected["gate.weight"], atol=1e-6)


def test_freq_weighted_k3_weights_by_counts(pair, make_advanced_mlp, tmp_path):
    a, b = pair
    c = make_advanced_mlp(4, 6, 4, seed=3)
    data_dir = str(tmp_path)
    # Members are a non-trivial (unordered, offset) subset to exercise indexing.
    _write_freq_artifacts(
        data_dir, layer=2, counts=[0.0, 2.0, 5.0, 1.0], group_members=[(1, 2, 3)]
    )
    ctx = StageContext(data_dir=data_dir, layer_index=2, group_index=0)

    merged = FrequencyWeightedAverage()(a, b, c, train_data=[], ctx=ctx)
    expected = _weighted_state((a, b, c), [2.0, 5.0, 1.0])
    for key, value in merged.state_dict().items():
        assert torch.allclose(value, expected[key], atol=1e-6)


def test_freq_weighted_all_zero_counts_fall_back_to_equal(pair, tmp_path):
    # Documented deviation from the reference's degenerate zero merge: a group
    # whose members were never routed averages with equal weights.
    a, b = pair
    data_dir = str(tmp_path)
    _write_freq_artifacts(data_dir, layer=1, counts=[0.0, 0.0], group_members=[(0, 1)])
    ctx = StageContext(data_dir=data_dir, layer_index=1, group_index=0)

    merged = FrequencyWeightedAverage()(a, b, train_data=[], ctx=ctx)
    expected = (a.gate.weight + b.gate.weight) / 2
    assert torch.allclose(merged.gate.weight, expected, atol=1e-6)


def test_freq_weighted_rejects_member_count_mismatch(pair, tmp_path):
    a, b = pair
    data_dir = str(tmp_path)
    # Partition group has 3 members but the merge receives 2 experts.
    _write_freq_artifacts(
        data_dir, layer=0, counts=[1.0, 1.0, 1.0], group_members=[(0, 1, 2)]
    )
    ctx = StageContext(data_dir=data_dir, layer_index=0, group_index=0)
    with pytest.raises(ValueError, match="members but the merge received"):
        FrequencyWeightedAverage()(a, b, train_data=[], ctx=ctx)


def test_freq_weighted_create_new_false_returns_same_object_for_singleton(pair):
    a, _ = pair
    assert FrequencyWeightedAverage()(a, create_new=False) is a
