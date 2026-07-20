"""Tests for the S8 additive loaders and the shared Best_* selector.

* the format-tolerant co-activation loader (reads the new versioned
  ``CoActivationMatrix`` and the legacy raw tensor identically);
* the new-module fusion-layer loader (``groups/`` Partition + ``FusedExpert``)
  builds a layer numerically identical to the legacy pickle loader;
* ``select_best_fused`` (the Best_* post-selector / metric compartment).
"""
import os

import torch

from common import layout
from common.artifacts import CoActivationMatrix
from common.fusion_layer import get_fusion_layer_by_index, get_fusion_layer_from_groups
from common.mlp import AdvancedMLP
from common.mlp_utils import select_best_fused
from common.pairing import compute_activation_correlation_matrix
from common.router import Qwen15RouterAdapter
from common.testing import N_EXPERTS, TOP_K
from common.testing.builders import Builders, expert_mlp
from common.testing.fakes import H

CPU = torch.device("cpu")


def test_coactivation_loader_handles_both_formats(tmp_path):
    matrix = torch.rand(N_EXPERTS, N_EXPERTS)
    matrix = matrix + matrix.t()

    raw_dir = tmp_path / "raw"
    raw_path = layout.activation_matrix_path(str(raw_dir), layout.TRAIN_SPLIT, 0)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    torch.save(matrix, raw_path)

    v1_dir = tmp_path / "v1"
    v1_path = layout.activation_matrix_path(str(v1_dir), layout.TRAIN_SPLIT, 0)
    os.makedirs(os.path.dirname(v1_path), exist_ok=True)
    CoActivationMatrix(matrix.clone(), split=layout.TRAIN_SPLIT).save(v1_path)

    legacy = compute_activation_correlation_matrix(0, str(raw_dir))
    versioned = compute_activation_correlation_matrix(0, str(v1_dir))
    assert torch.allclose(legacy, versioned)


def test_groups_loader_matches_legacy_loader(tmp_path):
    data_dir = str(tmp_path)
    layers = [0]
    builders = Builders(data_dir, layers, N_EXPERTS)
    # Shared gate / shared-expert / experts files for both loaders.
    builders.experts()
    pairs = builders.partition()        # groups/ Partition (new)
    builders.pairs()                    # pairs/ pickle list (legacy), same pairs
    builders.fused_experts_v1("v1", pairs)   # FusedExpert (new)
    builders.fusion_experts("legacy", pairs)  # pickled AdvancedMLP (same expert_mlp seeds)

    router = Qwen15RouterAdapter()
    legacy_layer = get_fusion_layer_by_index(
        0, layout.fusion_method_dir("legacy"), data_dir, device=CPU,
        use_default_experts=False, num_experts=N_EXPERTS, router_adapter=router, top_k=TOP_K,
    )
    new_layer = get_fusion_layer_from_groups(
        0, "v1", data_dir, device=CPU,
        use_default_experts=False, num_experts=N_EXPERTS, router_adapter=router, top_k=TOP_K,
    )
    assert new_layer.expert_groups == legacy_layer.expert_groups

    torch.manual_seed(0)
    x = torch.randn(1, 5, H)
    assert torch.allclose(new_layer.forward(x), legacy_layer.forward(x), atol=1e-6)


def test_select_best_fused_fallback_is_first_method():
    a, b = AdvancedMLP(4, 6, 4, device=CPU), AdvancedMLP(4, 6, 4, device=CPU)
    fused = {"Average": a, "ZipIt": b}
    chosen = select_best_fused(fused, None, {"MSE": False, "Cosine": True})
    # No scores -> the first registered method, not a hardcoded "Pruning".
    assert chosen["Best_MSE"] is a
    assert chosen["Best_Cosine"] is a


def test_select_best_fused_picks_winner_per_metric():
    a, b = AdvancedMLP(4, 6, 4, device=CPU), AdvancedMLP(4, 6, 4, device=CPU)
    fused = {"Average": a, "ZipIt": b}
    results = {
        "Average": {"MSE": 0.10, "Cosine": 0.20},
        "ZipIt": {"MSE": 0.05, "Cosine": 0.90},
    }
    chosen = select_best_fused(fused, results, {"MSE": False, "Cosine": True})
    assert chosen["Best_MSE"] is b      # lower MSE wins
    assert chosen["Best_Cosine"] is b   # higher Cosine wins


def test_select_best_fused_tie_break_reproduces_legacy_quirk():
    # The legacy stage-6 selector uses ``(score > best) == higher_is_better``: on
    # a tie a lower-is-better metric (hib=False) reassigns (last method wins),
    # while a higher-is-better metric (hib=True) keeps the first. select_best_fused
    # reproduces this exactly -- the golden enforces equivalence, so the quirk is
    # preserved deliberately, not "fixed".
    a, b = AdvancedMLP(4, 6, 4, device=CPU), AdvancedMLP(4, 6, 4, device=CPU)
    fused = {"Average": a, "ZipIt": b}
    results = {"Average": {"MSE": 0.5, "Cosine": 0.5}, "ZipIt": {"MSE": 0.5, "Cosine": 0.5}}
    chosen = select_best_fused(fused, results, {"MSE": False, "Cosine": True})
    assert chosen["Best_MSE"] is b       # lower-is-better tie -> last method
    assert chosen["Best_Cosine"] is a    # higher-is-better tie -> first method
