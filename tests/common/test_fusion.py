"""Reference tests for the fusion methods.

Beyond shape/fallback behavior, each faithful method is pinned to its paper
definition on a small analytically known case:

* ZipIt        -- unmerge U is the pseudo-inverse of merge M.
* Procrustes   -- the alignment recovers a known rotation and is function-
                  preserving for a permutation (Git Re-Basin).
* TIES         -- trim -> elect-sign -> disjoint-mean on a hand-built case.
* Fisher       -- precision-weighted average dominated by the higher Fisher.
"""

import pytest
import torch

from common import fusion
from common.fusion import (
    AverageFusion,
    DistillationFusion,
    FisherFusion,
    FusionMethod,
    LeastSquaresAdapter,
    ProcrustesFusion,
    PruningFusion,
    TiesFusion,
    ZipItFusion,
    build_fusion_methods,
    diagonal_empirical_fisher,
    fisher_weighted_average,
    orthogonal_procrustes,
)
from common.mlp import AdvancedMLP

CPU = torch.device("cpu")


@pytest.fixture
def pair(make_advanced_mlp):
    a = make_advanced_mlp(4, 6, 4, seed=1)
    b = make_advanced_mlp(4, 6, 4, seed=2)
    return a, b


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(32, 4)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def test_as_tensor_passthrough_and_stack():
    t = torch.randn(3, 4)
    assert fusion.as_tensor(t) is t
    stacked = fusion.as_tensor([torch.zeros(4), torch.ones(4)])
    assert stacked.shape == (2, 4)


def test_as_2d_tensor_promotes_and_moves():
    out = fusion.as_2d_tensor(torch.zeros(4), device=CPU, dtype=torch.float64)
    assert out.shape == (1, 4)
    assert out.dtype == torch.float64


def test_model_device_and_hidden_activations(make_advanced_mlp):
    mlp = make_advanced_mlp(4, 6, 4)
    assert fusion.model_device(mlp).type == "cpu"
    hidden = fusion.hidden_activations(mlp, torch.randn(5, 4))
    assert hidden.shape == (5, 6)


def test_batch_iter_chunks():
    chunks = list(fusion.batch_iter(torch.arange(10), 4))
    assert [len(c) for c in chunks] == [4, 4, 2]


def test_solve_linear_recovers_known_mapping():
    torch.manual_seed(0)
    x = torch.randn(64, 5)
    true_w = torch.randn(3, 5)
    y = x @ true_w.t()
    w = fusion.solve_linear(x, y, lambda_ridge=1e-6)
    assert w.shape == (3, 5)
    assert torch.allclose(x @ w.t(), y, atol=1e-3)


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #
def test_registry_has_pruning_not_projection():
    methods = build_fusion_methods()
    expected_base = {
        "ZipIt", "Average", "Pruning", "Procrustes", "Fisher", "TIES", "Distillation",
    }
    assert expected_base <= set(methods)
    assert "Projection" not in methods
    for name in expected_base:
        assert f"LS_{name}" in methods
    for method in methods.values():
        assert isinstance(method, FusionMethod)
        assert callable(method)


# --------------------------------------------------------------------------- #
# Average / Pruning (data-free).
# --------------------------------------------------------------------------- #
def test_average_is_exact_mean(pair, data):
    a, b = pair
    merged = AverageFusion()(a, b, data)
    expected = (a.gate.weight + b.gate.weight) / 2
    assert torch.allclose(merged.gate.weight, expected)


def test_average_create_new_does_not_mutate_input(pair, data):
    a, b = pair
    before = a.gate.weight.clone()
    AverageFusion()(a, b, data, create_new=True)
    assert torch.equal(a.gate.weight, before)


def test_pruning_returns_copy_of_first(pair, data):
    a, b = pair
    merged = PruningFusion()(a, b, data)
    assert torch.equal(merged.gate.weight, a.gate.weight)
    assert merged is not a


def test_pruning_keeps_a_even_on_empty_data(pair):
    # Pruning is data-free: empty data must NOT trigger the average fallback.
    a, b = pair
    merged = PruningFusion()(a, b, [])
    assert torch.equal(merged.gate.weight, a.gate.weight)
    assert not torch.allclose(merged.gate.weight, (a.gate.weight + b.gate.weight) / 2)


# --------------------------------------------------------------------------- #
# ZipIt.
# --------------------------------------------------------------------------- #
def test_zipit_unmerge_is_pseudo_inverse_of_merge():
    from common.mlp import Zipper

    zipper = Zipper(4)
    torch.manual_seed(0)
    zipper(torch.randn(64, 4))
    zipper.build_MU()
    # U == pinv(M): M @ U == I on the reduced space.
    assert torch.allclose(zipper.M @ zipper.U, torch.eye(2), atol=1e-5)
    assert torch.allclose(zipper.U, torch.linalg.pinv(zipper.M), atol=1e-5)


def test_zipit_produces_unzippable_model(pair, data):
    a, b = pair
    merged = ZipItFusion()(a, b, data)
    out = merged.forward_unzip(data)
    assert out.shape == (data.shape[0], 8)


def test_zipit_empty_falls_back_to_average(pair):
    a, b = pair
    merged = ZipItFusion()(a, b, [])
    expected = (a.gate.weight + b.gate.weight) / 2
    assert torch.allclose(merged.gate.weight, expected)


# --------------------------------------------------------------------------- #
# Procrustes.
# --------------------------------------------------------------------------- #
def test_orthogonal_procrustes_recovers_known_rotation():
    torch.manual_seed(0)
    source = torch.randn(40, 6)
    rotation, _ = torch.linalg.qr(torch.randn(6, 6))
    target = source @ rotation
    recovered = orthogonal_procrustes(target, source)
    assert torch.allclose(recovered, rotation, atol=1e-4)


def test_apply_permutation_is_function_preserving(make_advanced_mlp):
    mlp = make_advanced_mlp(4, 6, 4, seed=3)
    perm = torch.eye(6)[[2, 0, 1, 5, 4, 3]]  # a permutation matrix
    aligned = fusion._apply_hidden_rotation(mlp, perm)
    x = torch.randn(10, 4)
    assert torch.allclose(aligned.forward(x), mlp.forward(x), atol=1e-5)


def test_procrustes_merge_shape_and_fallback(pair, data):
    a, b = pair
    merged = ProcrustesFusion()(a, b, data)
    assert merged.forward(data).shape == (32, 4)
    empty = ProcrustesFusion()(a, b, [])
    assert torch.allclose(empty.gate.weight, (a.gate.weight + b.gate.weight) / 2)


# --------------------------------------------------------------------------- #
# TIES.
# --------------------------------------------------------------------------- #
def test_trim_zeros_smallest_fraction():
    w = torch.tensor([0.1, 5.0, -0.2, -8.0])
    trimmed = fusion._trim(w, trim_ratio=0.5)
    assert trimmed.tolist() == [0.0, 5.0, 0.0, -8.0]


def test_ties_elect_sign_and_disjoint_mean():
    # Agreement -> mean; conflict -> the sign-elected (larger) value wins.
    a = AdvancedMLP(1, 2, 1, device=CPU)
    b = AdvancedMLP(1, 2, 1, device=CPU)
    with torch.no_grad():
        a.gate.weight.copy_(torch.tensor([[2.0], [-1.0]]))
        b.gate.weight.copy_(torch.tensor([[1.0], [-3.0]]))
    merged = TiesFusion(trim_ratio=0.0)(a, b, torch.randn(4, 1))
    assert torch.allclose(merged.gate.weight, torch.tensor([[1.5], [-2.0]]), atol=1e-5)


def test_ties_sign_conflict_discards_minority():
    a = AdvancedMLP(1, 1, 1, device=CPU)
    b = AdvancedMLP(1, 1, 1, device=CPU)
    with torch.no_grad():
        a.gate.weight.copy_(torch.tensor([[2.0]]))
        b.gate.weight.copy_(torch.tensor([[-5.0]]))
    merged = TiesFusion(trim_ratio=0.0)(a, b, torch.randn(4, 1))
    # gamma = sign(2 - 5) = -1, so the +2 entry is dropped and -5 survives.
    assert merged.gate.weight.item() == pytest.approx(-5.0, abs=1e-5)


def test_ties_empty_falls_back_to_average(pair):
    a, b = pair
    merged = TiesFusion()(a, b, [])
    assert torch.allclose(merged.gate.weight, (a.gate.weight + b.gate.weight) / 2)


# --------------------------------------------------------------------------- #
# Fisher.
# --------------------------------------------------------------------------- #
def test_fisher_weighted_average_dominated_by_larger_fisher():
    state_a = {"w": torch.tensor([1.0, 2.0])}
    state_b = {"w": torch.tensor([9.0, 9.0])}
    fisher_a = {"w": torch.tensor([1e6, 1e-6])}
    fisher_b = {"w": torch.tensor([1e-6, 1e6])}
    merged = fisher_weighted_average(state_a, state_b, fisher_a, fisher_b)
    assert torch.allclose(merged["w"], torch.tensor([1.0, 9.0]), atol=1e-3)


def test_fisher_weighted_average_equal_fisher_is_mean():
    state_a = {"w": torch.tensor([1.0, 3.0])}
    state_b = {"w": torch.tensor([5.0, 1.0])}
    ones = {"w": torch.ones(2)}
    merged = fisher_weighted_average(state_a, state_b, ones, ones)
    assert torch.allclose(merged["w"], torch.tensor([3.0, 2.0]), atol=1e-5)


def test_diagonal_empirical_fisher_nonneg_and_keyed(pair, data):
    a, _ = pair
    fisher = diagonal_empirical_fisher(a, data)
    assert set(fisher) == {name for name, _ in a.named_parameters()}
    for value in fisher.values():
        assert torch.all(value >= 0)


def test_fisher_merge_preserves_shape_and_fallback(pair, data):
    a, b = pair
    merged = FisherFusion()(a, b, data)
    assert merged.gate.weight.shape == a.gate.weight.shape
    assert merged.forward(data).shape == (32, 4)
    empty = FisherFusion()(a, b, [])
    assert torch.allclose(empty.gate.weight, (a.gate.weight + b.gate.weight) / 2)


# --------------------------------------------------------------------------- #
# Distillation.
# --------------------------------------------------------------------------- #
def test_distillation_runs_and_preserves_shape(pair, data):
    a, b = pair
    student = DistillationFusion()(a, b, data)
    assert student.forward(data).shape == (32, 4)


def test_distillation_empty_falls_back_to_average(pair):
    a, b = pair
    merged = DistillationFusion()(a, b, [])
    assert torch.allclose(merged.gate.weight, (a.gate.weight + b.gate.weight) / 2)


def test_distillation_bf16_trains_in_fp32_and_casts_back(monkeypatch, make_advanced_mlp):
    """CR#6 regression guard. bf16 AdamW moments/grads underflow, so the loop must
    run in float32 and cast the student back to the experts' dtype afterwards.

    The guard is structural, not behavioral: on a toy MLP the bf16-underflow effect
    is negligible, so any output-error comparison passes on the unfixed bf16 loop
    too. Instead we assert the mechanism directly -- the optimizer is built on
    float32 params (float() ran before it) -- plus the castback (returned params
    are bf16). Reverting either half of the fix fails one of these assertions."""
    seen_optimizer_dtypes = set()
    real_adamw = torch.optim.AdamW

    def spy_adamw(params, *args, **kwargs):
        params = list(params)
        seen_optimizer_dtypes.update(p.dtype for p in params)
        return real_adamw(params, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", spy_adamw)

    torch.manual_seed(0)
    a = make_advanced_mlp(4, 6, 4, seed=1).to(torch.bfloat16)
    b = make_advanced_mlp(4, 6, 4, seed=2).to(torch.bfloat16)
    data = torch.randn(64, 4, dtype=torch.bfloat16)

    student = DistillationFusion()(a, b, data)

    # Mechanism: training ran in float32 (would be bf16 without the fix).
    assert seen_optimizer_dtypes == {torch.float32}
    # Castback: the merged student is returned in the experts' dtype.
    for name, param in student.named_parameters():
        assert param.dtype == torch.bfloat16, f"{name} came back as {param.dtype}"


def test_distillation_fp32_experts_unchanged_by_fp32_loop(make_advanced_mlp):
    """The fp32 castback path is a no-op for float32 experts: the merge stays
    deterministic and bit-identical to a rerun (guards the golden float32 path)."""
    a = make_advanced_mlp(4, 6, 4, seed=1)
    b = make_advanced_mlp(4, 6, 4, seed=2)
    data = torch.randn(32, 4)

    torch.manual_seed(0)
    first = DistillationFusion()(a, b, data)
    torch.manual_seed(0)
    second = DistillationFusion()(a, b, data)

    for key in first.state_dict():
        assert first.state_dict()[key].dtype == torch.float32
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])


# --------------------------------------------------------------------------- #
# Least-squares adapter.
# --------------------------------------------------------------------------- #
def test_ls_fits_projection_with_enough_data(pair):
    a, b = pair
    torch.manual_seed(0)
    train = torch.randn(64, 4)
    fused = LeastSquaresAdapter(AverageFusion())(a, b, train)
    assert fused.down is None
    assert fused.U is not None
    assert fused.forward_unzip(train).shape == (64, 8)


def test_ls_returns_early_with_little_data(pair):
    a, b = pair
    fused = LeastSquaresAdapter(AverageFusion())(a, b, torch.randn(4, 4))
    assert fused.down is not None


def test_ls_name_prefixes_inner():
    assert LeastSquaresAdapter(ZipItFusion()).name == "LS_ZipIt"
    assert LeastSquaresAdapter(PruningFusion()).name == "LS_Pruning"


class _PlugFusion(AverageFusion):
    name = "Plug"


def test_apply_meta_wrappers_covers_plugin_methods_and_is_idempotent():
    from common.registry import Registry

    reg = Registry("fusion")
    reg.register("Average", AverageFusion())
    fusion.apply_meta_wrappers(reg)
    assert "LS_Average" in reg

    # a base method registered after the first wrap is covered on re-apply
    reg.register("Plug", _PlugFusion())
    fusion.apply_meta_wrappers(reg)
    assert "LS_Plug" in reg

    # re-applying again is a no-op (no RegistryError, no duplicates)
    before = list(reg)
    fusion.apply_meta_wrappers(reg)
    assert list(reg) == before
