import pytest
import torch
import torch.nn.functional as F

from common.mlp import (
    AdvancedMLP,
    MLP,
    SimpleMLP,
    Zipper,
    greedy_correlation_matching,
    unite_matrix,
)
from common.numeric import solve_linear

CPU = torch.device("cpu")


def test_unite_matrix_block_diagonal_shape_and_values():
    a = torch.ones(2, 3)
    b = 2 * torch.ones(4, 5)

    united = unite_matrix(a, b)

    assert united.shape == (2 + 4, 3 + 5)
    assert torch.equal(united[:2, :3], a)
    assert torch.equal(united[2:, 3:], b)
    assert torch.count_nonzero(united[:2, 3:]) == 0
    assert torch.count_nonzero(united[2:, :3]) == 0


def test_zipper_forward_records_and_passes_through():
    z = Zipper(4)
    x = torch.randn(3, 4)
    out = z(x)
    assert torch.equal(out, x)
    assert len(z.activations) == 1


def test_zipper_forward_promotes_1d_input_to_2d():
    z = Zipper(4)
    out = z(torch.randn(4))
    assert out.shape == (1, 4)


def test_zipper_build_mu_shapes_and_relation():
    z = Zipper(4)
    torch.manual_seed(0)
    z(torch.randn(50, 4))
    z.build_MU()

    n = 4
    assert z.M.shape == (n // 2, n)
    assert z.U.shape == (n, n // 2)
    assert torch.allclose(z.M.sum(dim=1), torch.full((n // 2,), 1.0))
    assert torch.allclose(z.U, 2 * z.M.t())
    assert z.activations == []


def test_zipper_build_mu_requires_activations():
    z = Zipper(4)
    with pytest.raises(AssertionError):
        z.build_MU()



def test_greedy_correlation_matching_returns_disjoint_pairs():
    n = 6
    corr = torch.rand(n, n)
    corr = corr + corr.t()

    matches = greedy_correlation_matching(corr)

    assert len(matches) == n // 2
    seen = [idx for pair in matches for idx in pair]
    assert sorted(seen) == list(range(n))


def test_greedy_correlation_matching_picks_strongest_pair():
    corr = torch.full((4, 4), 0.01)
    corr[1, 3] = corr[3, 1] = 10.0
    corr.fill_diagonal_(0.0)

    matches = greedy_correlation_matching(corr)
    matches = [tuple(sorted(x)) for x in matches]
    assert (1, 3) in matches


def test_greedy_correlation_matching_ignores_diagonal():
    corr = torch.eye(4) * 100.0
    matches = greedy_correlation_matching(corr)
    for i, j in matches:
        assert i != j

def test_base_mlp_methods_are_abstract():
    base = MLP()
    with pytest.raises(NotImplementedError):
        base.forward(torch.zeros(1))
    with pytest.raises(NotImplementedError):
        base.zip_with(base)
    with pytest.raises(NotImplementedError):
        base.zip()


def test_simple_mlp_forward_shape():
    mlp = SimpleMLP(3, 4, 5, device=CPU)
    out = mlp.forward(torch.randn(8, 3))
    assert out.shape == (8, 5)


def test_simple_mlp_full_zip_pipeline_reconstructs_two_outputs():
    torch.manual_seed(0)
    a = SimpleMLP(3, 4, 4, device=CPU)
    b = SimpleMLP(3, 4, 4, device=CPU)

    x = torch.randn(20, 3)
    target = torch.cat([a.forward(x), b.forward(x)], dim=-1)

    a.zip_with(b)
    assert a.zipped is True
    a.train_with([row for row in x])
    a.zip()

    assert a.zip1 is None and a.zip2 is None
    assert a.zipped is False

    y = a.forward(x)
    recon = F.linear(y, a.U)
    assert recon.shape == target.shape


def test_advanced_mlp_forward_shape_and_dtype_roundtrip():
    mlp = AdvancedMLP(4, 6, 5, device=CPU)
    x = torch.randn(8, 4, dtype=torch.float32)
    out = mlp.forward(x)
    assert out.shape == (8, 5)
    assert out.dtype == torch.float32


def test_advanced_mlp_forward_unzip_without_u_duplicates_output():
    mlp = AdvancedMLP(4, 6, 4, device=CPU)
    x = torch.randn(5, 4)

    single = mlp.forward(x)
    doubled = mlp.forward_unzip(x)        
    assert doubled.shape == (5, 8)
    assert torch.allclose(doubled[:, :4], single, atol=1e-5)
    assert torch.allclose(doubled[:, 4:], single, atol=1e-5)


def test_advanced_mlp_forward_unzip_without_u_with_out_returns_single():
    mlp = AdvancedMLP(4, 6, 4, device=CPU)
    x = torch.randn(5, 4)
    out = mlp.forward_unzip(x, out=0)
    assert out.shape == (5, 4)
    assert torch.allclose(out, mlp.forward(x), atol=1e-5)


def test_advanced_mlp_full_zip_pipeline():
    torch.manual_seed(0)
    a = AdvancedMLP(4, 6, 4, device=CPU)
    b = AdvancedMLP(4, 6, 4, device=CPU)

    x = torch.randn(32, 4)

    a.zip_with(b)
    assert a.zipped is True
    a.train_with([row for row in x])
    a.zip()
    assert a.zip1 is None and a.zip2 is None
    assert a.U is not None

    full = a.forward_unzip(x)
    assert full.shape == (32, 8)

    side0 = a.forward_unzip(x, out=0)
    side1 = a.forward_unzip(x, out=1)
    assert side0.shape == (32, 4)
    assert side1.shape == (32, 4)
    assert torch.allclose(torch.cat([side0, side1], dim=-1), full, atol=1e-4)


def test_build_mu_rejects_odd_width():
    """build_MU is k=2 only: a concatenation of two experts has an even hidden
    width, so an odd activation width is a contract violation (assert)."""
    zipper = Zipper(3)
    torch.manual_seed(0)
    zipper(torch.randn(16, 3))
    with pytest.raises(AssertionError):
        zipper.build_MU()


def test_refit_unmerge_drops_down_and_installs_solved_u():
    """``refit_unmerge`` is the public LS seam: it nulls ``down`` and installs a
    least-squares ``U`` so ``forward_unzip`` reconstructs the given targets."""
    torch.manual_seed(0)
    mlp = AdvancedMLP(4, 6, 8, device=CPU)
    x = torch.randn(64, 4)
    targets = torch.randn(64, 8)

    returned = mlp.refit_unmerge(x, targets)
    assert returned is mlp
    assert mlp.down is None
    assert mlp.U is not None
    # U solves || forward(x) @ U^T - targets ||; equals the explicit solve.
    expected_u = solve_linear(mlp.forward(x).float(), targets.float())
    assert torch.allclose(mlp.U, expected_u, atol=1e-6)
    assert mlp.forward_unzip(x).shape == (64, 8)


# --------------------------------------------------------------------------- #
# dtype safety (CR#2): the merge path must not silently upcast bf16 to float32.
# --------------------------------------------------------------------------- #
def test_unite_matrix_preserves_dtype():
    """Block-diagonal padding inherits the operand dtype -- default-float32 zeros
    would promote bf16 weights to float32 through ``torch.cat``."""
    a = torch.ones(2, 3, dtype=torch.bfloat16)
    b = torch.ones(4, 5, dtype=torch.bfloat16)
    assert unite_matrix(a, b).dtype == torch.bfloat16


def test_build_mu_preserves_activation_dtype():
    """M and U inherit the activation dtype (bf16), not default float32."""
    zipper = Zipper(4)
    torch.manual_seed(0)
    zipper(torch.randn(16, 4, dtype=torch.bfloat16))
    zipper.build_MU()
    assert zipper.M.dtype == torch.bfloat16
    assert zipper.U.dtype == torch.bfloat16


def test_advanced_mlp_zip_pipeline_preserves_bf16():
    """CR#2 regression guard: merging two bf16 experts must keep every zipped
    parameter (gate/up/down and U) in bf16. The padding, M and the identity block
    all default to float32 unless pinned to the weight dtype, which ``torch.cat``
    would then propagate -- doubling the artifact and running inference in fp32."""
    torch.manual_seed(0)
    a = AdvancedMLP(4, 6, 4, device=CPU).to(torch.bfloat16)
    b = AdvancedMLP(4, 6, 4, device=CPU).to(torch.bfloat16)
    x = torch.randn(32, 4, dtype=torch.bfloat16)

    a.zip_with(b)
    assert a.gate.weight.dtype == torch.bfloat16
    a.train_with([row for row in x])
    a.zip()

    for name, param in a.named_parameters():
        assert param.dtype == torch.bfloat16, f"{name} upcast to {param.dtype}"
    assert a.U.dtype == torch.bfloat16
    assert a.forward_unzip(x).dtype == torch.bfloat16
