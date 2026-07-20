"""Offline calibration-signal computers for the REAP / HC-SMoE baselines (step 4).

Fake, CPU-only tests on tiny synthetic artifacts laid out exactly as stage 1
writes them (SavedExpert experts, pickled gate, mlp_inputs, raw top-k selections,
CoActivationMatrix). Each signal is checked against a hand-rolled reference
recomputed from the same on-disk modules.
"""

import os

import pytest
import torch
from torch import nn

from common import layout
from common.baselines import signals
from common.artifacts import CoActivationMatrix, SavedExpert
from common.mlp import AdvancedMLP

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# On-disk fixtures (stage-1 layout).
# --------------------------------------------------------------------------- #
def _save_experts(data_dir, layer, n_experts, hidden=4, inter=6):
    os.makedirs(layout.experts_dir(data_dir, layer), exist_ok=True)
    for i in range(n_experts):
        torch.manual_seed(100 + i)
        SavedExpert(AdvancedMLP(hidden, inter, hidden, device=CPU)).save(
            layout.expert_path(data_dir, layer, i)
        )


def _save_gate(data_dir, layer, hidden, n_experts, seed=7):
    torch.manual_seed(seed)
    os.makedirs(layout.experts_dir(data_dir, layer), exist_ok=True)
    torch.save(nn.Linear(hidden, n_experts, bias=False), layout.gate_path(data_dir, layer))


def _save_mlp_inputs(data_dir, layer, inputs, split=layout.TRAIN_SPLIT):
    os.makedirs(layout.mlp_inputs_dir(data_dir), exist_ok=True)
    torch.save(inputs, layout.mlp_inputs_path(data_dir, split, layer))


def _save_topk(data_dir, layer, selections, split=layout.TRAIN_SPLIT):
    os.makedirs(layout.activations_dir(data_dir), exist_ok=True)
    torch.save(selections, layout.activations_all_path(data_dir, split, layer))


def _coactivation_from_topk(rows, n_experts):
    """Rebuild the stage-1 co-activation matrix: every ordered pair in a row (incl. e==e)."""
    matrix = torch.zeros(n_experts, n_experts)
    for row in rows:
        for e1 in row:
            for e2 in row:
                matrix[e1, e2] += 1
    return matrix


def _save_coactivation(data_dir, layer, matrix, split=layout.TRAIN_SPLIT):
    os.makedirs(layout.activations_dir(data_dir), exist_ok=True)
    CoActivationMatrix(matrix, split=split, sample_size=int(matrix.sum())).save(
        layout.activation_matrix_path(data_dir, split, layer)
    )


def _load_saved_expert(data_dir, layer, e):
    return SavedExpert.load(layout.expert_path(data_dir, layer, e)).mlp


def _load_saved_gate(data_dir, layer):
    return torch.load(layout.gate_path(data_dir, layer), weights_only=False)


def _routed_mask(rows, n_experts):
    mask = torch.zeros(len(rows), n_experts, dtype=torch.bool)
    for t, row in enumerate(rows):
        mask[t, torch.as_tensor(row)] = True
    return mask


# --------------------------------------------------------------------------- #
# (c) selection frequencies == CoActivationMatrix diagonal.
# --------------------------------------------------------------------------- #
def test_selection_frequencies_match_coactivation_diagonal(tmp_path):
    n_experts = 5
    rows = [
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([1, 2]),
        torch.tensor([0, 3]),
        torch.tensor([2, 3]),
    ]  # expert 4 is never selected
    _save_topk(str(tmp_path), 0, rows)
    _save_coactivation(str(tmp_path), 0, _coactivation_from_topk(rows, n_experts))

    freq = signals.compute_selection_frequencies(0, str(tmp_path), n_experts)

    assert freq.dtype == torch.long
    assert freq.tolist() == [3, 2, 3, 2, 0]
    diag = CoActivationMatrix.load(
        layout.activation_matrix_path(str(tmp_path), layout.TRAIN_SPLIT, 0)
    ).matrix.diagonal().long()
    assert torch.equal(freq, diag)


def test_selection_frequencies_missing_topk_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        signals.compute_selection_frequencies(0, str(tmp_path), 3)


# --------------------------------------------------------------------------- #
# (b) characteristic activation = dense per-expert mean output.
# --------------------------------------------------------------------------- #
def test_characteristic_activation_is_dense_mean(tmp_path):
    n_experts, hidden = 3, 4
    _save_experts(str(tmp_path), 0, n_experts, hidden=hidden)
    inputs = torch.randn(12, hidden)
    _save_mlp_inputs(str(tmp_path), 0, inputs)

    ca = signals.compute_characteristic_activation(
        0, str(tmp_path), n_experts, CPU, num_samples=100, batch_size=5
    )

    assert ca.shape == (n_experts, hidden)
    assert ca.dtype == torch.float32
    for e in range(n_experts):
        with torch.no_grad():
            expected = _load_saved_expert(str(tmp_path), 0, e)(inputs).float().mean(dim=0)
        assert torch.allclose(ca[e], expected, atol=1e-5)


def test_characteristic_activation_subsample_keeps_shape(tmp_path):
    n_experts, hidden = 2, 4
    _save_experts(str(tmp_path), 0, n_experts, hidden=hidden)
    _save_mlp_inputs(str(tmp_path), 0, torch.randn(20, hidden))

    ca = signals.compute_characteristic_activation(
        0, str(tmp_path), n_experts, CPU, num_samples=5, batch_size=2,
        generator=torch.Generator().manual_seed(1),
    )
    assert ca.shape == (n_experts, hidden)


def test_characteristic_activation_missing_inputs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        signals.compute_characteristic_activation(0, str(tmp_path), 3, CPU)


# --------------------------------------------------------------------------- #
# (a) REAP saliency: masked mean of gate*||output|| products.
# --------------------------------------------------------------------------- #
def test_reap_saliency_matches_masked_reference(tmp_path):
    n_experts, hidden = 4, 4
    _save_experts(str(tmp_path), 0, n_experts, hidden=hidden)
    _save_gate(str(tmp_path), 0, hidden, n_experts)
    inputs = torch.randn(10, hidden)
    _save_mlp_inputs(str(tmp_path), 0, inputs)
    rows = [
        torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([1, 2]),
        torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 2]),
        torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 2]),
        torch.tensor([1, 2]),
    ]  # expert 3 is never routed
    _save_topk(str(tmp_path), 0, rows)

    sal = signals.compute_reap_saliency(
        0, str(tmp_path), n_experts, CPU, num_samples=100, batch_size=4
    )

    assert sal.shape == (n_experts,)
    assert sal.dtype == torch.float32
    with torch.no_grad():
        probs = _load_saved_gate(str(tmp_path), 0)(inputs).softmax(dim=-1, dtype=torch.float)
    routed = _routed_mask(rows, n_experts)
    for e in range(n_experts):
        with torch.no_grad():
            norms = _load_saved_expert(str(tmp_path), 0, e)(inputs).float().norm(dim=-1)
        mask = routed[:, e]
        expected = (
            (norms[mask].double() * probs[mask, e].double()).mean()
            if bool(mask.any())
            else torch.zeros((), dtype=torch.float64)
        )
        assert sal[e].item() == pytest.approx(float(expected), abs=1e-5)
    assert sal[3].item() == 0.0


def test_reap_saliency_never_routed_is_zero(tmp_path):
    n_experts, hidden = 3, 4
    _save_experts(str(tmp_path), 0, n_experts, hidden=hidden)
    _save_gate(str(tmp_path), 0, hidden, n_experts)
    _save_mlp_inputs(str(tmp_path), 0, torch.randn(6, hidden))
    # every token routes to experts {0, 1}; expert 2 is never selected.
    _save_topk(str(tmp_path), 0, [torch.tensor([0, 1]) for _ in range(6)])

    sal = signals.compute_reap_saliency(
        0, str(tmp_path), n_experts, CPU, num_samples=100, batch_size=3
    )

    assert sal[2].item() == 0.0
    assert sal[0].item() > 0.0 and sal[1].item() > 0.0


def test_reap_saliency_is_mean_of_products_not_product_of_means(tmp_path):
    # Data crafted so per-token norm and gate-prob co-vary: the mean of products
    # then differs from the product of means -- a test that fails if aggregation
    # is done in the wrong order.
    n_experts, hidden = 2, 4
    _save_experts(str(tmp_path), 0, n_experts, hidden=hidden)
    _save_gate(str(tmp_path), 0, hidden, n_experts)
    inputs = torch.tensor(
        [[3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1],
         [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
    )
    _save_mlp_inputs(str(tmp_path), 0, inputs)
    _save_topk(str(tmp_path), 0, [torch.tensor([0, 1]) for _ in range(inputs.shape[0])])

    sal = signals.compute_reap_saliency(
        0, str(tmp_path), n_experts, CPU, num_samples=100, batch_size=2
    )

    e = 0
    with torch.no_grad():
        norms = _load_saved_expert(str(tmp_path), 0, e)(inputs).float().norm(dim=-1).double()
        prob_e = _load_saved_gate(str(tmp_path), 0)(inputs).softmax(dim=-1, dtype=torch.float)[:, e].double()
    mean_of_products = (norms * prob_e).mean()
    product_of_means = norms.mean() * prob_e.mean()

    assert abs(float(mean_of_products) - float(product_of_means)) > 1e-4
    assert sal[e].item() == pytest.approx(float(mean_of_products), abs=1e-5)
    assert sal[e].item() != pytest.approx(float(product_of_means), abs=1e-5)


def test_reap_saliency_subsample_keeps_topk_aligned(tmp_path):
    # Under subsampling, inputs and top-k selections must be indexed by the SAME
    # permutation; a misaligned mask would score the wrong tokens per expert.
    n_experts, hidden, n = 3, 4, 40
    _save_experts(str(tmp_path), 0, n_experts, hidden=hidden)
    _save_gate(str(tmp_path), 0, hidden, n_experts)
    torch.manual_seed(0)
    inputs = torch.randn(n, hidden)
    _save_mlp_inputs(str(tmp_path), 0, inputs)
    rows = [torch.tensor([t % n_experts, (t + 1) % n_experts]) for t in range(n)]
    _save_topk(str(tmp_path), 0, rows)

    sal = signals.compute_reap_saliency(
        0, str(tmp_path), n_experts, CPU, num_samples=16, batch_size=8,
        generator=torch.Generator().manual_seed(123),
    )

    # Reproduce the exact subsample the function drew and align selections to it.
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(123))[:16]
    sub_inputs = inputs[idx]
    sub_rows = [rows[i] for i in idx.tolist()]
    routed = _routed_mask(sub_rows, n_experts)
    with torch.no_grad():
        probs = _load_saved_gate(str(tmp_path), 0)(sub_inputs).softmax(dim=-1, dtype=torch.float)
    for e in range(n_experts):
        with torch.no_grad():
            norms = _load_saved_expert(str(tmp_path), 0, e)(sub_inputs).float().norm(dim=-1)
        mask = routed[:, e]
        expected = (
            (norms[mask].double() * probs[mask, e].double()).mean()
            if bool(mask.any())
            else torch.zeros((), dtype=torch.float64)
        )
        assert sal[e].item() == pytest.approx(float(expected), abs=1e-5)


def test_reap_saliency_rejects_misaligned_topk(tmp_path):
    n_experts, hidden = 3, 4
    _save_experts(str(tmp_path), 0, n_experts, hidden=hidden)
    _save_gate(str(tmp_path), 0, hidden, n_experts)
    _save_mlp_inputs(str(tmp_path), 0, torch.randn(6, hidden))
    _save_topk(str(tmp_path), 0, [torch.tensor([0, 1]) for _ in range(5)])  # 5 != 6

    with pytest.raises(ValueError):
        signals.compute_reap_saliency(0, str(tmp_path), n_experts, CPU, num_samples=100)


def test_reap_saliency_missing_inputs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        signals.compute_reap_saliency(0, str(tmp_path), 3, CPU)


def test_reap_saliency_missing_gate_raises(tmp_path):
    _save_experts(str(tmp_path), 0, 3)
    _save_mlp_inputs(str(tmp_path), 0, torch.randn(6, 4))
    _save_topk(str(tmp_path), 0, [torch.tensor([0, 1]) for _ in range(6)])
    with pytest.raises(FileNotFoundError):
        signals.compute_reap_saliency(0, str(tmp_path), 3, CPU)
