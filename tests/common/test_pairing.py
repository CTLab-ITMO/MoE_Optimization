
import os
from functools import lru_cache

import pytest
import torch

from common import pairing
from common.mlp import AdvancedMLP

CPU = torch.device("cpu")


@pytest.fixture
def allow_pickled_load(monkeypatch):
    original = torch.load

    def loader(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", loader)


def _save_activation_matrix(data_dir, layer, matrix):
    act_dir = os.path.join(data_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)
    torch.save(matrix, os.path.join(act_dir, f"train_matrix_{layer}.pt"))


def _save_experts(data_dir, layer, n_experts, hidden=4, inter=6):
    experts_dir = os.path.join(data_dir, "experts", str(layer))
    os.makedirs(experts_dir, exist_ok=True)
    for i in range(n_experts):
        torch.manual_seed(i)
        expert = AdvancedMLP(hidden, inter, hidden, device=CPU)
        torch.save(expert, os.path.join(experts_dir, f"expert_{i}.pt"))


def _save_inputs(data_dir, layer, inputs):
    inputs_dir = os.path.join(data_dir, "mlp_inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    torch.save(inputs, os.path.join(inputs_dir, f"train_{layer}.pt"))


def test_random_pairs_covers_every_expert():
    pairs = pairing.random_pairs(8)
    assert len(pairs) == 4
    flat = [i for pair in pairs for i in pair]
    assert sorted(flat) == list(range(8))


def test_build_matches_returns_disjoint_pairs():
    n = 6
    corr = torch.rand(n, n)
    corr = corr + corr.t() 

    matches = pairing.build_matches(corr)

    assert len(matches) == n // 2
    seen = [idx for pair in matches for idx in pair]
    assert sorted(seen) == list(range(n))


def test_build_matches_picks_strongest_pair():
    corr = torch.full((4, 4), 0.01)
    corr[1, 3] = corr[3, 1] = 10.0
    corr.fill_diagonal_(0.0)

    matches = pairing.build_matches(corr)
    matches = [tuple(sorted(x)) for x in matches]
    assert (1, 3) in matches


def test_build_matches_ignores_diagonal():
    corr = torch.eye(4) * 100.0
    matches = pairing.build_matches(corr)
    for i, j in matches:
        assert i != j


def test_activation_correlation_matrix_is_normalized(tmp_path):
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _save_activation_matrix(str(tmp_path), 0, matrix)
    out = pairing.compute_activation_correlation_matrix(0, str(tmp_path))
    assert out.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert out.dtype == torch.float32


def test_output_similarity_matrix_happy_path(tmp_path, allow_pickled_load):
    n_experts = 4
    _save_experts(str(tmp_path), 0, n_experts)
    _save_inputs(str(tmp_path), 0, torch.randn(20, 4))

    sim = pairing.compute_output_similarity_matrix(
        0, str(tmp_path), n_experts, CPU, num_samples=10, batch_size=8
    )
    assert sim.shape == (n_experts, n_experts)
    assert torch.allclose(torch.diagonal(sim), torch.ones(n_experts), atol=1e-3)


def test_output_similarity_matrix_loads_safetensors_experts(tmp_path):
    # The new module writes experts as SavedExpert (safetensors); the similarity
    # signal must load them via SavedExpert, not the legacy pickle path (no
    # allow_pickled_load fixture here).
    from common import layout
    from common.artifacts import SavedExpert

    n_experts = 4
    os.makedirs(layout.experts_dir(str(tmp_path), 0), exist_ok=True)
    for i in range(n_experts):
        torch.manual_seed(i)
        SavedExpert(AdvancedMLP(4, 6, 4, device=CPU)).save(layout.expert_path(str(tmp_path), 0, i))
    _save_inputs(str(tmp_path), 0, torch.randn(20, 4))

    sim = pairing.compute_output_similarity_matrix(
        0, str(tmp_path), n_experts, CPU, num_samples=10, batch_size=8
    )
    assert sim.shape == (n_experts, n_experts)
    assert torch.allclose(torch.diagonal(sim), torch.ones(n_experts), atol=1e-3)


def test_output_similarity_matrix_accepts_list_and_3d_inputs(tmp_path, allow_pickled_load):
    n_experts = 2
    _save_experts(str(tmp_path), 1, n_experts)
    _save_inputs(str(tmp_path), 1, [torch.randn(5, 4) for _ in range(3)])

    sim = pairing.compute_output_similarity_matrix(
        1, str(tmp_path), n_experts, CPU, num_samples=4, batch_size=2
    )
    assert sim.shape == (n_experts, n_experts)


def test_output_similarity_matrix_matches_manual_mean_cosine(tmp_path, allow_pickled_load):
    # Reference check: entry (i, j) is the mean over tokens of cosine(out_i, out_j).
    n_experts = 2
    _save_experts(str(tmp_path), 2, n_experts)
    inputs = torch.randn(12, 4)
    _save_inputs(str(tmp_path), 2, inputs)

    sim = pairing.compute_output_similarity_matrix(
        2, str(tmp_path), n_experts, CPU, num_samples=12, batch_size=4
    )

    experts_dir = os.path.join(str(tmp_path), "experts", "2")
    expert_0 = torch.load(os.path.join(experts_dir, "expert_0.pt"), weights_only=False)
    expert_1 = torch.load(os.path.join(experts_dir, "expert_1.pt"), weights_only=False)
    with torch.no_grad():
        out_0 = torch.nn.functional.normalize(expert_0(inputs).float(), dim=-1)
        out_1 = torch.nn.functional.normalize(expert_1(inputs).float(), dim=-1)
    expected = (out_0 * out_1).sum(dim=-1).mean().item()

    assert sim[0, 1].item() == pytest.approx(expected, abs=1e-4)
    assert sim[1, 0].item() == pytest.approx(expected, abs=1e-4)


def test_output_similarity_matrix_missing_experts_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        pairing.compute_output_similarity_matrix(0, str(tmp_path), 4, CPU)


def test_output_similarity_matrix_missing_inputs(tmp_path):
    _save_experts(str(tmp_path), 0, 4)
    with pytest.raises(FileNotFoundError):
        pairing.compute_output_similarity_matrix(0, str(tmp_path), 4, CPU)


def test_get_pairs_random():
    pairs = pairing.get_pairs("random", layer=0, n_experts=6, data_dir="unused")
    assert len(pairs) == 3


def test_get_pairs_activation_correlation_greedy(tmp_path):
    _save_activation_matrix(str(tmp_path), 0, torch.rand(4, 4) + torch.eye(4))
    pairs = pairing.get_pairs(
        "activation_correlation", layer=0, n_experts=4, data_dir=str(tmp_path)
    )
    assert len(pairs) == 2


def test_get_pairs_activation_correlation_with_evolution(tmp_path):
    sim = torch.rand(4, 4)
    _save_activation_matrix(str(tmp_path), 0, sim + sim.t())
    pairs = pairing.get_pairs(
        "activation_correlation", layer=0, n_experts=4, data_dir=str(tmp_path),
        evolution_steps=2,
    )
    assert len(pairs) == 2


def test_get_pairs_output_similarity_defaults_device(tmp_path, allow_pickled_load):
    _save_experts(str(tmp_path), 0, 4)
    _save_inputs(str(tmp_path), 0, torch.randn(12, 4))
    pairs = pairing.get_pairs(
        "output_similarity", layer=0, n_experts=4, data_dir=str(tmp_path), device=None
    )
    assert len(pairs) == 2


def test_get_pairs_unknown_method_raises():
    with pytest.raises(ValueError):
        pairing.get_pairs("nope", layer=0, n_experts=4, data_dir="unused")


def test_get_pairs_random_uses_cpu_generator_under_cuda_ctx():
    # The random split's torch.randperm is a CPU op: even when ctx.device is cuda
    # the generator must be CPU, else randperm(..., generator=cuda_gen) raises.
    from common.context import StageContext

    ctx = StageContext(seed=1234, device=torch.device("cuda"), n_experts=8).for_layer(0)
    pairs = pairing.get_pairs("random", layer=0, n_experts=8, data_dir="unused", ctx=ctx)
    assert len(pairs) == 4


def test_get_pairs_forwards_output_similarity_knobs(monkeypatch):
    # §4.5: num_samples/batch_size are exposed at the public entry (get_pairs)
    # and forwarded to compute_output_similarity_matrix, not pinned to defaults.
    captured = {}

    def fake_sim(layer, data_dir, n_experts, device, num_samples, batch_size, generator=None):
        captured["num_samples"] = num_samples
        captured["batch_size"] = batch_size
        return torch.eye(n_experts)

    monkeypatch.setattr(pairing, "compute_output_similarity_matrix", fake_sim)
    pairing.get_pairs(
        "output_similarity", layer=0, n_experts=4, data_dir="unused",
        device=CPU, num_samples=123, batch_size=45,
    )
    assert captured == {"num_samples": 123, "batch_size": 45}


def _matrix_from_edges(n, edges):
    M = torch.zeros(n, n)
    for i, j, w in edges:
        M[i, j] = float(w)
        M[j, i] = float(w)
    return M


def _total_weight(corr, matches):
    return sum(float(corr[i, j]) for i, j in matches)


def _assert_valid_matching(matches, n):
    seen = set()
    for i, j in matches:
        assert 0 <= i < n and 0 <= j < n
        assert i != j
        assert i not in seen and j not in seen
        seen.add(i)
        seen.add(j)


def _max_weight_over_matchings(corr):
    n = corr.shape[0]
    M = [[float(corr[i, j]) for j in range(n)] for i in range(n)]

    @lru_cache(maxsize=None)
    def rec(nodes):
        if not nodes:
            return 0.0
        first, rest = nodes[0], nodes[1:]
        best = rec(rest)  
        for k in range(len(rest)):
            partner = rest[k]
            remaining = rest[:k] + rest[k + 1 :]
            w = M[first][partner] + rec(remaining)
            if w > best:
                best = w
        return best

    return rec(tuple(range(n)))


def test_build_matches_empty_matrix_returns_empty():
    assert pairing.build_matches(torch.zeros(0, 0)) == []


def test_build_matches_single_expert_has_no_pairs():
    assert pairing.build_matches(torch.zeros(1, 1)) == []


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [6, 8, 10])
def test_build_matches_random_is_optimal_and_perfect(n, seed):
    torch.manual_seed(seed)
    a = torch.rand(n, n) + 0.5
    corr = a + a.t()
    corr.fill_diagonal_(0.0)

    matches = pairing.build_matches(corr)

    _assert_valid_matching(matches, n)
    assert len(matches) == n // 2
    assert _total_weight(corr, matches) == pytest.approx(
        _max_weight_over_matchings(corr), abs=1e-4
    )


BLOSSOM_EDGE_CASES = [
    (6, [(0, 1, 8), (0, 2, 9), (1, 2, 10), (2, 3, 7), (0, 5, 5), (3, 4, 6)]),
    (6, [(0, 1, 9), (0, 2, 8), (1, 2, 10), (0, 3, 5), (3, 4, 4), (0, 5, 3)]),
    (6, [(0, 1, 9), (0, 2, 9), (1, 2, 10), (1, 3, 8), (2, 4, 8), (3, 4, 10), (4, 5, 6)]),
    (8, [(0, 1, 10), (0, 6, 10), (1, 2, 12), (2, 3, 20), (2, 4, 20),
         (3, 4, 25), (4, 5, 10), (5, 6, 10), (6, 7, 8)]),
    (8, [(0, 1, 8), (0, 2, 8), (1, 2, 10), (1, 3, 12), (2, 4, 12),
         (3, 4, 14), (3, 5, 12), (4, 6, 12), (5, 6, 14), (6, 7, 12)]),
    (8, [(0, 1, 23), (0, 4, 22), (0, 5, 15), (1, 2, 25),
         (2, 3, 22), (3, 4, 25), (3, 7, 14), (4, 6, 13)]),
    (10, [(0, 1, 45), (0, 4, 45), (1, 2, 50), (2, 3, 45), (3, 4, 50),
          (0, 5, 30), (2, 8, 35), (3, 7, 35), (4, 6, 26), (8, 9, 5)]),
    (10, [(0, 1, 45), (0, 4, 45), (1, 2, 50), (2, 3, 45), (3, 4, 50),
          (0, 5, 30), (2, 8, 35), (3, 7, 26), (4, 6, 40), (8, 9, 5)]),
]


@pytest.mark.parametrize("n,edges", BLOSSOM_EDGE_CASES)
def test_build_matches_blossom_cases_are_optimal(n, edges):
    corr = _matrix_from_edges(n, edges)

    matches = pairing.build_matches(corr)

    _assert_valid_matching(matches, n)
    assert _total_weight(corr, matches) == pytest.approx(
        _max_weight_over_matchings(corr), abs=1e-4
    )


def _max_weight_perfect_matching(corr):
    n = corr.shape[0]
    M = [[float(corr[i, j]) for j in range(n)] for i in range(n)]
    NEG = float("-inf")

    @lru_cache(maxsize=None)
    def rec(nodes):
        if not nodes:
            return 0.0
        first, rest = nodes[0], nodes[1:]
        best = NEG
        for k in range(len(rest)):
            partner = rest[k]
            sub = rec(rest[:k] + rest[k + 1 :])
            if sub != NEG and M[first][partner] + sub > best:
                best = M[first][partner] + sub
        return best

    return rec(tuple(range(n)))


def test_build_matches_returns_max_cardinality_pairing():
    corr = _matrix_from_edges(4, [(0, 1, 5.0), (2, 3, -3.0), (0, 2, -1.0), (1, 3, -1.0)])

    matches = pairing.build_matches(corr)

    _assert_valid_matching(matches, 4)
    assert len(matches) == 2
    assert _total_weight(corr, matches) == pytest.approx(
        _max_weight_perfect_matching(corr), abs=1e-4
    )
