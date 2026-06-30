"""Tests for the two-axis grouping abstraction (S6, §2.3/§3.7/§4.3/§4.5/§2.4/§2.6).

Covers the S6 gate:

* the canonical :data:`common.grouping.GROUPING_REGISTRY` holds one
  :class:`Grouping` per :class:`~common.names.PairingMethod`, in the pinned order
  the golden ``pairing_methods`` snapshot guards; config validation runs off it;
* the ``PairGrouping`` k == 2 contract (every produced group is a pair);
* **pair identity**: each registered strategy reproduces the legacy
  ``common.pairing.get_pairs`` exactly -- same group order *and* tuple
  orientation -- so the default ``activation_correlation`` is behaviour-equivalent
  to the old pairing (§2.6);
* the "signal x algorithm" decomposition: each ``AffinityProvider`` /
  ``Partitioner`` component matches the primitive it wraps;
* the ``output_similarity`` knobs (§4.5) are explicit strategy parameters and are
  forwarded to the matrix computation;
* the stochastic strategies draw from the per-layer ``ctx`` generator (S4), so
  ``random`` pairing is invariant to traversal order.
"""

import pytest
import torch

from common import layout, pairing
from common.context import StageContext
from common.grouping import (
    GROUPING_REGISTRY,
    AffinityProvider,
    CoActivationAffinity,
    Grouping,
    MatchingPartitioner,
    OutputSimilarityAffinity,
    PairGrouping,
    Partitioner,
    RandomPartitioner,
)
from common.mlp import AdvancedMLP
from common.names import PairingMethod
from common.registry import require_registered
from common.seeding import set_seed

CPU = torch.device("cpu")
SEED = 1234

ACT = str(PairingMethod.ACTIVATION_CORRELATION)
RAND = str(PairingMethod.RANDOM)
OUT = str(PairingMethod.OUTPUT_SIMILARITY)


@pytest.fixture
def allow_pickled_load(monkeypatch):
    """Let ``torch.load`` read the pickled fake experts the helpers save."""
    original = torch.load

    def loader(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", loader)


def _save_activation_matrix(data_dir, layer, matrix):
    path = layout.activation_matrix_path(data_dir, layout.TRAIN_SPLIT, layer)
    import os

    os.makedirs(layout.activations_dir(data_dir), exist_ok=True)
    torch.save(matrix, path)


def _save_experts(data_dir, layer, n_experts, hidden=4, inter=6):
    import os

    os.makedirs(layout.experts_dir(data_dir, layer), exist_ok=True)
    for i in range(n_experts):
        torch.manual_seed(i)
        expert = AdvancedMLP(hidden, inter, hidden, device=CPU)
        torch.save(expert, layout.expert_path(data_dir, layer, i))


def _save_inputs(data_dir, layer, inputs):
    import os

    os.makedirs(layout.mlp_inputs_dir(data_dir), exist_ok=True)
    torch.save(inputs, layout.mlp_inputs_path(data_dir, layout.TRAIN_SPLIT, layer))


def _symmetric_matrix(n, seed=0):
    torch.manual_seed(seed)
    m = torch.rand(n, n)
    m = m + m.t()
    m.fill_diagonal_(0.0)
    return m


def _ctx(data_dir=None, n_experts=None, layer=0, seed=SEED):
    return StageContext(
        seed=seed, device=CPU, data_dir=data_dir, n_experts=n_experts
    ).for_layer(layer)


# --------------------------------------------------------------------------- #
# Registry: pinned order, Grouping instances, config validation.
# --------------------------------------------------------------------------- #
def test_registry_names_match_pairing_methods_in_order():
    assert GROUPING_REGISTRY.names() == [str(method) for method in PairingMethod]


def test_registry_entries_are_pair_groupings_named_for_their_key():
    for name in GROUPING_REGISTRY.names():
        strategy = GROUPING_REGISTRY[name]
        assert isinstance(strategy, Grouping)
        assert isinstance(strategy, PairGrouping)
        assert strategy.name == name


def test_require_registered_drives_pairing_method_validation():
    require_registered(ACT, GROUPING_REGISTRY, field="PAIRING_METHOD")  # no raise
    with pytest.raises(ValueError, match="PAIRING_METHOD"):
        require_registered("nope", GROUPING_REGISTRY, field="PAIRING_METHOD")


# --------------------------------------------------------------------------- #
# PairGrouping k == 2 contract.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [ACT, RAND, OUT])
def test_pairgrouping_yields_only_pairs(name, tmp_path, allow_pickled_load):
    data_dir = str(tmp_path)
    n = 4
    _save_activation_matrix(data_dir, 0, _symmetric_matrix(n))
    _save_experts(data_dir, 0, n)
    _save_inputs(data_dir, 0, torch.randn(20, 4))

    partition = GROUPING_REGISTRY[name].partition(0, _ctx(data_dir, n))

    assert partition.num_groups == n // 2
    assert all(group.size == 2 for group in partition.groups)
    assert partition.method == name


# --------------------------------------------------------------------------- #
# Pair identity: registered strategy == legacy get_pairs (order + orientation).
# --------------------------------------------------------------------------- #
def test_activation_correlation_pairing_identical_to_get_pairs(tmp_path):
    data_dir = str(tmp_path)
    n = 6
    _save_activation_matrix(data_dir, 0, _symmetric_matrix(n))

    partition = GROUPING_REGISTRY[ACT].partition(0, _ctx(data_dir, n))
    expected = pairing.get_pairs(ACT, layer=0, n_experts=n, data_dir=data_dir)

    assert partition.to_pairs() == expected
    # Group order (i.pt index) and member orientation (out=0/1) preserved.
    assert [group.members for group in partition.groups] == [tuple(p) for p in expected]


def test_random_pairing_identical_to_get_pairs_with_same_ctx():
    n = 8
    ctx = _ctx(n_experts=n, layer=2)

    partition = GROUPING_REGISTRY[RAND].partition(2, ctx)
    expected = pairing.get_pairs(RAND, layer=2, n_experts=n, data_dir="unused", ctx=ctx)

    assert partition.to_pairs() == expected


def test_output_similarity_pairing_identical_to_get_pairs_under_subsample(
    tmp_path, allow_pickled_load
):
    # Force the generator-driven subsample path (inputs > num_samples) so the
    # identity proves the strategy and get_pairs draw from the *same* ctx
    # generator, not just that both skip sampling.
    data_dir = str(tmp_path)
    n = 4
    _save_experts(data_dir, 0, n)
    _save_inputs(data_dir, 0, torch.randn(40, 4))
    ctx = _ctx(data_dir, n)

    strategy = PairGrouping(OUT, OutputSimilarityAffinity(num_samples=8, batch_size=4), MatchingPartitioner())
    partition = strategy.partition(0, ctx)
    expected = pairing.get_pairs(
        OUT, layer=0, n_experts=n, data_dir=data_dir, device=CPU,
        num_samples=8, batch_size=4, ctx=ctx,
    )

    assert partition.to_pairs() == expected


def test_registered_output_similarity_strategy_produces_pairs(tmp_path, allow_pickled_load):
    data_dir = str(tmp_path)
    n = 4
    _save_experts(data_dir, 0, n)
    _save_inputs(data_dir, 0, torch.randn(20, 4))

    partition = GROUPING_REGISTRY[OUT].partition(0, _ctx(data_dir, n))

    assert partition.num_groups == n // 2
    assert partition.method == OUT


# --------------------------------------------------------------------------- #
# Two-axis decomposition: components match the primitives they wrap.
# --------------------------------------------------------------------------- #
def test_coactivation_affinity_matches_pairing_helper(tmp_path):
    data_dir = str(tmp_path)
    _save_activation_matrix(data_dir, 0, _symmetric_matrix(5))

    matrix = CoActivationAffinity().affinity(0, None, _ctx(data_dir, 5))
    expected = pairing.compute_activation_correlation_matrix(0, data_dir)

    assert torch.equal(matrix, expected)


def test_output_similarity_affinity_matches_pairing_helper(tmp_path, allow_pickled_load):
    data_dir = str(tmp_path)
    n = 3
    _save_experts(data_dir, 0, n)
    _save_inputs(data_dir, 0, torch.randn(16, 4))
    ctx = _ctx(data_dir, n)

    matrix = OutputSimilarityAffinity(num_samples=8, batch_size=4).affinity(0, n, ctx)
    expected = pairing.compute_output_similarity_matrix(
        0, data_dir, n, CPU, 8, 4,
        ctx.generator("pairing", method=PairingMethod.OUTPUT_SIMILARITY),
    )

    assert torch.equal(matrix, expected)


def test_matching_partitioner_matches_build_matches():
    matrix = _symmetric_matrix(8, seed=3)
    assert MatchingPartitioner().partition(matrix, None, None) == pairing.build_matches(matrix)


def test_random_partitioner_matches_random_pairs_with_same_ctx():
    n = 10
    ctx = _ctx(n_experts=n, layer=1)
    produced = RandomPartitioner().partition(None, n, ctx)
    expected = pairing.random_pairs(
        n, generator=ctx.generator("pairing", method=PairingMethod.RANDOM)
    )
    assert produced == expected


# --------------------------------------------------------------------------- #
# §4.5: output_similarity knobs are explicit strategy params, forwarded down.
# --------------------------------------------------------------------------- #
def test_output_similarity_affinity_forwards_knobs(monkeypatch):
    captured = {}

    def fake(layer, data_dir, n_experts, device, num_samples, batch_size, generator=None):
        captured["num_samples"] = num_samples
        captured["batch_size"] = batch_size
        return torch.eye(n_experts)

    monkeypatch.setattr(pairing, "compute_output_similarity_matrix", fake)
    OutputSimilarityAffinity(num_samples=123, batch_size=45).affinity(
        0, 4, _ctx("unused", 4)
    )
    assert captured == {"num_samples": 123, "batch_size": 45}


# --------------------------------------------------------------------------- #
# Order-invariance of the stochastic strategy (S4 carried through grouping).
# --------------------------------------------------------------------------- #
def _random_run(order, *, use_ctx):
    n = 8
    base = StageContext(seed=SEED, device=CPU, n_experts=n)
    strategy = GROUPING_REGISTRY[RAND]
    set_seed(SEED)
    out = {}
    for layer in order:
        ctx = base.for_layer(layer) if use_ctx else StageContext(n_experts=n).for_layer(layer)
        out[layer] = strategy.partition(layer, ctx).to_pairs()
    return out


def test_random_pairgrouping_is_order_invariant_with_ctx():
    layers = [0, 1, 2, 3]
    forward = _random_run(layers, use_ctx=True)
    reverse = _random_run(list(reversed(layers)), use_ctx=True)
    for layer in layers:
        assert forward[layer] == reverse[layer]


def test_random_pairgrouping_global_rng_is_order_dependent():
    """Adversarial: without ctx the global-RNG path depends on order, so the
    invariance above is a genuine property of the local seeds."""
    layers = [0, 1, 2, 3]
    forward = _random_run(layers, use_ctx=False)
    reverse = _random_run(list(reversed(layers)), use_ctx=False)
    assert any(forward[layer] != reverse[layer] for layer in layers)


# --------------------------------------------------------------------------- #
# GPU safety: the CPU randperm draws use a CPU generator even under a cuda ctx.
# --------------------------------------------------------------------------- #
def test_random_partitioner_uses_cpu_generator_under_cuda_ctx():
    # The random split's torch.randperm is a CPU op, so the generator must be CPU
    # even when ctx.device is cuda -- otherwise randperm(..., generator=cuda_gen)
    # raises on the GPU box. Reproduces the failure mode on any host (a cuda
    # Generator errors at build without a GPU, at the CPU randperm with one).
    cuda_ctx = StageContext(seed=SEED, device=torch.device("cuda"), n_experts=8).for_layer(0)
    pairs = RandomPartitioner().partition(None, 8, cuda_ctx)
    assert len(pairs) == 4


def test_output_similarity_affinity_draws_on_cpu_under_cuda_ctx(monkeypatch):
    # The subsample randperm runs before inputs.to(device), so its generator must
    # be CPU regardless of ctx.device. Capture the generator handed down and check
    # its device (avoids loading experts onto a real cuda device).
    captured = {}

    def fake(layer, data_dir, n_experts, device, num_samples, batch_size, generator=None):
        captured["gen_device"] = None if generator is None else generator.device
        return torch.eye(n_experts)

    monkeypatch.setattr(pairing, "compute_output_similarity_matrix", fake)
    cuda_ctx = StageContext(
        seed=SEED, device=torch.device("cuda"), n_experts=4, data_dir="unused"
    ).for_layer(0)
    OutputSimilarityAffinity(num_samples=8, batch_size=4).affinity(0, 4, cuda_ctx)

    assert captured["gen_device"] is not None and captured["gen_device"].type == "cpu"


# --------------------------------------------------------------------------- #
# The grouping interface is strict: explicit params, so a typo'd kwarg is caught.
# --------------------------------------------------------------------------- #
def test_concrete_grouping_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        GROUPING_REGISTRY[RAND].partition(0, _ctx(n_experts=8), bogus=1)


def test_concrete_partitioner_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        MatchingPartitioner().partition(torch.eye(4), 4, None, bogus=1)


def test_concrete_affinity_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        CoActivationAffinity().affinity(0, None, None, bogus=1)


# --------------------------------------------------------------------------- #
# Errors: clear failures for missing dynamic state; abstract bases.
# --------------------------------------------------------------------------- #
def test_missing_data_dir_raises_clear_error():
    with pytest.raises(ValueError, match="data_dir"):
        GROUPING_REGISTRY[ACT].partition(0, StageContext(n_experts=4).for_layer(0))


def test_matching_partitioner_requires_affinity():
    with pytest.raises(ValueError, match="affinity"):
        MatchingPartitioner().partition(None, 4, None)


def test_random_partitioner_requires_n_experts():
    with pytest.raises(ValueError, match="n_experts"):
        RandomPartitioner().partition(None, None, None)


def test_output_similarity_affinity_requires_n_experts(tmp_path):
    with pytest.raises(ValueError, match="n_experts"):
        OutputSimilarityAffinity().affinity(0, None, _ctx(str(tmp_path), None))


@pytest.mark.parametrize("cls", [Grouping, AffinityProvider, Partitioner])
def test_abstract_bases_cannot_be_instantiated(cls):
    with pytest.raises(TypeError):
        cls()
