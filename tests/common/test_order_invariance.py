"""Order-invariance gate (S4, §1.7/§3.3).

The latent bug: ``set_seed`` once per stage plus random draws inside nested
loops makes a stochastic op's result depend on how many draws happened before
it -- i.e. on the *traversal order* of the pipeline. The fix is a per-unit
:func:`common.seeding.seed_for` generator handed out by :class:`StageContext`.

Each property is proven in both directions:

* with a context (``seed_for`` generators), permuting the traversal order leaves
  every unit's result unchanged;
* without one (the legacy global RNG), permuting the order *changes* results --
  so the invariance above is a real property of the local seeds, not something
  that would hold anyway.

Covered stochastic points: Fisher's Gaussian ``eps`` (fusion) and ``random``
pairing (``torch.randperm``).
"""

import torch

from common.context import StageContext
from common.fusion import FisherFusion
from common.mlp import AdvancedMLP
from common.pairing import get_pairs
from common.seeding import set_seed

CPU = torch.device("cpu")
H = 4
INTER = 6
SEED = 1234
UNITS = [(0, 0), (0, 1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------- #
# Fisher (Gaussian eps).
# --------------------------------------------------------------------------- #
def _make_fisher_units():
    """Fixed experts + calibration data for each (layer, group) unit."""
    set_seed(SEED)
    experts = {}
    data = {}
    for unit in UNITS:
        experts[unit] = (
            AdvancedMLP(H, INTER, H, device=CPU),
            AdvancedMLP(H, INTER, H, device=CPU),
        )
        data[unit] = torch.randn(8, H)
    return experts, data


def _fisher_run(experts, data, order, *, use_ctx):
    """Merge each unit's pair with Fisher in ``order``; return per-unit weights.

    ``set_seed`` at the top mimics the stage-start global seeding the legacy path
    relies on; ``create_new=True`` (default) means originals are never mutated,
    so the only thing that can vary between two runs is the ``eps`` draw.
    """
    fusion = FisherFusion()
    base = StageContext(seed=SEED, device=CPU)
    set_seed(SEED)
    out = {}
    for unit in order:
        ctx = base.for_layer(unit[0]).for_group(unit[1]) if use_ctx else None
        mlp_a, mlp_b = experts[unit]
        fused = fusion(mlp_a, mlp_b, data[unit], ctx=ctx)
        out[unit] = fused.gate.weight.detach().clone()
    return out


def test_fisher_is_order_invariant_with_ctx():
    experts, data = _make_fisher_units()
    forward = _fisher_run(experts, data, UNITS, use_ctx=True)
    reverse = _fisher_run(experts, data, list(reversed(UNITS)), use_ctx=True)
    for unit in UNITS:
        assert torch.equal(forward[unit], reverse[unit]), (
            f"Fisher unit {unit} changed with traversal order despite seed_for"
        )


def test_fisher_global_rng_is_order_dependent():
    """Adversarial: the legacy global-RNG path DOES depend on order, so the
    invariance above is a genuine property of the local seeds."""
    experts, data = _make_fisher_units()
    forward = _fisher_run(experts, data, UNITS, use_ctx=False)
    reverse = _fisher_run(experts, data, list(reversed(UNITS)), use_ctx=False)
    assert any(not torch.equal(forward[unit], reverse[unit]) for unit in UNITS)


# --------------------------------------------------------------------------- #
# Random pairing (torch.randperm).
# --------------------------------------------------------------------------- #
N_EXPERTS = 8
LAYERS = [0, 1, 2, 3]


def _pairs_run(layers_order, *, use_ctx):
    base = StageContext(seed=SEED, device=CPU)
    set_seed(SEED)
    out = {}
    for layer in layers_order:
        ctx = base.for_layer(layer) if use_ctx else None
        out[layer] = get_pairs(
            "random", layer=layer, n_experts=N_EXPERTS, data_dir="unused", ctx=ctx
        )
    return out


def test_random_pairing_is_order_invariant_with_ctx():
    forward = _pairs_run(LAYERS, use_ctx=True)
    reverse = _pairs_run(list(reversed(LAYERS)), use_ctx=True)
    for layer in LAYERS:
        assert forward[layer] == reverse[layer], (
            f"random pairing layer {layer} changed with traversal order despite seed_for"
        )


def test_random_pairing_global_rng_is_order_dependent():
    forward = _pairs_run(LAYERS, use_ctx=False)
    reverse = _pairs_run(list(reversed(LAYERS)), use_ctx=False)
    assert any(forward[layer] != reverse[layer] for layer in LAYERS)
