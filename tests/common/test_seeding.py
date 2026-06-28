import random

import numpy as np
import torch

from common.seeding import generator_for, seed_for, set_seed


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_and_random_reproducible():
    set_seed(7)
    np_first = np.random.rand(3).tolist()
    py_first = [random.random() for _ in range(3)]

    set_seed(7)
    np_second = np.random.rand(3).tolist()
    py_second = [random.random() for _ in range(3)]

    assert np_first == np_second
    assert py_first == py_second


def test_different_seeds_give_different_sequences():
    set_seed(1)
    a = torch.rand(5)
    set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_set_seed_seeds_cuda_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))

    set_seed(42)

    assert calls
    assert set(calls) == {42}


# --------------------------------------------------------------------------- #
# seed_for: derived, order-independent local seeds (§3.3).
# --------------------------------------------------------------------------- #
def test_seed_for_is_stable_across_calls():
    a = seed_for(seed=42, stage="fusion", layer=3, group=1, method="Fisher")
    b = seed_for(seed=42, stage="fusion", layer=3, group=1, method="Fisher")
    assert a == b


def test_seed_for_is_process_independent():
    # hashlib-based, NOT the per-process-salted builtin hash(): a frozen literal
    # pins the seed mapping so any silent change to the hashing (which would
    # re-base every stochastic golden) is caught here.
    assert (
        seed_for(seed=42, stage="fusion", layer=0, group=0, method="Fisher")
        == 8353614451793807489
    )


def test_seed_for_distinct_for_distinct_coords():
    base = seed_for(seed=42, stage="fusion", layer=3, group=1, method="Fisher")
    assert base != seed_for(seed=42, stage="fusion", layer=4, group=1, method="Fisher")
    assert base != seed_for(seed=42, stage="fusion", layer=3, group=2, method="Fisher")
    assert base != seed_for(seed=42, stage="fusion", layer=3, group=1, method="Average")
    assert base != seed_for(seed=43, stage="fusion", layer=3, group=1, method="Fisher")
    assert base != seed_for(seed=42, stage="pairing", layer=3, group=1, method="Fisher")


def test_seed_for_skips_none_coords():
    assert seed_for(seed=1, stage="fusion", method="Fisher") == seed_for(
        seed=1, stage="fusion", layer=None, group=None, method="Fisher"
    )


def test_seed_for_returns_valid_torch_seed():
    s = seed_for(seed=42, stage="fusion", layer=0, group=0, method="Fisher")
    assert isinstance(s, int) and 0 <= s < (1 << 63)
    torch.Generator().manual_seed(s)  # must not raise


# --------------------------------------------------------------------------- #
# generator_for: a torch.Generator seeded by seed_for.
# --------------------------------------------------------------------------- #
def test_generator_for_is_reproducible():
    g1 = generator_for(seed=42, stage="fusion", layer=1, method="Fisher")
    g2 = generator_for(seed=42, stage="fusion", layer=1, method="Fisher")
    assert torch.equal(torch.randn(5, generator=g1), torch.randn(5, generator=g2))


def test_generator_for_distinct_coords_diverge():
    g1 = generator_for(seed=42, stage="fusion", layer=1, method="Fisher")
    g2 = generator_for(seed=42, stage="fusion", layer=2, method="Fisher")
    assert not torch.equal(torch.randn(5, generator=g1), torch.randn(5, generator=g2))


def test_generator_for_is_independent_of_global_rng():
    # The whole point: drawing from the generator does not depend on, and does
    # not disturb, the global RNG stream.
    set_seed(7)
    expected = torch.randn(5, generator=generator_for(seed=42, stage="s", layer=0))
    set_seed(999)
    torch.randn(100)  # perturb the global stream
    actual = torch.randn(5, generator=generator_for(seed=42, stage="s", layer=0))
    assert torch.equal(expected, actual)
