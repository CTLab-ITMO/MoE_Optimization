import random

import numpy as np
import torch

from common.seeding import set_seed


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
