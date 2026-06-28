"""Tests for the shared numerical constants and device fallback."""

import torch

from common import metrics, numeric


def test_eps_value_is_frozen():
    assert numeric.EPS == 1e-8


def test_metrics_eps_is_the_shared_constant():
    # metrics.py re-exports the single source of truth, not its own literal.
    assert metrics.EPS is numeric.EPS


def test_default_device_is_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert numeric._default_device() == torch.device("cpu")


def test_default_device_is_evaluated_at_call_time(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert numeric._default_device() == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert numeric._default_device() == torch.device("cpu")
