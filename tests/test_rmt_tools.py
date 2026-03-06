import torch

from common.rmt_tools import denoise_importance_gate_spikes
from common.spectrum_tools import whitened_mean_bias


def test_whitened_mean_bias_shapes() -> None:
    x = torch.randn(32, 8)
    out = whitened_mean_bias(x, alpha=0.2, lam=1e-3)
    assert out["bias"].shape == (8,)
    assert out["Sigma"].shape == (8, 8)


def test_gate_spike_denoise_shape() -> None:
    importance = torch.randn(8)
    gate_weight = torch.randn(8, 16)
    v_hat, info = denoise_importance_gate_spikes(importance, gate_weight, mp_factor=1.1)
    assert v_hat.shape == importance.shape
    assert info.rank >= 1
