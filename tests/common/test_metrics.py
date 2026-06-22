"""Reference tests for the proxy metrics.

Each metric is checked against an independent reference (torch built-ins or a
closed-form expression), not merely for "runs and returns a float".
"""

import math

import pytest
import torch
import torch.nn.functional as F

from common import metrics
from common.metrics import (
    CKA,
    JS,
    KL,
    MAE,
    MSE,
    RMSE,
    R2,
    Cosine,
    EntropyGap,
    Pearson,
    RelMSE,
    Top1,
    Top5,
    METRICS,
    METRICS_BY_NAME,
    best_metrics,
)


@pytest.fixture
def xy():
    torch.manual_seed(0)
    x = torch.randn(16, 5)
    y = x + 0.3 * torch.randn(16, 5)
    return x, y


# --------------------------------------------------------------------------- #
# Pointwise error / similarity.
# --------------------------------------------------------------------------- #
def test_mse_matches_closed_form(xy):
    x, y = xy
    assert MSE()(x, y) == pytest.approx(torch.mean((x - y) ** 2).item(), rel=1e-6)


def test_rmse_is_sqrt_of_mse(xy):
    x, y = xy
    assert RMSE()(x, y) == pytest.approx(math.sqrt(MSE()(x, y)), rel=1e-6)


def test_mae_matches_closed_form(xy):
    x, y = xy
    assert MAE()(x, y) == pytest.approx(torch.mean((x - y).abs()).item(), rel=1e-6)


def test_relmse_equals_mse_over_variance(xy):
    x, y = xy
    var = torch.mean((x - x.mean(dim=0, keepdim=True)) ** 2).item()
    assert RelMSE()(x, y) == pytest.approx(MSE()(x, y) / var, rel=1e-5)


def test_r2_complements_relmse(xy):
    # R2 (sum form) and RelMSE (mean form) share the same ratio -> R2 = 1 - RelMSE.
    x, y = xy
    assert R2()(x, y) == pytest.approx(1.0 - RelMSE()(x, y), rel=1e-4)


def test_r2_is_one_for_identity_and_negative_for_bad_fit(xy):
    x, _ = xy
    assert R2()(x, x) == pytest.approx(1.0, abs=1e-4)
    assert R2()(x, x + 100.0) < 0.0


def test_cosine_matches_torch(xy):
    x, y = xy
    ref = F.cosine_similarity(x, y).mean().item()
    assert Cosine()(x, y) == pytest.approx(ref, rel=1e-5)


def test_pearson_matches_corrcoef(xy):
    x, y = xy
    ref = torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()
    assert Pearson()(x, y) == pytest.approx(ref, rel=1e-4)


# --------------------------------------------------------------------------- #
# CKA.
# --------------------------------------------------------------------------- #
def test_cka_self_similarity_is_one():
    x = torch.randn(24, 6)
    assert CKA()(x, x) == pytest.approx(1.0, abs=1e-4)


def test_cka_invariant_to_isotropic_scaling():
    x = torch.randn(24, 6)
    assert CKA()(x, 7.0 * x) == pytest.approx(1.0, abs=1e-4)


def test_cka_invariant_to_orthogonal_feature_transform():
    x = torch.randn(24, 6)
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    assert CKA()(x, x @ q) == pytest.approx(1.0, abs=1e-4)


def test_cka_value_in_unit_interval(xy):
    x, y = xy
    value = CKA()(x, y)
    assert 0.0 <= value <= 1.0 + 1e-6


def test_center_gram_double_centers():
    k = torch.randn(5, 5)
    k = k @ k.t()
    centered = metrics.center_gram(k)
    assert torch.allclose(centered.sum(dim=0), torch.zeros(5), atol=1e-4)
    assert torch.allclose(centered.sum(dim=1), torch.zeros(5), atol=1e-4)


# --------------------------------------------------------------------------- #
# Distribution agreement.
# --------------------------------------------------------------------------- #
def test_kl_zero_for_equal_logits():
    logits = torch.randn(6, 4)
    assert KL()(logits, logits) == pytest.approx(0.0, abs=1e-5)


def test_kl_matches_manual_direction():
    # KL metric is KL(softmax(y) || softmax(x)).
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    p = F.softmax(x, dim=-1)
    q = F.softmax(y, dim=-1)
    manual = (q * (q.log() - p.log())).sum(dim=-1).mean().item()
    assert KL()(x, y) == pytest.approx(manual, rel=1e-4, abs=1e-6)


def test_js_zero_for_equal_and_symmetric():
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    assert JS()(x, x) == pytest.approx(0.0, abs=1e-5)
    assert JS()(x, y) == pytest.approx(JS()(y, x), rel=1e-5, abs=1e-7)


def test_js_is_nonnegative_and_bounded(xy):
    x, y = xy
    value = JS()(x, y)
    assert 0.0 <= value <= math.log(2) + 1e-4


def test_top1_and_top5_agreement():
    logits = torch.tensor([[3.0, 1.0, 2.0], [0.1, 5.0, 0.2]])
    assert Top1()(logits, logits) == pytest.approx(1.0)
    assert Top5()(logits, logits) == pytest.approx(1.0)
    swapped = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.1, 0.2]])
    assert Top1()(logits, swapped) == pytest.approx(0.0)


def test_entropy_gap_zero_for_equal_and_uniform_is_max():
    logits = torch.randn(4, 5)
    assert EntropyGap()(logits, logits) == pytest.approx(0.0, abs=1e-6)
    uniform = torch.zeros(1, 4)
    peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    assert metrics.prediction_entropy(uniform) > metrics.prediction_entropy(peaked)


# --------------------------------------------------------------------------- #
# Registry / direction flags.
# --------------------------------------------------------------------------- #
def test_registry_names_unique_and_complete():
    names = [m.name for m in METRICS]
    assert len(names) == len(set(names)) == 13
    assert set(METRICS_BY_NAME) == set(names)


def test_best_metrics_directions():
    assert best_metrics["MSE"] is False
    assert best_metrics["Cosine"] is True
    assert best_metrics["CKA"] is True
    assert best_metrics["KL"] is False
    # best_metrics is derived from the classes, so it must agree with them.
    for metric in METRICS:
        assert best_metrics[metric.name] is metric.higher_is_better
