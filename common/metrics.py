"""Metrics for comparing fused-expert outputs against originals.

Each metric is a class with ``name``, ``higher_is_better``, and ``log_scale``
flags. ``x`` is the original output; ``y`` is the fused approximation.

Two families:
- pointwise (MSE, RMSE, MAE, RelMSE, R2, Cosine, Pearson, CKA): compare tensors directly;
- distribution (KL, JS, Top1, Top5, EntropyGap): treat the last dimension as logits.

CKA: Kornblith et al., ICML 2019.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

# Stability epsilon for divisions and logs. Defined in common.numeric;
# re-exported here so callers can still reference ``metrics.EPS``.
from common.numeric import EPS
from common.registry import Registry


def center_gram(gram: torch.Tensor) -> torch.Tensor:
    """Double-center a Gram matrix (subtract row/column means, add grand mean)."""
    n = gram.size(0)
    unit = torch.ones((n, n), device=gram.device) / n
    return gram - unit @ gram - gram @ unit + unit @ gram @ unit


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy of softmax(logits) over the last dimension."""
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs.clamp_min(EPS)), dim=-1).mean()


def _topk_agreement(logits_p: torch.Tensor, logits_q: torch.Tensor, k: int) -> float:
    """Average overlap of the top-k indices of two logit tensors."""
    k = max(1, min(k, logits_p.shape[-1], logits_q.shape[-1]))
    top_p = torch.topk(logits_p, k=k, dim=-1).indices
    top_q = torch.topk(logits_q, k=k, dim=-1).indices
    if k == 1:
        return (top_p.squeeze(-1) == top_q.squeeze(-1)).float().mean().item()
    overlap = []
    for a, b in zip(top_p, top_q):
        overlap.append(len(set(a.tolist()) & set(b.tolist())) / float(k))
    return float(sum(overlap) / max(1, len(overlap)))


class Metric(ABC):
    """Compare a reference tensor ``x`` against an approximation ``y``."""

    name: str
    higher_is_better: bool
    # True if the metric looks better on a log scale. Used to derive
    # ``log_safe_metrics`` from the classes rather than a separate literal.
    log_scale: bool

    # Optional context (labels, device, etc.) for metrics that need shared state.
    # Built-in metrics ignore it; custom subclasses can use it without changing call sites.
    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor, ctx=None) -> float:
        raise NotImplementedError()


class MSE(Metric):
    name = "MSE"
    higher_is_better = False
    log_scale = True

    def __call__(self, x, y, ctx=None):
        return torch.square(x - y).mean().item()


class RMSE(Metric):
    name = "RMSE"
    higher_is_better = False
    log_scale = True

    def __call__(self, x, y, ctx=None):
        return torch.sqrt(torch.square(x - y).mean()).item()


class MAE(Metric):
    name = "MAE"
    higher_is_better = False
    log_scale = True

    def __call__(self, x, y, ctx=None):
        return torch.abs(x - y).mean().item()


class RelMSE(Metric):
    """MSE normalized by the variance of x. Approximately ``1 - R2``; in [0, inf), lower is better."""

    name = "RelMSE"
    higher_is_better = False
    log_scale = True

    def __call__(self, x, y, ctx=None):
        denom = torch.square(x - x.mean(dim=0, keepdim=True)).mean().clamp_min(EPS)
        return (torch.square(x - y).mean() / denom).item()


class R2(Metric):
    """Coefficient of determination over all elements."""

    name = "R2"
    higher_is_better = True
    log_scale = False

    def __call__(self, x, y, ctx=None):
        ss_res = torch.square(x - y).sum()
        ss_tot = torch.square(x - x.mean(dim=0, keepdim=True)).sum().clamp_min(EPS)
        return (1.0 - ss_res / ss_tot).item()


class Cosine(Metric):
    """Mean row-wise cosine similarity."""

    name = "Cosine"
    higher_is_better = True
    log_scale = False

    def __call__(self, x, y, ctx=None):
        return torch.cosine_similarity(x, y).mean().item()


class Pearson(Metric):
    """Pearson correlation over all flattened elements."""

    name = "Pearson"
    higher_is_better = True
    log_scale = False

    def __call__(self, x, y, ctx=None):
        x = x.flatten().float()
        y = y.flatten().float()
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(EPS))
        return (x.mul(y).sum() / denom).item()


class CKA(Metric):
    """Linear Centered Kernel Alignment (Kornblith et al., 2019)."""

    name = "CKA"
    higher_is_better = True
    log_scale = True

    def __call__(self, x, y, ctx=None):
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        k = x @ x.T
        l = y @ y.T

        kc = center_gram(k)
        lc = center_gram(l)

        hsic = (kc * lc).sum()
        norm_x = torch.sqrt((kc * kc).sum())
        norm_y = torch.sqrt((lc * lc).sum())
        return (hsic / (norm_x * norm_y + EPS)).mean().item()


class KL(Metric):
    """KL divergence ``KL(softmax(y) || softmax(x))`` over the last dimension."""

    name = "KL"
    higher_is_better = False
    log_scale = True

    def __call__(self, x, y, ctx=None):
        p = F.log_softmax(x, dim=-1)
        q = F.softmax(y, dim=-1)
        return F.kl_div(p, q, reduction="batchmean").item()


class JS(Metric):
    """Jensen-Shannon divergence between softmax(x) and softmax(y) (symmetric)."""

    name = "JS"
    higher_is_better = False
    log_scale = True

    def __call__(self, x, y, ctx=None):
        p = F.softmax(x, dim=-1)
        q = F.softmax(y, dim=-1)
        m = 0.5 * (p + q)
        js = 0.5 * (
            F.kl_div(m.log(), p, reduction="batchmean")
            + F.kl_div(m.log(), q, reduction="batchmean")
        )
        return js.mean().item()


class Top1(Metric):
    """Fraction of rows whose argmax matches."""

    name = "Top1"
    higher_is_better = True
    log_scale = False

    def __call__(self, x, y, ctx=None):
        return _topk_agreement(x, y, k=1)


class Top5(Metric):
    """Average top-5 index overlap (clamped to the output width)."""

    name = "Top5"
    higher_is_better = True
    log_scale = False

    def __call__(self, x, y, ctx=None):
        # _topk_agreement already clamps k to the output width.
        return _topk_agreement(x, y, k=5)


class EntropyGap(Metric):
    """Absolute difference of mean prediction entropies."""

    name = "EntropyGap"
    higher_is_better = False
    log_scale = True

    def __call__(self, x, y, ctx=None):
        return abs(prediction_entropy(x).item() - prediction_entropy(y).item())


# Metrics in display order; this list defines the default ``best_metrics`` set.
METRICS: list[Metric] = [
    MSE(),
    RMSE(),
    MAE(),
    RelMSE(),
    Cosine(),
    Pearson(),
    R2(),
    CKA(),
    JS(),
    KL(),
    Top1(),
    Top5(),
    EntropyGap(),
]

METRICS_BY_NAME: dict[str, Metric] = {metric.name: metric for metric in METRICS}

# name -> True if a greater value is better. Consumed by ``Best_<metric>`` selection.
best_metrics: dict[str, bool] = {
    metric.name: metric.higher_is_better for metric in METRICS
}

# Metric names that look better on a log scale. Derived from the ``log_scale``
# class flag so it stays in sync when metrics are added or renamed.
log_safe_metrics: set[str] = {metric.name for metric in METRICS if metric.log_scale}


def _build_metric_registry() -> Registry:
    """The canonical metric registry, in ``METRICS`` order (pinned for golden)."""
    registry = Registry("metric")
    for metric in METRICS:
        registry.register(metric.name, metric)
    return registry


# name -> Metric instance. Stages validate ``FILTER_METRICS`` against this
# and can add new metrics via ``register``.
METRIC_REGISTRY: Registry = _build_metric_registry()


def live_best_metrics() -> dict:
    """Return ``name -> higher_is_better`` from the live registry.

    Unlike the module-level ``best_metrics`` (frozen at import), this reflects
    any metrics added after import via ``register``.
    """
    return {name: METRIC_REGISTRY[name].higher_is_better for name in METRIC_REGISTRY}
