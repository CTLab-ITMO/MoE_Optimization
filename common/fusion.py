"""Expert-fusion methods as composable implementations.

Each strategy for merging a pair of experts into one :class:`~common.mlp.MLP`
is a subclass of :class:`FusionMethod`. Calling an instance with
``(mlp_a, mlp_b, train_data)`` returns the fused expert; data-dependent methods
fall back to a plain weight average when no calibration data is available.

The implementations are kept faithful to their source papers:

* ``ZipIt``        -- Stoica et al., "ZipIt! Merging Models from Different
                      Tasks without Training", 2023.
* ``Procrustes``   -- orthogonal Procrustes alignment (Schoenemann, 1966); for a
                      permutation it reduces to Git Re-Basin (Ainsworth et al.,
                      2023). For SiLU MLPs a general rotation is a linear
                      approximation, see :class:`ProcrustesFusion`.
* ``TIES``         -- Yadav et al., "TIES-Merging: Resolving Interference When
                      Merging Models", NeurIPS 2023.
* ``Fisher``       -- Matena & Raffel, "Merging Models with Fisher-Weighted
                      Averaging", NeurIPS 2022.
* ``LS_*``         -- least-squares output re-fit.
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from common.context import StageContext
from common.mlp import MLP, AdvancedMLP
from common.numeric import EPS


# --------------------------------------------------------------------------- #
# Tensor / model helpers shared by the fusion methods.
# --------------------------------------------------------------------------- #
def as_tensor(data) -> torch.Tensor:
    """Stack a list of tensors into one, or pass a tensor through unchanged."""
    if torch.is_tensor(data):
        return data
    return torch.stack(data, dim=0)


def as_2d_tensor(data, device=None, dtype=None) -> torch.Tensor:
    tensor = as_tensor(data)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if device is not None or dtype is not None:
        tensor = tensor.to(
            device=device if device is not None else tensor.device, dtype=dtype
        )
    return tensor


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def hidden_activations(model: AdvancedMLP, x: torch.Tensor) -> torch.Tensor:
    """SiLU-gated intermediate activations of an :class:`AdvancedMLP`."""
    x = x.to(device=model_device(model), dtype=model.gate.weight.dtype)
    return model.act(model.gate(x)) * model.up(x)


def batch_iter(x: torch.Tensor, batch_size: int):
    for start in range(0, len(x), batch_size):
        yield x[start : start + batch_size]


def solve_linear(x: torch.Tensor, y: torch.Tensor, lambda_ridge: float = 1e-3) -> torch.Tensor:
    """Ridge least-squares: return W minimizing ||x W^T - y|| (+ l2)."""
    d_in = x.shape[1]
    xtx = x.T @ x + lambda_ridge * torch.eye(d_in, device=x.device, dtype=x.dtype)
    rhs = x.T @ y
    return torch.linalg.solve(xtx, rhs).T


# --------------------------------------------------------------------------- #
# Base class.
# --------------------------------------------------------------------------- #
class FusionMethod(ABC):
    """Merge two experts into one.

    ``needs_data`` marks methods that require calibration data; when called with
    an empty ``train_data`` those fall back to :class:`AverageFusion`.
    """

    name: str
    needs_data: bool = True

    @abstractmethod
    def merge(
        self,
        mlp_a: MLP,
        mlp_b: MLP,
        train_data,
        create_new: bool = True,
        ctx: Optional[StageContext] = None,
    ) -> MLP:
        raise NotImplementedError()

    def __call__(
        self,
        mlp_a: MLP,
        mlp_b: MLP,
        train_data,
        create_new: bool = True,
        ctx: Optional[StageContext] = None,
    ) -> MLP:
        # ``ctx`` is an additive, optional carrier of dynamic state (seeding in
        # S4); methods that do not consume it ignore it. The signature is frozen
        # k-agnostic in S5; here it stays backward-compatible (a 4th positional
        # ``create_new`` call still works).
        if self.needs_data and len(train_data) == 0:
            return AverageFusion().merge(mlp_a, mlp_b, train_data, create_new, ctx)
        return self.merge(mlp_a, mlp_b, train_data, create_new, ctx)


# --------------------------------------------------------------------------- #
# Data-free methods.
# --------------------------------------------------------------------------- #
class AverageFusion(FusionMethod):
    """Element-wise average of the two experts' parameters."""

    name = "Average"
    needs_data = False

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        with torch.no_grad():
            new_mlp = copy.deepcopy(mlp_a) if create_new else mlp_a
            dict_a = mlp_a.state_dict()
            dict_b = mlp_b.state_dict()
            new_dict = {key: (dict_a[key] + dict_b[key]) / 2 for key in dict_a}
            new_mlp.load_state_dict(new_dict)
        return new_mlp


class PruningFusion(FusionMethod):
    """One-shot pruning baseline: keep expert A, drop expert B."""

    name = "Pruning"
    needs_data = False

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        return copy.deepcopy(mlp_a) if create_new else mlp_a


# --------------------------------------------------------------------------- #
# ZipIt.
# --------------------------------------------------------------------------- #
class ZipItFusion(FusionMethod):
    """ZipIt! merge: concatenate the experts, learn merge/unmerge matrices from
    feature correlations on ``train_data``, then collapse to a single expert
    that reconstructs either original output via ``forward_unzip(out=0|1)``."""

    name = "ZipIt"

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        with torch.no_grad():
            new_mlp = copy.deepcopy(mlp_a) if create_new else mlp_a
            new_mlp.zip_with(mlp_b)
            new_mlp.train_with(train_data)
            new_mlp.zip()
        return new_mlp


# --------------------------------------------------------------------------- #
# Procrustes alignment.
# --------------------------------------------------------------------------- #
def orthogonal_procrustes(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Return orthogonal R minimizing ``||target - source @ R||_F``.

    Closed form (Schoenemann, 1966): with ``source^T @ target = U S V^T`` the
    solution is ``R = U V^T``.
    """
    m = source.T @ target
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    return u @ vh


def _apply_hidden_rotation(model: AdvancedMLP, r: torch.Tensor) -> AdvancedMLP:
    """Rotate a copy of ``model`` so its hidden axis becomes ``hidden @ r``.

    Reduces to a Git Re-Basin permutation when ``r`` is a permutation matrix
    (function-preserving); for a general orthogonal ``r`` it is a linear
    approximation because SiLU does not commute with rotations.
    """
    new = copy.deepcopy(model)
    r = r.to(dtype=model.gate.weight.dtype, device=model.gate.weight.device)
    with torch.no_grad():
        new.gate.weight = nn.Parameter(r.T @ model.gate.weight)
        new.up.weight = nn.Parameter(r.T @ model.up.weight)
        new.down.weight = nn.Parameter(model.down.weight @ r)
    return new


class ProcrustesFusion(FusionMethod):
    """Align expert B's hidden units to A's by orthogonal Procrustes, then
    average. The alignment is computed from the SiLU-gated activations on
    ``train_data``; see :func:`orthogonal_procrustes`."""

    name = "Procrustes"

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        with torch.no_grad():
            data = as_2d_tensor(train_data, device=model_device(mlp_a))
            hidden_a = hidden_activations(mlp_a, data).float()
            hidden_b = hidden_activations(mlp_b, data).float()
            r = orthogonal_procrustes(hidden_a, hidden_b)
            aligned_b = _apply_hidden_rotation(mlp_b, r)
            return AverageFusion().merge(mlp_a, aligned_b, train_data, create_new=create_new)


# --------------------------------------------------------------------------- #
# TIES.
# --------------------------------------------------------------------------- #
def _trim(weights: torch.Tensor, trim_ratio: float) -> torch.Tensor:
    """Zero the smallest-magnitude ``trim_ratio`` fraction of ``weights``."""
    if trim_ratio <= 0:
        return weights
    threshold = torch.quantile(weights.abs().flatten(), trim_ratio)
    return torch.where(weights.abs() >= threshold, weights, torch.zeros_like(weights))


class TiesFusion(FusionMethod):
    """TIES-Merging (Yadav et al., 2023): Trim -> Elect-Sign -> Disjoint-Mean.

    Experts share no pretrained base, so the task vectors are the raw weights
    (base = 0). Per parameter tensor: (1) trim each expert's smallest-magnitude
    entries; (2) elect a sign per coordinate as ``sign(tau_a + tau_b)``;
    (3) average only the trimmed entries whose sign matches the elected sign.
    """

    name = "TIES"

    def __init__(self, trim_ratio: float = 0.2):
        self.trim_ratio = trim_ratio

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        with torch.no_grad():
            merged = copy.deepcopy(mlp_a) if create_new else mlp_a
            state_a = mlp_a.state_dict()
            state_b = mlp_b.state_dict()
            merged_state = {}

            for key in state_a:
                tau_a = _trim(state_a[key].float(), self.trim_ratio)
                tau_b = _trim(state_b[key].float(), self.trim_ratio)

                gamma = torch.sign(tau_a + tau_b)
                keep_a = (torch.sign(tau_a) == gamma) & (tau_a != 0)
                keep_b = (torch.sign(tau_b) == gamma) & (tau_b != 0)

                numerator = tau_a * keep_a + tau_b * keep_b
                count = keep_a.float() + keep_b.float()
                chosen = torch.where(
                    count > 0, numerator / count.clamp_min(1.0), torch.zeros_like(numerator)
                )
                merged_state[key] = chosen.to(dtype=state_a[key].dtype)

            merged.load_state_dict(merged_state, strict=True)
            return merged


# --------------------------------------------------------------------------- #
# Fisher-weighted averaging.
# --------------------------------------------------------------------------- #
def diagonal_empirical_fisher(
    model: AdvancedMLP,
    data: torch.Tensor,
    *,
    batch_size: int = 256,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    """Diagonal empirical Fisher of ``model`` under a unit-variance Gaussian
    output model (Matena & Raffel, 2022).

    Estimated with a Gaussian score sampler: for ``loss = sum(out * eps)`` with
    ``eps ~ N(0, I)``, ``E_eps[grad^2] = sum_k (d out_k / d theta)^2`` -- the
    per-parameter Fisher diagonal. Depends only on ``model``, not on its pair.

    ``generator`` seeds the ``eps`` draw: pass a per-pair
    :class:`torch.Generator` (from :meth:`StageContext.generator`) to make the
    estimate order-independent; ``None`` draws from the global RNG, which is
    bit-identical to the legacy ``torch.randn_like`` behaviour.
    """
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    fisher = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in params.items()}

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    batches = 0
    with torch.enable_grad():
        for batch in batch_iter(data, batch_size):
            model.zero_grad(set_to_none=True)
            out = model(batch)
            eps = torch.randn(
                out.shape, dtype=out.dtype, device=out.device, generator=generator
            )
            loss = (out * eps).sum()
            grads = torch.autograd.grad(
                loss, tuple(params.values()), allow_unused=True
            )
            for (name, _), grad in zip(params.items(), grads):
                if grad is not None:
                    fisher[name] += grad.detach().float().pow(2)
            batches += 1

    if was_training:
        model.train()
    denom = max(1, batches)
    return {name: value / denom for name, value in fisher.items()}


def fisher_weighted_average(
    state_a: dict, state_b: dict, fisher_a: dict, fisher_b: dict
) -> dict:
    """Precision-weighted merge ``theta* = (F_a*theta_a + F_b*theta_b)/(F_a+F_b)``."""
    merged = {}
    for key in state_a:
        imp_a = fisher_a.get(key)
        imp_b = fisher_b.get(key)
        if imp_a is None or imp_b is None:
            merged[key] = (state_a[key] + state_b[key]) / 2
            continue
        imp_a = imp_a.float()
        imp_b = imp_b.float()
        merged[key] = (
            (imp_a * state_a[key].float() + imp_b * state_b[key].float())
            / (imp_a + imp_b + EPS)
        ).to(dtype=state_a[key].dtype)
    return merged


class FisherFusion(FusionMethod):
    """Fisher-weighted averaging (Matena & Raffel, 2022): merge each parameter
    weighted by its diagonal empirical Fisher in the respective expert."""

    name = "Fisher"

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        data = as_2d_tensor(train_data, device=model_device(mlp_a))
        # Order-independent eps draw (S4): one per-pair generator feeds both
        # experts' Fisher estimates (A then B, so they still get distinct eps).
        # ctx=None keeps the legacy global-RNG behaviour.
        generator = (
            ctx.generator("fusion", method=self.name, device=model_device(mlp_a))
            if ctx is not None
            else None
        )
        fisher_a = diagonal_empirical_fisher(mlp_a, data, generator=generator)
        fisher_b = diagonal_empirical_fisher(mlp_b, data, generator=generator)

        merged = copy.deepcopy(mlp_a) if create_new else mlp_a
        merged_state = fisher_weighted_average(
            mlp_a.state_dict(), mlp_b.state_dict(), fisher_a, fisher_b
        )
        merged.load_state_dict(merged_state, strict=True)
        return merged


# --------------------------------------------------------------------------- #
# Distillation.
# --------------------------------------------------------------------------- #
class DistillationFusion(FusionMethod):
    """Train a student expert (initialized from the average) to mimic the
    pairwise teacher ensemble on outputs and hidden activations.

    All training hyperparameters are exposed as constructor arguments; the
    defaults reproduce the original hardcoded schedule. ``epochs_large`` is used
    when the calibration set has at least ``small_data_threshold`` samples,
    otherwise ``epochs_small``. ``hidden_loss_weight`` balances the hidden-state
    distillation term against the output term.
    """

    name = "Distillation"

    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        hidden_loss_weight: float = 0.25,
        max_batch_size: int = 256,
        epochs_large: int = 4,
        epochs_small: int = 8,
        small_data_threshold: int = 512,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_loss_weight = hidden_loss_weight
        self.max_batch_size = max_batch_size
        self.epochs_large = epochs_large
        self.epochs_small = epochs_small
        self.small_data_threshold = small_data_threshold

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        device = model_device(mlp_a)
        data = as_2d_tensor(train_data, device=device)
        batch_size = min(self.max_batch_size, max(1, len(data)))
        student = AverageFusion().merge(mlp_a, mlp_b, train_data, create_new=create_new)

        with torch.no_grad():
            teacher_out = 0.5 * (mlp_a(data) + mlp_b(data))
            teacher_hidden = 0.5 * (
                hidden_activations(mlp_a, data) + hidden_activations(mlp_b, data)
            )

        student.train()
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        epochs = (
            self.epochs_large if len(data) >= self.small_data_threshold else self.epochs_small
        )

        with torch.enable_grad():
            for _ in range(epochs):
                for batch_x, batch_out, batch_hidden in zip(
                    batch_iter(data, batch_size),
                    batch_iter(teacher_out, batch_size),
                    batch_iter(teacher_hidden, batch_size),
                ):
                    optimizer.zero_grad(set_to_none=True)
                    pred_out = student(batch_x)
                    pred_hidden = hidden_activations(student, batch_x)
                    loss = F.mse_loss(pred_out, batch_out) + self.hidden_loss_weight * F.mse_loss(
                        pred_hidden, batch_hidden
                    )
                    loss.backward()
                    optimizer.step()

        student.eval()
        return student


# --------------------------------------------------------------------------- #
# Least-squares output adapter (wraps another method).
# --------------------------------------------------------------------------- #
class LeastSquaresAdapter(FusionMethod):
    """Run ``inner``, then replace the fused expert's unmerge layer with a
    least-squares fit reconstructing both originals' concatenated outputs."""

    needs_data = False

    def __init__(self, inner: FusionMethod):
        self.inner = inner
        self.name = f"LS_{inner.name}"

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        fused = self.inner(mlp_a, mlp_b, train_data, create_new, ctx)
        if len(train_data) < 2 * mlp_a.down.out_features:
            return fused
        fused.down = None
        all_tokens = as_tensor(train_data)
        real_res = torch.cat([mlp_a.forward(all_tokens), mlp_b.forward(all_tokens)], dim=-1)
        new_res = fused.forward(all_tokens)
        fused.U = nn.Parameter(
            solve_linear(new_res.float(), real_res.float()).to(dtype=mlp_a.down.weight.dtype)
        )
        return fused


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #
BASE_FUSION_METHODS: list[FusionMethod] = [
    ZipItFusion(),
    AverageFusion(),
    PruningFusion(),
    ProcrustesFusion(),
    FisherFusion(),
    TiesFusion(),
    DistillationFusion(),
]


def build_fusion_methods() -> dict[str, FusionMethod]:
    """Name -> fusion instance, including an ``LS_<name>`` variant of each."""
    methods: dict[str, FusionMethod] = {m.name: m for m in BASE_FUSION_METHODS}
    for method in BASE_FUSION_METHODS:
        adapter = LeastSquaresAdapter(method)
        methods[adapter.name] = adapter
    return methods
