"""Expert-fusion strategies for merging a group of MoE experts.

Each strategy is a :class:`FusionMethod` subclass. Call an instance with the
experts as positional args (``method(mlp_a, mlp_b, train_data=data)``) to get
the fused expert. ``supported_arity`` is the per-class arity contract: most
methods are strictly pairwise (k=2); ``Average`` and ``Pruning`` are
k-agnostic (any k >= 1). Data-dependent methods fall back to a plain weight
average when no calibration data is given.

Implemented methods (faithful to source papers): ZipIt (Stoica 2023),
Procrustes alignment (Schoenemann 1966), TIES (Yadav NeurIPS 2023),
Fisher-weighted averaging (Matena & Raffel NeurIPS 2022), and LS_* variants
that re-fit the output layer by least squares.
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from common.context import StageContext
from common.fused_group import ZIP_UNMERGE_RECONSTRUCTION
from common.mlp import MLP, AdvancedMLP
from common.numeric import EPS, solve_linear
from common.registry import Registry


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


# ``solve_linear`` lives in ``common.numeric`` to avoid a circular import with
# ``common.mlp``; imported above so it remains reachable as ``fusion.solve_linear``.


# --------------------------------------------------------------------------- #
# Base class.
# --------------------------------------------------------------------------- #
class FusionMethod(ABC):
    """Merge a group of experts into one.

    ``needs_data`` marks methods that require calibration data; when called with
    an empty ``train_data`` those fall back to :class:`AverageFusion`.

    ``supported_arity`` is the per-class arity contract: ``2`` (default) for
    strictly pairwise methods, ``None`` for k-agnostic methods accepting any
    ``k >= 1``. Pairwise methods implement :meth:`merge`; k-agnostic methods
    override :meth:`merge_group` instead.

    ``wrappable=False`` opts a method out of meta-wrappers
    (:func:`apply_meta_wrappers`), e.g. when the pairwise ``LS_<name>`` variant
    would be meaningless for a k-agnostic method.

    ``reconstruction`` names the runtime reconstruction contract this method's
    result declares (a key in ``RECONSTRUCTION_REGISTRY``); it defaults to
    ``zip_unmerge`` (the pairwise unmerge path). ``FusedExpert.from_result``
    reads it and falls back to the size default when the declared kind is
    incompatible with the group size (e.g. a singleton).
    """

    name: str
    needs_data: bool = True
    supported_arity: Optional[int] = 2
    wrappable: bool = True
    reconstruction: str = ZIP_UNMERGE_RECONSTRUCTION

    @abstractmethod
    def merge(
        self,
        mlp_a: MLP,
        mlp_b: MLP,
        train_data,
        create_new: bool = True,
        ctx: Optional[StageContext] = None,
    ) -> MLP:
        # k=2 contract with explicit named params (catches subclass typos).
        # The variadic public interface lives on ``__call__``.
        raise NotImplementedError()

    def merge_group(
        self,
        mlps,
        train_data,
        create_new: bool = True,
        ctx: Optional[StageContext] = None,
    ) -> MLP:
        """Merge a sequence of experts; the k-agnostic seam behind ``__call__``.

        The default delegates a pair to :meth:`merge`; k-agnostic subclasses
        override this method. ``__call__`` validates arity first, so the
        guard here only protects direct ``merge_group`` callers.
        """
        if len(mlps) != 2:
            raise NotImplementedError(
                f"fusion method {self.name!r} supports only k=2 "
                f"(got {len(mlps)} experts)"
            )
        return self.merge(mlps[0], mlps[1], train_data, create_new, ctx)

    def _check_arity(self, count: int) -> None:
        if count < 1:
            raise ValueError(
                f"fusion method {self.name!r} needs at least one expert, got {count}"
            )
        if self.supported_arity is not None and count != self.supported_arity:
            raise NotImplementedError(
                f"fusion method {self.name!r} supports only "
                f"k={self.supported_arity} (got {count} experts)"
            )

    def __call__(self, *mlps, train_data=None, create_new: bool = True,
                 ctx: Optional[StageContext] = None) -> MLP:
        """Public fusion entry point; accepts experts as positional args.

        Arity is validated against ``supported_arity``: pairwise methods raise
        ``NotImplementedError`` on ``k != 2``, k-agnostic methods accept any
        ``k >= 1``. Keyword args: ``train_data`` (calibration rows),
        ``create_new``, ``ctx`` (optional dynamic state, e.g. for seeding).

        Backward compat: ``method(mlp_a, mlp_b, train_data)`` still works --
        with exactly three positionals, ``train_data`` unset and a
        non-``nn.Module`` last positional, that last positional is treated as
        ``train_data``. A third positional ``nn.Module`` is a third expert;
        for ``k >= 3`` pass ``train_data`` as a keyword only.
        """
        if (
            train_data is None
            and len(mlps) == 3
            and not isinstance(mlps[-1], nn.Module)
        ):
            *mlp_list, train_data = mlps
            mlps = tuple(mlp_list)
        self._check_arity(len(mlps))
        if train_data is None:
            train_data = []
        if self.needs_data and len(train_data) == 0:
            return AverageFusion().merge_group(mlps, train_data, create_new, ctx)
        return self.merge_group(mlps, train_data, create_new, ctx)


# --------------------------------------------------------------------------- #
# Data-free methods.
# --------------------------------------------------------------------------- #
class AverageFusion(FusionMethod):
    """Element-wise average of the group's parameters (k-agnostic; k=1 is identity)."""

    name = "Average"
    needs_data = False
    supported_arity = None

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        return self.merge_group((mlp_a, mlp_b), train_data, create_new, ctx)

    def merge_group(self, mlps, train_data, create_new=True, ctx=None):
        with torch.no_grad():
            new_mlp = copy.deepcopy(mlps[0]) if create_new else mlps[0]
            if len(mlps) == 1:
                return new_mlp
            states = [mlp.state_dict() for mlp in mlps]
            new_dict = {}
            for key in states[0]:
                total = states[0][key]
                for state in states[1:]:
                    total = total + state[key]
                new_dict[key] = total / len(mlps)
            new_mlp.load_state_dict(new_dict)
        return new_mlp


class PruningFusion(FusionMethod):
    """One-shot pruning baseline: keep the first expert, drop the rest (k-agnostic)."""

    name = "Pruning"
    needs_data = False
    supported_arity = None

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        return self.merge_group((mlp_a, mlp_b), train_data, create_new, ctx)

    def merge_group(self, mlps, train_data, create_new=True, ctx=None):
        return copy.deepcopy(mlps[0]) if create_new else mlps[0]


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
    """Return a copy of ``model`` with its hidden axis rotated by ``r``.

    Exact when ``r`` is a permutation matrix (function-preserving; Git Re-Basin);
    otherwise a linear approximation because SiLU does not commute with rotations.
    """
    new = copy.deepcopy(model)
    r = r.to(dtype=model.gate.weight.dtype, device=model.gate.weight.device)
    with torch.no_grad():
        new.gate.weight = nn.Parameter(r.T @ model.gate.weight)
        new.up.weight = nn.Parameter(r.T @ model.up.weight)
        new.down.weight = nn.Parameter(model.down.weight @ r)
    return new


class ProcrustesFusion(FusionMethod):
    """Align B's hidden units to A's via orthogonal Procrustes (computed from
    SiLU-gated activations on ``train_data``), then average the two experts."""

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

    Experts have no shared base, so task vectors are the raw weights. For each
    parameter: trim smallest-magnitude entries, elect a sign per coordinate as
    ``sign(tau_a + tau_b)``, then average only the entries matching that sign.
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
    """Diagonal empirical Fisher for ``model`` (Matena & Raffel, 2022).

    Uses a Gaussian score sampler: ``loss = sum(out * eps)``, ``eps ~ N(0, I)``.
    ``E[grad^2]`` gives the per-parameter Fisher diagonal.

    ``generator`` seeds the eps draw. Pass a per-pair ``torch.Generator``
    (from ``StageContext.generator``) for order-independent estimates; ``None``
    uses the global RNG.
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
        # One per-pair generator feeds both Fisher estimates for order-independence
        # (A and B still get distinct eps draws); ctx=None uses the global RNG.
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
    """Knowledge-distillation merge: train a student (initialized from the
    average) to mimic both experts' outputs and hidden activations.

    ``epochs_large`` applies when data >= ``small_data_threshold`` samples,
    else ``epochs_small``. ``hidden_loss_weight`` scales the hidden-state loss.
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
            teacher_out = (0.5 * (mlp_a(data) + mlp_b(data))).float()
            teacher_hidden = (
                0.5 * (hidden_activations(mlp_a, data) + hidden_activations(mlp_b, data))
            ).float()

        orig_dtype = next(student.parameters()).dtype
        student.float()
        train_x = data.float()

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
                    batch_iter(train_x, batch_size),
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
        student.to(dtype=orig_dtype)
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
        fused = self.inner(mlp_a, mlp_b, train_data=train_data, create_new=create_new, ctx=ctx)
        if len(train_data) < 2 * mlp_a.down.out_features:
            return fused
        all_tokens = as_tensor(train_data)
        targets = torch.cat([mlp_a.forward(all_tokens), mlp_b.forward(all_tokens)], dim=-1)
        # Use the public ``refit_unmerge`` instead of writing ``.down``/``.U``
        # directly; numerically equivalent but respects encapsulation.
        fused.refit_unmerge(all_tokens, targets)
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


# Meta-wrappers produce derived methods (e.g. LS_<name> refits the unmerge layer).
# Every registered base method gets a wrapped variant automatically.
FUSION_META_WRAPPERS = [LeastSquaresAdapter]


def _is_meta(method: FusionMethod) -> bool:
    return isinstance(method, tuple(FUSION_META_WRAPPERS))


def apply_meta_wrappers(registry: Registry) -> None:
    """Register a wrapped variant of every base (non-meta) method that lacks one.

    Methods with ``wrappable=False`` are skipped (the pairwise ``LS_<name>``
    wrapper is meaningless for them). Idempotent: already-wrapped methods are
    skipped. Order-stable: wrappers appear in the same order as their base
    methods in the registry.
    """
    base_names = [
        name for name in registry
        if not _is_meta(registry[name]) and registry[name].wrappable
    ]
    for wrapper in FUSION_META_WRAPPERS:
        for name in base_names:
            wrapped = wrapper(registry[name])
            if wrapped.name not in registry:
                registry.register(wrapped.name, wrapped)


def _build_fusion_registry() -> Registry:
    """Build the canonical registry with the base methods in
    ``BASE_FUSION_METHODS`` order.

    Order is intentionally pinned: ``Best_*`` uses the first registered method
    as its fallback, and golden snapshots depend on this exact sequence. The
    ``LS_<name>`` meta-wrappers are NOT applied here -- ``common/__init__.py``
    calls :func:`apply_meta_wrappers` after ``common.baselines`` registers
    ``FrequencyWeightedAverage`` so it takes index 7 before the wrappers.
    """
    registry = Registry("fusion")
    for method in BASE_FUSION_METHODS:
        registry.register(method.name, method)
    return registry


# Registry that stages read from: name -> instance. Extend by registering
# a new FusionMethod subclass in-tree.
FUSION_REGISTRY: Registry = _build_fusion_registry()


def build_fusion_methods() -> dict[str, FusionMethod]:
    """Snapshot of ``FUSION_REGISTRY`` as a plain ordered dict (name -> instance).

    Kept for callers that need a mutable dict; stages read the registry directly.
    """
    return dict(FUSION_REGISTRY)
