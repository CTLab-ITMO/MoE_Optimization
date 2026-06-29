"""Grouping as a two-axis, k-agnostic abstraction (§2.3, §3.7, §4.3, §4.5).

The old pipeline picked expert pairs through a single ``get_pairs`` dispatcher --
a chain of ``if method == ...`` branches. This module replaces that with a
registry of :class:`Grouping` strategies decomposed along **two independent
axes** (§3.7), so a new signal and a new algorithm combine without either
touching the other:

* :class:`AffinityProvider` -- the *signal*: an ``EXPERTS x EXPERTS`` affinity
  matrix for a layer. In-tree signals: :class:`CoActivationAffinity` (gate
  co-activation) and :class:`OutputSimilarityAffinity` (expert-output cosine).
* :class:`Partitioner` -- the *algorithm*: turns an affinity (or just the expert
  count) into a list of groups. In-tree algorithms: :class:`MatchingPartitioner`
  (max-weight matching) and :class:`RandomPartitioner`. k-means / spectral are
  deliberately left to other researchers (over-engineering for now, S6).

:class:`PairGrouping` is the generic matching-partitioner-over-an-affinity
specialization that yields disjoint pairs (a :class:`~common.artifacts.Partition`
whose every group has ``size == 2``); a pair is the ``k == 2`` case of the
k-agnostic :meth:`Grouping.partition` contract.

The three registered strategies reproduce the legacy ``get_pairs`` methods
exactly -- same group order and tuple orientation -- so the default
``activation_correlation + matching`` is behaviour-equivalent to the old default
(§2.6). They reuse the same primitives (:func:`common.pairing.build_matches`,
:func:`common.pairing.compute_activation_correlation_matrix`,
:func:`common.pairing.compute_output_similarity_matrix`,
:func:`common.pairing.random_pairs`), so identity is by construction; the
stochastic ones (``random`` / ``output_similarity``) draw from the per-layer
:class:`torch.Generator` carried by :class:`~common.context.StageContext` (S4),
making them order-invariant.

Per §3.7 the variadic ``**kwargs`` lives only on the *abstract* signatures (for
forward-compatible contract freezing); concrete implementations take explicit,
named parameters so a typo is caught rather than silently swallowed.

Library code only -- this module never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from common import pairing
from common.artifacts import Partition
from common.context import StageContext
from common.names import PairingMethod
from common.numeric import _default_device
from common.registry import Registry

# Entry-point group a third-party package uses to advertise a grouping plugin
# (pulled in only on an explicit ``GROUPING_REGISTRY.load_entry_points`` call --
# the canonical registry stays pinned in-tree; see :mod:`common.registry`).
GROUPING_ENTRY_POINT_GROUP = "moe.grouping"

# Default ``output_similarity`` knobs (§4.5): exposed as explicit
# ``OutputSimilarityAffinity.__init__`` parameters rather than pinned inside the
# matrix computation, so a strategy variant configures them by construction.
DEFAULT_OUTPUT_SIMILARITY_SAMPLES = 4096
DEFAULT_OUTPUT_SIMILARITY_BATCH_SIZE = 512

# The pairing draws below (``torch.randperm`` for random pairing and the
# output_similarity subsample) are CPU operations -- they index CPU tensors and
# build a Python pair list -- so their generator MUST live on the CPU regardless
# of the compute device. Deriving it from ``ctx.device`` (which is ``cuda`` on the
# GPU box) makes ``torch.randperm(..., generator=cuda_gen)`` raise. (Unlike
# Fisher's ``randn_like`` on a model tensor, which correctly draws on ``ctx.device``.)
_PAIRING_DRAW_DEVICE = torch.device("cpu")


def _require_data_dir(ctx: Optional[StageContext]) -> str:
    """Return ``ctx.data_dir`` or raise a clear error if it is unset."""
    data_dir = ctx.data_dir if ctx is not None else None
    if data_dir is None:
        raise ValueError(
            "grouping requires the artifact root: set ctx.data_dir on the "
            "StageContext passed to Grouping.partition"
        )
    return data_dir


# --------------------------------------------------------------------------- #
# Signal axis: AffinityProvider.
# --------------------------------------------------------------------------- #
class AffinityProvider(ABC):
    """Signal axis: an ``EXPERTS x EXPERTS`` affinity matrix for a layer.

    The matrix entry ``(i, j)`` scores how much experts ``i`` and ``j`` should be
    grouped together; a :class:`Partitioner` consumes it. ``**kwargs`` is the
    forward-compatible interface slot only (§3.7); concrete providers take
    explicit named parameters.
    """

    @abstractmethod
    def affinity(
        self, layer: int, n_experts: Optional[int], ctx: Optional[StageContext], **kwargs
    ) -> torch.Tensor:
        """Return the affinity matrix for ``layer`` (read via ``ctx``)."""
        raise NotImplementedError()


class CoActivationAffinity(AffinityProvider):
    """Gate co-activation signal: the normalized activation matrix (stage 1).

    Reproduces the ``activation_correlation`` signal of the legacy ``get_pairs``
    exactly -- it delegates to the same
    :func:`common.pairing.compute_activation_correlation_matrix`.
    """

    def affinity(self, layer, n_experts, ctx):
        return pairing.compute_activation_correlation_matrix(layer, _require_data_dir(ctx))


class OutputSimilarityAffinity(AffinityProvider):
    """Expert-output cosine-similarity signal (stages 3 + 4).

    The ``num_samples`` / ``batch_size`` knobs (§4.5) are explicit construction
    parameters of the strategy, so a variant is configured by registering a new
    instance rather than by reaching into the matrix computation. The subsample
    draw uses the order-invariant per-layer generator from ``ctx`` (S4).
    """

    def __init__(
        self,
        num_samples: int = DEFAULT_OUTPUT_SIMILARITY_SAMPLES,
        batch_size: int = DEFAULT_OUTPUT_SIMILARITY_BATCH_SIZE,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def affinity(self, layer, n_experts, ctx):
        if n_experts is None:
            raise ValueError("OutputSimilarityAffinity requires n_experts (set ctx.n_experts)")
        device = ctx.device if ctx is not None and ctx.device is not None else _default_device()
        generator = (
            ctx.generator(
                "pairing",
                method=PairingMethod.OUTPUT_SIMILARITY,
                device=_PAIRING_DRAW_DEVICE,
            )
            if ctx is not None
            else None
        )
        return pairing.compute_output_similarity_matrix(
            layer,
            _require_data_dir(ctx),
            n_experts,
            device,
            self.num_samples,
            self.batch_size,
            generator,
        )


# --------------------------------------------------------------------------- #
# Algorithm axis: Partitioner.
# --------------------------------------------------------------------------- #
class Partitioner(ABC):
    """Algorithm axis: build a list of expert groups.

    Consumes an affinity matrix (``None`` for signal-free algorithms such as
    random) and the expert count, and returns a list of member tuples. ``**kwargs``
    is the forward-compatible interface slot only (§3.7).
    """

    @abstractmethod
    def partition(
        self,
        affinity: Optional[torch.Tensor],
        n_experts: Optional[int],
        ctx: Optional[StageContext],
        **kwargs,
    ) -> list[tuple[int, ...]]:
        """Return the list of groups (each an ordered tuple of expert indices)."""
        raise NotImplementedError()


class MatchingPartitioner(Partitioner):
    """Max-weight matching over an affinity matrix -> disjoint pairs.

    Delegates to :func:`common.pairing.build_matches`, so the group order and the
    tuple orientation ``(pair[0], pair[1])`` are identical to the legacy path.
    """

    def partition(self, affinity, n_experts, ctx):
        if affinity is None:
            raise ValueError("MatchingPartitioner requires an affinity matrix")
        return pairing.build_matches(affinity)


class RandomPartitioner(Partitioner):
    """Random disjoint pairing of the experts (signal-free).

    Delegates to :func:`common.pairing.random_pairs` with the order-invariant
    per-layer generator from ``ctx`` (S4), so the split is reproducible and
    independent of pipeline traversal order.
    """

    def partition(self, affinity, n_experts, ctx):
        if n_experts is None:
            raise ValueError("RandomPartitioner requires n_experts (set ctx.n_experts)")
        generator = (
            ctx.generator("pairing", method=PairingMethod.RANDOM, device=_PAIRING_DRAW_DEVICE)
            if ctx is not None
            else None
        )
        return pairing.random_pairs(n_experts, generator=generator)


# --------------------------------------------------------------------------- #
# Grouping (top-level strategy).
# --------------------------------------------------------------------------- #
class Grouping(ABC):
    """k-agnostic grouping strategy: partition a layer's experts into groups.

    The frozen public contract is :meth:`partition`, returning a
    :class:`~common.artifacts.Partition` of arbitrarily sized groups. A new
    grouping (clusters, overlapping groups, hierarchies) is a new subclass /
    registry entry, never a change to this base. ``**kwargs`` is the
    forward-compatible interface slot only (§3.7).
    """

    name: str

    @abstractmethod
    def partition(self, layer: int, ctx: Optional[StageContext], **kwargs) -> Partition:
        """Partition ``layer``'s experts into a :class:`Partition`."""
        raise NotImplementedError()


class PairGrouping(Grouping):
    """Disjoint-pairs grouping: a :class:`Partitioner` over an :class:`AffinityProvider`.

    The generic ``k == 2`` specialization (§2.3): every produced group is a pair.
    Composing one affinity with one partitioner is the whole "signal x algorithm"
    decomposition; the three in-tree strategies differ only in which pair they are
    built from.
    """

    def __init__(
        self,
        name: str,
        affinity: Optional[AffinityProvider],
        partitioner: Partitioner,
    ):
        self.name = name
        self.affinity = affinity
        self.partitioner = partitioner

    def partition(self, layer, ctx, *, n_experts: Optional[int] = None) -> Partition:
        """Return the pairwise :class:`Partition` for ``layer``.

        ``n_experts`` defaults to ``ctx.n_experts``; the resolved count is handed
        to the affinity provider and the partitioner. The signal-free random
        algorithm needs it explicitly; for matching it is inferred from the
        affinity matrix when otherwise unknown.
        """
        n = n_experts if n_experts is not None else (ctx.n_experts if ctx is not None else None)
        matrix = self.affinity.affinity(layer, n, ctx) if self.affinity is not None else None
        if n is None and matrix is not None:
            n = int(matrix.shape[0])
        pairs = self.partitioner.partition(matrix, n, ctx)
        return Partition.from_pairs(pairs, method=self.name)


# --------------------------------------------------------------------------- #
# Canonical grouping registry (pinned order; golden ``pairing_methods`` snapshot).
# --------------------------------------------------------------------------- #
def _build_grouping_registry() -> Registry:
    """The canonical grouping registry: name -> :class:`Grouping` instance.

    Pinned to the :class:`~common.names.PairingMethod` order so the golden
    ``pairing_methods`` snapshot is an exact match and config validation of
    ``PAIRING_METHOD`` is driven from :meth:`Registry.names` (§1.6/§4.3).
    Extensible without a fork via ``register`` / ``load_entry_points``.
    """
    registry = Registry("grouping")
    registry.register(
        str(PairingMethod.ACTIVATION_CORRELATION),
        PairGrouping(
            str(PairingMethod.ACTIVATION_CORRELATION),
            CoActivationAffinity(),
            MatchingPartitioner(),
        ),
    )
    registry.register(
        str(PairingMethod.RANDOM),
        PairGrouping(str(PairingMethod.RANDOM), None, RandomPartitioner()),
    )
    registry.register(
        str(PairingMethod.OUTPUT_SIMILARITY),
        PairGrouping(
            str(PairingMethod.OUTPUT_SIMILARITY),
            OutputSimilarityAffinity(),
            MatchingPartitioner(),
        ),
    )
    return registry


# The single source of truth the grouping stage reads from (§3.7): a
# name -> Grouping Mapping, extensible via ``register`` / ``load_entry_points``.
GROUPING_REGISTRY: Registry = _build_grouping_registry()
