"""Grouping as a two-axis, k-agnostic abstraction.

Expert grouping is decomposed along two independent axes, so a new signal and a
new algorithm combine without either touching the other:

* :class:`AffinityProvider` -- the *signal*: an ``EXPERTS x EXPERTS`` affinity
  matrix for a layer. In-tree signals: :class:`CoActivationAffinity` (gate
  co-activation) and :class:`OutputSimilarityAffinity` (expert-output cosine).
* :class:`Partitioner` -- the *algorithm*: turns an affinity (or just the expert
  count) into a list of groups. In-tree algorithms: :class:`MatchingPartitioner`
  (max-weight matching) and :class:`RandomPartitioner`. k-means / spectral are
  deliberately left to other researchers.

:class:`PairGrouping` is the generic matching-partitioner-over-an-affinity
specialization that yields disjoint pairs (a :class:`~common.artifacts.Partition`
whose every group has ``size == 2``); a pair is the ``k == 2`` case of the
k-agnostic :meth:`Grouping.partition` contract.

Each registered strategy reuses the shared primitives
(:func:`common.pairing.build_matches`,
:func:`common.pairing.compute_activation_correlation_matrix`,
:func:`common.pairing.compute_output_similarity_matrix`,
:func:`common.pairing.random_pairs`); the stochastic ones
(``random`` / ``output_similarity``) draw from the per-layer
:class:`torch.Generator` carried by :class:`~common.context.StageContext`,
making them order-invariant.

Concrete implementations take explicit, named parameters so a typo is caught
rather than silently swallowed.

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

# Default ``output_similarity`` knobs: exposed as explicit
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
    grouped together; a :class:`Partitioner` consumes it. Concrete providers take
    explicit named parameters.
    """

    @abstractmethod
    def affinity(
        self, layer: int, n_experts: Optional[int], ctx: Optional[StageContext]
    ) -> torch.Tensor:
        """Return the affinity matrix for ``layer`` (read via ``ctx``)."""
        raise NotImplementedError()


class CoActivationAffinity(AffinityProvider):
    """Gate co-activation signal: the normalized activation matrix.

    Delegates to
    :func:`common.pairing.compute_activation_correlation_matrix`.
    """

    def affinity(self, layer, n_experts, ctx):
        return pairing.compute_activation_correlation_matrix(layer, _require_data_dir(ctx))


class OutputSimilarityAffinity(AffinityProvider):
    """Expert-output cosine-similarity signal.

    The ``num_samples`` / ``batch_size`` knobs are explicit construction
    parameters of the strategy, so a variant is configured by registering a new
    instance rather than by reaching into the matrix computation. The subsample
    draw uses the order-invariant per-layer generator from ``ctx``.
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
    random) and the expert count, and returns a list of member tuples.
    """

    @abstractmethod
    def partition(
        self,
        affinity: Optional[torch.Tensor],
        n_experts: Optional[int],
        ctx: Optional[StageContext],
    ) -> list[tuple[int, ...]]:
        """Return the list of groups (each an ordered tuple of expert indices)."""
        raise NotImplementedError()


class MatchingPartitioner(Partitioner):
    """Max-weight matching over an affinity matrix -> disjoint pairs.

    Delegates to :func:`common.pairing.build_matches`, which fixes the group
    order and the tuple orientation ``(pair[0], pair[1])``.
    """

    def partition(self, affinity, n_experts, ctx):
        if affinity is None:
            raise ValueError("MatchingPartitioner requires an affinity matrix")
        return pairing.build_matches(affinity)


class RandomPartitioner(Partitioner):
    """Random disjoint pairing of the experts (signal-free).

    Delegates to :func:`common.pairing.random_pairs` with the order-invariant
    per-layer generator from ``ctx``, so the split is reproducible and
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
    registry entry.
    """

    name: str

    @abstractmethod
    def partition(self, layer: int, ctx: Optional[StageContext]) -> Partition:
        """Partition ``layer``'s experts into a :class:`Partition`."""
        raise NotImplementedError()


class PairGrouping(Grouping):
    """Disjoint-pairs grouping: a :class:`Partitioner` over an :class:`AffinityProvider`.

    The generic ``k == 2`` specialization: every produced group is a pair.
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
    ``PAIRING_METHOD`` is driven from :meth:`Registry.names`.
    Extended by registering a new subclass in-tree via ``register``.
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


# The single source of truth the grouping stage reads from: a
# name -> Grouping Mapping, extended by registering a new subclass in-tree.
GROUPING_REGISTRY: Registry = _build_grouping_registry()
