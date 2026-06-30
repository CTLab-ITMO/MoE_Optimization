"""Expert grouping abstracted on two independent axes.

* :class:`AffinityProvider` -- signal: an ``EXPERTS x EXPERTS`` affinity matrix for a layer.
  In-tree: :class:`CoActivationAffinity` (gate co-activation), :class:`OutputSimilarityAffinity` (output cosine).
* :class:`Partitioner` -- algorithm: turns an affinity matrix into a list of groups.
  In-tree: :class:`MatchingPartitioner` (max-weight matching), :class:`RandomPartitioner`.

:class:`PairGrouping` composes one affinity with one partitioner to produce disjoint pairs (the ``k == 2`` case).

Stochastic strategies draw from the per-layer :class:`torch.Generator` in :class:`~common.context.StageContext`
to stay order-invariant. Library code only -- never imports an experiment ``config``.
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

# Default knobs for output_similarity; passed as explicit constructor params to OutputSimilarityAffinity.
DEFAULT_OUTPUT_SIMILARITY_SAMPLES = 4096
DEFAULT_OUTPUT_SIMILARITY_BATCH_SIZE = 512

# Pairing draws (randperm, subsample) operate on CPU tensors and build Python lists,
# so the generator must be CPU even when ctx.device is cuda -- a CUDA generator
# would raise inside torch.randperm.
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
    """Signal axis: produces an ``EXPERTS x EXPERTS`` affinity matrix for a layer.

    Entry ``(i, j)`` scores how well experts ``i`` and ``j`` fit together; consumed by a :class:`Partitioner`.
    """

    @abstractmethod
    def affinity(
        self, layer: int, n_experts: Optional[int], ctx: Optional[StageContext]
    ) -> torch.Tensor:
        """Return the affinity matrix for ``layer`` (read via ``ctx``)."""
        raise NotImplementedError()


class CoActivationAffinity(AffinityProvider):
    """Gate co-activation signal; delegates to :func:`common.pairing.compute_activation_correlation_matrix`."""

    def affinity(self, layer, n_experts, ctx):
        return pairing.compute_activation_correlation_matrix(layer, _require_data_dir(ctx))


class OutputSimilarityAffinity(AffinityProvider):
    """Expert-output cosine-similarity signal.

    ``num_samples``/``batch_size`` are constructor params; configure a variant by registering a new instance.
    The subsample draw uses the per-layer generator from ``ctx`` for order-invariance.
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
    """Algorithm axis: turns an affinity matrix into a list of expert groups.

    Signal-free algorithms (e.g., random) pass ``None`` for the affinity.
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
    """Max-weight matching -> disjoint pairs.

    Delegates to :func:`common.pairing.build_matches`, which fixes group order and tuple orientation.
    """

    def partition(self, affinity, n_experts, ctx):
        if affinity is None:
            raise ValueError("MatchingPartitioner requires an affinity matrix")
        return pairing.build_matches(affinity)


class RandomPartitioner(Partitioner):
    """Random disjoint pairing (signal-free).

    Uses the per-layer generator from ``ctx`` for reproducible, order-invariant splits.
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

    Public contract: :meth:`partition` returns a :class:`~common.artifacts.Partition`.
    Add a new strategy by subclassing and registering.
    """

    name: str

    @abstractmethod
    def partition(self, layer: int, ctx: Optional[StageContext]) -> Partition:
        """Partition ``layer``'s experts into a :class:`Partition`."""
        raise NotImplementedError()


class PairGrouping(Grouping):
    """Disjoint-pairs grouping: one :class:`AffinityProvider` combined with one :class:`Partitioner`.

    The ``k == 2`` specialization -- every group is a pair. The three in-tree strategies
    differ only in which affinity/partitioner pair they use.
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

        ``n_experts`` defaults to ``ctx.n_experts``. Random pairing requires an explicit count;
        matching infers it from the affinity matrix shape if not provided.
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
    """Build the canonical grouping registry (name -> :class:`Grouping` instance).

    Registered in :class:`~common.names.PairingMethod` order. Extend by calling ``register`` with a new subclass.
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


# Registry the grouping stage reads from; extend by registering a new subclass.
GROUPING_REGISTRY: Registry = _build_grouping_registry()
