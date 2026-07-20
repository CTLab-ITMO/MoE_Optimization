"""HC-SMoE clustering grouping + frequency-weighted merge (moved from
``common.grouping`` and ``common.fusion``).

Registers the ``hcsmoe_clustering`` grouping strategy and the
``FrequencyWeightedAverage`` fusion method on import.
"""
import copy
from typing import Optional

import numpy as np
import torch

from common import layout
from common.artifacts import CoActivationMatrix, Group, Partition
from common.baselines import signals
from common.fused_group import IDENTITY_RECONSTRUCTION
from common.fusion import FUSION_REGISTRY, FusionMethod
from common.grouping import (
    GROUPING_REGISTRY,
    Grouping,
    _PAIRING_DRAW_DEVICE,
    _require_data_dir,
)
from common.names import PairingMethod
from common.numeric import _default_device

DEFAULT_HCSMOE_SAMPLES = 4096
DEFAULT_HCSMOE_BATCH_SIZE = 512
# Default merge ratio: int(E * (1 - ratio)) clusters survive per layer.
DEFAULT_HCSMOE_RATIO = 0.5


def _average_linkage(distances: torch.Tensor) -> np.ndarray:
    """UPGMA (size-weighted average linkage) agglomerative clustering.

    ``distances`` is a symmetric ``[n, n]`` pairwise-distance matrix. Returns a
    SciPy-compatible linkage matrix of shape ``(n - 1, 4)`` -- each row
    ``[idx_i, idx_j, dist, size]`` with ``idx_i < idx_j``; the merge at step ``s``
    forms a new cluster indexed ``n + s`` (so :func:`_linkage_to_labels` decodes it).

    The Lance-Williams update for the merged cluster ``u = i union j`` is
    size-weighted -- ``d(k, u) = (n_i * d(k, i) + n_j * d(k, j)) / (n_i + n_j)`` --
    i.e. true UPGMA, NOT the unweighted ``(d(k, i) + d(k, j)) / 2`` (that is WPGMA).
    Accumulation is float64; ties break to the lexicographically smallest
    ``(idx_i, idx_j)`` cluster-id pair (SciPy leaves the tie order unspecified).

    Implemented without SciPy (absent from this environment); equivalent to
    ``scipy.cluster.hierarchy.linkage(squareform(D), method="average")``.
    """
    d = distances.detach().cpu().to(torch.float64).numpy()
    n = int(d.shape[0])
    total = 2 * n - 1
    # Working distances keyed by cluster id (originals 0..n-1, merges n..2n-2).
    work = np.full((total, total), np.inf, dtype=np.float64)
    work[:n, :n] = d
    np.fill_diagonal(work, np.inf)
    sizes = np.ones(total, dtype=np.float64)
    # ``active`` stays ascending: each step removes two ids and appends the new
    # id (n + step), which is strictly larger than every id created so far.
    active = list(range(n))
    linkage = np.empty((n - 1, 4), dtype=np.float64)
    for step in range(n - 1):
        best_dist = np.inf
        best_i = best_j = -1
        m = len(active)
        for a in range(m):
            i = active[a]
            for b in range(a + 1, m):
                j = active[b]  # i < j because ``active`` is ascending
                if work[i, j] < best_dist:  # strict -> first (lexicographic) min wins
                    best_dist = work[i, j]
                    best_i, best_j = i, j
        new_id = n + step
        sizes[new_id] = sizes[best_i] + sizes[best_j]
        linkage[step] = (best_i, best_j, best_dist, sizes[new_id])
        for k in active:
            if k == best_i or k == best_j:
                continue
            merged = (
                sizes[best_i] * work[best_i, k] + sizes[best_j] * work[best_j, k]
            ) / sizes[new_id]
            work[k, new_id] = merged
            work[new_id, k] = merged
        active.remove(best_i)
        active.remove(best_j)
        active.append(new_id)
    return linkage


def _linkage_to_labels(linkage_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
    """Convert a SciPy-format linkage matrix into flat 0-based cluster labels.

    Verbatim port of ``linkage_to_labels`` from the REAP reference
    (``reap/src/reap/cluster.py``, Apache-2.0): replay the recorded merges until
    ``num_clusters`` clusters remain, then remap the surviving cluster ids to
    consecutive integers ``0..num_clusters-1``.
    """
    n_samples = linkage_matrix.shape[0] + 1
    if num_clusters < 1 or num_clusters > n_samples:
        raise ValueError(
            f"Number of clusters must be between 1 and {n_samples} (the number of samples)."
        )
    labels = np.arange(n_samples)
    num_merges = n_samples - num_clusters
    for i in range(num_merges):
        cluster_1_id = int(linkage_matrix[i, 0])
        cluster_2_id = int(linkage_matrix[i, 1])
        new_cluster_id = n_samples + i
        mask = (labels == cluster_1_id) | (labels == cluster_2_id)
        labels[mask] = new_cluster_id
    unique_final_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_final_labels)}
    final_labels = np.array([label_map[label] for label in labels])
    return final_labels


class HCSMoEClusterGrouping(Grouping):
    """HC-SMoE agglomerative clustering as a k-agnostic grouping strategy.

    Signal: the per-expert characteristic activation -- the dense mean output
    vector (:func:`common.baselines.signals.compute_characteristic_activation`). Experts are
    clustered by pairwise Euclidean distance with UPGMA (size-weighted average
    linkage, :func:`_average_linkage`); the dendrogram is cut so
    ``int(n_experts * (1 - ratio))`` clusters remain per layer. Each cluster
    becomes one group (arbitrary size, including singletons, so ``k != 2`` from
    step 3 is required). Every expert is covered by exactly one group, so -- unlike
    REAP -- the stage-4 router is left intact (no mask).

    ``ratio`` / ``num_samples`` / ``batch_size`` are constructor params; register a
    variant (e.g. a different ratio) by registering a new instance.
    """

    def __init__(
        self,
        name: str,
        ratio: float = DEFAULT_HCSMOE_RATIO,
        num_samples: int = DEFAULT_HCSMOE_SAMPLES,
        batch_size: int = DEFAULT_HCSMOE_BATCH_SIZE,
    ):
        self.name = name
        self.ratio = ratio
        self.num_samples = num_samples
        self.batch_size = batch_size

    def partition(self, layer, ctx) -> Partition:
        n = ctx.n_experts if ctx is not None else None
        if n is None:
            raise ValueError(
                "HCSMoEClusterGrouping requires n_experts (set ctx.n_experts)"
            )
        device = (
            ctx.device if ctx is not None and ctx.device is not None else _default_device()
        )
        generator = (
            ctx.generator(
                "pairing",
                method=PairingMethod.HCSMOE_CLUSTERING,
                device=_PAIRING_DRAW_DEVICE,
            )
            if ctx is not None
            else None
        )
        features = signals.compute_characteristic_activation(
            layer,
            _require_data_dir(ctx),
            n,
            device,
            self.num_samples,
            self.batch_size,
            generator,
        )
        groups = self._cluster(features, n)
        return Partition(groups=groups, method=self.name)

    def _cluster(self, features: torch.Tensor, n_experts: int) -> tuple[Group, ...]:
        """Cluster experts by UPGMA over pairwise Euclidean CA distance.

        Returns groups ordered by ascending minimum member index (members
        ascending); every expert is covered by exactly one group.
        """
        n_clusters = max(1, min(n_experts, int(n_experts * (1.0 - self.ratio))))
        if n_experts <= 1 or n_clusters >= n_experts:
            return tuple(Group(members=(expert,)) for expert in range(n_experts))
        feats = features.to(torch.float64)
        # Direct pairwise Euclidean norm (matches the reference ``norm(x - y)``;
        # avoids the mm-based expansion's near-zero instability).
        distances = (feats.unsqueeze(1) - feats.unsqueeze(0)).norm(dim=-1)
        linkage = _average_linkage(distances)
        labels = _linkage_to_labels(linkage, n_clusters)
        buckets: dict[int, list[int]] = {}
        for expert, label in enumerate(labels.tolist()):
            buckets.setdefault(int(label), []).append(expert)
        ordered = sorted(buckets.values(), key=lambda members: members[0])
        return tuple(Group(members=tuple(members)) for members in ordered)


class FrequencyWeightedAverage(FusionMethod):
    """HC-SMoE merge: average a cluster's experts weighted by their top-k
    selection frequency (Chen et al., 2024, "Retraining-Free Merging of Sparse
    MoE via Hierarchical Clustering").

    Each member's weight is its raw top-k selection count over the training
    calibration set -- the diagonal of the train ``CoActivationMatrix`` (stage 1
    increments ``matrix[e, e]`` once per top-k membership; the identity to
    :func:`common.baselines.signals.compute_selection_frequencies` is asserted by test).
    The fused parameters are ``sum_i w_i * theta_i / sum_i w_i`` over gate/up/down.
    The reference weights with ``counts / total_tokens``, but the per-layer
    ``1 / total_tokens`` cancels in that ratio, so raw counts reproduce it exactly
    (up to a negligible epsilon the reference adds to the denominator).

    k-agnostic: a singleton cluster is identity. The router is left untouched --
    merging writes only expert weights (unlike REAP there is no router mask), so
    this is exactly the :class:`~common.fusion_layer.FusionLayer` semantics.

    Counts and group membership come from the artifact store via ``ctx``: the
    diagonal of the train CoActivationMatrix (``ctx.data_dir`` + ``ctx.layer_index``)
    and the member indices from that layer's :class:`~common.artifacts.Partition`
    (``ctx.group_index``). When the artifact context is absent (a direct call with
    no ``ctx``, e.g. the generic k-agnostic arity checks) or the whole group's
    counts are zero, the method falls back to equal weights -- a documented
    departure from the reference's degenerate zero merge.

    ``needs_data=False`` (weights come from counts, not ``train_data`` -- an empty
    ``train_data`` must NOT trigger the Average fallback in ``__call__``);
    ``wrappable=False`` (the pairwise ``LS_`` wrapper is meaningless here).
    """

    name = "FrequencyWeightedAverage"
    needs_data = False
    supported_arity = None
    wrappable = False
    reconstruction = IDENTITY_RECONSTRUCTION

    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        return self.merge_group((mlp_a, mlp_b), train_data, create_new, ctx)

    def merge_group(self, mlps, train_data, create_new=True, ctx=None):
        with torch.no_grad():
            new_mlp = copy.deepcopy(mlps[0]) if create_new else mlps[0]
            if len(mlps) == 1:
                return new_mlp
            weights = self._member_weights(len(mlps), ctx)
            states = [mlp.state_dict() for mlp in mlps]
            new_dict = {}
            for key in states[0]:
                total = states[0][key].float() * weights[0]
                for state, weight in zip(states[1:], weights[1:]):
                    total = total + state[key].float() * weight
                new_dict[key] = total.to(dtype=states[0][key].dtype)
            new_mlp.load_state_dict(new_dict)
        return new_mlp

    def _member_weights(self, count: int, ctx) -> torch.Tensor:
        """Per-member merge weights (length ``count``, float32, summing to 1).

        Frequency counts normalized to a convex combination; uniform weights when
        the artifact context is absent or the group's counts are all zero.
        """
        counts = self._frequency_counts(count, ctx)
        if counts is None or float(counts.sum()) <= 0.0:
            return torch.full((count,), 1.0 / count, dtype=torch.float32)
        counts = counts.to(torch.float32)
        return counts / counts.sum()

    def _frequency_counts(self, count: int, ctx) -> Optional[torch.Tensor]:
        """Raw top-k selection counts for the group's members, or ``None`` when the
        artifact context (data_dir + layer_index + group_index) is unavailable.

        Reads the diagonal of the train CoActivationMatrix and the member indices
        from the layer's Partition (variant A). Raises if the Partition group's
        size disagrees with the number of experts passed to the merge.
        """
        if (
            ctx is None
            or ctx.data_dir is None
            or ctx.layer_index is None
            or ctx.group_index is None
        ):
            return None
        frequencies = CoActivationMatrix.load(
            layout.activation_matrix_path(
                ctx.data_dir, layout.TRAIN_SPLIT, ctx.layer_index
            )
        ).matrix.diagonal()
        members = Partition.load(
            layout.group_path(ctx.data_dir, ctx.layer_index)
        ).groups[ctx.group_index].members
        if len(members) != count:
            raise ValueError(
                f"FrequencyWeightedAverage: layer {ctx.layer_index} group "
                f"{ctx.group_index} has {len(members)} members but the merge "
                f"received {count} experts"
            )
        return frequencies[list(members)]


GROUPING_REGISTRY.register(
    str(PairingMethod.HCSMOE_CLUSTERING),
    HCSMoEClusterGrouping(str(PairingMethod.HCSMOE_CLUSTERING)),
)
FUSION_REGISTRY.register(FrequencyWeightedAverage().name, FrequencyWeightedAverage())
