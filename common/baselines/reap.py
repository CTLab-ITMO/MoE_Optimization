"""REAP saliency-pruning grouping (moved from ``common.grouping``).

Registers the ``reap_saliency`` grouping strategy on import.
"""
import torch

from common.artifacts import Group, Partition
from common.baselines import signals
from common.grouping import (
    GROUPING_REGISTRY,
    Grouping,
    _PAIRING_DRAW_DEVICE,
    _require_data_dir,
)
from common.names import PairingMethod
from common.numeric import _default_device
from common.router import RouterAdapter

DEFAULT_SALIENCY_SAMPLES = 4096
DEFAULT_SALIENCY_BATCH_SIZE = 512
# Default prune ratio: fraction of experts dropped per layer (n_pruned = int(E * ratio)).
DEFAULT_PRUNE_RATIO = 0.5


class SaliencyPruneGrouping(Grouping):
    """REAP router-weighted saliency pruning as a k-agnostic grouping strategy.

    Signal: the per-expert REAP saliency vector
    (:func:`common.baselines.signals.compute_reap_saliency`). ``n_pruned = int(n_experts * ratio)``
    lowest-saliency experts are dropped per layer, on the raw saliency with no
    per-layer normalization (matching the reference budget). Tie-break is explicit
    and deterministic -- ``torch.argsort(saliency, stable=True)[:n_pruned]`` drops
    the lowest index first among equal scores; a never-routed expert scores 0 and
    is dropped first. Survivors become singleton (``k == 1``) groups; dropped
    experts belong to no group, so the stage-4 layer builder masks them out of the
    router (step 5.2).

    ``ratio`` / ``num_samples`` / ``batch_size`` are constructor params; register a
    variant (e.g. a different ratio) by registering a new instance.
    """

    def __init__(
        self,
        name: str,
        ratio: float = DEFAULT_PRUNE_RATIO,
        num_samples: int = DEFAULT_SALIENCY_SAMPLES,
        batch_size: int = DEFAULT_SALIENCY_BATCH_SIZE,
    ):
        self.name = name
        self.ratio = ratio
        self.num_samples = num_samples
        self.batch_size = batch_size

    def partition(self, layer, ctx) -> Partition:
        n = ctx.n_experts if ctx is not None else None
        if n is None:
            raise ValueError(
                "SaliencyPruneGrouping requires n_experts (set ctx.n_experts)"
            )
        device = (
            ctx.device if ctx is not None and ctx.device is not None else _default_device()
        )
        generator = (
            ctx.generator(
                "pairing",
                method=PairingMethod.REAP_SALIENCY,
                device=_PAIRING_DRAW_DEVICE,
            )
            if ctx is not None
            else None
        )
        saliency = signals.compute_reap_saliency(
            layer,
            _require_data_dir(ctx),
            n,
            device,
            self.num_samples,
            self.batch_size,
            generator,
        )
        survivors = self._survivors(saliency, n)
        groups = tuple(Group(members=(int(expert),)) for expert in survivors)
        return Partition(groups=groups, method=self.name)

    def _survivors(self, saliency: torch.Tensor, n_experts: int) -> list[int]:
        """Return the kept (survivor) expert indices in ascending order.

        Drops ``int(n_experts * ratio)`` lowest-saliency experts. Ties break to the
        lowest index via a stable argsort (a never-routed expert, saliency 0, sorts
        first and is dropped first). Survivors are ascending so the per-group file
        layout is deterministic.
        """
        n_pruned = int(n_experts * self.ratio)
        dropped = {
            int(expert)
            for expert in torch.argsort(saliency, stable=True)[:n_pruned].tolist()
        }
        return [expert for expert in range(n_experts) if expert not in dropped]


class MaskedRouterAdapter(RouterAdapter):
    """Prune experts out of the router via an additive -inf logit mask (REAP).

    Wraps a base RouterAdapter. Dropped experts get a -inf additive logit bias
    before the base adapter's softmax + top-k, so top-k reselects among survivors
    and survivor weights are the softmax renormalized over the survivor logits --
    the numeric equivalent of deleting the gate rows (REAP mask-and-renorm). A tuple
    gate's ready weights/selected were computed on the UNMASKED logits, so they are
    discarded: the decision is always recomputed from the masked bare logits through
    the base adapter's bare-logits path (which for Qwen1.5, normalize_topk=False,
    leaves the top-k slice unnormalized; a Mixtral base would renormalize -- the base
    adapter's own policy applies).
    """

    def __init__(self, base, logit_mask):
        self.base = base
        self.logit_mask = logit_mask  # shape (num_experts,), 0 for survivors, -inf for dropped

    @classmethod
    def from_uncovered(cls, base, uncovered, num_experts, top_k=None):
        """Build from a partition's uncovered (dropped) expert indices.

        Validates: at least one uncovered expert, and survivors >= top_k (so routing
        can still select). Returns a mask that is -inf exactly on the uncovered set.
        """
        uncovered = sorted({int(e) for e in uncovered})
        if not uncovered:
            raise ValueError("MaskedRouterAdapter requires at least one uncovered expert")
        survivors = num_experts - len(uncovered)
        if top_k is not None and survivors < top_k:
            raise ValueError(
                f"masked router leaves {survivors} survivors, fewer than top_k={top_k}"
            )
        mask = torch.zeros(num_experts)
        mask[uncovered] = float("-inf")
        return cls(base, mask)

    def route(self, gate_output, *, top_k):
        logits = gate_output[0] if isinstance(gate_output, tuple) else gate_output
        masked = logits + self.logit_mask.to(dtype=logits.dtype, device=logits.device)
        return self.base.route(masked, top_k=top_k)


GROUPING_REGISTRY.register(
    str(PairingMethod.REAP_SALIENCY),
    SaliencyPruneGrouping(str(PairingMethod.REAP_SALIENCY)),
)
