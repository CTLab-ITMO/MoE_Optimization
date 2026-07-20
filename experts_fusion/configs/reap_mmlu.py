"""REAP baseline: MMLU x router-weighted expert-activation pruning.

REAP (router-weighted expert activation pruning) ports as a grouping strategy that
keeps the high-saliency experts as singletons and drops the rest; the layer masks
the pruned experts' router logits to -inf (mask-and-renorm, not redirect). The
survivors' "fusion" is the identity ``Pruning`` method, so ``FILTER_FUSIONS`` and
``INSPECT_FUSION_METHOD`` both narrow to ``Pruning`` (the base default ``ZipIt``
would fail stage 4's INSPECT-vs-produced check). The prune ratio (0.5) is fixed at
registration of the ``reap_saliency`` instance, not a config knob.

Reuses the E1 ablation surface from ``_base`` (model, DI adapters, sizes, MMLU
dataset triple) so the baseline numbers are directly comparable to our fusion.
Run with ``python -m experts_fusion.run --config reap_mmlu``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import __all__  # noqa: F401 -- re-export config surface
from common.fusion import PruningFusion
from common.names import PairingMethod

DATA_DIR = "./data/ef_reap_mmlu"
PAIRING_METHOD = PairingMethod.REAP_SALIENCY
# REAP survivors are singletons fused by identity; drop every other method so the
# baseline runs only the pruning path (Best_* are already suppressed by _base's
# FILTER_METRICS=[]).
FILTER_FUSIONS = [PruningFusion.name]
INSPECT_FUSION_METHOD = PruningFusion.name
