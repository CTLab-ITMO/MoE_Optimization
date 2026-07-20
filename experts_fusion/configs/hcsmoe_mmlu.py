"""HC-SMoE baseline: MMLU x hierarchical-clustering expert merging.

HC-SMoE (hierarchical clustering of Mixture-of-Experts) ports as a grouping
strategy that agglomeratively clusters experts on their characteristic activations
(UPGMA / average-linkage) into variable-size groups, then merges each cluster with
the ``FrequencyWeightedAverage`` fusion method (frequency-weighted mean of the
gate/up/down weights). The router is left untouched -- exactly our ``FusionLayer``
semantics, so there is no router mask (unlike REAP). ``FILTER_FUSIONS`` and
``INSPECT_FUSION_METHOD`` narrow to ``FrequencyWeightedAverage`` (the base default
``ZipIt`` would fail stage 4's INSPECT-vs-produced check). The cluster-count ratio
(0.5) is fixed at registration of the ``hcsmoe_clustering`` instance, not a config
knob.

Reuses the E1 ablation surface from ``_base`` (model, DI adapters, sizes, MMLU
dataset triple) so the baseline numbers are directly comparable to our fusion.
Run with ``python -m experts_fusion.run --config hcsmoe_mmlu``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import __all__  # noqa: F401 -- re-export config surface
from common.baselines.hcsmoe import FrequencyWeightedAverage
from common.names import PairingMethod

DATA_DIR = "./data/ef_hcsmoe_mmlu"
PAIRING_METHOD = PairingMethod.HCSMOE_CLUSTERING
# HC-SMoE clusters merge via frequency-weighted averaging; drop every other method
# so the baseline runs only that path (Best_* are already suppressed by _base's
# FILTER_METRICS=[]).
FILTER_FUSIONS = [FrequencyWeightedAverage.name]
INSPECT_FUSION_METHOD = FrequencyWeightedAverage.name
