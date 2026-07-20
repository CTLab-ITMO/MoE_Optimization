"""REAP / HC-SMoE baseline strategies (grouping + fusion), registered on import."""
from common.baselines import signals  # noqa: F401
from common.baselines import reap  # noqa: F401  registers reap_saliency grouping
from common.baselines import hcsmoe  # noqa: F401  registers hcsmoe_clustering grouping + FrequencyWeightedAverage
