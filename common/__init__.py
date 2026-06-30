"""Reusable MoE expert-fusion primitives.

The extension interfaces and their registries are re-exported here so they can be
imported straight from ``common`` (``from common import FusionMethod,
FUSION_REGISTRY``); the implementations live in the submodules. See
``common/EXTENDING.md`` for how to add your own strategy in-tree.
"""
from common.context import StageContext
from common.registry import Registry, require_all_registered, require_registered
from common.fusion import FUSION_REGISTRY, FusionMethod
from common.grouping import (
    GROUPING_REGISTRY,
    AffinityProvider,
    Grouping,
    Partitioner,
)
from common.metrics import METRIC_REGISTRY, Metric
from common.dataset_schema import SCHEMA_REGISTRY, DatasetSchema
from common.mlp import ACTIVATION_REGISTRY, MLP, AdvancedMLP
from common.fused_group import FusedExpert, FusedGroup
from common.router import RouterAdapter
from common.expert_adapter import ExpertAdapter, SharedExpertAdapter
from common.qa_parser import QAParser
from common.artifacts import CoActivationMatrix, GroupActivationIndex, Partition
from common.fusion_layer import FusionLayer

__all__ = [
    # dynamic state + registry machinery
    "StageContext",
    "Registry",
    "require_registered",
    "require_all_registered",
    # extension axes: interface + registry
    "FusionMethod",
    "FUSION_REGISTRY",
    "Grouping",
    "AffinityProvider",
    "Partitioner",
    "GROUPING_REGISTRY",
    "Metric",
    "METRIC_REGISTRY",
    "DatasetSchema",
    "SCHEMA_REGISTRY",
    "ACTIVATION_REGISTRY",
    # model-family adapters (DI)
    "RouterAdapter",
    "ExpertAdapter",
    "SharedExpertAdapter",
    "QAParser",
    # runtime + artifact types
    "MLP",
    "AdvancedMLP",
    "FusedGroup",
    "FusedExpert",
    "FusionLayer",
    "Partition",
    "GroupActivationIndex",
    "CoActivationMatrix",
]
