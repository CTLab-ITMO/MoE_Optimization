from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple


class TraceSource(str, Enum):
    ARTIFACT = "artifact"
    HF_DATASET = "hf_dataset"
    SYNTHETIC = "synthetic"


class ExperimentStageId(str, Enum):
    RUNTIME = "runtime"
    COLLECT_TRACES = "collect_traces"
    H1_DIAGNOSTICS = "h1_diagnostics"
    H2_BIAS = "h2_bias"
    H3_ROUTING = "h3_routing"
    BASELINE_EVALUATION = "baseline_evaluation"
    CANDIDATE_EVALUATION = "candidate_evaluation"
    ARTIFACTS = "artifacts"


class CandidateAction(str, Enum):
    KEEP = "keep"
    PRUNE = "prune"
    MERGE = "merge"
    ABSTAIN = "abstain"


class CandidateMethod(str, Enum):
    RANDOM_MERGE = "random_merge"
    USAGE_PRUNE = "usage_prune"
    FUNCTIONAL_COSINE = "functional_cosine"
    RMT_GEOMETRY = "rmt_geometry"


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_name: str
    stages: Tuple[ExperimentStageId, ...]


@dataclass(frozen=True)
class CalibrationTraceContract:
    source: TraceSource
    layer: int
    split: str
    hidden_states_path: str
    routing_probs_path: str
    token_batches_path: str
    rows: int
    tokens: int
    sha256: str


@dataclass(frozen=True)
class CandidatePlanContract:
    candidate_id: str
    layer: int
    method: CandidateMethod
    action: CandidateAction
    compression_fraction: float
    groups: Tuple[Tuple[int, ...], ...]
    seed: int


@dataclass(frozen=True)
class EvaluationMetricsContract:
    nll: float
    perplexity: float
    tokens_per_second: float
    peak_memory_mb: float
    model_parameter_mb: float
    evaluated_tokens: int
    elapsed_seconds: float


@dataclass(frozen=True)
class EvaluationComparisonContract:
    candidate_id: str
    baseline: EvaluationMetricsContract
    candidate: EvaluationMetricsContract
    relative_perplexity_degradation: float
    throughput_gain: float
    memory_reduction: float
    accepted_quality_budgets: Mapping[str, bool]
