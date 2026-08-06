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


class PilotStep(str, Enum):
    TRACES_CALIBRATION = "traces_calibration"
    TRACES_VALIDATION = "traces_validation"
    TRACES_TEST = "traces_test"
    H1_DIAGNOSTICS = "h1_diagnostics"
    BASELINE_VALIDATION = "baseline_validation"
    BASELINE_TEST = "baseline_test"
    VALIDATE_CANDIDATES = "validate_candidates"
    SELECT_CANDIDATES = "select_candidates"
    TEST_SELECTED = "test_selected"
    BUILD_SUMMARY = "build_summary"


class PilotStepStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class CandidateAction(str, Enum):
    KEEP = "keep"
    PRUNE = "prune"
    MERGE = "merge"
    ABSTAIN = "abstain"


class CandidateMethod(str, Enum):
    RANDOM_MERGE = "random_merge"
    USAGE_PRUNE = "usage_prune"
    USAGE_GUIDED_MERGE = "usage_guided_merge"
    FUNCTIONAL_COSINE = "functional_cosine"
    RMT_GEOMETRY = "rmt_geometry"
    RMT_FUNCTIONAL_USAGE = "rmt_functional_usage"


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_name: str
    stages: Tuple[ExperimentStageId, ...]


@dataclass(frozen=True)
class MoeLayerRuntimeContract:
    layer: int
    kind: str
    routed_experts: int
    shared_experts: int
    configured_routed_top_k: int
    requested_routed_top_k: int
    effective_routed_top_k: int
    top_k_updated: bool


@dataclass(frozen=True)
class TracePayloadMetadata:
    source: TraceSource
    layer: int
    split: str
    routed_experts: int
    shared_experts: int
    routed_top_k: int


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
    routed_experts: int
    shared_experts: int
    routed_top_k: int


@dataclass(frozen=True)
class CandidatePlanContract:
    candidate_id: str
    layer: int
    method: CandidateMethod
    action: CandidateAction
    compression_fraction: float
    groups: Tuple[Tuple[int, ...], ...]
    seed: int
    diagnostics: Tuple[Tuple[str, float], ...] = ()


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


@dataclass(frozen=True)
class PilotExecutionStateContract:
    schema_version: str
    config_hash: str
    completed_steps: Tuple[PilotStep, ...]
    completed_validation_candidate_ids: Tuple[str, ...]
    failed_validation_candidate_ids: Tuple[str, ...]
    selected_candidate_ids: Tuple[str, ...]
    completed_test_candidate_ids: Tuple[str, ...]
    failed_test_candidate_ids: Tuple[str, ...]


@dataclass(frozen=True)
class PilotStepResultContract:
    step: PilotStep
    status: PilotStepStatus
    processed_items: int
    remaining_items: int
    artifact_paths: Tuple[str, ...] = ()
