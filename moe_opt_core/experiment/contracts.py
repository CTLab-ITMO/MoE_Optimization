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


class MultiLayerStep(str, Enum):
    TRACES_CALIBRATION = "traces_calibration"
    TRACES_VALIDATION = "traces_validation"
    TRACES_TEST = "traces_test"
    LAYER_DIAGNOSTICS = "layer_diagnostics"
    BASELINE_VALIDATION = "baseline_validation"
    SENSITIVITY_VALIDATION = "sensitivity_validation"
    BUILD_RECIPES = "build_recipes"
    MULTILAYER_VALIDATION = "multilayer_validation"
    SELECT_CONFIGURATIONS = "select_configurations"
    BASELINE_TEST = "baseline_test"
    TEST_SELECTED = "test_selected"
    MATERIALIZE = "materialize"
    BENCHMARK_MATERIALIZED = "benchmark_materialized"
    BUILD_SUMMARY = "build_summary"


class TraceRefreshMode(str, Enum):
    STATIC = "static"
    ADAPTIVE = "adaptive"


class MultiLayerRecipeKind(str, Enum):
    CUMULATIVE = "cumulative"
    HYBRID = "hybrid"
    GLOBAL_BUDGET = "global_budget"


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


class MergeWeightingPolicy(str, Enum):
    UNIFORM = "uniform"
    USAGE_PROPORTIONAL = "usage_proportional"
    USAGE_BOUNDED = "usage_bounded"
    FUNCTIONAL_RIDGE = "functional_ridge"


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
    merge_weighting: MergeWeightingPolicy = MergeWeightingPolicy.UNIFORM
    group_merge_weights: Tuple[Tuple[float, ...], ...] = ()


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
class EvaluationRepeatSummaryContract:
    repeat_count: int
    aggregation: str
    perplexity_std: float
    tokens_per_second_median: float
    tokens_per_second_p05: float
    tokens_per_second_p95: float


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
class ValidationRobustnessContract:
    slice_count: int
    relative_perplexity_degradations: Tuple[float, ...]
    mean_relative_perplexity_degradation: float
    std_relative_perplexity_degradation: float
    worst_relative_perplexity_degradation: float
    std_penalty: float
    robust_score: float
    max_slice_degradation: float | None
    passes_max_slice_degradation: bool


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


@dataclass(frozen=True)
class LayerCandidateScoreContract:
    layer: int
    plan: CandidatePlanContract
    relative_perplexity_degradation: float
    robust_score: float
    measured: bool = True
    score_source_layer: int | None = None


@dataclass(frozen=True)
class LayerCompressionChoiceContract:
    layer: int
    plan: CandidatePlanContract | None
    compression_fraction: float
    predicted_degradation: float
    score_source_layer: int


@dataclass(frozen=True)
class MultiLayerRecipeContract:
    recipe_id: str
    kind: MultiLayerRecipeKind
    trace_refresh_mode: TraceRefreshMode
    choices: Tuple[LayerCompressionChoiceContract, ...]
    target_compression: float
    achieved_compression: float
    predicted_degradation: float


@dataclass(frozen=True)
class MultiLayerValidationScoreContract:
    recipe_id: str
    kind: MultiLayerRecipeKind
    trace_refresh_mode: TraceRefreshMode
    methods: Tuple[str, ...]
    layer_count: int
    target_compression: float
    achieved_compression: float
    relative_perplexity_degradation: float
    robust_score: float


@dataclass(frozen=True)
class ScalarDistributionContract:
    count: int
    median: float
    p05: float
    p95: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ReplicatedRecipeObservationContract:
    seed: int
    recipe_id: str
    record_type: str
    methods: Tuple[str, ...]
    layer_count: int
    target_compression: float
    achieved_compression: float
    relative_perplexity_degradation: float
    throughput_gain: float
    memory_reduction: float


@dataclass(frozen=True)
class ReplicatedRecipeSummaryContract:
    recipe_id: str
    record_type: str
    methods: Tuple[str, ...]
    layer_count: int
    target_compression: float
    achieved_compression: float
    seeds: Tuple[int, ...]
    degradation: ScalarDistributionContract
    throughput_gain: ScalarDistributionContract
    memory_reduction: ScalarDistributionContract
    quality_pass_rate: float


@dataclass(frozen=True)
class SensitivityVariantObservationContract:
    seed: int
    layer: int
    method: str
    compression_fraction: float
    candidate_id: str
    spectral_rank: int | None
    merge_weighting: str
    robust_score: float


@dataclass(frozen=True)
class SensitivityCellSummaryContract:
    layer: int
    method: str
    compression_fraction: float
    seeds: Tuple[int, ...]
    robust_score: ScalarDistributionContract
    winning_candidate_ids: Tuple[str, ...]
    spectral_rank_counts: Tuple[Tuple[int, int], ...]
    merge_weighting_counts: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class MultiLayerCampaignSummaryContract:
    schema_version: str
    config_name: str
    expected_seeds: Tuple[int, ...]
    completed_seeds: Tuple[int, ...]
    missing_seeds: Tuple[int, ...]
    recipe_summaries: Tuple[ReplicatedRecipeSummaryContract, ...]
    sensitivity_summaries: Tuple[SensitivityCellSummaryContract, ...]


@dataclass(frozen=True)
class ProgressiveStageContract:
    layer_count: int
    recipe_count: int
    completed_count: int
    accepted_count: int
    passed: bool


@dataclass(frozen=True)
class ProgressiveValidationPlanContract:
    current_layer_count: int | None
    pending_recipe_ids: Tuple[str, ...]
    skipped_recipe_ids: Tuple[str, ...]
    stages: Tuple[ProgressiveStageContract, ...]


@dataclass(frozen=True)
class MultiLayerExecutionStateContract:
    schema_version: str
    config_hash: str
    completed_steps: Tuple[MultiLayerStep, ...]
    completed_sensitivity_ids: Tuple[str, ...]
    failed_sensitivity_ids: Tuple[str, ...]
    completed_validation_recipe_ids: Tuple[str, ...]
    failed_validation_recipe_ids: Tuple[str, ...]
    skipped_validation_recipe_ids: Tuple[str, ...]
    selected_recipe_ids: Tuple[str, ...]
    completed_test_recipe_ids: Tuple[str, ...]
    failed_test_recipe_ids: Tuple[str, ...]


@dataclass(frozen=True)
class MultiLayerStepResultContract:
    step: MultiLayerStep
    status: PilotStepStatus
    processed_items: int
    remaining_items: int
    artifact_paths: Tuple[str, ...] = ()
