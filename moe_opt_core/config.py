from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    name: str
    kind: str
    load_source: str = "hf"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = False


@dataclass(frozen=True)
class H1Config:
    enabled: bool
    layer: int
    mlp_inputs_path: str
    experts_dir: str
    output_pairs_path: str
    output_diagnostics_path: str
    output_merged_dir: str
    experts_source: str = "saved_experts"
    auto_generate_inputs_if_missing: bool = False
    mode: str = "pair"
    merge_method: str = "geometry"
    proj_dim: int = 8
    n_inputs: int = 4096
    batch_size: int = 64
    mp_factor: float = 1.1
    cluster_count: int = 0
    geometry_rank: int = 64
    shrink_tau: float = 0.0
    null_trials: int = 32
    bootstrap_trials: int = 32
    null_quantile: float = 0.95
    stability_max_angle_deg: float = 30.0


@dataclass(frozen=True)
class H2Config:
    enabled: bool
    layer: int
    mode: str
    output_bias_path: str
    importance_matrix_path: str = ""
    importance_vector_path: str = ""
    gate_weight_path: str = ""
    auto_from_model: bool = True
    alpha: float = 0.2
    lam: float = 1e-3
    mp_factor: float = 1.1
    normalize: str = "maxabs"
    scale: float = 0.1


@dataclass(frozen=True)
class H3Config:
    enabled: bool
    routing_probs_path: str
    output_diagnostics_path: str
    auto_from_model: bool = True
    tau: float = 0.92
    k_min: int = 1
    k_max: int = 8


@dataclass(frozen=True)
class ProfileConfig:
    enabled: bool = True
    warmup_steps: int = 10
    measure_steps: int = 50


@dataclass(frozen=True)
class CalibrationConfig:
    source: str = "artifact"
    dataset_name: str = ""
    dataset_config: str = ""
    dataset_split: str = "train"
    text_column: str = ""
    output_dir: str = "artifacts/traces"
    max_samples: int = 256
    max_length: int = 512
    batch_size: int = 1
    calibration_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 123


@dataclass(frozen=True)
class HybridPairingVariantConfig:
    name: str
    rmt_weight: float
    functional_weight: float
    usage_weight: float


@dataclass(frozen=True)
class HybridPairingConfig:
    rmt_weight: float = 0.5
    functional_weight: float = 0.35
    usage_weight: float = 0.15
    variants: Tuple[HybridPairingVariantConfig, ...] = ()


@dataclass(frozen=True)
class MergeWeightingVariantConfig:
    name: str
    policy: str
    minimum_weight: float = 0.0
    ridge_lambda: float = 0.0


@dataclass(frozen=True)
class MergeWeightingConfig:
    methods: Tuple[str, ...] = ()
    variants: Tuple[MergeWeightingVariantConfig, ...] = ()


@dataclass(frozen=True)
class SpectralRankAssignmentConfig:
    method: str
    compression_fraction: float
    rank: int


@dataclass(frozen=True)
class SpectralRankPolicyConfig:
    include_useful_rank: bool = False
    include_mp_rank: bool = False
    explained_variance_thresholds: Tuple[float, ...] = ()


@dataclass(frozen=True)
class ValidationSelectionConfig:
    slice_count: int = 1
    std_penalty: float = 0.0
    max_slice_degradation: float | None = None


@dataclass(frozen=True)
class MultiLayerMaterializationConfig:
    enabled: bool = True
    output_dir: str = "artifacts/materialized"
    selected_plan_count: int = 1
    strategy: str = "selected_order"
    save_layer_weights: bool = True


@dataclass(frozen=True)
class MultiLayerProtocolConfig:
    replication_seeds: Tuple[int, ...] = ()
    validation_repeats: int = 1
    test_repeats: int = 1
    materialized_repeats: int = 1
    aggregation: str = "median"


@dataclass(frozen=True)
class MultiLayerSelectionConfig:
    strategy: str = "global_winners"
    per_method_per_cell: int = 1
    include_global_winners: bool = True
    balance_layer_counts: bool = False
    balance_refresh_modes: bool = True
    max_selected: int = 0


@dataclass(frozen=True)
class ProgressiveScalingConfig:
    enabled: bool = False
    layer_counts: Tuple[int, ...] = ()
    min_accepted_recipes: int = 1
    stop_on_failure: bool = True


@dataclass(frozen=True)
class MultiLayerConfig:
    enabled: bool = False
    sensitivity_layers: Tuple[int, ...] = ()
    methods: Tuple[str, ...] = (
        "usage_prune",
        "functional_cosine",
        "rmt_functional_usage",
    )
    compression_fractions: Tuple[float, ...] = (0.1, 0.25)
    cumulative_layer_counts: Tuple[int, ...] = (1, 3, 6)
    include_all_layers: bool = True
    trace_refresh_modes: Tuple[str, ...] = ("static", "adaptive")
    global_compression_targets: Tuple[float, ...] = (0.05, 0.1, 0.2)
    quality_budget: float = 0.03
    max_layer_degradation: float = 0.05
    candidates_per_invocation: int = 2
    max_step_seconds: float = 1800.0
    include_hybrid_recipes: bool = False
    protocol: MultiLayerProtocolConfig = MultiLayerProtocolConfig()
    selection: MultiLayerSelectionConfig = MultiLayerSelectionConfig()
    progressive_scaling: ProgressiveScalingConfig = ProgressiveScalingConfig()
    materialization: MultiLayerMaterializationConfig = MultiLayerMaterializationConfig()


@dataclass(frozen=True)
class PilotConfig:
    enabled: bool = False
    layer: int = 0
    methods: Tuple[str, ...] = (
        "random_merge",
        "usage_prune",
        "functional_cosine",
        "rmt_geometry",
        "rmt_functional_usage",
    )
    compression_fractions: Tuple[float, ...] = (0.1, 0.25, 0.4)
    quality_budgets: Tuple[float, ...] = (0.01, 0.03, 0.05)
    seeds: Tuple[int, ...] = (123, 456, 789)
    max_eval_batches: int = 32
    require_real_traces: bool = True
    candidates_per_invocation: int = 2
    max_step_seconds: float = 1800.0
    spectral_pairing_ranks: Tuple[int, ...] = ()
    spectral_rank_schedule: Tuple[SpectralRankAssignmentConfig, ...] = ()
    spectral_rank_policy: SpectralRankPolicyConfig = SpectralRankPolicyConfig()
    hybrid_pairing: HybridPairingConfig = HybridPairingConfig()
    merge_weighting: MergeWeightingConfig = MergeWeightingConfig()
    validation_selection: ValidationSelectionConfig = ValidationSelectionConfig()


@dataclass(frozen=True)
class ArtifactSpecConfig:
    root_dir: str
    manifest_path: str
    schema_version: str = "1.0"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    runtime_top_k: int
    runtime_expected_routed_experts: int | None
    runtime_expected_shared_experts: int | None
    runtime_strict_architecture: bool
    layers: List[int]
    constants: Dict[str, Dict[str, Any]]
    artifacts: ArtifactSpecConfig
    profile: ProfileConfig
    calibration: CalibrationConfig
    pilot: PilotConfig
    multilayer: MultiLayerConfig
    h1: H1Config
    h2: H2Config
    h3: H3Config


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _require(data: Dict[str, Any], keys: List[str], ctx: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing keys in {ctx}: {missing}")


def validate_experiment_config_dict(data: Dict[str, Any]) -> None:
    _require(data, ["name", "model", "runtime", "layers", "constants", "artifacts", "stages"], "experiment")
    _require(data["model"], ["name", "kind"], "model")
    _require(data["runtime"], ["top_k"], "runtime")
    _require(data["artifacts"], ["root_dir", "manifest_path"], "artifacts")
    _require(data["stages"], ["h1", "h2", "h3"], "stages")

    h1 = data["stages"]["h1"]
    _require(
        h1,
        [
            "enabled",
            "layer",
            "mlp_inputs_path",
            "experts_dir",
            "output_pairs_path",
            "output_diagnostics_path",
            "output_merged_dir",
        ],
        "stages.h1",
    )

    h2 = data["stages"]["h2"]
    _require(h2, ["enabled", "layer", "mode", "output_bias_path"], "stages.h2")

    h3 = data["stages"]["h3"]
    _require(h3, ["enabled", "routing_probs_path", "output_diagnostics_path"], "stages.h3")

    top_k = int(data["runtime"]["top_k"])
    if top_k < 1:
        raise ValueError("runtime.top_k must be >= 1")
    expected_routed = data["runtime"].get("expected_routed_experts")
    if expected_routed is not None and int(expected_routed) < 1:
        raise ValueError("runtime.expected_routed_experts must be >= 1")
    expected_shared = data["runtime"].get("expected_shared_experts")
    if expected_shared is not None and int(expected_shared) < 0:
        raise ValueError("runtime.expected_shared_experts must be >= 0")
    if expected_routed is not None and top_k > int(expected_routed):
        raise ValueError("runtime.top_k must not exceed runtime.expected_routed_experts")

    if not isinstance(data["layers"], list) or len(data["layers"]) == 0:
        raise ValueError("layers must be a non-empty list")

    profile = data.get("profile", {})
    profile_warmup_steps = int(profile.get("warmup_steps", 10))
    profile_measure_steps = int(profile.get("measure_steps", 50))
    if profile_warmup_steps < 0:
        raise ValueError("profile.warmup_steps must be non-negative")
    if profile_measure_steps < 1:
        raise ValueError("profile.measure_steps must be positive")

    model_load_source = str(data["model"].get("load_source", "hf"))
    if model_load_source not in {"hf", "local"}:
        raise ValueError("model.load_source must be one of: hf, local")

    h1_mode = str(h1.get("mode", "pair"))
    if h1_mode not in {"pair", "cluster"}:
        raise ValueError("stages.h1.mode must be 'pair' or 'cluster'")
    if int(h1.get("null_trials", 32)) < 1 or int(h1.get("bootstrap_trials", 32)) < 1:
        raise ValueError("stages.h1 null_trials and bootstrap_trials must be positive")
    if not 0.0 < float(h1.get("null_quantile", 0.95)) < 1.0:
        raise ValueError("stages.h1.null_quantile must be in (0, 1)")

    h2_mode = str(h2.get("mode", "gate_spike"))
    if h2_mode not in {"covariance", "gate_spike"}:
        raise ValueError("stages.h2.mode must be 'covariance' or 'gate_spike'")

    k_min = int(h3.get("k_min", 1))
    k_max = int(h3.get("k_max", 8))
    if k_min < 1 or k_max < 1 or k_min > k_max:
        raise ValueError("stages.h3 requires 1 <= k_min <= k_max")

    calibration = data.get("calibration", {})
    source = str(calibration.get("source", "artifact"))
    if source not in {"artifact", "hf_dataset", "synthetic"}:
        raise ValueError("calibration.source must be one of: artifact, hf_dataset, synthetic")
    if source == "hf_dataset" and not str(calibration.get("dataset_name", "")).strip():
        raise ValueError("calibration.dataset_name is required for source='hf_dataset'")
    if int(calibration.get("max_samples", 256)) < 3:
        raise ValueError("calibration.max_samples must be >= 3")
    if int(calibration.get("max_length", 512)) < 2:
        raise ValueError("calibration.max_length must be >= 2")
    if int(calibration.get("batch_size", 1)) < 1:
        raise ValueError("calibration.batch_size must be >= 1")

    fractions = (
        float(calibration.get("calibration_fraction", 0.6)),
        float(calibration.get("validation_fraction", 0.2)),
        float(calibration.get("test_fraction", 0.2)),
    )
    if any(value <= 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("calibration split fractions must be positive and sum to 1")

    pilot = data.get("pilot", {})
    allowed_methods = {
        "random_merge",
        "usage_prune",
        "usage_guided_merge",
        "functional_cosine",
        "rmt_geometry",
        "rmt_functional_usage",
    }
    methods = tuple(str(item) for item in pilot.get("methods", PilotConfig().methods))
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        raise ValueError(f"pilot.methods contains unsupported methods: {unknown_methods}")
    compression_fractions = tuple(
        float(value) for value in pilot.get("compression_fractions", (0.1, 0.25, 0.4))
    )
    for value in compression_fractions:
        if not 0.0 < float(value) < 1.0:
            raise ValueError("pilot.compression_fractions values must be in (0, 1)")
    for value in pilot.get("quality_budgets", (0.01, 0.03, 0.05)):
        if float(value) < 0.0:
            raise ValueError("pilot.quality_budgets values must be non-negative")
    validation_selection = pilot.get("validation_selection", {})
    if not isinstance(validation_selection, dict):
        raise ValueError("pilot.validation_selection must be an object")
    validation_slice_count = int(validation_selection.get("slice_count", 1))
    validation_std_penalty = float(validation_selection.get("std_penalty", 0.0))
    validation_max_slice_degradation = validation_selection.get("max_slice_degradation")
    max_eval_batches = int(pilot.get("max_eval_batches", 32))
    if max_eval_batches < 1:
        raise ValueError("pilot.max_eval_batches must be >= 1")
    if validation_slice_count < 1:
        raise ValueError("pilot.validation_selection.slice_count must be positive")
    if validation_slice_count > max_eval_batches:
        raise ValueError(
            "pilot.validation_selection.slice_count must not exceed pilot.max_eval_batches"
        )
    if validation_slice_count > profile_measure_steps:
        raise ValueError(
            "pilot.validation_selection.slice_count must not exceed profile.measure_steps"
        )
    if validation_std_penalty < 0.0:
        raise ValueError("pilot.validation_selection.std_penalty must be non-negative")
    if (
        validation_max_slice_degradation is not None
        and float(validation_max_slice_degradation) < 0.0
    ):
        raise ValueError(
            "pilot.validation_selection.max_slice_degradation must be non-negative"
        )
    if bool(pilot.get("enabled", False)):
        if not methods:
            raise ValueError("pilot.methods must not be empty when pilot is enabled")
        if not pilot.get("compression_fractions", (0.1, 0.25, 0.4)):
            raise ValueError("pilot.compression_fractions must not be empty when pilot is enabled")
        if not pilot.get("quality_budgets", (0.01, 0.03, 0.05)):
            raise ValueError("pilot.quality_budgets must not be empty when pilot is enabled")
        if not pilot.get("seeds", (123, 456, 789)):
            raise ValueError("pilot.seeds must not be empty when pilot is enabled")
        pilot_layer = int(pilot.get("layer", h1["layer"]))
        if pilot_layer not in [int(layer) for layer in data["layers"]]:
            raise ValueError("pilot.layer must be included in layers")
        if not bool(h1["enabled"]):
            raise ValueError("pilot requires stages.h1.enabled=true")
        if int(pilot.get("candidates_per_invocation", 2)) < 1:
            raise ValueError("pilot.candidates_per_invocation must be >= 1")
        if float(pilot.get("max_step_seconds", 1800.0)) <= 0.0:
            raise ValueError("pilot.max_step_seconds must be positive")
        if bool(pilot.get("require_real_traces", True)) and source == "synthetic":
            raise ValueError("pilot.require_real_traces=true is incompatible with calibration.source='synthetic'")
    spectral_pairing_ranks = tuple(int(rank) for rank in pilot.get("spectral_pairing_ranks", ()))
    if any(rank < 1 for rank in spectral_pairing_ranks):
        raise ValueError("pilot.spectral_pairing_ranks values must be positive")
    if len(set(spectral_pairing_ranks)) != len(spectral_pairing_ranks):
        raise ValueError("pilot.spectral_pairing_ranks values must be unique")
    if expected_routed is not None and any(rank > int(expected_routed) for rank in spectral_pairing_ranks):
        raise ValueError("pilot.spectral_pairing_ranks must not exceed runtime.expected_routed_experts")
    rank_policy = pilot.get("spectral_rank_policy", {})
    if not isinstance(rank_policy, dict):
        raise ValueError("pilot.spectral_rank_policy must be an object")
    explained_variance_thresholds = tuple(
        float(value)
        for value in rank_policy.get("explained_variance_thresholds", ())
    )
    if any(not 0.0 < value <= 1.0 for value in explained_variance_thresholds):
        raise ValueError(
            "pilot.spectral_rank_policy.explained_variance_thresholds values "
            "must be in (0, 1]"
        )
    if len(set(explained_variance_thresholds)) != len(
        explained_variance_thresholds
    ):
        raise ValueError(
            "pilot.spectral_rank_policy.explained_variance_thresholds values "
            "must be unique"
        )
    rank_schedule = pilot.get("spectral_rank_schedule", ())
    if not isinstance(rank_schedule, (list, tuple)):
        raise ValueError("pilot.spectral_rank_schedule must be an array")
    schedule_keys: list[tuple[str, float]] = []
    scheduled_methods: set[str] = set()
    for index, item in enumerate(rank_schedule):
        if not isinstance(item, dict):
            raise ValueError("pilot.spectral_rank_schedule entries must be objects")
        _require(
            item,
            ["method", "compression_fraction", "rank"],
            f"pilot.spectral_rank_schedule[{index}]",
        )
        method = str(item["method"])
        fraction = float(item["compression_fraction"])
        rank = int(item["rank"])
        if method not in {"rmt_geometry", "rmt_functional_usage"}:
            raise ValueError("pilot.spectral_rank_schedule only supports RMT methods")
        if method not in methods:
            raise ValueError("pilot.spectral_rank_schedule methods must be enabled in pilot.methods")
        if not any(abs(fraction - value) <= 1e-12 for value in compression_fractions):
            raise ValueError(
                "pilot.spectral_rank_schedule fractions must be enabled in "
                "pilot.compression_fractions"
            )
        if rank not in spectral_pairing_ranks:
            raise ValueError(
                "pilot.spectral_rank_schedule ranks must be enabled in "
                "pilot.spectral_pairing_ranks"
            )
        schedule_keys.append((method, fraction))
        scheduled_methods.add(method)
    if len(set(schedule_keys)) != len(schedule_keys):
        raise ValueError("pilot.spectral_rank_schedule assignments must be unique")
    for method in scheduled_methods:
        assigned = {
            fraction for scheduled_method, fraction in schedule_keys if scheduled_method == method
        }
        if any(not any(abs(fraction - value) <= 1e-12 for value in assigned) for fraction in compression_fractions):
            raise ValueError(
                "pilot.spectral_rank_schedule must assign every compression fraction "
                f"for method {method}"
            )

    merge_weighting = pilot.get("merge_weighting", {})
    if not isinstance(merge_weighting, dict):
        raise ValueError("pilot.merge_weighting must be an object")
    weighted_methods = tuple(str(item) for item in merge_weighting.get("methods", ()))
    if len(set(weighted_methods)) != len(weighted_methods):
        raise ValueError("pilot.merge_weighting methods must be unique")
    unsupported_weighted_methods = sorted(
        set(weighted_methods) - (allowed_methods - {"usage_prune"})
    )
    if unsupported_weighted_methods:
        raise ValueError(
            "pilot.merge_weighting only supports merge methods: "
            f"{unsupported_weighted_methods}"
        )
    if any(method not in methods for method in weighted_methods):
        raise ValueError("pilot.merge_weighting methods must be enabled in pilot.methods")
    merge_weight_variants = merge_weighting.get("variants", ())
    if not isinstance(merge_weight_variants, (list, tuple)):
        raise ValueError("pilot.merge_weighting.variants must be an array")
    if bool(weighted_methods) != bool(merge_weight_variants):
        raise ValueError(
            "pilot.merge_weighting methods and variants must either both be empty or both be set"
        )
    merge_variant_names: list[str] = []
    allowed_weighting_policies = {
        "uniform",
        "usage_proportional",
        "usage_bounded",
        "functional_ridge",
    }
    for index, item in enumerate(merge_weight_variants):
        if not isinstance(item, dict):
            raise ValueError("pilot.merge_weighting variants must be objects")
        _require(item, ["name", "policy"], f"pilot.merge_weighting.variants[{index}]")
        name = str(item["name"])
        policy = str(item["policy"])
        minimum_weight = float(item.get("minimum_weight", 0.0))
        ridge_lambda = float(item.get("ridge_lambda", 0.0))
        if re.fullmatch(r"[a-z0-9_]+", name) is None:
            raise ValueError("pilot.merge_weighting variant names must match [a-z0-9_]+")
        if policy not in allowed_weighting_policies:
            raise ValueError("pilot.merge_weighting contains an unsupported policy")
        if not 0.0 <= minimum_weight < 0.5:
            raise ValueError("pilot.merge_weighting minimum_weight must be in [0, 0.5)")
        if policy == "usage_bounded" and minimum_weight <= 0.0:
            raise ValueError("usage_bounded merge weighting requires a positive minimum_weight")
        if policy != "usage_bounded" and minimum_weight != 0.0:
            raise ValueError(
                "pilot.merge_weighting minimum_weight is only valid for usage_bounded"
            )
        if ridge_lambda < 0.0:
            raise ValueError("pilot.merge_weighting ridge_lambda must be non-negative")
        if policy == "functional_ridge" and ridge_lambda <= 0.0:
            raise ValueError(
                "functional_ridge merge weighting requires a positive ridge_lambda"
            )
        if policy != "functional_ridge" and ridge_lambda != 0.0:
            raise ValueError(
                "pilot.merge_weighting ridge_lambda is only valid for functional_ridge"
            )
        merge_variant_names.append(name)
    if len(set(merge_variant_names)) != len(merge_variant_names):
        raise ValueError("pilot.merge_weighting variant names must be unique")

    hybrid = pilot.get("hybrid_pairing", {})
    hybrid_weights = (
        float(hybrid.get("rmt_weight", 0.5)),
        float(hybrid.get("functional_weight", 0.35)),
        float(hybrid.get("usage_weight", 0.15)),
    )
    if any(weight < 0.0 for weight in hybrid_weights):
        raise ValueError("pilot.hybrid_pairing weights must be non-negative")
    if sum(hybrid_weights) <= 0.0:
        raise ValueError("pilot.hybrid_pairing must contain at least one positive weight")
    variants = hybrid.get("variants", ())
    if not isinstance(variants, (list, tuple)):
        raise ValueError("pilot.hybrid_pairing.variants must be an array")
    for index, item in enumerate(variants):
        if not isinstance(item, dict):
            raise ValueError("pilot.hybrid_pairing variants must be objects")
        _require(
            item,
            ["name", "rmt_weight", "functional_weight", "usage_weight"],
            f"pilot.hybrid_pairing.variants[{index}]",
        )
    variant_names = tuple(str(item.get("name", "")) for item in variants)
    if any(not name for name in variant_names):
        raise ValueError("pilot.hybrid_pairing variant names must not be empty")
    if any(re.fullmatch(r"[a-z0-9_]+", name) is None for name in variant_names):
        raise ValueError("pilot.hybrid_pairing variant names must match [a-z0-9_]+")
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("pilot.hybrid_pairing variant names must be unique")
    for item in variants:
        weights = (
            float(item.get("rmt_weight", 0.0)),
            float(item.get("functional_weight", 0.0)),
            float(item.get("usage_weight", 0.0)),
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("pilot.hybrid_pairing variant weights must be non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("pilot.hybrid_pairing variants must contain at least one positive weight")

    configured_layers = tuple(int(layer) for layer in data["layers"])
    if len(set(configured_layers)) != len(configured_layers):
        raise ValueError("layers values must be unique")
    if any(layer < 0 for layer in configured_layers):
        raise ValueError("layers values must be non-negative")

    multilayer = data.get("multilayer", {})
    if not isinstance(multilayer, dict):
        raise ValueError("multilayer must be an object")
    multilayer_methods = tuple(
        str(item) for item in multilayer.get("methods", MultiLayerConfig().methods)
    )
    supported_multilayer_methods = {
        "usage_prune",
        "functional_cosine",
        "rmt_functional_usage",
    }
    unknown_multilayer_methods = sorted(
        set(multilayer_methods) - supported_multilayer_methods
    )
    if unknown_multilayer_methods:
        raise ValueError(
            "multilayer.methods contains unsupported methods: "
            f"{unknown_multilayer_methods}"
        )
    if len(set(multilayer_methods)) != len(multilayer_methods):
        raise ValueError("multilayer.methods values must be unique")
    multilayer_fractions = tuple(
        float(value)
        for value in multilayer.get(
            "compression_fractions", MultiLayerConfig().compression_fractions
        )
    )
    if any(not 0.0 < value < 1.0 for value in multilayer_fractions):
        raise ValueError("multilayer.compression_fractions values must be in (0, 1)")
    if len(set(multilayer_fractions)) != len(multilayer_fractions):
        raise ValueError("multilayer.compression_fractions values must be unique")
    sensitivity_layers = tuple(
        int(layer) for layer in multilayer.get("sensitivity_layers", ())
    )
    if len(set(sensitivity_layers)) != len(sensitivity_layers):
        raise ValueError("multilayer.sensitivity_layers values must be unique")
    unknown_sensitivity_layers = sorted(set(sensitivity_layers) - set(configured_layers))
    if unknown_sensitivity_layers:
        raise ValueError(
            "multilayer.sensitivity_layers must be included in layers: "
            f"{unknown_sensitivity_layers}"
        )
    effective_layer_count = len(configured_layers)
    cumulative_counts = tuple(
        int(value)
        for value in multilayer.get(
            "cumulative_layer_counts", MultiLayerConfig().cumulative_layer_counts
        )
    )
    if any(value < 1 for value in cumulative_counts):
        raise ValueError("multilayer.cumulative_layer_counts values must be positive")
    if len(set(cumulative_counts)) != len(cumulative_counts):
        raise ValueError("multilayer.cumulative_layer_counts values must be unique")
    if bool(multilayer.get("enabled", False)) and any(
        value > effective_layer_count for value in cumulative_counts
    ):
        raise ValueError(
            "multilayer.cumulative_layer_counts values must not exceed the "
            "number of configured layers"
        )
    refresh_modes = tuple(
        str(value)
        for value in multilayer.get(
            "trace_refresh_modes", MultiLayerConfig().trace_refresh_modes
        )
    )
    if not refresh_modes or set(refresh_modes) - {"static", "adaptive"}:
        raise ValueError(
            "multilayer.trace_refresh_modes must contain only: static, adaptive"
        )
    if len(set(refresh_modes)) != len(refresh_modes):
        raise ValueError("multilayer.trace_refresh_modes values must be unique")
    global_targets = tuple(
        float(value)
        for value in multilayer.get(
            "global_compression_targets", MultiLayerConfig().global_compression_targets
        )
    )
    if any(not 0.0 < value < 1.0 for value in global_targets):
        raise ValueError("multilayer.global_compression_targets values must be in (0, 1)")
    if len(set(global_targets)) != len(global_targets):
        raise ValueError("multilayer.global_compression_targets values must be unique")
    if float(multilayer.get("quality_budget", 0.03)) < 0.0:
        raise ValueError("multilayer.quality_budget must be non-negative")
    if float(multilayer.get("max_layer_degradation", 0.05)) < 0.0:
        raise ValueError("multilayer.max_layer_degradation must be non-negative")
    if int(multilayer.get("candidates_per_invocation", 2)) < 1:
        raise ValueError("multilayer.candidates_per_invocation must be positive")
    if float(multilayer.get("max_step_seconds", 1800.0)) <= 0.0:
        raise ValueError("multilayer.max_step_seconds must be positive")
    protocol = multilayer.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("multilayer.protocol must be an object")
    replication_seeds = tuple(int(value) for value in protocol.get("replication_seeds", ()))
    if len(set(replication_seeds)) != len(replication_seeds):
        raise ValueError("multilayer.protocol.replication_seeds values must be unique")
    for field in ("validation_repeats", "test_repeats", "materialized_repeats"):
        if int(protocol.get(field, 1)) < 1:
            raise ValueError(f"multilayer.protocol.{field} must be positive")
    if str(protocol.get("aggregation", "median")) != "median":
        raise ValueError("multilayer.protocol.aggregation must be median")

    selection = multilayer.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("multilayer.selection must be an object")
    if str(selection.get("strategy", "global_winners")) not in {
        "global_winners",
        "method_balanced",
    }:
        raise ValueError(
            "multilayer.selection.strategy must be global_winners or method_balanced"
        )
    if int(selection.get("per_method_per_cell", 1)) < 1:
        raise ValueError(
            "multilayer.selection.per_method_per_cell must be positive"
        )
    max_selected = int(selection.get("max_selected", 0))
    if max_selected < 0:
        raise ValueError("multilayer.selection.max_selected must be non-negative")
    if (
        str(selection.get("strategy", "global_winners")) == "method_balanced"
        and max_selected
    ):
        balanced_counts = set(cumulative_counts)
        if bool(multilayer.get("include_all_layers", True)):
            balanced_counts.add(effective_layer_count)
        family_count = len(multilayer_methods) + int(
            bool(multilayer.get("include_hybrid_recipes", False))
        )
        required_quota = (
            int(selection.get("per_method_per_cell", 1))
            * family_count
            * len(multilayer_fractions)
            * (
                len(balanced_counts)
                if bool(selection.get("balance_layer_counts", False))
                else 1
            )
            * (
                len(refresh_modes)
                if bool(selection.get("balance_refresh_modes", True))
                else 1
            )
        )
        if max_selected < required_quota:
            raise ValueError(
                "multilayer.selection.max_selected is smaller than the configured "
                f"method-balanced quota ({required_quota})"
            )

    progressive = multilayer.get("progressive_scaling", {})
    if not isinstance(progressive, dict):
        raise ValueError("multilayer.progressive_scaling must be an object")
    progressive_counts = tuple(
        int(value) for value in progressive.get("layer_counts", ())
    )
    if any(value < 1 or value > effective_layer_count for value in progressive_counts):
        raise ValueError(
            "multilayer.progressive_scaling.layer_counts values must be within "
            "the configured layer count"
        )
    if len(set(progressive_counts)) != len(progressive_counts):
        raise ValueError(
            "multilayer.progressive_scaling.layer_counts values must be unique"
        )
    if bool(progressive.get("enabled", False)) and not progressive_counts:
        raise ValueError(
            "multilayer.progressive_scaling.layer_counts must not be empty when enabled"
        )
    if int(progressive.get("min_accepted_recipes", 1)) < 1:
        raise ValueError(
            "multilayer.progressive_scaling.min_accepted_recipes must be positive"
        )
    if bool(progressive.get("enabled", False)) and set(progressive_counts) != set(
        cumulative_counts
    ):
        raise ValueError(
            "multilayer.progressive_scaling.layer_counts must match "
            "multilayer.cumulative_layer_counts"
        )
    materialization = multilayer.get("materialization", {})
    if not isinstance(materialization, dict):
        raise ValueError("multilayer.materialization must be an object")
    if int(materialization.get("selected_plan_count", 1)) < 1:
        raise ValueError(
            "multilayer.materialization.selected_plan_count must be positive"
        )
    if materialization.get("strategy", "selected_order") not in {
        "selected_order",
        "method_balanced",
    }:
        raise ValueError(
            "multilayer.materialization.strategy must be one of: "
            "selected_order, method_balanced"
        )
    if bool(materialization.get("enabled", True)) and not str(
        materialization.get("output_dir", "artifacts/materialized")
    ).strip():
        raise ValueError(
            "multilayer.materialization.output_dir is required when materialization is enabled"
        )
    if bool(multilayer.get("enabled", False)):
        if not bool(pilot.get("enabled", False)):
            raise ValueError("multilayer requires pilot.enabled=true")
        if not bool(h1["enabled"]):
            raise ValueError("multilayer requires stages.h1.enabled=true")
        if not multilayer_methods:
            raise ValueError("multilayer.methods must not be empty")
        if not multilayer_fractions:
            raise ValueError("multilayer.compression_fractions must not be empty")



def _load_constants(constants_section: Dict[str, str], config_path: Path) -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for domain, rel_path in constants_section.items():
        p = (config_path.parent / rel_path).resolve() if not Path(rel_path).is_absolute() else Path(rel_path)
        loaded[domain] = _load_json(p)
    return loaded


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_json(config_path)
    validate_experiment_config_dict(raw)

    constants = _load_constants(raw["constants"], config_path)

    profile_raw = raw.get("profile", {})
    profile = ProfileConfig(
        enabled=bool(profile_raw.get("enabled", True)),
        warmup_steps=int(profile_raw.get("warmup_steps", 10)),
        measure_steps=int(profile_raw.get("measure_steps", 50)),
    )

    calibration_raw = raw.get("calibration", {})
    calibration = CalibrationConfig(
        source=str(calibration_raw.get("source", "artifact")),
        dataset_name=str(calibration_raw.get("dataset_name", "")),
        dataset_config=str(calibration_raw.get("dataset_config", "")),
        dataset_split=str(calibration_raw.get("dataset_split", "train")),
        text_column=str(calibration_raw.get("text_column", "")),
        output_dir=str(calibration_raw.get("output_dir", "artifacts/traces")),
        max_samples=int(calibration_raw.get("max_samples", 256)),
        max_length=int(calibration_raw.get("max_length", 512)),
        batch_size=int(calibration_raw.get("batch_size", 1)),
        calibration_fraction=float(calibration_raw.get("calibration_fraction", 0.6)),
        validation_fraction=float(calibration_raw.get("validation_fraction", 0.2)),
        test_fraction=float(calibration_raw.get("test_fraction", 0.2)),
        seed=int(calibration_raw.get("seed", 123)),
    )

    pilot_raw = raw.get("pilot", {})
    pilot_defaults = PilotConfig()
    hybrid_raw = pilot_raw.get("hybrid_pairing", {})
    hybrid_defaults = HybridPairingConfig()
    merge_weighting_raw = pilot_raw.get("merge_weighting", {})
    spectral_rank_policy_raw = pilot_raw.get("spectral_rank_policy", {})
    spectral_rank_policy_defaults = SpectralRankPolicyConfig()
    validation_selection_raw = pilot_raw.get("validation_selection", {})
    validation_selection_defaults = ValidationSelectionConfig()
    hybrid_variants = tuple(
        HybridPairingVariantConfig(
            name=str(item["name"]),
            rmt_weight=float(item["rmt_weight"]),
            functional_weight=float(item["functional_weight"]),
            usage_weight=float(item["usage_weight"]),
        )
        for item in hybrid_raw.get("variants", ())
    )
    merge_weighting_variants = tuple(
        MergeWeightingVariantConfig(
            name=str(item["name"]),
            policy=str(item["policy"]),
            minimum_weight=float(item.get("minimum_weight", 0.0)),
            ridge_lambda=float(item.get("ridge_lambda", 0.0)),
        )
        for item in merge_weighting_raw.get("variants", ())
    )
    spectral_rank_schedule = tuple(
        SpectralRankAssignmentConfig(
            method=str(item["method"]),
            compression_fraction=float(item["compression_fraction"]),
            rank=int(item["rank"]),
        )
        for item in pilot_raw.get("spectral_rank_schedule", ())
    )
    pilot = PilotConfig(
        enabled=bool(pilot_raw.get("enabled", False)),
        layer=int(pilot_raw.get("layer", raw["stages"]["h1"]["layer"])),
        methods=tuple(str(item) for item in pilot_raw.get("methods", pilot_defaults.methods)),
        compression_fractions=tuple(
            float(item) for item in pilot_raw.get("compression_fractions", pilot_defaults.compression_fractions)
        ),
        quality_budgets=tuple(
            float(item) for item in pilot_raw.get("quality_budgets", pilot_defaults.quality_budgets)
        ),
        seeds=tuple(int(item) for item in pilot_raw.get("seeds", pilot_defaults.seeds)),
        max_eval_batches=int(pilot_raw.get("max_eval_batches", 32)),
        require_real_traces=bool(pilot_raw.get("require_real_traces", True)),
        candidates_per_invocation=int(pilot_raw.get("candidates_per_invocation", 2)),
        max_step_seconds=float(pilot_raw.get("max_step_seconds", 1800.0)),
        spectral_pairing_ranks=tuple(
            int(item) for item in pilot_raw.get("spectral_pairing_ranks", ())
        ),
        spectral_rank_schedule=spectral_rank_schedule,
        spectral_rank_policy=SpectralRankPolicyConfig(
            include_useful_rank=bool(
                spectral_rank_policy_raw.get(
                    "include_useful_rank",
                    spectral_rank_policy_defaults.include_useful_rank,
                )
            ),
            include_mp_rank=bool(
                spectral_rank_policy_raw.get(
                    "include_mp_rank",
                    spectral_rank_policy_defaults.include_mp_rank,
                )
            ),
            explained_variance_thresholds=tuple(
                float(item)
                for item in spectral_rank_policy_raw.get(
                    "explained_variance_thresholds",
                    spectral_rank_policy_defaults.explained_variance_thresholds,
                )
            ),
        ),
        hybrid_pairing=HybridPairingConfig(
            rmt_weight=float(hybrid_raw.get("rmt_weight", hybrid_defaults.rmt_weight)),
            functional_weight=float(
                hybrid_raw.get("functional_weight", hybrid_defaults.functional_weight)
            ),
            usage_weight=float(hybrid_raw.get("usage_weight", hybrid_defaults.usage_weight)),
            variants=hybrid_variants,
        ),
        merge_weighting=MergeWeightingConfig(
            methods=tuple(str(item) for item in merge_weighting_raw.get("methods", ())),
            variants=merge_weighting_variants,
        ),
        validation_selection=ValidationSelectionConfig(
            slice_count=int(
                validation_selection_raw.get(
                    "slice_count", validation_selection_defaults.slice_count
                )
            ),
            std_penalty=float(
                validation_selection_raw.get(
                    "std_penalty", validation_selection_defaults.std_penalty
                )
            ),
            max_slice_degradation=(
                float(validation_selection_raw["max_slice_degradation"])
                if validation_selection_raw.get("max_slice_degradation") is not None
                else None
            ),
        ),
    )

    multilayer_raw = raw.get("multilayer", {})
    multilayer_defaults = MultiLayerConfig()
    protocol_raw = multilayer_raw.get("protocol", {})
    protocol_defaults = MultiLayerProtocolConfig()
    selection_raw = multilayer_raw.get("selection", {})
    selection_defaults = MultiLayerSelectionConfig()
    progressive_raw = multilayer_raw.get("progressive_scaling", {})
    progressive_defaults = ProgressiveScalingConfig()
    materialization_raw = multilayer_raw.get("materialization", {})
    materialization_defaults = MultiLayerMaterializationConfig()
    multilayer = MultiLayerConfig(
        enabled=bool(multilayer_raw.get("enabled", multilayer_defaults.enabled)),
        sensitivity_layers=tuple(
            int(item)
            for item in multilayer_raw.get(
                "sensitivity_layers", multilayer_defaults.sensitivity_layers
            )
        ),
        methods=tuple(
            str(item)
            for item in multilayer_raw.get("methods", multilayer_defaults.methods)
        ),
        compression_fractions=tuple(
            float(item)
            for item in multilayer_raw.get(
                "compression_fractions", multilayer_defaults.compression_fractions
            )
        ),
        cumulative_layer_counts=tuple(
            int(item)
            for item in multilayer_raw.get(
                "cumulative_layer_counts", multilayer_defaults.cumulative_layer_counts
            )
        ),
        include_all_layers=bool(
            multilayer_raw.get("include_all_layers", multilayer_defaults.include_all_layers)
        ),
        trace_refresh_modes=tuple(
            str(item)
            for item in multilayer_raw.get(
                "trace_refresh_modes", multilayer_defaults.trace_refresh_modes
            )
        ),
        global_compression_targets=tuple(
            float(item)
            for item in multilayer_raw.get(
                "global_compression_targets",
                multilayer_defaults.global_compression_targets,
            )
        ),
        quality_budget=float(
            multilayer_raw.get("quality_budget", multilayer_defaults.quality_budget)
        ),
        max_layer_degradation=float(
            multilayer_raw.get(
                "max_layer_degradation", multilayer_defaults.max_layer_degradation
            )
        ),
        candidates_per_invocation=int(
            multilayer_raw.get(
                "candidates_per_invocation",
                multilayer_defaults.candidates_per_invocation,
            )
        ),
        max_step_seconds=float(
            multilayer_raw.get(
                "max_step_seconds", multilayer_defaults.max_step_seconds
            )
        ),
        include_hybrid_recipes=bool(
            multilayer_raw.get(
                "include_hybrid_recipes",
                multilayer_defaults.include_hybrid_recipes,
            )
        ),
        protocol=MultiLayerProtocolConfig(
            replication_seeds=tuple(
                int(item)
                for item in protocol_raw.get(
                    "replication_seeds", protocol_defaults.replication_seeds
                )
            ),
            validation_repeats=int(
                protocol_raw.get(
                    "validation_repeats", protocol_defaults.validation_repeats
                )
            ),
            test_repeats=int(
                protocol_raw.get("test_repeats", protocol_defaults.test_repeats)
            ),
            materialized_repeats=int(
                protocol_raw.get(
                    "materialized_repeats", protocol_defaults.materialized_repeats
                )
            ),
            aggregation=str(
                protocol_raw.get("aggregation", protocol_defaults.aggregation)
            ),
        ),
        selection=MultiLayerSelectionConfig(
            strategy=str(
                selection_raw.get("strategy", selection_defaults.strategy)
            ),
            per_method_per_cell=int(
                selection_raw.get(
                    "per_method_per_cell",
                    selection_defaults.per_method_per_cell,
                )
            ),
            include_global_winners=bool(
                selection_raw.get(
                    "include_global_winners",
                    selection_defaults.include_global_winners,
                )
            ),
            balance_layer_counts=bool(
                selection_raw.get(
                    "balance_layer_counts",
                    selection_defaults.balance_layer_counts,
                )
            ),
            balance_refresh_modes=bool(
                selection_raw.get(
                    "balance_refresh_modes",
                    selection_defaults.balance_refresh_modes,
                )
            ),
            max_selected=int(
                selection_raw.get(
                    "max_selected", selection_defaults.max_selected
                )
            ),
        ),
        progressive_scaling=ProgressiveScalingConfig(
            enabled=bool(
                progressive_raw.get("enabled", progressive_defaults.enabled)
            ),
            layer_counts=tuple(
                int(item)
                for item in progressive_raw.get(
                    "layer_counts", progressive_defaults.layer_counts
                )
            ),
            min_accepted_recipes=int(
                progressive_raw.get(
                    "min_accepted_recipes",
                    progressive_defaults.min_accepted_recipes,
                )
            ),
            stop_on_failure=bool(
                progressive_raw.get(
                    "stop_on_failure", progressive_defaults.stop_on_failure
                )
            ),
        ),
        materialization=MultiLayerMaterializationConfig(
            enabled=bool(
                materialization_raw.get(
                    "enabled", materialization_defaults.enabled
                )
            ),
            output_dir=str(
                materialization_raw.get(
                    "output_dir", materialization_defaults.output_dir
                )
            ),
            selected_plan_count=int(
                materialization_raw.get(
                    "selected_plan_count",
                    materialization_defaults.selected_plan_count,
                )
            ),
            strategy=str(
                materialization_raw.get(
                    "strategy",
                    materialization_defaults.strategy,
                )
            ),
            save_layer_weights=bool(
                materialization_raw.get(
                    "save_layer_weights",
                    materialization_defaults.save_layer_weights,
                )
            ),
        ),
    )

    artifacts = ArtifactSpecConfig(
        root_dir=str(raw["artifacts"]["root_dir"]),
        manifest_path=str(raw["artifacts"]["manifest_path"]),
        schema_version=str(raw["artifacts"].get("schema_version", "1.0")),
    )

    model_raw = raw["model"]
    model = ModelConfig(
        name=str(model_raw["name"]),
        kind=str(model_raw["kind"]),
        load_source=str(model_raw.get("load_source", "hf")),
        torch_dtype=str(model_raw.get("torch_dtype", "auto")),
        device_map=str(model_raw.get("device_map", "auto")),
        local_files_only=bool(model_raw.get("local_files_only", False)),
        trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
    )

    h1_raw = raw["stages"]["h1"]
    h1 = H1Config(
        enabled=bool(h1_raw["enabled"]),
        layer=int(h1_raw["layer"]),
        mlp_inputs_path=str(h1_raw["mlp_inputs_path"]),
        experts_dir=str(h1_raw["experts_dir"]),
        output_pairs_path=str(h1_raw["output_pairs_path"]),
        output_diagnostics_path=str(h1_raw["output_diagnostics_path"]),
        output_merged_dir=str(h1_raw["output_merged_dir"]),
        experts_source=str(h1_raw.get("experts_source", "saved_experts")),
        auto_generate_inputs_if_missing=bool(h1_raw.get("auto_generate_inputs_if_missing", False)),
        mode=str(h1_raw.get("mode", "pair")),
        merge_method=str(h1_raw.get("merge_method", "geometry")),
        proj_dim=int(h1_raw.get("proj_dim", 8)),
        n_inputs=int(h1_raw.get("n_inputs", 4096)),
        batch_size=int(h1_raw.get("batch_size", 64)),
        mp_factor=float(h1_raw.get("mp_factor", 1.1)),
        cluster_count=int(h1_raw.get("cluster_count", 0)),
        geometry_rank=int(h1_raw.get("geometry_rank", 64)),
        shrink_tau=float(h1_raw.get("shrink_tau", 0.0)),
        null_trials=int(h1_raw.get("null_trials", 32)),
        bootstrap_trials=int(h1_raw.get("bootstrap_trials", 32)),
        null_quantile=float(h1_raw.get("null_quantile", 0.95)),
        stability_max_angle_deg=float(h1_raw.get("stability_max_angle_deg", 30.0)),
    )

    h2_raw = raw["stages"]["h2"]
    h2 = H2Config(
        enabled=bool(h2_raw["enabled"]),
        layer=int(h2_raw["layer"]),
        mode=str(h2_raw["mode"]),
        output_bias_path=str(h2_raw["output_bias_path"]),
        importance_matrix_path=str(h2_raw.get("importance_matrix_path", "")),
        importance_vector_path=str(h2_raw.get("importance_vector_path", "")),
        gate_weight_path=str(h2_raw.get("gate_weight_path", "")),
        auto_from_model=bool(h2_raw.get("auto_from_model", True)),
        alpha=float(h2_raw.get("alpha", 0.2)),
        lam=float(h2_raw.get("lam", 1e-3)),
        mp_factor=float(h2_raw.get("mp_factor", 1.1)),
        normalize=str(h2_raw.get("normalize", "maxabs")),
        scale=float(h2_raw.get("scale", 0.1)),
    )

    h3_raw = raw["stages"]["h3"]
    h3 = H3Config(
        enabled=bool(h3_raw["enabled"]),
        routing_probs_path=str(h3_raw["routing_probs_path"]),
        output_diagnostics_path=str(h3_raw["output_diagnostics_path"]),
        auto_from_model=bool(h3_raw.get("auto_from_model", True)),
        tau=float(h3_raw.get("tau", 0.92)),
        k_min=int(h3_raw.get("k_min", 1)),
        k_max=int(h3_raw.get("k_max", 8)),
    )

    return ExperimentConfig(
        name=str(raw["name"]),
        model=model,
        runtime_top_k=int(raw["runtime"]["top_k"]),
        runtime_expected_routed_experts=(
            int(raw["runtime"]["expected_routed_experts"])
            if raw["runtime"].get("expected_routed_experts") is not None
            else None
        ),
        runtime_expected_shared_experts=(
            int(raw["runtime"]["expected_shared_experts"])
            if raw["runtime"].get("expected_shared_experts") is not None
            else None
        ),
        runtime_strict_architecture=bool(raw["runtime"].get("strict_architecture", False)),
        layers=[int(x) for x in raw["layers"]],
        constants=constants,
        artifacts=artifacts,
        profile=profile,
        calibration=calibration,
        pilot=pilot,
        multilayer=multilayer,
        h1=h1,
        h2=h2,
        h3=h3,
    )



def config_hash(config: ExperimentConfig) -> str:
    config_payload = dict(config.__dict__)
    pilot_payload = dict(config.pilot.__dict__)
    if config.pilot.validation_selection == ValidationSelectionConfig():
        pilot_payload.pop("validation_selection")
    if not config.pilot.spectral_rank_schedule:
        pilot_payload.pop("spectral_rank_schedule")
    if config.pilot.spectral_rank_policy == SpectralRankPolicyConfig():
        pilot_payload.pop("spectral_rank_policy")
    if config.pilot.merge_weighting == MergeWeightingConfig():
        pilot_payload.pop("merge_weighting")
    config_payload["pilot"] = pilot_payload
    if config.multilayer == MultiLayerConfig():
        config_payload.pop("multilayer")
    payload = json.dumps(config_payload, default=lambda x: x.__dict__, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()
