from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
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
class PilotConfig:
    enabled: bool = False
    layer: int = 0
    methods: Tuple[str, ...] = (
        "random_merge",
        "usage_prune",
        "functional_cosine",
        "rmt_geometry",
    )
    compression_fractions: Tuple[float, ...] = (0.1, 0.25, 0.4)
    quality_budgets: Tuple[float, ...] = (0.01, 0.03, 0.05)
    seeds: Tuple[int, ...] = (123, 456, 789)
    max_eval_batches: int = 32


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
    layers: List[int]
    constants: Dict[str, Dict[str, Any]]
    artifacts: ArtifactSpecConfig
    profile: ProfileConfig
    calibration: CalibrationConfig
    pilot: PilotConfig
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

    if not isinstance(data["layers"], list) or len(data["layers"]) == 0:
        raise ValueError("layers must be a non-empty list")

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
    allowed_methods = {"random_merge", "usage_prune", "functional_cosine", "rmt_geometry"}
    methods = tuple(str(item) for item in pilot.get("methods", sorted(allowed_methods)))
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        raise ValueError(f"pilot.methods contains unsupported methods: {unknown_methods}")
    for value in pilot.get("compression_fractions", (0.1, 0.25, 0.4)):
        if not 0.0 < float(value) < 1.0:
            raise ValueError("pilot.compression_fractions values must be in (0, 1)")
    for value in pilot.get("quality_budgets", (0.01, 0.03, 0.05)):
        if float(value) < 0.0:
            raise ValueError("pilot.quality_budgets values must be non-negative")
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
        if int(pilot.get("max_eval_batches", 32)) < 1:
            raise ValueError("pilot.max_eval_batches must be >= 1")



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
        layers=[int(x) for x in raw["layers"]],
        constants=constants,
        artifacts=artifacts,
        profile=profile,
        calibration=calibration,
        pilot=pilot,
        h1=h1,
        h2=h2,
        h3=h3,
    )



def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(config.__dict__, default=lambda x: x.__dict__, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()
