from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from moe_opt_core.artifacts import write_artifact_manifest as write_artifact_manifest_file, write_run_summary
from moe_opt_core.config import ExperimentConfig, config_hash
from moe_opt_core.experiment import ExpectedExperimentFailure, ExperimentFailureCode, TraceSource
from moe_opt_core.pipelines.h1_merge import merge_cluster_modules, run_h1_pipeline
from moe_opt_core.pipelines.h2_bias import run_h2_pipeline
from moe_opt_core.pipelines.h3_routing import run_h3_pipeline
from moe_opt_runner.runtime import RuntimeContext


def _layer_handle(ctx: RuntimeContext, layer: int):
    handle = ctx.handles_by_layer.get(layer)
    if handle is None:
        raise RuntimeError(f"Layer {layer} not found in discovered MoE layers: {sorted(ctx.handles_by_layer.keys())}")
    return handle


def _load_experts_from_dir(experts_dir: str, layer: int):
    base = Path(experts_dir) / str(layer)
    experts = []
    idx = 0
    while True:
        p = base / f"{idx}.pt"
        if not p.exists():
            break
        experts.append(torch.load(p, map_location="cpu", weights_only=False))
        idx += 1
    if not experts:
        raise FileNotFoundError(f"No experts found in {base}")
    return experts


def _ensure_h1_inputs(config: ExperimentConfig, ctx: Optional[RuntimeContext]) -> str:
    p = Path(config.h1.mlp_inputs_path)
    if p.exists():
        return str(p)
    if config.calibration.source != TraceSource.SYNTHETIC.value:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.TRACE_ARTIFACT_MISSING,
            f"H1 trace inputs not found: {p}",
            context={
                "calibration_source": config.calibration.source,
                "hint": "Run the collect_traces stage or provide an artifact path",
            },
        )
    if not config.h1.auto_generate_inputs_if_missing:
        raise FileNotFoundError(f"Synthetic H1 inputs not found: {p}")
    if ctx is None:
        raise RuntimeError("Runtime context is required to auto-generate H1 inputs")

    seed = int(config.constants["training"]["seed"])
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = _layer_handle(ctx, config.h1.layer)
    gate_w = ctx.adapter.get_gate_weight(h.gate).detach().cpu()
    d = int(gate_w.shape[1])
    x = torch.randn((config.h1.n_inputs, d), generator=g, dtype=torch.float32)

    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, p)
    return str(p)


def _ensure_h2_inputs(config: ExperimentConfig, ctx: Optional[RuntimeContext]) -> tuple[str, str, str]:
    h2 = config.h2
    imp_mat = h2.importance_matrix_path
    imp_vec = h2.importance_vector_path
    gate = h2.gate_weight_path

    if h2.mode == "covariance" and imp_mat and Path(imp_mat).exists():
        return imp_mat, imp_vec, gate

    if h2.mode == "gate_spike" and imp_vec and gate and Path(imp_vec).exists() and Path(gate).exists():
        return imp_mat, imp_vec, gate

    if not h2.auto_from_model:
        return imp_mat, imp_vec, gate
    if ctx is None:
        raise RuntimeError("Runtime context is required for auto-generated H2 inputs")

    handle = _layer_handle(ctx, h2.layer)
    gate_w = ctx.adapter.get_gate_weight(handle.gate).detach().cpu()
    seed = int(config.constants["training"]["seed"])

    if not gate:
        gate = str(Path(config.artifacts.root_dir) / f"h2_gate_weight_layer{h2.layer}.pt")
    gate_path = Path(gate)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gate_w, gate_path)

    if h2.mode == "gate_spike":
        if not imp_vec:
            imp_vec = str(Path(config.artifacts.root_dir) / f"h2_importance_vec_layer{h2.layer}.pt")
        imp_path = Path(imp_vec)
        imp_path.parent.mkdir(parents=True, exist_ok=True)
        vec = gate_w.norm(dim=1)
        torch.save(vec, imp_path)
    else:
        if config.calibration.source != TraceSource.SYNTHETIC.value:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.TRACE_ARTIFACT_MISSING,
                "H2 covariance mode requires a real importance matrix artifact",
                context={"importance_matrix_path": imp_mat},
            )
        if not imp_mat:
            imp_mat = str(Path(config.artifacts.root_dir) / f"h2_importance_matrix_layer{h2.layer}.pt")
        mpath = Path(imp_mat)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = 1024
        e = gate_w.shape[0]
        x = torch.randn((n, e), generator=g)
        torch.save(x, mpath)

    return imp_mat, imp_vec, gate


def _ensure_h3_inputs(config: ExperimentConfig, ctx: Optional[RuntimeContext]) -> str:
    p = Path(config.h3.routing_probs_path)
    if p.exists():
        return str(p)
    if config.calibration.source != TraceSource.SYNTHETIC.value:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.TRACE_ARTIFACT_MISSING,
            f"H3 routing probabilities not found: {p}",
            context={"hint": "Run the collect_traces stage or provide an artifact path"},
        )
    if not config.h3.auto_from_model:
        raise FileNotFoundError(f"Synthetic H3 routing probs not found: {p}")
    if ctx is None:
        raise RuntimeError("Runtime context is required to auto-generate H3 routing probabilities")

    h = _layer_handle(ctx, config.h1.layer)
    gate_w = ctx.adapter.get_gate_weight(h.gate).detach().cpu()
    seed = int(config.constants["training"]["seed"])
    g = torch.Generator(device="cpu").manual_seed(seed)

    t = 2048
    d = gate_w.shape[1]
    x = torch.randn((t, d), generator=g)
    logits = x @ gate_w.t()
    probs = torch.softmax(logits, dim=-1)

    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probs, p)
    return str(p)


def run_h1_stage(config: ExperimentConfig, ctx: Optional[RuntimeContext] = None) -> Dict[str, object]:
    h1 = config.h1
    if h1.experts_source == "hf_model":
        if ctx is None:
            raise RuntimeError("Runtime context is required for experts_source='hf_model'")
        experts = list(_layer_handle(ctx, h1.layer).experts)
    else:
        experts = _load_experts_from_dir(h1.experts_dir, h1.layer)

    mlp_inputs_path = _ensure_h1_inputs(config, ctx)
    result = run_h1_pipeline(
        experts=experts,
        mlp_inputs_path=mlp_inputs_path,
        n_inputs=h1.n_inputs,
        mode=h1.mode,
        proj_dim=h1.proj_dim,
        batch_size=h1.batch_size,
        seed=int(config.constants["training"]["seed"]),
        mp_factor=h1.mp_factor,
        cluster_count=h1.cluster_count,
        output_pairs_path=h1.output_pairs_path,
        output_diagnostics_path=h1.output_diagnostics_path,
        validation_inputs_path=str(
            Path(config.calibration.output_dir) / "validation" / f"layer_{h1.layer}.pt"
        ),
        null_trials=h1.null_trials,
        bootstrap_trials=h1.bootstrap_trials,
        null_quantile=h1.null_quantile,
        stability_max_angle_deg=h1.stability_max_angle_deg,
    )

    merged_count = 0
    if h1.mode == "cluster" and result.clusters:
        out_dir = Path(h1.output_merged_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, cluster in enumerate(result.clusters):
            merged = merge_cluster_modules(
                experts=experts,
                cluster=cluster,
                method=h1.merge_method,
                rank=h1.geometry_rank,
                shrink_tau=h1.shrink_tau,
            )
            torch.save(merged, out_dir / f"cluster_{idx}.pt")
            merged_count += 1

    return {
        "pairs": len(result.pairs),
        "clusters": len(result.clusters),
        "merged_clusters": merged_count,
        "spike_rank": result.diagnostics.get("rank", 0),
        "embedding_path": result.diagnostics.get("embedding_path", ""),
    }


def run_h2_stage(config: ExperimentConfig, ctx: Optional[RuntimeContext] = None) -> Dict[str, object]:
    h2 = config.h2
    imp_mat, imp_vec, gate = _ensure_h2_inputs(config, ctx)
    diag_path = str(Path(h2.output_bias_path).with_suffix(".json"))

    result = run_h2_pipeline(
        mode=h2.mode,
        output_bias_path=h2.output_bias_path,
        output_diagnostics_path=diag_path,
        importance_matrix_path=imp_mat,
        importance_vector_path=imp_vec,
        gate_weight_path=gate,
        alpha=h2.alpha,
        lam=h2.lam,
        mp_factor=h2.mp_factor,
        normalize=h2.normalize,
        scale=h2.scale,
    )
    return {
        "bias_norm": float(torch.linalg.norm(result.bias).item()),
        "method": result.diagnostics.get("method", h2.mode),
    }


def run_h3_stage(config: ExperimentConfig, ctx: Optional[RuntimeContext] = None) -> Dict[str, object]:
    h3 = config.h3
    routing_probs_path = _ensure_h3_inputs(config, ctx)
    result = run_h3_pipeline(
        routing_probs_path=routing_probs_path,
        output_diagnostics_path=h3.output_diagnostics_path,
        tau=h3.tau,
        k_min=h3.k_min,
        k_max=h3.k_max,
    )
    return dict(result.diagnostics)


def write_artifact_manifest(config: ExperimentConfig, stage_outputs: Dict[str, Dict[str, object]]) -> None:
    write_artifact_manifest_file(
        schema_version=config.artifacts.schema_version,
        config_hash=config_hash(config),
        root_dir=config.artifacts.root_dir,
        manifest_path=config.artifacts.manifest_path,
        paths={
            "h1_pairs": config.h1.output_pairs_path,
            "h1_diag": config.h1.output_diagnostics_path,
            "h2_bias": config.h2.output_bias_path,
            "h3_diag": config.h3.output_diagnostics_path,
            "trace_manifest": Path(config.calibration.output_dir) / "trace_manifest.json",
            "pilot_runs": Path(config.artifacts.root_dir) / "pilot_runs.jsonl",
            "pilot_summary": Path(config.artifacts.root_dir) / "pilot_summary.json",
        },
    )
    write_run_summary(
        root_dir=config.artifacts.root_dir,
        summary_name="run_summary.json",
        payload=stage_outputs,
    )




