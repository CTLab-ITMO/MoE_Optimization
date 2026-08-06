from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from moe_opt_core.config import ExperimentConfig, config_hash
from moe_opt_core.experiment import CandidateMethod
from moe_opt_core.experiment.candidates import build_candidate_plans
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode
from moe_opt_core.experiment.evaluation import compare_evaluations
from moe_opt_core.experiment.trace_validation import validate_trace_payload
from moe_opt_runner.evaluation import append_incremental_record, evaluate_language_model
from moe_opt_runner.runtime import RuntimeContext
from moe_opt_runner.transforms import reversible_one_layer_transform


def _require_path(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.TRACE_ARTIFACT_MISSING,
            f"Missing {purpose}: {path}",
        )
    return path


def _metrics_dict(metrics) -> dict[str, object]:
    return asdict(metrics)


def _plan_dict(plan) -> dict[str, object]:
    return {
        "candidate_id": plan.candidate_id,
        "layer": plan.layer,
        "method": plan.method.value,
        "action": plan.action.value,
        "compression_fraction": plan.compression_fraction,
        "groups": [list(group) for group in plan.groups],
        "seed": plan.seed,
    }


def _load_pilot_artifacts(config: ExperimentConfig) -> dict[str, object]:
    layer = config.pilot.layer
    trace_path = _require_path(
        Path(config.calibration.output_dir) / "calibration" / f"layer_{layer}.pt",
        "calibration trace",
    )
    embedding_path = _require_path(
        Path(config.h1.output_diagnostics_path).with_suffix(".pt"),
        "H1 embedding artifact",
    )
    validation_batches_path = _require_path(
        Path(config.calibration.output_dir) / "validation" / "token_batches.pt",
        "validation token batches",
    )
    test_batches_path = _require_path(
        Path(config.calibration.output_dir) / "test" / "token_batches.pt",
        "test token batches",
    )
    return {
        "trace": torch.load(trace_path, map_location="cpu"),
        "embedding": torch.load(embedding_path, map_location="cpu"),
        "validation_batches": torch.load(validation_batches_path, map_location="cpu"),
        "test_batches": torch.load(test_batches_path, map_location="cpu"),
    }


def _evaluate(
    config: ExperimentConfig,
    ctx: RuntimeContext,
    batches,
):
    return evaluate_language_model(
        ctx.model,
        batches,
        warmup_steps=config.profile.warmup_steps,
        measure_steps=min(config.profile.measure_steps, config.pilot.max_eval_batches),
    )


def run_one_layer_pilot(config: ExperimentConfig, ctx: RuntimeContext) -> Dict[str, object]:
    if not config.h1.enabled:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.INVALID_CONFIG,
            "The one-layer pilot requires stages.h1.enabled=true",
        )
    artifacts = _load_pilot_artifacts(config)
    handle = ctx.handles_by_layer.get(config.pilot.layer)
    if handle is None:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.LAYER_UNAVAILABLE,
            f"Pilot layer {config.pilot.layer} was not discovered",
        )
    runtime_contract = ctx.runtime_contracts_by_layer.get(config.pilot.layer)
    if runtime_contract is None:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            f"Pilot layer {config.pilot.layer} has no validated runtime contract",
        )

    trace = artifacts["trace"]
    trace_metadata = validate_trace_payload(
        trace,
        runtime_contract,
        require_real_traces=config.pilot.require_real_traces,
    )
    embedding = artifacts["embedding"]
    num_experts = runtime_contract.routed_experts
    if len(handle.experts) != num_experts:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            "Pilot expert collection does not match the routed expert contract",
        )
    plans = build_candidate_plans(
        num_experts=num_experts,
        layer=config.pilot.layer,
        methods=config.pilot.methods,
        compression_fractions=config.pilot.compression_fractions,
        seeds=config.pilot.seeds,
        rmt_pairs=embedding["rmt_pairs"],
        functional_embedding=embedding["raw_embedding"],
        routing_probs=trace["routing_probs"],
    )

    record_path = Path(config.artifacts.root_dir) / "pilot_runs.jsonl"
    run_hash = config_hash(config)
    baseline_validation = _evaluate(config, ctx, artifacts["validation_batches"])
    baseline_test = _evaluate(config, ctx, artifacts["test_batches"])
    append_incremental_record(
        record_path,
        {
            "config_hash": run_hash,
            "record_type": "baseline",
            "split": "validation",
            "metrics": _metrics_dict(baseline_validation),
        },
    )
    append_incremental_record(
        record_path,
        {
            "config_hash": run_hash,
            "record_type": "baseline",
            "split": "test",
            "metrics": _metrics_dict(baseline_test),
        },
    )

    validation_results = []
    expert_cache = {}
    progress = tqdm(plans, desc="Evaluate one-layer candidates")
    for plan in progress:
        progress.set_postfix_str(plan.candidate_id)
        merge_method = "geometry" if plan.method == CandidateMethod.RMT_GEOMETRY else "baseline"
        try:
            with reversible_one_layer_transform(
                ctx,
                plan,
                merge_method=merge_method,
                geometry_rank=config.h1.geometry_rank,
                shrink_tau=config.h1.shrink_tau,
                expert_cache=expert_cache,
            ) as transform:
                metrics = _evaluate(config, ctx, artifacts["validation_batches"])
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            append_incremental_record(
                record_path,
                {
                    "config_hash": run_hash,
                    "record_type": "failed_candidate",
                    "split": "validation",
                    "plan": _plan_dict(plan),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            continue
        comparison = compare_evaluations(
            candidate_id=plan.candidate_id,
            baseline=baseline_validation,
            candidate=metrics,
            quality_budgets=config.pilot.quality_budgets,
        )
        row = {
            "config_hash": run_hash,
            "record_type": "candidate",
            "split": "validation",
            "plan": _plan_dict(plan),
            "transform": asdict(transform),
            "metrics": _metrics_dict(metrics),
            "relative_perplexity_degradation": comparison.relative_perplexity_degradation,
            "throughput_gain": comparison.throughput_gain,
            "memory_reduction": comparison.memory_reduction,
            "accepted_quality_budgets": dict(comparison.accepted_quality_budgets),
        }
        append_incremental_record(record_path, row)
        validation_results.append((plan, comparison, row))

    eligible = [
        item
        for item in validation_results
        if item[1].relative_perplexity_degradation <= max(config.pilot.quality_budgets)
    ]
    winners = {}
    for item in eligible:
        plan, comparison, _row = item
        key = (plan.method.value, plan.compression_fraction)
        current = winners.get(key)
        if current is None or comparison.candidate.perplexity < current[1].candidate.perplexity:
            winners[key] = item

    test_results = []
    for plan, _validation_comparison, _validation_row in tqdm(
        winners.values(),
        desc="Evaluate selected candidates on test",
    ):
        merge_method = "geometry" if plan.method == CandidateMethod.RMT_GEOMETRY else "baseline"
        try:
            with reversible_one_layer_transform(
                ctx,
                plan,
                merge_method=merge_method,
                geometry_rank=config.h1.geometry_rank,
                shrink_tau=config.h1.shrink_tau,
                expert_cache=expert_cache,
            ) as transform:
                metrics = _evaluate(config, ctx, artifacts["test_batches"])
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            append_incremental_record(
                record_path,
                {
                    "config_hash": run_hash,
                    "record_type": "failed_selected_candidate",
                    "split": "test",
                    "plan": _plan_dict(plan),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            continue
        comparison = compare_evaluations(
            candidate_id=plan.candidate_id,
            baseline=baseline_test,
            candidate=metrics,
            quality_budgets=config.pilot.quality_budgets,
        )
        row = {
            "config_hash": run_hash,
            "record_type": "selected_candidate",
            "split": "test",
            "plan": _plan_dict(plan),
            "transform": asdict(transform),
            "metrics": _metrics_dict(metrics),
            "relative_perplexity_degradation": comparison.relative_perplexity_degradation,
            "throughput_gain": comparison.throughput_gain,
            "memory_reduction": comparison.memory_reduction,
            "accepted_quality_budgets": dict(comparison.accepted_quality_budgets),
        }
        append_incremental_record(record_path, row)
        test_results.append(row)

    summary = {
        "layer": config.pilot.layer,
        "trace_metadata": {
            "source": trace_metadata.source.value,
            "routed_experts": trace_metadata.routed_experts,
            "shared_experts": trace_metadata.shared_experts,
            "routed_top_k": trace_metadata.routed_top_k,
        },
        "shared_experts_modified": False,
        "candidate_count": len(plans),
        "selected_candidate_count": len(test_results),
        "record_path": str(record_path),
        "baseline_validation": _metrics_dict(baseline_validation),
        "baseline_test": _metrics_dict(baseline_test),
        "selected_test_results": test_results,
    }
    summary_path = Path(config.artifacts.root_dir) / "pilot_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
