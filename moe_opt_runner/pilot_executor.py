from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Dict, Mapping, Sequence

import torch
from tqdm import tqdm

from moe_opt_core.config import ExperimentConfig, config_hash
from moe_opt_core.experiment import (
    CandidateMethod,
    EvaluationMetricsContract,
    PilotExecutionStateContract,
    PilotStep,
    PilotStepResultContract,
    PilotStepStatus,
)
from moe_opt_core.experiment.candidates import build_candidate_plans
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode
from moe_opt_core.experiment.evaluation import compare_evaluations
from moe_opt_core.experiment.pilot_planning import (
    build_pilot_execution_state,
    next_pilot_step,
    pending_candidate_ids,
    select_validation_winners,
)
from moe_opt_core.experiment.pilot_validation import (
    aggregate_evaluation_metrics,
    partition_validation_batches,
    summarize_validation_degradations,
)
from moe_opt_core.experiment.trace_validation import validate_trace_payload
from moe_opt_core.pipelines.rmt_diagnostics import spectral_embeddings_at_ranks
from moe_opt_runner.evaluation import append_incremental_record, evaluate_language_model
from moe_opt_runner.pilot_artifacts import (
    read_incremental_records,
    read_pilot_state_config_hash,
    read_selection,
    write_json_atomic,
    write_pilot_state,
    write_selection,
)
from moe_opt_runner.runtime import RuntimeContext
from moe_opt_runner.transforms import reversible_one_layer_transform


def _require_path(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.TRACE_ARTIFACT_MISSING,
            f"Missing {purpose}: {path}",
        )
    return path


def _metrics_dict(metrics: EvaluationMetricsContract) -> dict[str, object]:
    return asdict(metrics)


def _metrics_contract(payload: Mapping[str, object]) -> EvaluationMetricsContract:
    return EvaluationMetricsContract(
        nll=float(payload["nll"]),
        perplexity=float(payload["perplexity"]),
        tokens_per_second=float(payload["tokens_per_second"]),
        peak_memory_mb=float(payload["peak_memory_mb"]),
        model_parameter_mb=float(payload["model_parameter_mb"]),
        evaluated_tokens=int(payload["evaluated_tokens"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
    )


def _plan_dict(plan) -> dict[str, object]:
    payload = {
        "candidate_id": plan.candidate_id,
        "layer": plan.layer,
        "method": plan.method.value,
        "action": plan.action.value,
        "compression_fraction": plan.compression_fraction,
        "groups": [list(group) for group in plan.groups],
        "seed": plan.seed,
        "diagnostics": dict(plan.diagnostics),
    }
    if plan.group_merge_weights:
        payload["merge_weighting"] = plan.merge_weighting.value
        payload["group_merge_weights"] = [
            list(weights) for weights in plan.group_merge_weights
        ]
    return payload


def pilot_step_result_dict(result: PilotStepResultContract) -> dict[str, object]:
    return {
        "step": result.step.value,
        "status": result.status.value,
        "processed_items": result.processed_items,
        "remaining_items": result.remaining_items,
        "artifact_paths": list(result.artifact_paths),
    }


class ResumablePilotExecutor:
    def __init__(self, config: ExperimentConfig, ctx: RuntimeContext | None = None) -> None:
        self.config = config
        self.ctx = ctx
        self.run_hash = config_hash(config)
        self.root = Path(config.artifacts.root_dir)
        self.record_path = self.root / "pilot_runs.jsonl"
        self.selection_path = self.root / "pilot_selection.json"
        self.state_path = self.root / "pilot_state.json"
        self.summary_path = self.root / "pilot_summary.json"

    def _require_context(self) -> RuntimeContext:
        if self.ctx is None:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.MODEL_UNAVAILABLE,
                "This pilot step requires a loaded model runtime",
            )
        return self.ctx

    def _batch_path(self, split: str) -> Path:
        return Path(self.config.calibration.output_dir) / split / "token_batches.pt"

    def _trace_path(self) -> Path:
        return (
            Path(self.config.calibration.output_dir)
            / "calibration"
            / f"layer_{self.config.pilot.layer}.pt"
        )

    def _embedding_path(self) -> Path:
        return Path(self.config.h1.output_diagnostics_path).with_suffix(".pt")

    def _records(self) -> tuple[Mapping[str, object], ...]:
        return read_incremental_records(self.record_path)

    def _selected_ids(self) -> tuple[str, ...]:
        return read_selection(self.selection_path, config_hash=self.run_hash)

    def _artifact_steps(self) -> tuple[PilotStep, ...]:
        completed: list[PilotStep] = []
        trace_root = Path(self.config.calibration.output_dir)
        for split, step in (
            ("calibration", PilotStep.TRACES_CALIBRATION),
            ("validation", PilotStep.TRACES_VALIDATION),
            ("test", PilotStep.TRACES_TEST),
        ):
            trace_paths = (
                trace_root / "trace_manifest.json",
                trace_root / split / f"layer_{self.config.pilot.layer}.pt",
                self._batch_path(split),
            )
            if all(path.exists() for path in trace_paths):
                completed.append(step)
        h1_paths = (
            Path(self.config.h1.output_diagnostics_path),
            self._embedding_path(),
            Path(self.config.h1.output_pairs_path),
        )
        if all(path.exists() for path in h1_paths):
            completed.append(PilotStep.H1_DIAGNOSTICS)
        return tuple(completed)

    def _load_trace_and_embedding(self):
        trace = torch.load(_require_path(self._trace_path(), "calibration trace"), map_location="cpu")
        embedding = torch.load(
            _require_path(self._embedding_path(), "H1 embedding artifact"),
            map_location="cpu",
        )
        return trace, embedding

    def _build_plans(self, *, num_experts: int, trace, embedding):
        spectral_embeddings = spectral_embeddings_at_ranks(
            embedding["raw_embedding"],
            self.config.pilot.spectral_pairing_ranks,
        )
        return build_candidate_plans(
            num_experts=num_experts,
            layer=self.config.pilot.layer,
            methods=self.config.pilot.methods,
            compression_fractions=self.config.pilot.compression_fractions,
            seeds=self.config.pilot.seeds,
            rmt_pairs=embedding["rmt_pairs"],
            spectral_embedding=embedding["spectral_embedding"],
            functional_embedding=embedding["raw_embedding"],
            routing_probs=trace["routing_probs"],
            hybrid_weights=(
                self.config.pilot.hybrid_pairing.rmt_weight,
                self.config.pilot.hybrid_pairing.functional_weight,
                self.config.pilot.hybrid_pairing.usage_weight,
            ),
            spectral_embeddings=spectral_embeddings,
            hybrid_weight_variants=tuple(
                (
                    variant.name,
                    (
                        variant.rmt_weight,
                        variant.functional_weight,
                        variant.usage_weight,
                    ),
                )
                for variant in self.config.pilot.hybrid_pairing.variants
            ),
            spectral_rank_schedule=tuple(
                (
                    assignment.method,
                    assignment.compression_fraction,
                    assignment.rank,
                )
                for assignment in self.config.pilot.spectral_rank_schedule
            ),
            merge_weighting_methods=self.config.pilot.merge_weighting.methods,
            merge_weighting_variants=tuple(
                (variant.name, variant.policy, variant.minimum_weight)
                for variant in self.config.pilot.merge_weighting.variants
            ),
        )

    def _load_trace_and_plans(self):
        ctx = self._require_context()
        trace, embedding = self._load_trace_and_embedding()
        handle = ctx.handles_by_layer.get(self.config.pilot.layer)
        runtime_contract = ctx.runtime_contracts_by_layer.get(self.config.pilot.layer)
        if handle is None or runtime_contract is None:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
                f"Pilot layer {self.config.pilot.layer} has no validated runtime contract",
            )
        validate_trace_payload(
            trace,
            runtime_contract,
            require_real_traces=self.config.pilot.require_real_traces,
        )
        if len(handle.experts) != runtime_contract.routed_experts:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
                "Pilot expert collection does not match the routed expert contract",
            )
        plans = self._build_plans(
            num_experts=runtime_contract.routed_experts,
            trace=trace,
            embedding=embedding,
        )
        return trace, plans

    def _load_plans_without_runtime(self):
        trace, embedding = self._load_trace_and_embedding()
        metadata = trace.get("metadata", {})
        observed_experts = metadata.get("routed_experts") if isinstance(metadata, Mapping) else None
        num_experts = self.config.runtime_expected_routed_experts or observed_experts
        if num_experts is None:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Cannot build candidate plans without a routed expert count",
            )
        if observed_experts is not None and int(observed_experts) != int(num_experts):
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
                "Trace routed expert count does not match experiment config",
            )
        return self._build_plans(
            num_experts=int(num_experts),
            trace=trace,
            embedding=embedding,
        )

    def _state(
        self,
        *,
        all_plan_ids: Sequence[str] | None = None,
    ) -> PilotExecutionStateContract:
        stored_hash = read_pilot_state_config_hash(self.state_path)
        if stored_hash is not None and stored_hash != self.run_hash:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Pilot state belongs to another experiment config",
                context={
                    "expected_config_hash": self.run_hash,
                    "observed_config_hash": stored_hash,
                },
            )
        plan_ids = tuple(all_plan_ids) if all_plan_ids is not None else ()
        return build_pilot_execution_state(
            config_hash=self.run_hash,
            records=self._records(),
            all_validation_candidate_ids=plan_ids,
            selected_candidate_ids=self._selected_ids(),
            artifact_steps=self._artifact_steps(),
            selection_written=self.selection_path.exists(),
            summary_written=self.summary_path.exists(),
        )

    def _save_result(
        self,
        result: PilotStepResultContract,
        *,
        all_plan_ids: Sequence[str] | None = None,
    ) -> PilotStepResultContract:
        write_pilot_state(self.state_path, self._state(all_plan_ids=all_plan_ids), result)
        return result

    def status(self) -> dict[str, object]:
        plan_ids: tuple[str, ...] = ()
        if self._trace_path().exists() and self._embedding_path().exists():
            plans = self._load_plans_without_runtime()
            plan_ids = tuple(plan.candidate_id for plan in plans)
        state = self._state(all_plan_ids=plan_ids)
        remaining_validation = pending_candidate_ids(
            plan_ids,
            state,
            split="validation",
            retry_failed=False,
        )
        remaining_test = pending_candidate_ids(
            state.selected_candidate_ids,
            state,
            split="test",
            retry_failed=False,
        )
        payload = asdict(state)
        payload["completed_steps"] = [step.value for step in state.completed_steps]
        payload["total_validation_candidates"] = len(plan_ids)
        payload["remaining_validation_candidates"] = len(remaining_validation)
        payload["remaining_test_candidates"] = len(remaining_test)
        payload["candidate_grid_available"] = bool(plan_ids)
        next_step = next_pilot_step(
            state.completed_steps,
            remaining_validation_candidates=len(remaining_validation),
            remaining_test_candidates=len(remaining_test),
        )
        payload["next_step"] = next_step.value if next_step is not None else None
        payload["state_path"] = str(self.state_path)
        return payload

    def is_step_complete(self, step: PilotStep) -> bool:
        return step in self._state().completed_steps

    def has_selection(self) -> bool:
        return self.selection_path.exists()

    def begin_step(self, step: PilotStep) -> None:
        write_pilot_state(
            self.state_path,
            self._state(),
            PilotStepResultContract(
                step=step,
                status=PilotStepStatus.PARTIAL,
                processed_items=0,
                remaining_items=0,
            ),
        )

    def skip_step(
        self,
        step: PilotStep,
        artifact_paths: Sequence[str | Path] = (),
    ) -> PilotStepResultContract:
        return self._save_result(
            PilotStepResultContract(
                step=step,
                status=PilotStepStatus.SKIPPED,
                processed_items=0,
                remaining_items=0,
                artifact_paths=tuple(str(path) for path in artifact_paths),
            )
        )

    def record_artifact_step(
        self,
        step: PilotStep,
        artifact_paths: Sequence[str | Path],
    ) -> PilotStepResultContract:
        missing = [str(path) for path in artifact_paths if not Path(path).exists()]
        if missing:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.TRACE_ARTIFACT_MISSING,
                f"Pilot step {step.value} did not create all expected artifacts",
                context={"missing": missing},
            )
        return self._save_result(
            PilotStepResultContract(
                step=step,
                status=PilotStepStatus.COMPLETED,
                processed_items=1,
                remaining_items=0,
                artifact_paths=tuple(str(path) for path in artifact_paths),
            )
        )

    def _evaluate(
        self,
        batches,
        *,
        warmup_steps: int | None = None,
        measure_steps: int | None = None,
    ) -> EvaluationMetricsContract:
        ctx = self._require_context()
        return evaluate_language_model(
            ctx.model,
            batches,
            warmup_steps=(
                self.config.profile.warmup_steps if warmup_steps is None else warmup_steps
            ),
            measure_steps=(
                min(self.config.profile.measure_steps, self.config.pilot.max_eval_batches)
                if measure_steps is None
                else measure_steps
            ),
        )

    def _uses_validation_slices(self, split: str) -> bool:
        selection = self.config.pilot.validation_selection
        return split == "validation" and (
            selection.slice_count > 1
            or selection.std_penalty > 0.0
            or selection.max_slice_degradation is not None
        )

    def _validation_batch_slices(self, batches) -> tuple[tuple[object, ...], ...]:
        evaluation_batch_count = min(
            len(batches),
            self.config.profile.measure_steps,
            self.config.pilot.max_eval_batches,
        )
        evaluation_batches = tuple(batches[:evaluation_batch_count])
        try:
            return partition_validation_batches(
                evaluation_batches,
                self.config.pilot.validation_selection.slice_count,
            )
        except ValueError as exc:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.EVALUATION_FAILED,
                "Validation data does not contain enough batches for the configured slices",
                context={
                    "available_batches": evaluation_batch_count,
                    "slice_count": self.config.pilot.validation_selection.slice_count,
                },
            ) from exc

    def _evaluate_validation_slices(
        self,
        batches,
    ) -> tuple[EvaluationMetricsContract, ...]:
        return tuple(
            self._evaluate(
                batch_slice,
                warmup_steps=0,
                measure_steps=len(batch_slice),
            )
            for batch_slice in self._validation_batch_slices(batches)
        )

    def _baseline_metrics(self, split: str) -> EvaluationMetricsContract:
        for row in reversed(self._records()):
            if (
                row.get("config_hash") == self.run_hash
                and row.get("record_type") == "baseline"
                and row.get("split") == split
                and isinstance(row.get("metrics"), Mapping)
            ):
                return _metrics_contract(row["metrics"])
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.PILOT_STATE_MISMATCH,
            f"Baseline for split '{split}' has not been evaluated",
        )

    def _baseline_validation_slice_metrics(self) -> tuple[EvaluationMetricsContract, ...]:
        for row in reversed(self._records()):
            if (
                row.get("config_hash") != self.run_hash
                or row.get("record_type") != "baseline"
                or row.get("split") != "validation"
            ):
                continue
            validation_slices = row.get("validation_slices")
            if not isinstance(validation_slices, Mapping):
                break
            metrics = validation_slices.get("metrics")
            if isinstance(metrics, Sequence):
                return tuple(
                    _metrics_contract(item)
                    for item in metrics
                    if isinstance(item, Mapping)
                )
            break
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.PILOT_STATE_MISMATCH,
            "Validation baseline does not contain slice metrics",
        )

    def run_baseline(self, split: str) -> PilotStepResultContract:
        if split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")
        if split == "test" and not self.selection_path.exists():
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Test baseline is locked until validation selection is written",
            )
        step = PilotStep.BASELINE_VALIDATION if split == "validation" else PilotStep.BASELINE_TEST
        if step in self._state().completed_steps:
            return self._save_result(
                PilotStepResultContract(step, PilotStepStatus.SKIPPED, 0, 0, (str(self.record_path),))
            )
        batches = torch.load(
            _require_path(self._batch_path(split), f"{split} token batches"),
            map_location="cpu",
        )
        validation_slice_metrics: tuple[EvaluationMetricsContract, ...] = ()
        if self._uses_validation_slices(split):
            validation_slice_metrics = self._evaluate_validation_slices(batches)
            metrics = aggregate_evaluation_metrics(validation_slice_metrics)
        else:
            metrics = self._evaluate(batches)
        record: dict[str, object] = {
            "config_hash": self.run_hash,
            "record_type": "baseline",
            "split": split,
            "metrics": _metrics_dict(metrics),
        }
        if validation_slice_metrics:
            record["validation_slices"] = {
                "slice_count": len(validation_slice_metrics),
                "metrics": [_metrics_dict(item) for item in validation_slice_metrics],
            }
        append_incremental_record(
            self.record_path,
            record,
        )
        return self._save_result(
            PilotStepResultContract(step, PilotStepStatus.COMPLETED, 1, 0, (str(self.record_path),))
        )

    def _evaluate_plan(
        self,
        plan,
        batches,
        baseline,
        expert_cache,
        *,
        split: str,
    ) -> Mapping[str, object]:
        ctx = self._require_context()
        merge_method = "geometry" if plan.method == CandidateMethod.RMT_GEOMETRY else "baseline"
        validation_slice_metrics: tuple[EvaluationMetricsContract, ...] = ()
        with reversible_one_layer_transform(
            ctx,
            plan,
            merge_method=merge_method,
            geometry_rank=self.config.h1.geometry_rank,
            shrink_tau=self.config.h1.shrink_tau,
            expert_cache=expert_cache,
        ) as transform:
            if self._uses_validation_slices(split):
                validation_slice_metrics = self._evaluate_validation_slices(batches)
                metrics = aggregate_evaluation_metrics(validation_slice_metrics)
            else:
                metrics = self._evaluate(batches)
        comparison = compare_evaluations(
            candidate_id=plan.candidate_id,
            baseline=baseline,
            candidate=metrics,
            quality_budgets=self.config.pilot.quality_budgets,
        )
        result = {
            "config_hash": self.run_hash,
            "plan": _plan_dict(plan),
            "transform": asdict(transform),
            "metrics": _metrics_dict(metrics),
            "relative_perplexity_degradation": comparison.relative_perplexity_degradation,
            "throughput_gain": comparison.throughput_gain,
            "memory_reduction": comparison.memory_reduction,
            "accepted_quality_budgets": dict(comparison.accepted_quality_budgets),
        }
        if validation_slice_metrics:
            baseline_slice_metrics = self._baseline_validation_slice_metrics()
            if len(validation_slice_metrics) != len(baseline_slice_metrics):
                raise ExpectedExperimentFailure(
                    ExperimentFailureCode.PILOT_STATE_MISMATCH,
                    "Candidate and baseline validation slice counts do not match",
                    context={
                        "candidate_slice_count": len(validation_slice_metrics),
                        "baseline_slice_count": len(baseline_slice_metrics),
                    },
                )
            slice_comparisons = tuple(
                compare_evaluations(
                    candidate_id=plan.candidate_id,
                    baseline=baseline_slice,
                    candidate=candidate_slice,
                    quality_budgets=self.config.pilot.quality_budgets,
                )
                for baseline_slice, candidate_slice in zip(
                    baseline_slice_metrics,
                    validation_slice_metrics,
                )
            )
            robustness = summarize_validation_degradations(
                tuple(
                    item.relative_perplexity_degradation for item in slice_comparisons
                ),
                std_penalty=self.config.pilot.validation_selection.std_penalty,
                max_slice_degradation=(
                    self.config.pilot.validation_selection.max_slice_degradation
                ),
            )
            result["validation_slices"] = [
                {
                    "slice_index": index,
                    "metrics": _metrics_dict(candidate_slice),
                    "relative_perplexity_degradation": (
                        slice_comparison.relative_perplexity_degradation
                    ),
                }
                for index, (candidate_slice, slice_comparison) in enumerate(
                    zip(validation_slice_metrics, slice_comparisons)
                )
            ]
            result["validation_robustness"] = asdict(robustness)
        return result

    def _run_candidate_batch(
        self,
        *,
        split: str,
        plans,
        max_items: int,
        max_seconds: float,
        retry_failed: bool,
    ) -> PilotStepResultContract:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        if max_seconds <= 0.0:
            raise ValueError("max_seconds must be positive")
        step = PilotStep.VALIDATE_CANDIDATES if split == "validation" else PilotStep.TEST_SELECTED
        all_plan_ids = tuple(plan.candidate_id for plan in plans)
        state = self._state(all_plan_ids=all_plan_ids if split == "validation" else None)
        pending_ids = pending_candidate_ids(
            all_plan_ids,
            state,
            split=split,
            retry_failed=retry_failed,
        )
        if not pending_ids:
            return self._save_result(
                PilotStepResultContract(step, PilotStepStatus.SKIPPED, 0, 0, (str(self.record_path),)),
                all_plan_ids=all_plan_ids if split == "validation" else None,
            )

        plan_by_id = {plan.candidate_id: plan for plan in plans}
        batches = torch.load(
            _require_path(self._batch_path(split), f"{split} token batches"),
            map_location="cpu",
        )
        baseline = self._baseline_metrics(split)
        expert_cache: Dict[tuple[object, ...], object] = {}
        processed = 0
        started = time.perf_counter()
        progress = tqdm(pending_ids, desc=f"Pilot {split} batch", total=len(pending_ids))
        for candidate_id in progress:
            if processed >= max_items:
                break
            if processed > 0 and time.perf_counter() - started >= max_seconds:
                break
            plan = plan_by_id[candidate_id]
            progress.set_postfix_str(candidate_id)
            try:
                row = dict(
                    self._evaluate_plan(
                        plan,
                        batches,
                        baseline,
                        expert_cache,
                        split=split,
                    )
                )
                row.update(
                    {
                        "record_type": "candidate" if split == "validation" else "selected_candidate",
                        "split": split,
                    }
                )
            except Exception as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row = {
                    "config_hash": self.run_hash,
                    "record_type": (
                        "failed_candidate" if split == "validation" else "failed_selected_candidate"
                    ),
                    "split": split,
                    "plan": _plan_dict(plan),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            append_incremental_record(self.record_path, row)
            processed += 1
            refreshed = self._state(all_plan_ids=all_plan_ids if split == "validation" else None)
            remaining = len(
                pending_candidate_ids(
                    all_plan_ids,
                    refreshed,
                    split=split,
                    retry_failed=retry_failed,
                )
            )
            write_pilot_state(
                self.state_path,
                refreshed,
                PilotStepResultContract(
                    step,
                    PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
                    processed,
                    remaining,
                    (str(self.record_path),),
                ),
            )

        final_state = self._state(all_plan_ids=all_plan_ids if split == "validation" else None)
        remaining = len(
            pending_candidate_ids(
                all_plan_ids,
                final_state,
                split=split,
                retry_failed=retry_failed,
            )
        )
        return self._save_result(
            PilotStepResultContract(
                step,
                PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
                processed,
                remaining,
                (str(self.record_path),),
            ),
            all_plan_ids=all_plan_ids if split == "validation" else None,
        )

    def run_validation_batch(
        self,
        *,
        max_items: int,
        max_seconds: float,
        retry_failed: bool = False,
    ) -> PilotStepResultContract:
        if self.selection_path.exists():
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Validation is locked after candidate selection",
            )
        self._baseline_metrics("validation")
        _trace, plans = self._load_trace_and_plans()
        return self._run_candidate_batch(
            split="validation",
            plans=plans,
            max_items=max_items,
            max_seconds=max_seconds,
            retry_failed=retry_failed,
        )

    def select_candidates(self) -> PilotStepResultContract:
        if self.selection_path.exists():
            return self.skip_step(PilotStep.SELECT_CANDIDATES, (self.selection_path,))
        plans = (
            self._load_trace_and_plans()[1]
            if self.ctx is not None
            else self._load_plans_without_runtime()
        )
        plan_ids = tuple(plan.candidate_id for plan in plans)
        state = self._state(all_plan_ids=plan_ids)
        if PilotStep.VALIDATE_CANDIDATES not in state.completed_steps:
            pending = pending_candidate_ids(plan_ids, state, split="validation", retry_failed=False)
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Candidate selection requires a completed validation grid",
                context={"remaining_validation_candidates": len(pending)},
            )
        selected = select_validation_winners(
            records=self._records(),
            config_hash=self.run_hash,
            candidate_order=plan_ids,
            max_quality_budget=max(self.config.pilot.quality_budgets),
        )
        write_selection(self.selection_path, config_hash=self.run_hash, candidate_ids=selected)
        return self._save_result(
            PilotStepResultContract(
                PilotStep.SELECT_CANDIDATES,
                PilotStepStatus.COMPLETED,
                len(selected),
                0,
                (str(self.selection_path),),
            ),
            all_plan_ids=plan_ids,
        )

    def run_test_batch(
        self,
        *,
        max_items: int,
        max_seconds: float,
        retry_failed: bool = False,
    ) -> PilotStepResultContract:
        self._baseline_metrics("test")
        selected_ids = self._selected_ids()
        if not self.selection_path.exists():
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Selected candidates artifact has not been created",
            )
        _trace, all_plans = self._load_trace_and_plans()
        plan_by_id = {plan.candidate_id: plan for plan in all_plans}
        unknown = [candidate_id for candidate_id in selected_ids if candidate_id not in plan_by_id]
        if unknown:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Selection contains unknown candidate ids",
                context={"unknown_candidate_ids": unknown},
            )
        plans = tuple(plan_by_id[candidate_id] for candidate_id in selected_ids)
        return self._run_candidate_batch(
            split="test",
            plans=plans,
            max_items=max_items,
            max_seconds=max_seconds,
            retry_failed=retry_failed,
        )

    def build_summary(self) -> PilotStepResultContract:
        state = self._state()
        if PilotStep.TEST_SELECTED not in state.completed_steps:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Pilot summary requires completed selected-candidate test evaluation",
            )
        trace = torch.load(_require_path(self._trace_path(), "calibration trace"), map_location="cpu")
        metadata = trace.get("metadata", {})
        records = self._records()
        selected_ids = set(state.selected_candidate_ids)
        selected_validation_results = [
            dict(row)
            for row in records
            if row.get("config_hash") == self.run_hash
            and row.get("record_type") == "candidate"
            and row.get("split") == "validation"
            and isinstance(row.get("plan"), Mapping)
            and str(row["plan"].get("candidate_id")) in selected_ids
        ]
        selected_test_results = [
            dict(row)
            for row in records
            if row.get("config_hash") == self.run_hash
            and row.get("record_type") == "selected_candidate"
            and row.get("split") == "test"
        ]
        baseline_validation_slices = (
            self._baseline_validation_slice_metrics()
            if self._uses_validation_slices("validation")
            else ()
        )
        summary = {
            "layer": self.config.pilot.layer,
            "trace_metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            "shared_experts_modified": False,
            "validation_candidate_count": (
                len(state.completed_validation_candidate_ids) + len(state.failed_validation_candidate_ids)
            ),
            "candidate_count": (
                len(state.completed_validation_candidate_ids) + len(state.failed_validation_candidate_ids)
            ),
            "selected_candidate_count": len(state.selected_candidate_ids),
            "completed_test_candidate_count": len(state.completed_test_candidate_ids),
            "failed_test_candidate_count": len(state.failed_test_candidate_ids),
            "record_path": str(self.record_path),
            "selection_path": str(self.selection_path),
            "state_path": str(self.state_path),
            "validation_selection": asdict(self.config.pilot.validation_selection),
            "baseline_validation": _metrics_dict(self._baseline_metrics("validation")),
            "baseline_validation_slices": [
                _metrics_dict(item) for item in baseline_validation_slices
            ],
            "baseline_test": _metrics_dict(self._baseline_metrics("test")),
            "selected_validation_results": selected_validation_results,
            "selected_test_results": selected_test_results,
        }
        write_json_atomic(self.summary_path, summary)
        return self._save_result(
            PilotStepResultContract(
                PilotStep.BUILD_SUMMARY,
                PilotStepStatus.COMPLETED,
                len(selected_test_results),
                0,
                (str(self.summary_path),),
            )
        )


def run_one_layer_pilot(config: ExperimentConfig, ctx: RuntimeContext) -> Dict[str, object]:
    executor = ResumablePilotExecutor(config, ctx)
    executor.run_baseline("validation")
    executor.run_validation_batch(max_items=10**9, max_seconds=float("inf"))
    executor.select_candidates()
    executor.run_baseline("test")
    executor.run_test_batch(max_items=10**9, max_seconds=float("inf"))
    executor.build_summary()
    return json.loads(executor.summary_path.read_text(encoding="utf-8"))


__all__ = ["ResumablePilotExecutor", "pilot_step_result_dict", "run_one_layer_pilot"]
