from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import torch
from tqdm import tqdm

from moe_opt_core.config import ExperimentConfig, config_hash
from moe_opt_core.experiment import (
    EvaluationMetricsContract,
    LayerCandidateScoreContract,
    MultiLayerRecipeContract,
    MultiLayerStep,
    MultiLayerStepResultContract,
    PilotStepStatus,
)
from moe_opt_core.experiment.evaluation import compare_evaluations
from moe_opt_core.experiment.multilayer_planning import (
    allocate_global_budget,
    build_cumulative_recipes,
    build_multilayer_execution_state,
    cumulative_layer_prefixes,
    pending_multilayer_ids,
    next_multilayer_step,
    resolve_sensitivity_layers,
    score_layer_plans,
)
from moe_opt_core.experiment.pilot_validation import (
    aggregate_evaluation_metrics,
    partition_validation_batches,
    summarize_validation_degradations,
)
from moe_opt_core.experiment.serialization import (
    candidate_plan_from_dict,
    candidate_plan_to_dict,
    multilayer_recipe_from_dict,
    multilayer_recipe_to_dict,
)
from moe_opt_runner.candidate_factory import build_layer_candidate_plans
from moe_opt_runner.evaluation import append_incremental_record, evaluate_language_model
from moe_opt_runner.layer_diagnostics import (
    layer_diagnostic_paths,
    run_layer_diagnostics,
)
from moe_opt_runner.materialization import (
    load_materialized_recipe_bundle,
    materialize_recipe_bundle,
)
from moe_opt_runner.pilot_artifacts import read_incremental_records, write_json_atomic
from moe_opt_runner.runtime import RuntimeContext
from moe_opt_runner.tracing import collect_traces_from_batches
from moe_opt_runner.transforms import (
    reversible_multi_layer_transform,
    reversible_one_layer_transform,
)


RuntimeFactory = Callable[[], RuntimeContext]


def multilayer_step_result_dict(
    result: MultiLayerStepResultContract,
) -> dict[str, object]:
    return {
        "step": result.step.value,
        "status": result.status.value,
        "processed_items": result.processed_items,
        "remaining_items": result.remaining_items,
        "artifact_paths": list(result.artifact_paths),
    }


def _metrics_from_dict(payload: Mapping[str, object]) -> EvaluationMetricsContract:
    return EvaluationMetricsContract(
        nll=float(payload["nll"]),
        perplexity=float(payload["perplexity"]),
        tokens_per_second=float(payload["tokens_per_second"]),
        peak_memory_mb=float(payload["peak_memory_mb"]),
        model_parameter_mb=float(payload["model_parameter_mb"]),
        evaluated_tokens=int(payload["evaluated_tokens"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
    )


def _directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (
        1024.0 * 1024.0
    )


class ResumableMultiLayerExecutor:
    def __init__(self, config: ExperimentConfig, ctx: RuntimeContext | None = None) -> None:
        if not config.multilayer.enabled:
            raise ValueError("The multi-layer executor requires multilayer.enabled=true")
        self.config = config
        self.ctx = ctx
        self.run_hash = config_hash(config)
        self.root = Path(config.artifacts.root_dir) / "multilayer"
        self.record_path = self.root / "multilayer_runs.jsonl"
        self.state_path = self.root / "multilayer_state.json"
        self.recipes_path = self.root / "multilayer_recipes.json"
        self.selection_path = self.root / "multilayer_selection.json"
        self.summary_path = self.root / "multilayer_summary.json"
        self.layer_artifact_root = self.root / "layer_artifacts"

    @property
    def sensitivity_layers(self) -> tuple[int, ...]:
        return resolve_sensitivity_layers(
            self.config.layers,
            self.config.multilayer.sensitivity_layers,
        )

    def _require_context(self) -> RuntimeContext:
        if self.ctx is None:
            raise RuntimeError("This multi-layer step requires a loaded model runtime")
        return self.ctx

    def _records(self) -> tuple[Mapping[str, object], ...]:
        return read_incremental_records(self.record_path)

    def _batch_path(self, split: str) -> Path:
        return Path(self.config.calibration.output_dir) / split / "token_batches.pt"

    def _load_batches(self, split: str):
        path = self._batch_path(split)
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} token batches: {path}")
        return torch.load(path, map_location="cpu")

    def _trace_paths(self, split: str) -> tuple[Path, ...]:
        root = Path(self.config.calibration.output_dir)
        return (
            root / "trace_manifest.json",
            root / split / "token_batches.pt",
            *(root / split / f"layer_{layer}.pt" for layer in self.config.layers),
        )

    def _diagnostic_paths(self, layer: int) -> tuple[Path, ...]:
        return layer_diagnostic_paths(self.layer_artifact_root, layer)

    def _selected_ids(self) -> tuple[str, ...]:
        if not self.selection_path.exists():
            return ()
        payload = json.loads(self.selection_path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != self.run_hash:
            raise ValueError("Multi-layer selection belongs to another config")
        return tuple(str(item) for item in payload.get("selected_recipe_ids", ()))

    def _artifact_steps(self) -> tuple[MultiLayerStep, ...]:
        completed: set[MultiLayerStep] = set()
        for split, step in (
            ("calibration", MultiLayerStep.TRACES_CALIBRATION),
            ("validation", MultiLayerStep.TRACES_VALIDATION),
            ("test", MultiLayerStep.TRACES_TEST),
        ):
            if all(path.exists() for path in self._trace_paths(split)):
                completed.add(step)
        if all(
            all(path.exists() for path in self._diagnostic_paths(layer))
            for layer in self.config.layers
        ):
            completed.add(MultiLayerStep.LAYER_DIAGNOSTICS)
        records = self._records()
        if self._baseline_row("validation", records=records) is not None:
            completed.add(MultiLayerStep.BASELINE_VALIDATION)
        if self._baseline_row("test", records=records) is not None:
            completed.add(MultiLayerStep.BASELINE_TEST)
        if self.recipes_path.exists():
            completed.add(MultiLayerStep.BUILD_RECIPES)
        if self.selection_path.exists():
            completed.add(MultiLayerStep.SELECT_CONFIGURATIONS)
        if self.summary_path.exists():
            completed.add(MultiLayerStep.BUILD_SUMMARY)
        return tuple(step for step in MultiLayerStep if step in completed)

    def _state(self):
        return build_multilayer_execution_state(
            config_hash=self.run_hash,
            records=self._records(),
            artifact_steps=self._artifact_steps(),
            selected_recipe_ids=self._selected_ids(),
        )

    def _write_state(self, result: MultiLayerStepResultContract) -> None:
        state = asdict(self._state())
        state["completed_steps"] = [step.value for step in self._state().completed_steps]
        state["last_result"] = multilayer_step_result_dict(result)
        write_json_atomic(self.state_path, state)

    def _result(
        self,
        step: MultiLayerStep,
        status: PilotStepStatus,
        processed: int,
        remaining: int,
        paths: Sequence[Path | str] = (),
    ) -> MultiLayerStepResultContract:
        result = MultiLayerStepResultContract(
            step=step,
            status=status,
            processed_items=processed,
            remaining_items=remaining,
            artifact_paths=tuple(str(path) for path in paths),
        )
        self._write_state(result)
        return result

    def record_trace_step(
        self,
        split: str,
        *,
        processed_items: int = 1,
    ) -> MultiLayerStepResultContract:
        step = MultiLayerStep(f"traces_{split}")
        paths = self._trace_paths(split)
        remaining = sum(
            not (
                Path(self.config.calibration.output_dir)
                / split
                / f"layer_{layer}.pt"
            ).exists()
            for layer in self.config.layers
        )
        return self._result(
            step,
            PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
            processed_items,
            remaining,
            tuple(path for path in paths if path.exists()),
        )

    def pending_trace_layers(self, split: str) -> tuple[int, ...]:
        root = Path(self.config.calibration.output_dir) / split
        return tuple(
            layer
            for layer in self.config.layers
            if not (root / f"layer_{layer}.pt").exists()
        )

    def _estimated_sensitivity_ids(self) -> tuple[str, ...]:
        identifiers = []
        merge_methods = set(self.config.pilot.merge_weighting.methods)
        merge_variant_count = max(1, len(self.config.pilot.merge_weighting.variants))
        hybrid_variant_count = max(1, len(self.config.pilot.hybrid_pairing.variants))
        scheduled_methods = {
            assignment.method for assignment in self.config.pilot.spectral_rank_schedule
        }
        for layer in self.sensitivity_layers:
            for method in self.config.multilayer.methods:
                for fraction in self.config.multilayer.compression_fractions:
                    rank_count = 1
                    if method in {"rmt_geometry", "rmt_functional_usage"}:
                        rank_count = (
                            1
                            if method in scheduled_methods
                            else max(1, len(self.config.pilot.spectral_pairing_ranks))
                        )
                    hybrid_count = (
                        hybrid_variant_count
                        if method == "rmt_functional_usage"
                        else 1
                    )
                    weighting_count = (
                        merge_variant_count if method in merge_methods else 1
                    )
                    for index in range(rank_count * hybrid_count * weighting_count):
                        identifiers.append(
                            f"estimated-l{layer}-{method}-f{fraction:.6g}-v{index}"
                        )
        return tuple(identifiers)

    def status(self) -> dict[str, object]:
        state = self._state()
        plans_by_layer = self._load_plans_by_layer(require_all=False)
        sensitivity_ids = tuple(
            plan.candidate_id
            for layer in self.sensitivity_layers
            for plan in plans_by_layer.get(layer, ())
        )
        if not sensitivity_ids:
            sensitivity_ids = self._estimated_sensitivity_ids()
        recipes = self._load_recipes() if self.recipes_path.exists() else ()
        remaining_diagnostics = sum(
            not all(path.exists() for path in self._diagnostic_paths(layer))
            for layer in self.config.layers
        )
        remaining_sensitivity = len(
            pending_multilayer_ids(
                sensitivity_ids,
                completed_ids=state.completed_sensitivity_ids,
                failed_ids=state.failed_sensitivity_ids,
                retry_failed=False,
            )
        )
        remaining_validation = len(
            pending_multilayer_ids(
                tuple(recipe.recipe_id for recipe in recipes),
                completed_ids=state.completed_validation_recipe_ids,
                failed_ids=state.failed_validation_recipe_ids,
                retry_failed=False,
            )
        )
        remaining_test = len(
            pending_multilayer_ids(
                state.selected_recipe_ids,
                completed_ids=state.completed_test_recipe_ids,
                failed_ids=state.failed_test_recipe_ids,
                retry_failed=False,
            )
        )
        materialization_ids = self._materialization_ids()
        materialization_root = Path(self.config.multilayer.materialization.output_dir)
        remaining_materializations = sum(
            not (materialization_root / recipe_id / "manifest.json").exists()
            for recipe_id in materialization_ids
        )
        benchmarked = {
            str(row["recipe_id"])
            for row in self._records()
            if row.get("config_hash") == self.run_hash
            and row.get("record_type") == "materialized_benchmark"
        }
        remaining_benchmarks = sum(
            recipe_id not in benchmarked for recipe_id in materialization_ids
        )
        next_step = next_multilayer_step(
            state.completed_steps,
            remaining_diagnostic_layers=remaining_diagnostics,
            remaining_sensitivity_candidates=remaining_sensitivity,
            remaining_validation_recipes=remaining_validation,
            remaining_test_recipes=remaining_test,
            remaining_materializations=remaining_materializations,
            remaining_materialized_benchmarks=remaining_benchmarks,
            selection_written=self.selection_path.exists(),
            selected_recipe_count=len(state.selected_recipe_ids),
        )
        return {
            "config_hash": self.run_hash,
            "completed_steps": [step.value for step in state.completed_steps],
            "sensitivity_layers": list(self.sensitivity_layers),
            "completed_sensitivity": len(state.completed_sensitivity_ids),
            "remaining_diagnostic_layers": remaining_diagnostics,
            "remaining_sensitivity": remaining_sensitivity,
            "recipes": len(recipes),
            "remaining_validation_recipes": remaining_validation,
            "selected_recipe_ids": list(state.selected_recipe_ids),
            "remaining_test_recipes": remaining_test,
            "remaining_materializations": remaining_materializations,
            "remaining_materialized_benchmarks": remaining_benchmarks,
            "next_step": next_step.value if next_step is not None else None,
        }

    def run_layer_diagnostics_batch(
        self,
        *,
        max_items: int,
        max_seconds: float,
    ) -> MultiLayerStepResultContract:
        required = {
            MultiLayerStep.TRACES_CALIBRATION,
            MultiLayerStep.TRACES_VALIDATION,
        }
        if not required <= set(self._artifact_steps()):
            raise RuntimeError(
                "Layer diagnostics require calibration and validation traces"
            )
        ctx = self._require_context()
        pending = tuple(
            layer
            for layer in self.config.layers
            if not all(path.exists() for path in self._diagnostic_paths(layer))
        )
        if not pending:
            return self._result(
                MultiLayerStep.LAYER_DIAGNOSTICS,
                PilotStepStatus.SKIPPED,
                0,
                0,
                (self.layer_artifact_root,),
            )
        processed = 0
        started = time.perf_counter()
        for layer in tqdm(pending, desc="Layer diagnostics"):
            if processed >= max_items or (
                processed and time.perf_counter() - started >= max_seconds
            ):
                break
            run_layer_diagnostics(
                self.config,
                ctx,
                layer=layer,
                trace_root=self.config.calibration.output_dir,
                output_root=self.layer_artifact_root,
            )
            processed += 1
            remaining = len(pending) - processed
            self._result(
                MultiLayerStep.LAYER_DIAGNOSTICS,
                PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
                processed,
                remaining,
                (self.layer_artifact_root,),
            )
        remaining = sum(
            not all(path.exists() for path in self._diagnostic_paths(layer))
            for layer in self.config.layers
        )
        return self._result(
            MultiLayerStep.LAYER_DIAGNOSTICS,
            PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
            processed,
            remaining,
            (self.layer_artifact_root,),
        )

    def _load_plans_by_layer(self, *, require_all: bool) -> dict[int, tuple]:
        plans: dict[int, tuple] = {}
        trace_root = Path(self.config.calibration.output_dir) / "calibration"
        for layer in self.config.layers:
            trace_path = trace_root / f"layer_{layer}.pt"
            embedding_path = self._diagnostic_paths(layer)[2]
            if not trace_path.exists() or not embedding_path.exists():
                if require_all:
                    raise FileNotFoundError(
                        f"Missing trace or embedding for layer {layer}"
                    )
                continue
            trace = torch.load(trace_path, map_location="cpu")
            embedding = torch.load(embedding_path, map_location="cpu")
            metadata = trace.get("metadata", {})
            num_experts = int(
                metadata.get(
                    "routed_experts",
                    self.config.runtime_expected_routed_experts or 0,
                )
            )
            plans[layer] = build_layer_candidate_plans(
                self.config,
                layer=layer,
                num_experts=num_experts,
                trace=trace,
                embedding=embedding,
                methods=self.config.multilayer.methods,
                compression_fractions=self.config.multilayer.compression_fractions,
                namespace_layer=True,
            )
        return plans

    def _evaluate(self, batches) -> EvaluationMetricsContract:
        return evaluate_language_model(
            self._require_context().model,
            batches,
            warmup_steps=self.config.profile.warmup_steps,
            measure_steps=min(
                self.config.profile.measure_steps,
                self.config.pilot.max_eval_batches,
            ),
        )

    def _validation_slices(self, batches):
        limited = tuple(batches[: self.config.pilot.max_eval_batches])
        return partition_validation_batches(
            limited,
            self.config.pilot.validation_selection.slice_count,
        )

    def _evaluate_validation(self, batches):
        metrics = tuple(
            evaluate_language_model(
                self._require_context().model,
                batch_slice,
                warmup_steps=0,
                measure_steps=len(batch_slice),
            )
            for batch_slice in self._validation_slices(batches)
        )
        return aggregate_evaluation_metrics(metrics), metrics

    def _baseline_row(
        self,
        split: str,
        *,
        records: Sequence[Mapping[str, object]] | None = None,
    ) -> Mapping[str, object] | None:
        for row in reversed(tuple(records) if records is not None else self._records()):
            if (
                row.get("config_hash") == self.run_hash
                and row.get("record_type") == "multilayer_baseline"
                and row.get("split") == split
            ):
                return row
        return None

    def run_baseline(self, split: str) -> MultiLayerStepResultContract:
        step = (
            MultiLayerStep.BASELINE_VALIDATION
            if split == "validation"
            else MultiLayerStep.BASELINE_TEST
        )
        trace_step = MultiLayerStep(f"traces_{split}")
        if trace_step not in self._artifact_steps():
            raise RuntimeError(f"The {split} baseline requires completed traces")
        if split == "test" and not self.selection_path.exists():
            raise RuntimeError("Test baseline is locked until configuration selection")
        if self._baseline_row(split) is not None:
            return self._result(step, PilotStepStatus.SKIPPED, 0, 0, (self.record_path,))
        batches = self._load_batches(split)
        if split == "validation":
            metrics, slices = self._evaluate_validation(batches)
        else:
            metrics, slices = self._evaluate(batches), ()
        row: dict[str, object] = {
            "config_hash": self.run_hash,
            "record_type": "multilayer_baseline",
            "split": split,
            "metrics": asdict(metrics),
        }
        if slices:
            row["validation_slice_metrics"] = [asdict(item) for item in slices]
        append_incremental_record(self.record_path, row)
        return self._result(step, PilotStepStatus.COMPLETED, 1, 0, (self.record_path,))

    def _baseline_metrics(self, split: str):
        row = self._baseline_row(split)
        if row is None or not isinstance(row.get("metrics"), Mapping):
            raise RuntimeError(f"Missing {split} multi-layer baseline")
        metrics = _metrics_from_dict(row["metrics"])
        slices = tuple(
            _metrics_from_dict(item)
            for item in row.get("validation_slice_metrics", ())
            if isinstance(item, Mapping)
        )
        return metrics, slices

    def _comparison_record(
        self,
        *,
        identifier: str,
        split: str,
        metrics: EvaluationMetricsContract,
        slice_metrics: Sequence[EvaluationMetricsContract],
    ) -> dict[str, object]:
        baseline, baseline_slices = self._baseline_metrics(split)
        comparison = compare_evaluations(
            candidate_id=identifier,
            baseline=baseline,
            candidate=metrics,
            quality_budgets=self.config.pilot.quality_budgets,
        )
        result: dict[str, object] = {
            "metrics": asdict(metrics),
            "relative_perplexity_degradation": comparison.relative_perplexity_degradation,
            "throughput_gain": comparison.throughput_gain,
            "memory_reduction": comparison.memory_reduction,
            "accepted_quality_budgets": dict(comparison.accepted_quality_budgets),
        }
        if slice_metrics:
            degradations = tuple(
                compare_evaluations(
                    candidate_id=identifier,
                    baseline=baseline_slice,
                    candidate=candidate_slice,
                    quality_budgets=self.config.pilot.quality_budgets,
                ).relative_perplexity_degradation
                for baseline_slice, candidate_slice in zip(
                    baseline_slices,
                    slice_metrics,
                )
            )
            result["validation_robustness"] = asdict(
                summarize_validation_degradations(
                    degradations,
                    std_penalty=self.config.pilot.validation_selection.std_penalty,
                    max_slice_degradation=(
                        self.config.pilot.validation_selection.max_slice_degradation
                    ),
                )
            )
        return result

    def run_sensitivity_batch(
        self,
        *,
        max_items: int,
        max_seconds: float,
        retry_failed: bool,
    ) -> MultiLayerStepResultContract:
        completed = set(self._artifact_steps())
        if MultiLayerStep.LAYER_DIAGNOSTICS not in completed:
            raise RuntimeError("Sensitivity validation requires all layer diagnostics")
        if MultiLayerStep.BASELINE_VALIDATION not in completed:
            raise RuntimeError("Sensitivity validation requires validation baseline")
        plans_by_layer = self._load_plans_by_layer(require_all=True)
        plans = tuple(
            plan
            for layer in self.sensitivity_layers
            for plan in plans_by_layer[layer]
        )
        state = self._state()
        pending = pending_multilayer_ids(
            tuple(plan.candidate_id for plan in plans),
            completed_ids=state.completed_sensitivity_ids,
            failed_ids=state.failed_sensitivity_ids,
            retry_failed=retry_failed,
        )
        return self._run_item_batch(
            step=MultiLayerStep.SENSITIVITY_VALIDATION,
            pending_ids=pending,
            max_items=max_items,
            max_seconds=max_seconds,
            evaluate=lambda item_id: self._evaluate_sensitivity(
                next(plan for plan in plans if plan.candidate_id == item_id)
            ),
            success_type="sensitivity_candidate",
            failure_type="failed_sensitivity_candidate",
            id_field="candidate_id",
        )

    def _evaluate_sensitivity(self, plan) -> dict[str, object]:
        batches = self._load_batches("validation")
        with reversible_one_layer_transform(
            self._require_context(),
            plan,
            merge_method="baseline",
            geometry_rank=self.config.h1.geometry_rank,
            shrink_tau=self.config.h1.shrink_tau,
            expert_cache={},
        ) as transform:
            metrics, slices = self._evaluate_validation(batches)
        return {
            "plan": candidate_plan_to_dict(plan),
            "transform": asdict(transform),
            **self._comparison_record(
                identifier=plan.candidate_id,
                split="validation",
                metrics=metrics,
                slice_metrics=slices,
            ),
        }

    def _run_item_batch(
        self,
        *,
        step: MultiLayerStep,
        pending_ids: Sequence[str],
        max_items: int,
        max_seconds: float,
        evaluate,
        success_type: str,
        failure_type: str,
        id_field: str,
    ) -> MultiLayerStepResultContract:
        if not pending_ids:
            return self._result(step, PilotStepStatus.SKIPPED, 0, 0, (self.record_path,))
        processed = 0
        started = time.perf_counter()
        for item_id in tqdm(pending_ids, desc=step.value):
            if processed >= max_items or (
                processed and time.perf_counter() - started >= max_seconds
            ):
                break
            try:
                row = dict(evaluate(item_id))
                row["record_type"] = success_type
            except Exception as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row = {
                    "record_type": failure_type,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            row.update(
                {
                    "config_hash": self.run_hash,
                    id_field: item_id,
                }
            )
            append_incremental_record(self.record_path, row)
            processed += 1
            self._result(
                step,
                PilotStepStatus.PARTIAL,
                processed,
                len(pending_ids) - processed,
                (self.record_path,),
            )
        remaining = len(pending_ids) - processed
        return self._result(
            step,
            PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
            processed,
            remaining,
            (self.record_path,),
        )

    def _sensitivity_scores(self) -> tuple[LayerCandidateScoreContract, ...]:
        scores = []
        for row in self._records():
            if (
                row.get("config_hash") != self.run_hash
                or row.get("record_type") != "sensitivity_candidate"
                or not isinstance(row.get("plan"), Mapping)
            ):
                continue
            plan = candidate_plan_from_dict(row["plan"])
            robustness = row.get("validation_robustness")
            robust_score = float(row["relative_perplexity_degradation"])
            if isinstance(robustness, Mapping):
                if not bool(robustness.get("passes_max_slice_degradation", True)):
                    continue
                robust_score = float(robustness["robust_score"])
            scores.append(
                LayerCandidateScoreContract(
                    layer=plan.layer,
                    plan=plan,
                    relative_perplexity_degradation=float(
                        row["relative_perplexity_degradation"]
                    ),
                    robust_score=robust_score,
                )
            )
        return tuple(scores)

    def build_recipes(self) -> MultiLayerStepResultContract:
        if self.recipes_path.exists():
            return self._result(
                MultiLayerStep.BUILD_RECIPES,
                PilotStepStatus.SKIPPED,
                0,
                0,
                (self.recipes_path,),
            )
        plans_by_layer = self._load_plans_by_layer(require_all=True)
        expected_ids = tuple(
            plan.candidate_id
            for layer in self.sensitivity_layers
            for plan in plans_by_layer[layer]
        )
        state = self._state()
        pending = pending_multilayer_ids(
            expected_ids,
            completed_ids=state.completed_sensitivity_ids,
            failed_ids=state.failed_sensitivity_ids,
            retry_failed=False,
        )
        if pending:
            raise RuntimeError(
                "Cannot build recipes before sensitivity validation finishes: "
                f"{len(pending)} pending"
            )
        scored = score_layer_plans(
            plans_by_layer=plans_by_layer,
            measured_scores=self._sensitivity_scores(),
        )
        prefixes = cumulative_layer_prefixes(
            self.config.layers,
            self.config.multilayer.cumulative_layer_counts,
            include_all=self.config.multilayer.include_all_layers,
        )
        recipes = list(
            build_cumulative_recipes(
                layer_prefixes=prefixes,
                scored_plans=scored,
                methods=self.config.multilayer.methods,
                compression_fractions=self.config.multilayer.compression_fractions,
                refresh_modes=self.config.multilayer.trace_refresh_modes,
            )
        )
        planning_failures = []
        for refresh in self.config.multilayer.trace_refresh_modes:
            for target in self.config.multilayer.global_compression_targets:
                try:
                    recipes.append(
                        allocate_global_budget(
                            target_compression=target,
                            scored_plans=scored,
                            all_layers=self.config.layers,
                            quality_budget=self.config.multilayer.quality_budget,
                            max_layer_degradation=(
                                self.config.multilayer.max_layer_degradation
                            ),
                            refresh_mode=refresh,
                        )
                    )
                except ValueError as exc:
                    planning_failures.append(
                        {
                            "trace_refresh_mode": refresh,
                            "target_compression": target,
                            "error": str(exc),
                        }
                    )
        write_json_atomic(
            self.recipes_path,
            {
                "schema_version": "1.0",
                "config_hash": self.run_hash,
                "recipes": [multilayer_recipe_to_dict(recipe) for recipe in recipes],
                "planning_failures": planning_failures,
            },
        )
        return self._result(
            MultiLayerStep.BUILD_RECIPES,
            PilotStepStatus.COMPLETED,
            len(recipes),
            0,
            (self.recipes_path,),
        )

    def _load_recipes(self) -> tuple[MultiLayerRecipeContract, ...]:
        payload = json.loads(self.recipes_path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != self.run_hash:
            raise ValueError("Multi-layer recipes belong to another config")
        return tuple(
            multilayer_recipe_from_dict(item)
            for item in payload.get("recipes", ())
            if isinstance(item, Mapping)
        )

    @contextmanager
    def _recipe_transform(self, recipe: MultiLayerRecipeContract):
        plans = tuple(
            choice.plan for choice in recipe.choices if choice.plan is not None
        )
        if recipe.trace_refresh_mode.value == "static":
            with reversible_multi_layer_transform(
                self._require_context(),
                plans,
                merge_method="baseline",
                geometry_rank=self.config.h1.geometry_rank,
                shrink_tau=self.config.h1.shrink_tau,
            ) as transforms:
                yield recipe, transforms
            return

        calibration_batches = self._load_batches("calibration")
        resolved_choices = []
        with ExitStack() as stack:
            transforms = []
            for position, choice in enumerate(recipe.choices):
                if choice.plan is None:
                    continue
                adaptive_root = (
                    self.root
                    / "adaptive"
                    / recipe.recipe_id
                    / f"step_{position:02d}_layer_{choice.layer}"
                )
                trace_root = adaptive_root / "traces"
                collect_traces_from_batches(
                    self.config,
                    self._require_context(),
                    {"calibration": calibration_batches},
                    layers=(choice.layer,),
                    output_root=trace_root,
                    persist_legacy_aliases=False,
                )
                diagnostic_root = adaptive_root / "diagnostics"
                run_layer_diagnostics(
                    self.config,
                    self._require_context(),
                    layer=choice.layer,
                    trace_root=trace_root,
                    output_root=diagnostic_root,
                )
                trace = torch.load(
                    trace_root / "calibration" / f"layer_{choice.layer}.pt",
                    map_location="cpu",
                )
                embedding = torch.load(
                    layer_diagnostic_paths(diagnostic_root, choice.layer)[2],
                    map_location="cpu",
                )
                candidates = build_layer_candidate_plans(
                    self.config,
                    layer=choice.layer,
                    num_experts=int(trace["metadata"]["routed_experts"]),
                    trace=trace,
                    embedding=embedding,
                    methods=(choice.plan.method.value,),
                    compression_fractions=(choice.plan.compression_fraction,),
                    namespace_layer=True,
                )
                adaptive_plan = next(
                    (
                        plan
                        for plan in candidates
                        if plan.candidate_id == choice.plan.candidate_id
                    ),
                    candidates[0],
                )
                transform = stack.enter_context(
                    reversible_one_layer_transform(
                        self._require_context(),
                        adaptive_plan,
                        merge_method="baseline",
                        geometry_rank=self.config.h1.geometry_rank,
                        shrink_tau=self.config.h1.shrink_tau,
                        expert_cache={},
                    )
                )
                transforms.append(transform)
                resolved_choices.append(replace(choice, plan=adaptive_plan))
            yield replace(recipe, choices=tuple(resolved_choices)), tuple(transforms)

    def run_recipe_batch(
        self,
        *,
        split: str,
        max_items: int,
        max_seconds: float,
        retry_failed: bool,
    ) -> MultiLayerStepResultContract:
        if not self.recipes_path.exists():
            raise RuntimeError("Multi-layer validation requires built recipes")
        if split == "validation" and self._baseline_row("validation") is None:
            raise RuntimeError("Multi-layer validation requires validation baseline")
        if split == "test":
            if not self.selection_path.exists():
                raise RuntimeError("Test evaluation is locked until configuration selection")
            if self._baseline_row("test") is None:
                raise RuntimeError("Test evaluation requires test baseline")
        recipes = self._load_recipes()
        state = self._state()
        if split == "validation":
            step = MultiLayerStep.MULTILAYER_VALIDATION
            candidate_ids = tuple(recipe.recipe_id for recipe in recipes)
            completed = state.completed_validation_recipe_ids
            failed = state.failed_validation_recipe_ids
            success_type = "multilayer_candidate"
            failure_type = "failed_multilayer_candidate"
        else:
            step = MultiLayerStep.TEST_SELECTED
            candidate_ids = state.selected_recipe_ids
            completed = state.completed_test_recipe_ids
            failed = state.failed_test_recipe_ids
            success_type = "selected_multilayer_candidate"
            failure_type = "failed_selected_multilayer_candidate"
        pending = pending_multilayer_ids(
            candidate_ids,
            completed_ids=completed,
            failed_ids=failed,
            retry_failed=retry_failed,
        )
        recipe_by_id = {recipe.recipe_id: recipe for recipe in recipes}
        return self._run_item_batch(
            step=step,
            pending_ids=pending,
            max_items=max_items,
            max_seconds=max_seconds,
            evaluate=lambda recipe_id: self._evaluate_recipe(
                recipe_by_id[recipe_id], split=split
            ),
            success_type=success_type,
            failure_type=failure_type,
            id_field="recipe_id",
        )

    def _evaluate_recipe(
        self,
        recipe: MultiLayerRecipeContract,
        *,
        split: str,
    ) -> dict[str, object]:
        batches = self._load_batches(split)
        with self._recipe_transform(recipe) as (resolved_recipe, transforms):
            if split == "validation":
                metrics, slices = self._evaluate_validation(batches)
            else:
                metrics, slices = self._evaluate(batches), ()
        return {
            "split": split,
            "recipe": multilayer_recipe_to_dict(resolved_recipe),
            "transforms": [asdict(item) for item in transforms],
            **self._comparison_record(
                identifier=recipe.recipe_id,
                split=split,
                metrics=metrics,
                slice_metrics=slices,
            ),
        }

    def select_configurations(self) -> MultiLayerStepResultContract:
        if self.selection_path.exists():
            return self._result(
                MultiLayerStep.SELECT_CONFIGURATIONS,
                PilotStepStatus.SKIPPED,
                0,
                0,
                (self.selection_path,),
            )
        recipes = self._load_recipes()
        state = self._state()
        pending = pending_multilayer_ids(
            tuple(recipe.recipe_id for recipe in recipes),
            completed_ids=state.completed_validation_recipe_ids,
            failed_ids=state.failed_validation_recipe_ids,
            retry_failed=False,
        )
        if pending:
            raise RuntimeError(
                "Cannot select configurations before validation finishes: "
                f"{len(pending)} pending"
            )
        winners: dict[tuple[str, float, str], tuple[str, float]] = {}
        for row in self._records():
            if (
                row.get("config_hash") != self.run_hash
                or row.get("record_type") != "multilayer_candidate"
                or not isinstance(row.get("recipe"), Mapping)
            ):
                continue
            degradation = float(row["relative_perplexity_degradation"])
            robustness = row.get("validation_robustness")
            score = degradation
            if isinstance(robustness, Mapping):
                if not bool(robustness.get("passes_max_slice_degradation", True)):
                    continue
                score = float(robustness["robust_score"])
            if score > self.config.multilayer.quality_budget:
                continue
            recipe = row["recipe"]
            key = (
                str(recipe["kind"]),
                round(float(recipe["achieved_compression"]), 6),
                str(recipe["trace_refresh_mode"]),
            )
            candidate = (str(row["recipe_id"]), score)
            current = winners.get(key)
            if current is None or (candidate[1], candidate[0]) < (current[1], current[0]):
                winners[key] = candidate
        selected = tuple(
            candidate_id
            for candidate_id, _ in sorted(
                winners.values(), key=lambda item: (item[1], item[0])
            )
        )
        write_json_atomic(
            self.selection_path,
            {
                "schema_version": "1.0",
                "config_hash": self.run_hash,
                "selected_recipe_ids": list(selected),
            },
        )
        return self._result(
            MultiLayerStep.SELECT_CONFIGURATIONS,
            PilotStepStatus.COMPLETED,
            len(selected),
            0,
            (self.selection_path,),
        )

    def _materialization_ids(self) -> tuple[str, ...]:
        return self._selected_ids()[: self.config.multilayer.materialization.selected_plan_count]

    def _resolved_recipe_by_id(self) -> dict[str, MultiLayerRecipeContract]:
        recipes = {recipe.recipe_id: recipe for recipe in self._load_recipes()}
        preferred_types = (
            "multilayer_candidate",
            "selected_multilayer_candidate",
        )
        records = self._records()
        for record_type in preferred_types:
            for row in records:
                if (
                    row.get("config_hash") != self.run_hash
                    or row.get("record_type") != record_type
                    or not isinstance(row.get("recipe"), Mapping)
                ):
                    continue
                recipe = multilayer_recipe_from_dict(row["recipe"])
                recipes[recipe.recipe_id] = recipe
        return recipes

    def materialize_batch(
        self,
        runtime_factory: RuntimeFactory,
        *,
        max_items: int,
    ) -> MultiLayerStepResultContract:
        recipe_by_id = self._resolved_recipe_by_id()
        output_root = Path(self.config.multilayer.materialization.output_dir)
        pending = tuple(
            recipe_id
            for recipe_id in self._materialization_ids()
            if not (output_root / recipe_id / "manifest.json").exists()
        )
        processed = 0
        for recipe_id in tqdm(pending[:max_items], desc="Materialize recipes"):
            ctx = runtime_factory()
            bundle = materialize_recipe_bundle(
                ctx,
                recipe_by_id[recipe_id],
                output_root,
                config_hash=self.run_hash,
                merge_method="baseline",
                geometry_rank=self.config.h1.geometry_rank,
                shrink_tau=self.config.h1.shrink_tau,
                save_layer_weights=(
                    self.config.multilayer.materialization.save_layer_weights
                ),
            )
            append_incremental_record(
                self.record_path,
                {
                    "config_hash": self.run_hash,
                    "record_type": "materialized_recipe",
                    "recipe_id": recipe_id,
                    "bundle_path": str(bundle),
                    "artifact_size_mb": _directory_size_mb(bundle),
                },
            )
            processed += 1
            del ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        remaining = len(pending) - processed
        return self._result(
            MultiLayerStep.MATERIALIZE,
            PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
            processed,
            remaining,
            (output_root,),
        )

    def benchmark_materialized_batch(
        self,
        runtime_factory: RuntimeFactory,
        *,
        max_items: int,
    ) -> MultiLayerStepResultContract:
        completed = {
            str(row["recipe_id"])
            for row in self._records()
            if row.get("config_hash") == self.run_hash
            and row.get("record_type") == "materialized_benchmark"
        }
        pending = tuple(
            recipe_id for recipe_id in self._materialization_ids() if recipe_id not in completed
        )
        output_root = Path(self.config.multilayer.materialization.output_dir)
        batches = self._load_batches("test")
        processed = 0
        for recipe_id in tqdm(pending[:max_items], desc="Benchmark materialized"):
            ctx = runtime_factory()
            load_materialized_recipe_bundle(
                ctx,
                output_root / recipe_id,
                expected_config_hash=self.run_hash,
                merge_method="baseline",
                geometry_rank=self.config.h1.geometry_rank,
                shrink_tau=self.config.h1.shrink_tau,
            )
            self.ctx = ctx
            device = next(ctx.model.parameters()).device
            steady_state_mb = (
                float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0))
                if device.type == "cuda"
                else 0.0
            )
            metrics = self._evaluate(batches)
            comparison = self._comparison_record(
                identifier=recipe_id,
                split="test",
                metrics=metrics,
                slice_metrics=(),
            )
            row = {
                "config_hash": self.run_hash,
                "record_type": "materialized_benchmark",
                "recipe_id": recipe_id,
                **comparison,
                "steady_state_cuda_allocated_mb": steady_state_mb,
                "artifact_size_mb": _directory_size_mb(output_root / recipe_id),
                "measurement_mode": "fresh_runtime_per_recipe_without_reversible_snapshot",
            }
            append_incremental_record(self.record_path, row)
            processed += 1
            self.ctx = None
            del ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        remaining = len(pending) - processed
        return self._result(
            MultiLayerStep.BENCHMARK_MATERIALIZED,
            PilotStepStatus.COMPLETED if remaining == 0 else PilotStepStatus.PARTIAL,
            processed,
            remaining,
            (self.record_path,),
        )

    def build_summary(self) -> MultiLayerStepResultContract:
        records = self._records()
        payload = {
            "schema_version": "1.0",
            "config_hash": self.run_hash,
            "sensitivity_layers": list(self.sensitivity_layers),
            "selected_recipe_ids": list(self._selected_ids()),
            "record_counts": {
                record_type: sum(row.get("record_type") == record_type for row in records)
                for record_type in sorted(
                    {str(row.get("record_type")) for row in records}
                )
            },
            "records_path": str(self.record_path),
            "recipes_path": str(self.recipes_path),
            "selection_path": str(self.selection_path),
            "materialization_root": self.config.multilayer.materialization.output_dir,
        }
        write_json_atomic(self.summary_path, payload)
        return self._result(
            MultiLayerStep.BUILD_SUMMARY,
            PilotStepStatus.COMPLETED,
            1,
            0,
            (self.summary_path,),
        )
