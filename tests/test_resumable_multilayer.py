from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
    MultiLayerStep,
    PilotStepStatus,
    LayerCompressionChoiceContract,
    MultiLayerRecipeContract,
    MultiLayerRecipeKind,
    TraceRefreshMode,
)
from moe_opt_core.experiment.serialization import multilayer_recipe_to_dict
from moe_opt_runner.evaluation import append_incremental_record
from moe_opt_runner.multilayer_executor import ResumableMultiLayerExecutor


def _plan(layer: int, suffix: str = "a") -> CandidatePlanContract:
    return CandidatePlanContract(
        candidate_id=f"l{layer}-functional_cosine-f0.100-s123-{suffix}",
        layer=layer,
        method=CandidateMethod.FUNCTIONAL_COSINE,
        action=CandidateAction.MERGE,
        compression_fraction=0.1,
        groups=((0, 1), (2,), (3,)),
        seed=123,
    )


def _config(tmp_path: Path):
    config = load_experiment_config(
        "configs/experiments/h1_rmt_multilayer_representative_pilot.json"
    )
    root = tmp_path / "artifacts"
    return replace(
        config,
        layers=[0, 1],
        artifacts=replace(
            config.artifacts,
            root_dir=str(root),
            manifest_path=str(root / "manifest.json"),
        ),
        calibration=replace(
            config.calibration,
            output_dir=str(root / "traces"),
        ),
        multilayer=replace(
            config.multilayer,
            sensitivity_layers=(0,),
            methods=("functional_cosine",),
            compression_fractions=(0.1,),
            cumulative_layer_counts=(1, 2),
            global_compression_targets=(0.05,),
            quality_budget=0.1,
            max_layer_degradation=0.1,
            materialization=replace(
                config.multilayer.materialization,
                output_dir=str(root / "materialized"),
            ),
        ),
    )


def test_sensitivity_batch_resumes_without_duplicate_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    first = ResumableMultiLayerExecutor(config)
    plans = {0: (_plan(0, "a"), _plan(0, "b")), 1: (_plan(1),)}
    completed_prerequisites = (
        MultiLayerStep.LAYER_DIAGNOSTICS,
        MultiLayerStep.BASELINE_VALIDATION,
    )
    monkeypatch.setattr(first, "_artifact_steps", lambda: completed_prerequisites)
    monkeypatch.setattr(first, "_load_plans_by_layer", lambda **_: plans)
    monkeypatch.setattr(
        first,
        "_evaluate_sensitivity",
        lambda plan: {"plan": {"candidate_id": plan.candidate_id}},
    )
    partial = first.run_sensitivity_batch(
        max_items=1,
        max_seconds=60.0,
        retry_failed=False,
    )
    assert partial.status == PilotStepStatus.PARTIAL
    assert partial.remaining_items == 1

    resumed = ResumableMultiLayerExecutor(config)
    monkeypatch.setattr(resumed, "_artifact_steps", lambda: completed_prerequisites)
    monkeypatch.setattr(resumed, "_load_plans_by_layer", lambda **_: plans)
    monkeypatch.setattr(
        resumed,
        "_evaluate_sensitivity",
        lambda plan: {"plan": {"candidate_id": plan.candidate_id}},
    )
    completed = resumed.run_sensitivity_batch(
        max_items=2,
        max_seconds=60.0,
        retry_failed=False,
    )
    assert completed.status == PilotStepStatus.COMPLETED
    records = resumed._records()
    ids = [
        row["candidate_id"]
        for row in records
        if row.get("record_type") == "sensitivity_candidate"
    ]
    assert ids == [plans[0][0].candidate_id, plans[0][1].candidate_id]


def test_recipe_build_uses_measured_layer_scores_and_writes_resumable_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    executor = ResumableMultiLayerExecutor(config)
    plans = {0: (_plan(0),), 1: (_plan(1),)}
    monkeypatch.setattr(executor, "_load_plans_by_layer", lambda **_: plans)
    append_incremental_record(
        executor.record_path,
        {
            "config_hash": executor.run_hash,
            "record_type": "sensitivity_candidate",
            "candidate_id": plans[0][0].candidate_id,
            "plan": {
                "candidate_id": plans[0][0].candidate_id,
                "layer": 0,
                "method": "functional_cosine",
                "action": "merge",
                "compression_fraction": 0.1,
                "groups": [[0, 1], [2], [3]],
                "seed": 123,
                "diagnostics": {},
                "merge_weighting": "uniform",
                "group_merge_weights": [],
            },
            "relative_perplexity_degradation": 0.01,
            "validation_robustness": {"robust_score": 0.015},
        },
    )
    result = executor.build_recipes()
    assert result.status == PilotStepStatus.COMPLETED
    recipes = executor._load_recipes()
    assert recipes
    assert {recipe.trace_refresh_mode.value for recipe in recipes} == {
        "static",
        "adaptive",
    }
    skipped = executor.build_recipes()
    assert skipped.status == PilotStepStatus.SKIPPED


def test_status_is_model_free_and_points_to_first_missing_step(tmp_path: Path) -> None:
    executor = ResumableMultiLayerExecutor(_config(tmp_path))
    status = executor.status()
    assert status["next_step"] == MultiLayerStep.TRACES_CALIBRATION.value
    assert status["remaining_diagnostic_layers"] == 2
    assert status["remaining_sensitivity"] == 1


def test_fresh_status_does_not_require_recipes_for_balanced_materialization(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        multilayer=replace(
            config.multilayer,
            materialization=replace(
                config.multilayer.materialization,
                strategy="method_balanced",
            ),
        ),
    )
    status = ResumableMultiLayerExecutor(config).status()
    assert status["next_step"] == MultiLayerStep.TRACES_CALIBRATION.value
    assert status["remaining_materializations"] == 0


def test_trace_step_reports_partial_until_every_layer_artifact_exists(
    tmp_path: Path,
) -> None:
    executor = ResumableMultiLayerExecutor(_config(tmp_path))
    trace_root = Path(executor.config.calibration.output_dir)
    split_root = trace_root / "calibration"
    split_root.mkdir(parents=True)
    (trace_root / "trace_manifest.json").write_text("[]", encoding="utf-8")
    (split_root / "token_batches.pt").write_bytes(b"batches")
    (split_root / "layer_0.pt").write_bytes(b"trace")
    partial = executor.record_trace_step("calibration", processed_items=1)
    assert partial.status == PilotStepStatus.PARTIAL
    assert partial.remaining_items == 1
    assert executor.pending_trace_layers("calibration") == (1,)
    (split_root / "layer_1.pt").write_bytes(b"trace")
    completed = executor.record_trace_step("calibration", processed_items=1)
    assert completed.status == PilotStepStatus.COMPLETED
    assert completed.remaining_items == 0


def test_materialization_prefers_resolved_test_recipe(tmp_path: Path, monkeypatch) -> None:
    executor = ResumableMultiLayerExecutor(_config(tmp_path))
    static = MultiLayerRecipeContract(
        recipe_id="adaptive",
        kind=MultiLayerRecipeKind.CUMULATIVE,
        trace_refresh_mode=TraceRefreshMode.ADAPTIVE,
        choices=(
            LayerCompressionChoiceContract(0, _plan(0, "static"), 0.1, 0.01, 0),
        ),
        target_compression=0.1,
        achieved_compression=0.05,
        predicted_degradation=0.01,
    )
    resolved = replace(
        static,
        choices=(
            LayerCompressionChoiceContract(0, _plan(0, "resolved"), 0.1, 0.01, 0),
        ),
    )
    monkeypatch.setattr(executor, "_load_recipes", lambda: (static,))
    append_incremental_record(
        executor.record_path,
        {
            "config_hash": executor.run_hash,
            "record_type": "selected_multilayer_candidate",
            "recipe_id": "adaptive",
            "recipe": multilayer_recipe_to_dict(resolved),
        },
    )
    selected = executor._resolved_recipe_by_id()["adaptive"]
    assert selected.choices[0].plan.candidate_id.endswith("-resolved")
