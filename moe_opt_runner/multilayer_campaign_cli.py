from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from moe_opt_core.config import ExperimentConfig, config_hash, load_experiment_config
from moe_opt_core.experiment import (
    ReplicatedRecipeObservationContract,
    SensitivityVariantObservationContract,
)
from moe_opt_core.experiment.campaign_analysis import build_campaign_summary
from moe_opt_core.experiment.protocol import derive_replication_config
from moe_opt_core.experiment.serialization import multilayer_recipe_from_dict
from moe_opt_runner.pilot_artifacts import read_incremental_records, write_json_atomic


def _configured_seeds(config: ExperimentConfig) -> tuple[int, ...]:
    configured = config.multilayer.protocol.replication_seeds
    return configured or (config.calibration.seed,)


def _spectral_rank(plan: Mapping[str, object]) -> int | None:
    diagnostics = plan.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        for key in ("spectral_rank", "rank"):
            value = diagnostics.get(key)
            if value is not None:
                return int(value)
    match = re.search(r"(?:^|-)r(\d+)(?:-|$)", str(plan.get("candidate_id", "")))
    return int(match.group(1)) if match else None


def _read_seed_records(
    config: ExperimentConfig,
) -> tuple[tuple[Mapping[str, object], ...], bool]:
    root = Path(config.artifacts.root_dir) / "multilayer"
    summary_exists = (root / "multilayer_summary.json").exists()
    records = tuple(read_incremental_records(root / "multilayer_runs.jsonl"))
    expected_hash = config_hash(config)
    return (
        tuple(row for row in records if row.get("config_hash") == expected_hash),
        summary_exists,
    )


def _recipe_observations(
    seed: int,
    records: Sequence[Mapping[str, object]],
) -> tuple[ReplicatedRecipeObservationContract, ...]:
    recipe_by_id = {}
    for row in records:
        payload = row.get("recipe")
        if isinstance(payload, Mapping) and row.get("recipe_id") is not None:
            recipe_by_id[str(row["recipe_id"])] = multilayer_recipe_from_dict(payload)
    latest = {
        (str(row["record_type"]), str(row["recipe_id"])): row
        for row in records
        if row.get("record_type")
        in {"selected_multilayer_candidate", "materialized_benchmark"}
        and row.get("recipe_id") is not None
    }
    observations: list[ReplicatedRecipeObservationContract] = []
    for (record_type, recipe_id), row in sorted(latest.items()):
        recipe = recipe_by_id.get(recipe_id)
        if recipe is None:
            raise ValueError(f"Missing recipe payload for {recipe_id} in seed {seed}")
        observations.append(
            ReplicatedRecipeObservationContract(
                seed=seed,
                recipe_id=recipe_id,
                record_type=record_type,
                methods=tuple(
                    sorted(
                        {
                            choice.plan.method.value
                            for choice in recipe.choices
                            if choice.plan is not None
                        }
                    )
                ),
                layer_count=len(recipe.choices),
                target_compression=recipe.target_compression,
                achieved_compression=recipe.achieved_compression,
                relative_perplexity_degradation=float(
                    row["relative_perplexity_degradation"]
                ),
                throughput_gain=float(row["throughput_gain"]),
                memory_reduction=float(row["memory_reduction"]),
            )
        )
    return tuple(observations)


def _sensitivity_observations(
    seed: int,
    records: Sequence[Mapping[str, object]],
) -> tuple[SensitivityVariantObservationContract, ...]:
    latest = {
        str(row["candidate_id"]): row
        for row in records
        if row.get("record_type") == "sensitivity_candidate"
        and row.get("candidate_id") is not None
    }
    observations: list[SensitivityVariantObservationContract] = []
    for candidate_id, row in sorted(latest.items()):
        plan = row.get("plan")
        if not isinstance(plan, Mapping):
            raise ValueError(f"Missing candidate plan for {candidate_id}")
        robustness = row.get("validation_robustness")
        robust_score = float(row["relative_perplexity_degradation"])
        if isinstance(robustness, Mapping):
            robust_score = float(robustness["robust_score"])
        observations.append(
            SensitivityVariantObservationContract(
                seed=seed,
                layer=int(plan["layer"]),
                method=str(plan["method"]),
                compression_fraction=float(plan["compression_fraction"]),
                candidate_id=candidate_id,
                spectral_rank=_spectral_rank(plan),
                merge_weighting=str(plan.get("merge_weighting", "uniform")),
                robust_score=robust_score,
            )
        )
    return tuple(observations)


def aggregate_multilayer_campaign(
    config: ExperimentConfig,
    *,
    allow_partial: bool,
) -> tuple[dict[str, object], Path]:
    seeds = _configured_seeds(config)
    completed: list[int] = []
    recipes: list[ReplicatedRecipeObservationContract] = []
    sensitivity: list[SensitivityVariantObservationContract] = []
    for seed in seeds:
        replicated = derive_replication_config(config, seed)
        records, is_complete = _read_seed_records(replicated)
        if not is_complete:
            continue
        completed.append(seed)
        recipes.extend(_recipe_observations(seed, records))
        sensitivity.extend(_sensitivity_observations(seed, records))
    missing = tuple(seed for seed in seeds if seed not in completed)
    if missing and not allow_partial:
        raise RuntimeError(f"Campaign is incomplete; missing seeds: {list(missing)}")
    summary = build_campaign_summary(
        config_name=config.name,
        expected_seeds=seeds,
        completed_seeds=completed,
        recipe_observations=recipes,
        sensitivity_observations=sensitivity,
        quality_budget=config.multilayer.quality_budget,
    )
    output_path = Path(config.artifacts.root_dir) / "campaign_summary.json"
    return asdict(summary), output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate a replicated RMT-MoE campaign")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    payload, output_path = aggregate_multilayer_campaign(
        config,
        allow_partial=args.allow_partial,
    )
    if not args.no_write:
        write_json_atomic(output_path, payload)
    print(
        json.dumps(
            {"output_path": str(output_path), **payload},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
