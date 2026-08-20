from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from moe_opt_core.config import ExperimentConfig


def resolve_replication_seed(
    config: ExperimentConfig,
    requested_seed: int | None,
) -> int | None:
    configured = config.multilayer.protocol.replication_seeds
    if requested_seed is not None:
        if configured and int(requested_seed) not in configured:
            raise ValueError(
                f"replication seed {requested_seed} is not enabled in the config"
            )
        return int(requested_seed)
    if len(configured) > 1:
        raise ValueError(
            "the config contains multiple replication seeds; select one with "
            "--replication-seed"
        )
    return configured[0] if configured else None


def derive_replication_config(
    config: ExperimentConfig,
    seed: int,
) -> ExperimentConfig:
    seed = int(seed)
    root = Path(config.artifacts.root_dir) / f"seed_{seed}"
    trace_root = root / "traces"
    training = dict(config.constants.get("training", {}))
    training["seed"] = seed
    constants = dict(config.constants)
    constants["training"] = training
    return replace(
        config,
        name=f"{config.name}_seed_{seed}",
        constants=constants,
        artifacts=replace(
            config.artifacts,
            root_dir=str(root),
            manifest_path=str(root / "artifacts_manifest.json"),
        ),
        calibration=replace(
            config.calibration,
            output_dir=str(trace_root),
            seed=seed,
        ),
        multilayer=replace(
            config.multilayer,
            materialization=replace(
                config.multilayer.materialization,
                output_dir=str(root / "materialized"),
            ),
        ),
        h1=replace(
            config.h1,
            mlp_inputs_path=str(trace_root / "h1_calibration_inputs.pt"),
            output_pairs_path=str(root / "pairs.pt"),
            output_diagnostics_path=str(root / "h1_diagnostics.json"),
            output_merged_dir=str(root / "merged"),
        ),
        h2=replace(
            config.h2,
            output_bias_path=str(root / "h2_bias.pt"),
        ),
        h3=replace(
            config.h3,
            routing_probs_path=str(root / "routing_probs.pt"),
            output_diagnostics_path=str(root / "h3_diagnostics.json"),
        ),
    )


def materialize_replication_config(
    config: ExperimentConfig,
    requested_seed: int | None,
) -> ExperimentConfig:
    seed = resolve_replication_seed(config, requested_seed)
    return config if seed is None else derive_replication_config(config, seed)
