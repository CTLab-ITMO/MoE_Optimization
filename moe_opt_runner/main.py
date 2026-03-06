from __future__ import annotations

import argparse
import json
from typing import Dict

from moe_opt_core.artifacts import write_artifact_manifest, write_run_summary
from moe_opt_core.config import config_hash, load_experiment_config
from moe_opt_core.metrics import StageTimer


def _write_manifest_without_stage_module(config, stage_outputs: Dict[str, object]) -> None:
    write_artifact_manifest(
        schema_version=config.artifacts.schema_version,
        config_hash=config_hash(config),
        root_dir=config.artifacts.root_dir,
        manifest_path=config.artifacts.manifest_path,
        paths={
            "h1_pairs": config.h1.output_pairs_path,
            "h1_diag": config.h1.output_diagnostics_path,
            "h2_bias": config.h2.output_bias_path,
            "h3_diag": config.h3.output_diagnostics_path,
        },
    )
    write_run_summary(
        root_dir=config.artifacts.root_dir,
        summary_name="run_summary.json",
        payload=stage_outputs,
    )


def run_experiment_from_config(config_path: str) -> Dict[str, object]:
    config = load_experiment_config(config_path)
    timer = StageTimer()
    outputs: Dict[str, object] = {"experiment": config.name, "stages": {}}

    if config.h1.enabled or config.h2.enabled or config.h3.enabled:
        from moe_opt_runner.stages import run_h1_stage, run_h2_stage, run_h3_stage, write_artifact_manifest

        ctx = None
        if config.model.load_source == "hf":
            from moe_opt_runner.runtime import build_runtime_context

            with timer.time("runtime_context"):
                ctx = build_runtime_context(config)

        if config.h1.enabled:
            with timer.time("h1"):
                outputs["stages"]["h1"] = run_h1_stage(config, ctx=ctx)

        if config.h2.enabled:
            with timer.time("h2"):
                outputs["stages"]["h2"] = run_h2_stage(config, ctx=ctx)

        if config.h3.enabled:
            with timer.time("h3"):
                outputs["stages"]["h3"] = run_h3_stage(config, ctx=ctx)

        write_artifact_manifest(config, outputs["stages"])
    else:
        _write_manifest_without_stage_module(config, outputs["stages"])

    outputs["timing_s"] = timer.elapsed_s
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoE optimization experiment from JSON config")
    parser.add_argument("--config", required=True, type=str, help="Path to experiment JSON")
    parser.add_argument("--validate-only", action="store_true", help="Only validate config")
    args = parser.parse_args()

    if args.validate_only:
        cfg = load_experiment_config(args.config)
        print(f"Config is valid: {cfg.name}")
        return

    outputs = run_experiment_from_config(args.config)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
