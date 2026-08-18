from __future__ import annotations

from pathlib import Path

from moe_opt_core.config import ExperimentConfig
from moe_opt_core.pipelines.h1_merge import run_h1_pipeline
from moe_opt_runner.runtime import RuntimeContext


def layer_diagnostic_paths(
    root: str | Path,
    layer: int,
) -> tuple[Path, Path, Path]:
    layer_root = Path(root) / f"layer_{int(layer)}"
    diagnostics = layer_root / "diagnostics.json"
    return layer_root / "pairs.pt", diagnostics, diagnostics.with_suffix(".pt")


def run_layer_diagnostics(
    config: ExperimentConfig,
    ctx: RuntimeContext,
    *,
    layer: int,
    trace_root: str | Path,
    output_root: str | Path,
):
    handle = ctx.handles_by_layer.get(layer)
    if handle is None:
        raise ValueError(f"Layer {layer} is not available")
    pairs_path, diagnostics_path, _ = layer_diagnostic_paths(output_root, layer)
    return run_h1_pipeline(
        experts=list(handle.experts),
        mlp_inputs_path=str(Path(trace_root) / "calibration" / f"layer_{layer}.pt"),
        n_inputs=config.h1.n_inputs,
        mode="pair",
        proj_dim=config.h1.proj_dim,
        batch_size=config.h1.batch_size,
        seed=int(config.constants["training"]["seed"]),
        mp_factor=config.h1.mp_factor,
        cluster_count=0,
        output_pairs_path=str(pairs_path),
        output_diagnostics_path=str(diagnostics_path),
        validation_inputs_path=str(
            Path(trace_root) / "validation" / f"layer_{layer}.pt"
        ),
        null_trials=config.h1.null_trials,
        bootstrap_trials=config.h1.bootstrap_trials,
        null_quantile=config.h1.null_quantile,
        stability_max_angle_deg=config.h1.stability_max_angle_deg,
    )
