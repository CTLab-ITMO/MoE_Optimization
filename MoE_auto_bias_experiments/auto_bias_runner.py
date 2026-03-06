from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))


_PROFILES: dict[str, dict[str, Any]] = {
    "mixtral": {
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "model_kind": "mixtral",
        "runtime_top_k": 2,
    },
    "qwen3": {
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "model_kind": "qwen",
        "runtime_top_k": 8,
    },
    "phi": {
        "model_name": "microsoft/Phi-3.5-MoE-instruct",
        "model_kind": "mixtral",
        "runtime_top_k": 2,
    },
    "qwen1_5": {
        "model_name": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "model_kind": "qwen",
        "runtime_top_k": 8,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "experiments" / "h123_refactor_baseline.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_metrics(metrics_path: Path, run_name: str, result: dict[str, Any]) -> None:
    stage_h2 = result.get("stages", {}).get("h2", {})
    line = {
        "run": run_name,
        "bias_norm": stage_h2.get("bias_norm"),
        "method": stage_h2.get("method"),
        "timing_s": result.get("timing_s", {}),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _write_auto_bias_contract(
    *,
    run_cfg_path: Path,
    root: Path,
    exp_cfg,
    status: str,
    profile: str,
    metrics_path: str | None,
    summary_path: Path | None,
) -> Path:
    from moe_opt_core.artifacts import write_artifact_manifest, write_run_summary
    from moe_opt_core.config import config_hash

    manifest_path = root / "auto_bias_manifest.json"
    stage_summary = Path(exp_cfg.artifacts.root_dir) / "run_summary.json"

    write_artifact_manifest(
        schema_version=exp_cfg.artifacts.schema_version,
        config_hash=config_hash(exp_cfg),
        root_dir=exp_cfg.artifacts.root_dir,
        manifest_path=manifest_path,
        paths={
            "run_config": run_cfg_path,
            "stage_manifest": exp_cfg.artifacts.manifest_path,
            "stage_summary": stage_summary,
            "auto_bias_summary": summary_path,
            "metrics_jsonl": metrics_path,
        },
    )

    write_run_summary(
        root_dir=exp_cfg.artifacts.root_dir,
        summary_name="auto_bias_runner_summary.json",
        payload={
            "status": status,
            "profile": profile,
            "run_config": str(run_cfg_path),
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path) if summary_path is not None else None,
            "metrics_path": metrics_path,
        },
    )
    return manifest_path


def _build_config(
    base_cfg: dict[str, Any],
    *,
    profile: str,
    model_name: str | None,
    model_kind: str | None,
    runtime_top_k: int | None,
    layer: int | None,
    h2_mode: str,
    output_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    prof = _PROFILES[profile]

    cfg["name"] = f"auto_bias_{profile}"

    cfg.setdefault("model", {})
    cfg["model"]["name"] = model_name or prof["model_name"]
    cfg["model"]["kind"] = model_kind or prof["model_kind"]
    cfg["model"].setdefault("load_source", "hf")
    cfg["model"].setdefault("torch_dtype", "auto")
    cfg["model"].setdefault("device_map", "auto")

    cfg.setdefault("runtime", {})
    cfg["runtime"]["top_k"] = int(runtime_top_k or prof["runtime_top_k"])

    if layer is None:
        layer = int(cfg.get("stages", {}).get("h2", {}).get("layer", cfg.get("layers", [0])[0]))
    cfg["layers"] = [int(layer)]

    cfg.setdefault("artifacts", {})
    cfg["artifacts"]["root_dir"] = str(output_root)
    cfg["artifacts"]["manifest_path"] = str(output_root / "artifacts_manifest.json")
    cfg["artifacts"].setdefault("schema_version", "1.0")

    stages = cfg.setdefault("stages", {})
    h1 = stages.setdefault("h1", {})
    h2 = stages.setdefault("h2", {})
    h3 = stages.setdefault("h3", {})

    h1["enabled"] = False
    h3["enabled"] = False

    h2["enabled"] = True
    h2["layer"] = int(layer)
    h2["mode"] = str(h2_mode)
    h2["auto_from_model"] = True
    h2["output_bias_path"] = str(output_root / f"h2_bias_layer{layer}.pt")

    if h2_mode == "covariance":
        h2["importance_matrix_path"] = str(output_root / f"importance_matrix_layer{layer}.pt")
        h2["importance_vector_path"] = ""
        h2["gate_weight_path"] = ""
    else:
        h2["importance_vector_path"] = str(output_root / f"importance_vec_layer{layer}.pt")
        h2["gate_weight_path"] = str(output_root / f"gate_weight_layer{layer}.pt")
        h2["importance_matrix_path"] = ""

    h1["layer"] = int(layer)
    h1["output_pairs_path"] = str(output_root / f"pairs_layer{layer}.pt")
    h1["output_diagnostics_path"] = str(output_root / f"h1_diag_layer{layer}.json")
    h1["output_merged_dir"] = str(output_root / "merged")
    h1.setdefault("mlp_inputs_path", str(output_root / f"routing_inputs_layer{layer}.pt"))
    h1.setdefault("experts_dir", "data/experts")

    h3.setdefault("routing_probs_path", str(output_root / f"routing_probs_layer{layer}.pt"))
    h3["output_diagnostics_path"] = str(output_root / f"h3_diag_layer{layer}.json")

    return cfg


def run_auto_bias(
    *,
    config_path: str,
    profile: str,
    model_name: str | None,
    model_kind: str | None,
    runtime_top_k: int | None,
    layer: int | None,
    h2_mode: str,
    output_root: str | None,
    metrics_path: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if profile not in _PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Available: {sorted(_PROFILES.keys())}")

    cfg_path = Path(config_path).resolve()
    base_cfg = _load_json(cfg_path)

    constants = base_cfg.get("constants", {})
    for key, raw in list(constants.items()):
        pth = Path(str(raw))
        if not pth.is_absolute():
            constants[key] = str((cfg_path.parent / pth).resolve())

    root = Path(output_root) if output_root else (_repo_root() / "artifacts" / "auto_bias" / profile)
    cfg = _build_config(
        base_cfg,
        profile=profile,
        model_name=model_name,
        model_kind=model_kind,
        runtime_top_k=runtime_top_k,
        layer=layer,
        h2_mode=h2_mode,
        output_root=root,
    )

    run_cfg_path = root / "run_config.json"
    _dump_json(run_cfg_path, cfg)

    from moe_opt_core.config import load_experiment_config

    exp_cfg = load_experiment_config(run_cfg_path)

    if dry_run:
        manifest_path = _write_auto_bias_contract(
            run_cfg_path=run_cfg_path,
            root=root,
            exp_cfg=exp_cfg,
            status="validated",
            profile=profile,
            metrics_path=metrics_path,
            summary_path=None,
        )
        return {
            "status": "validated",
            "profile": profile,
            "config": str(run_cfg_path),
            "output_root": str(root),
            "manifest": str(manifest_path),
        }

    from moe_opt_runner.main import run_experiment_from_config

    result = run_experiment_from_config(str(run_cfg_path))
    summary_path = root / "auto_bias_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if metrics_path:
        _append_metrics(Path(metrics_path), cfg["name"], result)

    manifest_path = _write_auto_bias_contract(
        run_cfg_path=run_cfg_path,
        root=root,
        exp_cfg=exp_cfg,
        status="ok",
        profile=profile,
        metrics_path=metrics_path,
        summary_path=summary_path,
    )

    return {
        "status": "ok",
        "profile": profile,
        "config": str(run_cfg_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "result": result,
    }


def build_parser(default_profile: str = "mixtral") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run H2 auto-bias from JSON config")
    parser.add_argument("--config", type=str, default=str(_default_config_path()))
    parser.add_argument("--profile", type=str, default=default_profile, choices=sorted(_PROFILES.keys()))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-kind", type=str, default=None)
    parser.add_argument("--runtime-top-k", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--h2-mode", type=str, default="gate_spike", choices=["gate_spike", "covariance"])
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default="WES_metrics.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(default_profile: str = "mixtral") -> None:
    parser = build_parser(default_profile=default_profile)
    args = parser.parse_args()
    out = run_auto_bias(
        config_path=args.config,
        profile=args.profile,
        model_name=args.model_name,
        model_kind=args.model_kind,
        runtime_top_k=args.runtime_top_k,
        layer=args.layer,
        h2_mode=args.h2_mode,
        output_root=args.output_root,
        metrics_path=args.metrics_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
