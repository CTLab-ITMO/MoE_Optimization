from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CASES = [
    "baseline",
    "h1",
    "h2",
    "h3",
    "h1_h2",
    "h1_h3",
    "h2_h3",
    "h1_h2_h3",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _normalize_constants_paths(raw: dict[str, Any], config_path: Path) -> None:
    constants = raw.get("constants", {})
    for key, value in list(constants.items()):
        p = Path(str(value))
        if not p.is_absolute():
            constants[key] = str((config_path.parent / p).resolve())


def _parse_case(case: str) -> dict[str, bool]:
    if case == "baseline":
        return {"h1": False, "h2": False, "h3": False}

    flags = {"h1": False, "h2": False, "h3": False}
    for name in case.split("_"):
        if name in flags:
            flags[name] = True
        elif name:
            raise ValueError(f"Unknown stage token {name!r} in case {case!r}")
    return flags


def _retarget_stage_paths(cfg: dict[str, Any], case_root: Path, layer: int) -> None:
    artifacts = cfg.setdefault("artifacts", {})
    artifacts["root_dir"] = str(case_root)
    artifacts["manifest_path"] = str(case_root / "artifacts_manifest.json")
    artifacts.setdefault("schema_version", "1.0")

    stages = cfg.setdefault("stages", {})
    h1 = stages.setdefault("h1", {})
    h2 = stages.setdefault("h2", {})
    h3 = stages.setdefault("h3", {})

    h1["layer"] = int(h1.get("layer", layer))
    h1["mlp_inputs_path"] = str(case_root / f"routing_inputs_layer{layer}.pt")
    h1["output_pairs_path"] = str(case_root / f"pairs_layer{layer}.pt")
    h1["output_diagnostics_path"] = str(case_root / f"h1_diag_layer{layer}.json")
    h1["output_merged_dir"] = str(case_root / "merged")

    h2["layer"] = int(h2.get("layer", layer))
    h2["output_bias_path"] = str(case_root / f"h2_bias_layer{layer}.pt")
    if str(h2.get("mode", "gate_spike")) == "covariance":
        h2["importance_matrix_path"] = str(case_root / f"importance_matrix_layer{layer}.pt")
        h2["importance_vector_path"] = ""
        h2["gate_weight_path"] = ""
    else:
        h2["importance_vector_path"] = str(case_root / f"importance_vec_layer{layer}.pt")
        h2["gate_weight_path"] = str(case_root / f"gate_weight_layer{layer}.pt")
        h2["importance_matrix_path"] = ""

    h3["routing_probs_path"] = str(case_root / f"routing_probs_layer{layer}.pt")
    h3["output_diagnostics_path"] = str(case_root / f"h3_diag_layer{layer}.json")


def _build_case_config(base_cfg: dict[str, Any], *, base_name: str, case: str, case_root: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    flags = _parse_case(case)
    layer = int(cfg.get("layers", [0])[0])

    cfg["name"] = f"{base_name}__{case}"
    cfg["layers"] = [layer]

    stages = cfg.setdefault("stages", {})
    stages.setdefault("h1", {})["enabled"] = flags["h1"]
    stages.setdefault("h2", {})["enabled"] = flags["h2"]
    stages.setdefault("h3", {})["enabled"] = flags["h3"]

    _retarget_stage_paths(cfg, case_root, layer)
    return cfg


def _collect_summary(case: str, result: dict[str, Any], status: str, error: str | None = None) -> dict[str, Any]:
    return {
        "case": case,
        "status": status,
        "error": error,
        "timing_s": result.get("timing_s", {}),
        "stages": result.get("stages", {}),
    }


def run_ablation(
    *,
    config_path: str,
    output_root: str | None,
    cases: list[str],
    execute: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    cfg_path = Path(config_path).resolve()
    base_cfg = _load_json(cfg_path)
    _normalize_constants_paths(base_cfg, cfg_path)

    base_name = str(base_cfg.get("name", "experiment"))
    root = Path(output_root) if output_root else (Path("artifacts") / "ablations" / base_name)
    root.mkdir(parents=True, exist_ok=True)

    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    plan: list[dict[str, Any]] = []
    for case in cases:
        case_root = root / case
        cfg = _build_case_config(base_cfg, base_name=base_name, case=case, case_root=case_root)
        cfg_path_out = configs_dir / f"{case}.json"
        _dump_json(cfg_path_out, cfg)
        plan.append(
            {
                "case": case,
                "config_path": str(cfg_path_out),
                "artifact_root": str(case_root),
                "enabled": {
                    "h1": bool(cfg["stages"]["h1"].get("enabled", False)),
                    "h2": bool(cfg["stages"]["h2"].get("enabled", False)),
                    "h3": bool(cfg["stages"]["h3"].get("enabled", False)),
                },
            }
        )

    plan_payload = {
        "base_config": str(cfg_path),
        "output_root": str(root),
        "execute": bool(execute),
        "cases": plan,
    }
    _dump_json(root / "ablation_plan.json", plan_payload)

    if not execute:
        return {
            "status": "planned",
            "plan_path": str(root / "ablation_plan.json"),
            "output_root": str(root),
            "cases": len(plan),
        }

    from moe_opt_runner.main import run_experiment_from_config

    report_rows: list[dict[str, Any]] = []
    for item in plan:
        case = str(item["case"])
        case_cfg_path = str(item["config_path"])
        try:
            result = run_experiment_from_config(case_cfg_path)
            report_rows.append(_collect_summary(case, result, status="ok"))
        except Exception as exc:  # noqa: BLE001
            report_rows.append(_collect_summary(case, {}, status="failed", error=str(exc)))
            if not continue_on_error:
                break

    report = {
        "base_config": str(cfg_path),
        "output_root": str(root),
        "cases": report_rows,
    }
    _dump_json(root / "ablation_report.json", report)
    return {
        "status": "executed",
        "plan_path": str(root / "ablation_plan.json"),
        "report_path": str(root / "ablation_report.json"),
        "output_root": str(root),
        "cases": len(report_rows),
    }


def _parse_cases(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_CASES)
    out = [x.strip() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--cases produced empty case set")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally execute H1/H2/H3 ablation matrix")
    parser.add_argument("--config", required=True, type=str, help="Base experiment JSON")
    parser.add_argument("--output-root", type=str, default=None, help="Root for generated ablation configs and reports")
    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(DEFAULT_CASES),
        help="Comma-separated case list: baseline,h1,h2,h3,h1_h2,h1_h3,h2_h3,h1_h2_h3",
    )
    parser.add_argument("--execute", action="store_true", help="Run generated configs (requires runtime dependencies)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue other cases after a failed case")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out = run_ablation(
        config_path=args.config,
        output_root=args.output_root,
        cases=_parse_cases(args.cases),
        execute=args.execute,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
