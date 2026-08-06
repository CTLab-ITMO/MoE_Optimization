from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from typing import Dict

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment.runtime_validation import (
    infer_configured_routed_expert_count,
    infer_configured_routed_top_k,
    infer_configured_shared_expert_count,
)


def build_preflight_report(
    config_path: str,
    *,
    require_cuda: bool = False,
    check_model_config: bool = False,
) -> Dict[str, object]:
    config = load_experiment_config(config_path)
    report: Dict[str, object] = {
        "config": config.name,
        "python": platform.python_version(),
        "model": config.model.name,
        "calibration_source": config.calibration.source,
        "dependencies": {},
        "errors": [],
    }
    for dependency in ("torch", "transformers", "datasets"):
        available = importlib.util.find_spec(dependency) is not None
        report["dependencies"][dependency] = available
        if dependency in {"torch", "transformers"} and not available:
            report["errors"].append(f"missing_dependency:{dependency}")
        if dependency == "datasets" and config.calibration.source == "hf_dataset" and not available:
            report["errors"].append("missing_dependency:datasets")

    if report["dependencies"]["torch"]:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
        report["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if require_cuda and not torch.cuda.is_available():
            report["errors"].append("cuda_required_but_unavailable")
    else:
        report["cuda_available"] = False

    if check_model_config and report["dependencies"]["transformers"]:
        try:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(
                config.model.name,
                local_files_only=config.model.local_files_only,
                trust_remote_code=config.model.trust_remote_code,
            )
            report["model_config_available"] = True
            architecture = {
                "routed_experts": infer_configured_routed_expert_count(model_config),
                "shared_experts": infer_configured_shared_expert_count(model_config),
                "routed_top_k": infer_configured_routed_top_k(model_config),
            }
            report["model_architecture"] = architecture
            expectations = {
                "routed_experts": config.runtime_expected_routed_experts,
                "shared_experts": config.runtime_expected_shared_experts,
                "routed_top_k": config.runtime_top_k,
            }
            report["expected_model_architecture"] = expectations
            for name, expected in expectations.items():
                observed = architecture[name]
                if expected is not None and observed is not None and observed != expected:
                    report["errors"].append(
                        f"model_architecture_mismatch:{name}:expected={expected}:observed={observed}"
                    )
        except Exception as exc:
            report["model_config_available"] = False
            report["errors"].append(f"model_config_unavailable:{type(exc).__name__}")

    report["ready"] = not report["errors"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a MoE experiment runtime before model loading")
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-model-config", action="store_true")
    args = parser.parse_args()
    report = build_preflight_report(
        args.config,
        require_cuda=args.require_cuda,
        check_model_config=args.check_model_config,
    )
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
