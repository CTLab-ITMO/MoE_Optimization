# Refactor PR Overview: MoE Optimization Core/Runner Unification

## 1) PR Goal

This PR finalizes the infrastructure refactor of the MoE optimization codebase toward:

- a single JSON-driven experiment interface;
- strict separation of algorithmic core and legacy scripts;
- explicit artifact contracts for reproducibility;
- fair H1/H2/H3 ablations from one base config;
- lower engineering risk (import-time side effects, scattered defaults, runtime inconsistencies).

## 2) Main Problems Before Refactor

Before the refactor, the project had several systemic risks:

- experiment orchestration and algorithms were mixed inside scripts;
- experiment defaults were scattered across multiple files;
- artifact outputs were not consistently declared;
- legacy pipeline had fragile behavior (`if pairs`, import-time config loading);
- fair method comparison (H1/H2/H3) from one baseline config was cumbersome.

## 3) What Changed

### 3.1 Architecture and Execution

- The split is now explicit and stable:
  - `moe_opt_core`: algorithms, contracts, typed config models.
  - `moe_opt_runner`: stage orchestration and CLI execution.
- Legacy scripts are kept as thin wrappers over JSON configs.

### 3.2 JSON as Source of Truth

- Experiments are launched from `configs/experiments/*.json`.
- Constants are centralized in `configs/constants/constant_repo_*.json`.
- Validation-first workflow via `--validate-only` is supported.

### 3.3 Unified H2 Auto-Bias Flow

- Added unified runner:
  - `MoE_auto_bias_experiments/auto_bias_runner.py`
- Kept profile wrappers (`WES_*`) as compatibility entrypoints.
- Supports both H2 modes (`gate_spike`, `covariance`) without duplicated logic.

### 3.4 Unified Fusion Flow

- Added shared fusion runner:
  - `Qwen_MoE_experts_fusion_experiment/fusion_runner.py`
- `6_fusion.py` and `6_fusion_parallel.py` now delegate to it.
- Removed duplicated fusion orchestration paths.

### 3.5 Artifact Contract Hardening

- Added shared helpers in `moe_opt_core.artifacts`:
  - `write_artifact_manifest(...)`
  - `write_run_summary(...)`
- Core runner, legacy flow, and auto-bias flow now emit contract-based manifests.

Current outputs:

- core runner: `artifacts_manifest.json`, `run_summary.json`
- auto-bias: `auto_bias_manifest.json`, `auto_bias_runner_summary.json`
- legacy wrappers: `legacy_manifest_<step>.json` (+ optional `legacy_summary_<step>.json`)

### 3.6 H1/H2/H3 Ablation Matrix from One Config

- Added `moe_opt_runner/ablation.py`.
- Builds case matrix from one base config:
  - `baseline`, `h1`, `h2`, `h3`, `h1_h2`, `h1_h3`, `h2_h3`, `h1_h2_h3`
- Produces:
  - `ablation_plan.json` (generated configs and paths)
  - `ablation_report.json` (when executed)

This enables fair comparisons: same base config, controlled stage toggles.

### 3.7 Critical Defect Fixes

- Fixed fragile legacy pair check (`if pairs`) with safe explicit condition.
- Reduced import-time side effects:
  - `Qwen_MoE_experts_fusion_experiment/config.py` is now lazy-loaded and cached.
  - `fusion_layer.py` no longer uses top-level `from config import ...` routing constants.
- Removed `torch.cuda.empty_cache()` from hot loops and kept it only at controlled boundaries.

## 4) Backward Compatibility

- Legacy CLI entrypoints remain available.
- Legacy scripts depending on `Qwen_MoE_experts_fusion_experiment/config.py` still support:
  - `MOE_EXPERIMENT_CONFIG=/path/to/experiment.json`

## 5) Not Included in This PR

- Kernel-level GPU optimization work.
- Full runtime validation on large real models (requires target environment with `torch` + weights).
- Full `pytest` run (intentionally skipped in this workflow).

## 6) Validation Performed in This PR

Performed smoke/static validation:

- `python -m py_compile` and `python -m compileall` for project modules.
- `python -m moe_opt_runner.main --validate-only` for experiment configs.
- `--help` checks for legacy and new CLIs.
- Auto-bias dry-runs for profile wrappers.
- Ablation plan generation via `moe_opt_runner.ablation`.

## 7) Usage After Merge

### Core runner

```bash
python -m moe_opt_runner.main --config configs/experiments/h123_refactor_baseline.json --validate-only
python -m moe_opt_runner.main --config configs/experiments/h123_refactor_baseline.json
```

### Auto-bias

```bash
python MoE_auto_bias_experiments/auto_bias_runner.py --profile mixtral --config configs/experiments/h2_autobias_template.json --dry-run
python MoE_auto_bias_experiments/auto_bias_runner.py --profile qwen3 --config configs/experiments/h2_autobias_template.json
```

### Ablation matrix

```bash
python -m moe_opt_runner.ablation --config configs/experiments/h123_refactor_baseline.json
python -m moe_opt_runner.ablation --config configs/experiments/h123_refactor_baseline.json --execute --continue-on-error
```

## 8) Residual Risk and Follow-up

- A runtime validation pass is still required in target GPU/model environment.
- Add or extend regression metric checks for non-degradation guarantees.
- Optional further cleanup: migrate remaining historical scripts to pure wrapper mode.

## 9) Summary

This PR is focused on engineering reliability, reproducibility, and clearer experiment control.
It establishes a robust base for further RMT-driven H1/H2/H3 research and production-like benchmarking.