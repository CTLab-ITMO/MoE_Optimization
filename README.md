# MoE_Optimization

This repository contains experiments for Mixture-of-Experts (MoE) optimization.

## Architecture

- `moe_opt_core/`: shared algorithmic core for H1/H2/H3, typed configs, artifact contracts, metrics.
- `moe_opt_runner/`: JSON-driven orchestration.
- `configs/experiments/*.json`: single source of truth for experiment runs.
- `configs/constants/constant_repo_*.json`: grouped constants (training, routing, visualization, io).

## Quick Start

```bash
python -m moe_opt_runner.main --config configs/experiments/h123_refactor_baseline.json --validate-only
python -m moe_opt_runner.main --config configs/experiments/h123_refactor_baseline.json
```

Run the environment gate before a real GPU pilot:

```bash
python -m moe_opt_runner.preflight --config configs/experiments/h1_rmt_one_layer_pilot.json --require-cuda --check-model-config
python -m moe_opt_runner.pilot_cli --config configs/experiments/h1_rmt_one_layer_pilot.json --step status
```

Run the first Qwen pilot through the bounded commands in
`docs/6. Runbook первой RMT-MoE итерации.md`; do not use the monolithic runner
for the server experiment.

For Qwen1.5-MoE the runtime contract treats `top_k=4` as routed top-k and tracks
the four always-active shared experts separately. Preflight and trace validation
stop the run if the loaded model does not match the configured 60 routed + 4
shared architecture.

Install the experiment-only dependencies without replacing the CUDA-specific Torch
build already present in the environment:

```bash
python -m pip install -r requirements-experiment.txt
```

The one-layer pilot uses real Hugging Face dataset traces, finite-size RMT diagnostics,
reversible candidate materialization, validation-driven selection, and held-out test
evaluation. Candidate results are appended to `pilot_runs.jsonl`, so completed runs
remain available if a later candidate fails.

The follow-up held-out hybrid ablation is configured in
`configs/experiments/h1_rmt_hybrid_heldout_one_layer_pilot.json`. It uses a fresh
dataset split and compares pure RMT geometry with a deterministic
`rmt_functional_usage` pairing score. Validation traces must be collected before
`h1_diagnostics`; the resumable status planner enforces this dependency.

The detailed launch and interpretation protocol is documented in
`docs/6. Runbook первой RMT-MoE итерации.md`.

The robust rank-selection follow-up is configured in
`configs/experiments/h1_rmt_multislice_rank_validation_one_layer_pilot.json`.
It reuses the held-out traces, evaluates each candidate on three deterministic
validation slices, and ranks candidates by mean degradation plus a dispersion
penalty before the test split is unlocked.

Note: full run requires `torch` and model weights available in the environment.

## Legacy Thin Wrappers

```bash
python Qwen_MoE_experts_fusion_experiment/1_generate_data.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/2_find_best_matches.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/2_find_pairs_h1_spikes.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/3_save_experts.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/4_write_mlp_inputs.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/5_activations.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/6_fusion.py --config configs/experiments/h123_refactor_baseline.json --layer 0
python Qwen_MoE_experts_fusion_experiment/6_fusion_parallel.py --config configs/experiments/h123_refactor_baseline.json --layer 0
python Qwen_MoE_experts_fusion_experiment/expert_unfolding_spikes_pairs.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/7_final_benchmark.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/8_generation.py --config configs/experiments/h123_refactor_baseline.json
python Qwen_MoE_experts_fusion_experiment/run_from_config.py --config configs/experiments/h123_refactor_baseline.json
```

## Auto-Bias (H2)

Unified runner:

```bash
python MoE_auto_bias_experiments/auto_bias_runner.py --profile mixtral --config configs/experiments/h2_autobias_template.json --dry-run
python MoE_auto_bias_experiments/auto_bias_runner.py --profile qwen3 --config configs/experiments/h2_autobias_template.json
```

Profile wrappers:

```bash
python MoE_auto_bias_experiments/WES_auto_bias_mixtral.py --dry-run
python MoE_auto_bias_experiments/WES_auto_bias_qwen3.py --dry-run
python MoE_auto_bias_experiments/WES_auto_bias_phi.py --dry-run
python MoE_auto_bias_experiments/WES_auto_bias_qwen1.5.py --dry-run
```

Legacy scripts that import `Qwen_MoE_experts_fusion_experiment/config.py` can be redirected to another experiment config via `MOE_EXPERIMENT_CONFIG`.

## Artifact Contracts

- Core runner writes artifacts_manifest.json + run_summary.json.
- Auto-bias runner additionally writes auto_bias_manifest.json + auto_bias_runner_summary.json.
- Legacy wrappers write legacy_manifest_<step>.json (+ optional legacy_summary_<step>.json) into artifacts.root_dir.
## Ablation Matrix

Generate comparable H1/H2/H3 experiment configs from one base JSON:

```bash
python -m moe_opt_runner.ablation --config configs/experiments/h123_refactor_baseline.json
```

Execute all generated cases (requires runtime dependencies/model access):

```bash
python -m moe_opt_runner.ablation --config configs/experiments/h123_refactor_baseline.json --execute --continue-on-error
```


