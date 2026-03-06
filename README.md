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


