# Experiment launch scripts

Run every command from the repository root. One config is one isolated
experiment: `python -m experts_fusion.run --config <name>` runs all four pipeline
stages and writes outputs to the config's own `DATA_DIR`.

The scripts below are intentionally sequential. They stop on the first failed
experiment and preserve completed `DATA_DIR` trees, so rerunning the same script
can continue through the pipeline's freshness checks.

## Main fusion experiments

This is the primary compression approach: activation-correlation grouping on
Qwen1.5-MoE, evaluated on three datasets. Each config runs every fusion method
selected by its `FILTER_FUSIONS`; the current experiment configs exclude
`Distillation` and `Best_*` pseudo-methods.

```bash
#!/usr/bin/env bash
set -euo pipefail

for config in \
    mmlu_activation_correlation \
    commonsense_qa_activation_correlation \
    arc_challenge_activation_correlation
do
    python -m experts_fusion.run --config "$config"
done
```

The outputs are written to:

- `./data/ef_mmlu_activation_correlation`
- `./data/ef_commonsense_qa_activation_correlation`
- `./data/ef_arc_challenge_activation_correlation`

## Grouping ablations

These runs compare the primary grouping strategy with random grouping and
output-similarity grouping on MMLU and CommonsenseQA.

```bash
#!/usr/bin/env bash
set -euo pipefail

for config in \
    mmlu_random \
    mmlu_output_similarity \
    commonsense_qa_random \
    commonsense_qa_output_similarity
do
    python -m experts_fusion.run --config "$config"
done
```

Together with the MMLU and CommonsenseQA runs from the previous section, this
forms the complete `{mmlu, commonsense_qa}` ×
`{activation_correlation, random, output_similarity}` grouping grid.

## Second-model variation

This runs the primary grouping and fusion pipeline on OLMoE-1B-7B using
ARC-Challenge.

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m experts_fusion.run --config olmoe_arc_challenge
```

The output is written to `./data/ef_olmoe_arc_challenge`.

## REAP baseline

These configs are provided by the `reap-baselines` branch. They run 50% REAP
expert pruning with router masking and the identity `Pruning` fusion path.

```bash
#!/usr/bin/env bash
set -euo pipefail

for config in \
    reap_mmlu \
    reap_commonsense_qa \
    reap_arc_challenge
do
    python -m experts_fusion.run --config "$config"
done
```

## HC-SMoE baseline

These configs are provided by the `reap-baselines` branch. They run
characteristic-activation hierarchical clustering with a 50% target cluster
count and `FrequencyWeightedAverage` fusion.

```bash
#!/usr/bin/env bash
set -euo pipefail

for config in \
    hcsmoe_mmlu \
    hcsmoe_commonsense_qa \
    hcsmoe_arc_challenge
do
    python -m experts_fusion.run --config "$config"
done
```

## Result collection

After all runs have completed, assemble the sequential-replacement results into
one CSV file:

```bash
python -m experts_fusion.report \
    ./data/ef_mmlu_activation_correlation \
    ./data/ef_commonsense_qa_activation_correlation \
    ./data/ef_arc_challenge_activation_correlation \
    ./data/ef_mmlu_random \
    ./data/ef_mmlu_output_similarity \
    ./data/ef_commonsense_qa_random \
    ./data/ef_commonsense_qa_output_similarity \
    ./data/ef_olmoe_arc_challenge \
    ./data/ef_reap_mmlu \
    ./data/ef_reap_commonsense_qa \
    ./data/ef_reap_arc_challenge \
    ./data/ef_hcsmoe_mmlu \
    ./data/ef_hcsmoe_commonsense_qa \
    ./data/ef_hcsmoe_arc_challenge \
    --mode seq \
    --format csv \
    --out experiments.csv
```
