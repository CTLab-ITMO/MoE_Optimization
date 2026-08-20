from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import EvaluationMetricsContract
from moe_opt_core.experiment.pilot_validation import summarize_evaluation_repeats
from moe_opt_core.experiment.protocol import (
    derive_replication_config,
    resolve_replication_seed,
)


def _metrics(nll: float, throughput: float) -> EvaluationMetricsContract:
    return EvaluationMetricsContract(
        nll=nll,
        perplexity=2.0**nll,
        tokens_per_second=throughput,
        peak_memory_mb=100.0,
        model_parameter_mb=50.0,
        evaluated_tokens=128,
        elapsed_seconds=128.0 / throughput,
    )


def test_repeat_summary_uses_median_and_preserves_spread() -> None:
    aggregate, summary = summarize_evaluation_repeats(
        (_metrics(1.0, 100.0), _metrics(1.1, 200.0), _metrics(1.2, 300.0))
    )
    assert aggregate.nll == pytest.approx(1.1)
    assert aggregate.tokens_per_second == pytest.approx(200.0)
    assert summary.repeat_count == 3
    assert summary.tokens_per_second_p05 == pytest.approx(110.0)
    assert summary.tokens_per_second_p95 == pytest.approx(290.0)
    assert summary.perplexity_std > 0.0


def test_replication_config_isolates_artifacts_and_updates_all_random_seeds() -> None:
    config = load_experiment_config(
        "configs/experiments/h1_rmt_multilayer_representative_pilot.json"
    )
    replicated = derive_replication_config(config, 2027)
    assert replicated.calibration.seed == 2027
    assert replicated.constants["training"]["seed"] == 2027
    assert replicated.artifacts.root_dir.endswith("seed_2027")
    assert Path(replicated.calibration.output_dir).parts[-2:] == ("seed_2027", "traces")
    assert Path(replicated.multilayer.materialization.output_dir).parts[-2:] == (
        "seed_2027",
        "materialized",
    )
    assert config.constants["training"]["seed"] != 2027


def test_multiple_replication_seeds_require_an_explicit_cli_choice() -> None:
    config = load_experiment_config(
        "configs/experiments/h1_rmt_multilayer_representative_pilot.json"
    )
    config = replace(
        config,
        multilayer=replace(
            config.multilayer,
            protocol=replace(
                config.multilayer.protocol,
                replication_seeds=(2026, 2027, 2028),
            ),
        ),
    )
    with pytest.raises(ValueError, match="--replication-seed"):
        resolve_replication_seed(config, None)
    assert resolve_replication_seed(config, 2027) == 2027


def test_full_model_campaign_loads_rank_protocol_and_progressive_policy() -> None:
    config = load_experiment_config(
        "configs/experiments/h1_rmt_multilayer_full_model.json"
    )
    assert config.pilot.spectral_pairing_ranks == (1, 2, 4, 8, 16)
    assert config.pilot.spectral_rank_policy.include_useful_rank
    assert config.pilot.spectral_rank_policy.include_mp_rank
    assert config.multilayer.protocol.replication_seeds == (2026, 2027, 2028)
    assert config.multilayer.selection.strategy == "method_balanced"
    assert config.multilayer.selection.balance_layer_counts
    assert not config.multilayer.selection.balance_refresh_modes
    assert config.multilayer.materialization.strategy == "method_balanced"
    assert config.multilayer.progressive_scaling.layer_counts == (3, 6, 12, 24)
