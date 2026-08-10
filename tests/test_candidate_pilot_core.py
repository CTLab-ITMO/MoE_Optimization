from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from common.moe_utils import MoeHandle
from moe_opt_core.adapters import HFModelAdapter
from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
    MoeLayerRuntimeContract,
)
from moe_opt_core.experiment.candidates import build_candidate_plans, validate_candidate_plan
from moe_opt_core.experiment.evaluation import compare_evaluations
from moe_opt_runner.evaluation import evaluate_language_model
from moe_opt_runner.runtime import RuntimeContext
from moe_opt_runner.pilot import run_one_layer_pilot
from moe_opt_runner.pilot_executor import ResumablePilotExecutor
from moe_opt_runner.transforms import GroupedLinearGate, reversible_one_layer_transform


class TinyMoe(nn.Module):
    def __init__(self, hidden_dim: int, experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(experts)])
        self.num_experts = experts
        self.top_k = min(2, experts)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.gate(hidden), dim=-1)
        outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        return torch.sum(outputs * probs.unsqueeze(-1), dim=-2)


class TinyCausalLm(nn.Module):
    def __init__(self, vocab: int = 16, hidden_dim: int = 6, experts: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden_dim)
        self.moe = TinyMoe(hidden_dim, experts)
        self.lm_head = nn.Linear(hidden_dim, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.lm_head(self.moe(self.embedding(input_ids)))
        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _runtime() -> RuntimeContext:
    model = TinyCausalLm()
    handle = MoeHandle(layer_idx=0, moe=model.moe, gate=model.moe.gate, experts=model.moe.experts, kind="tiny")
    return RuntimeContext(
        model=model,
        adapter=HFModelAdapter(model=model),
        handles_by_layer={0: handle},
        runtime_contracts_by_layer={
            0: MoeLayerRuntimeContract(
                layer=0,
                kind="tiny",
                routed_experts=4,
                shared_experts=0,
                configured_routed_top_k=2,
                requested_routed_top_k=2,
                effective_routed_top_k=2,
                top_k_updated=False,
            )
        },
    )


def test_candidate_plans_are_deterministic_and_valid() -> None:
    kwargs = {
        "num_experts": 6,
        "layer": 0,
        "methods": [
            "random_merge",
            "usage_prune",
            "usage_guided_merge",
            "functional_cosine",
            "rmt_geometry",
            "rmt_functional_usage",
        ],
        "compression_fractions": [0.25],
        "seeds": [7, 11],
        "rmt_pairs": [(0, 1), (2, 3), (4, 5)],
        "spectral_embedding": torch.arange(6, dtype=torch.float32).reshape(6, 1),
        "functional_embedding": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "routing_probs": torch.softmax(torch.randn(10, 6, generator=torch.Generator().manual_seed(3)), dim=-1),
    }
    first = build_candidate_plans(**kwargs)
    second = build_candidate_plans(**kwargs)
    assert first == second
    assert len(first) == 7
    for plan in first:
        validate_candidate_plan(plan, num_experts=6)


def test_usage_guided_merge_pairs_low_usage_experts_with_retained_partners() -> None:
    routing_probs = torch.tensor(
        [
            [0.01, 0.02, 0.30, 0.67],
            [0.01, 0.02, 0.30, 0.67],
        ]
    )
    plans = build_candidate_plans(
        num_experts=4,
        layer=0,
        methods=("usage_guided_merge",),
        compression_fractions=(0.5,),
        seeds=(7,),
        rmt_pairs=((0, 1), (2, 3)),
        spectral_embedding=torch.ones((4, 1)),
        functional_embedding=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        ),
        routing_probs=routing_probs,
    )
    assert plans[0].method == CandidateMethod.USAGE_GUIDED_MERGE
    assert plans[0].action == CandidateAction.MERGE
    assert plans[0].groups == ((0, 2), (1, 3))
    diagnostics = dict(plans[0].diagnostics)
    assert diagnostics["mean_source_usage"] == pytest.approx(0.015)
    assert diagnostics["mean_partner_usage"] == pytest.approx(0.485)
    assert diagnostics["mean_functional_similarity"] == pytest.approx(1.0)


def test_rank_and_weight_ablation_variants_have_unique_candidate_ids() -> None:
    common = {
        "num_experts": 4,
        "layer": 0,
        "compression_fractions": (0.25,),
        "seeds": (7,),
        "rmt_pairs": ((0, 1), (2, 3)),
        "spectral_embedding": torch.ones((4, 1)),
        "spectral_embeddings": {
            1: torch.arange(4, dtype=torch.float32).reshape(4, 1),
            2: torch.arange(8, dtype=torch.float32).reshape(4, 2),
        },
        "functional_embedding": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "routing_probs": torch.full((3, 4), 0.25),
    }
    plans = build_candidate_plans(
        methods=("rmt_geometry", "rmt_functional_usage"),
        hybrid_weight_variants=(
            ("balanced", (0.5, 0.35, 0.15)),
            ("usage_heavy", (0.2, 0.3, 0.5)),
        ),
        **common,
    )
    candidate_ids = tuple(plan.candidate_id for plan in plans)
    assert len(candidate_ids) == 6
    assert len(set(candidate_ids)) == 6
    assert "rmt_geometry-r1-f0.250-s7" in candidate_ids
    assert "rmt_functional_usage-r2-wusage_heavy-f0.250-s7" in candidate_ids
    for plan in plans:
        assert dict(plan.diagnostics)["spectral_pairing_rank"] in {1.0, 2.0}


def test_rank_ablation_config_materializes_thirty_unique_plans() -> None:
    config = load_experiment_config(
        "configs/experiments/h1_rmt_pairing_rank_ablation_one_layer_pilot.json"
    )
    executor = ResumablePilotExecutor(config)
    raw_embedding = torch.randn(60, 64, generator=torch.Generator().manual_seed(31))
    plans = executor._build_plans(
        num_experts=60,
        trace={"routing_probs": torch.softmax(torch.randn(32, 60), dim=-1)},
        embedding={
            "raw_embedding": raw_embedding,
            "spectral_embedding": raw_embedding[:, :1],
            "rmt_pairs": tuple((idx, idx + 1) for idx in range(0, 60, 2)),
        },
    )
    assert len(plans) == 30
    assert len({plan.candidate_id for plan in plans}) == 30
    assert sum(plan.method == CandidateMethod.USAGE_GUIDED_MERGE for plan in plans) == 3
    assert sum(plan.method == CandidateMethod.RMT_GEOMETRY for plan in plans) == 12
    assert sum(plan.method == CandidateMethod.RMT_FUNCTIONAL_USAGE for plan in plans) == 12


def test_hybrid_pairing_uses_routing_load_as_a_merge_safety_term() -> None:
    routing_probs = torch.tensor(
        [
            [0.45, 0.45, 0.05, 0.05],
            [0.45, 0.45, 0.05, 0.05],
        ]
    )
    plans = build_candidate_plans(
        num_experts=4,
        layer=0,
        methods=("rmt_functional_usage",),
        compression_fractions=(0.25,),
        seeds=(7,),
        rmt_pairs=((0, 1), (2, 3)),
        spectral_embedding=torch.tensor([[0.0], [0.1], [10.0], [10.1]]),
        functional_embedding=torch.ones((4, 2)),
        routing_probs=routing_probs,
        hybrid_weights=(0.1, 0.1, 0.8),
    )
    assert plans[0].groups[0] == (2, 3)
    assert plans[0].method == CandidateMethod.RMT_FUNCTIONAL_USAGE
    diagnostics = dict(plans[0].diagnostics)
    assert set(diagnostics) == {
        "mean_hybrid_score",
        "mean_rmt_similarity",
        "mean_functional_similarity",
        "mean_usage_safety",
    }


def test_reversible_transform_restores_layer() -> None:
    ctx = _runtime()
    original_gate = ctx.handles_by_layer[0].gate
    original_experts = ctx.handles_by_layer[0].experts
    plan = CandidatePlanContract(
        candidate_id="merge",
        layer=0,
        method=CandidateMethod.RMT_GEOMETRY,
        action=CandidateAction.MERGE,
        compression_fraction=0.25,
        groups=((0, 1), (2,), (3,)),
        seed=1,
    )
    with reversible_one_layer_transform(
        ctx,
        plan,
        merge_method="baseline",
        geometry_rank=2,
        shrink_tau=0.0,
    ) as info:
        assert len(ctx.handles_by_layer[0].experts) == 3
        assert ctx.model.moe.gate.out_features == 3
        assert info.active_experts == 3
    assert ctx.handles_by_layer[0].gate is original_gate
    assert ctx.handles_by_layer[0].experts is original_experts
    assert ctx.model.moe.num_experts == 4


def test_reversible_transform_never_replaces_shared_expert() -> None:
    ctx = _runtime()
    ctx.model.moe.shared_expert = nn.Linear(6, 6)
    ctx.model.moe.shared_expert_gate = nn.Linear(6, 1)
    ctx.runtime_contracts_by_layer[0] = replace(
        ctx.runtime_contracts_by_layer[0],
        shared_experts=4,
    )
    shared_expert = ctx.model.moe.shared_expert
    shared_expert_gate = ctx.model.moe.shared_expert_gate
    plan = CandidatePlanContract(
        candidate_id="shared-isolation",
        layer=0,
        method=CandidateMethod.RMT_GEOMETRY,
        action=CandidateAction.MERGE,
        compression_fraction=0.25,
        groups=((0, 1), (2,), (3,)),
        seed=1,
    )
    with reversible_one_layer_transform(
        ctx,
        plan,
        merge_method="baseline",
        geometry_rank=2,
        shrink_tau=0.0,
    ) as info:
        assert ctx.model.moe.shared_expert is shared_expert
        assert ctx.model.moe.shared_expert_gate is shared_expert_gate
        assert info.shared_experts_preserved == 4
    assert ctx.model.moe.shared_expert is shared_expert
    assert ctx.model.moe.shared_expert_gate is shared_expert_gate


def test_grouped_gate_preserves_group_probability_mass() -> None:
    gate = nn.Linear(3, 4, bias=True)
    grouped = GroupedLinearGate(gate, ((0, 2), (1,), (3,)))
    hidden = torch.randn(5, 3, generator=torch.Generator().manual_seed(8))
    source_probs = torch.softmax(gate(hidden), dim=-1)
    expected = torch.stack(
        (source_probs[:, [0, 2]].sum(dim=1), source_probs[:, 1], source_probs[:, 3]),
        dim=1,
    )
    assert torch.allclose(torch.softmax(grouped(hidden), dim=-1), expected, atol=1e-6)


def test_evaluator_and_quality_budget_comparison() -> None:
    ctx = _runtime()
    input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    batches = [{"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids}]
    metrics = evaluate_language_model(ctx.model, batches, warmup_steps=0, measure_steps=1)
    comparison = compare_evaluations(
        candidate_id="identity",
        baseline=metrics,
        candidate=metrics,
        quality_budgets=(0.01, 0.03),
    )
    assert metrics.evaluated_tokens == 6
    assert comparison.relative_perplexity_degradation == 0.0
    assert all(comparison.accepted_quality_budgets.values())


def test_one_layer_pilot_writes_incremental_records_and_restores_model(tmp_path: Path) -> None:
    config = load_experiment_config("configs/experiments/h123_refactor_baseline.json")
    trace_root = tmp_path / "traces"
    artifact_root = tmp_path / "artifacts"
    h1 = replace(
        config.h1,
        enabled=True,
        layer=0,
        output_diagnostics_path=str(artifact_root / "h1_diag_0.json"),
        geometry_rank=2,
    )
    calibration = replace(config.calibration, source="artifact", output_dir=str(trace_root))
    pilot = replace(
        config.pilot,
        enabled=True,
        layer=0,
        methods=("rmt_geometry",),
        compression_fractions=(0.25,),
        quality_budgets=(1e9,),
        seeds=(7,),
        max_eval_batches=3,
        validation_selection=replace(
            config.pilot.validation_selection,
            slice_count=3,
            std_penalty=1.0,
            max_slice_degradation=1e9,
        ),
    )
    profile = replace(config.profile, warmup_steps=0, measure_steps=3)
    artifacts = replace(
        config.artifacts,
        root_dir=str(artifact_root),
        manifest_path=str(artifact_root / "manifest.json"),
    )
    config = replace(
        config,
        layers=[0],
        h1=h1,
        calibration=calibration,
        pilot=pilot,
        profile=profile,
        artifacts=artifacts,
    )

    (trace_root / "calibration").mkdir(parents=True)
    routing_probs = torch.softmax(torch.randn(12, 4), dim=-1)
    torch.save(
        {
            "metadata": {
                "source": "artifact",
                "layer": 0,
                "split": "calibration",
                "routed_experts": 4,
                "shared_experts": 0,
                "routed_top_k": 2,
            },
            "routing_probs": routing_probs,
            "selected_experts": torch.topk(routing_probs, k=2, dim=-1).indices,
        },
        trace_root / "calibration" / "layer_0.pt",
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids}
    batches = [batch, batch, batch]
    for split in ("validation", "test"):
        split_dir = trace_root / split
        split_dir.mkdir(parents=True)
        torch.save(batches, split_dir / "token_batches.pt")
    artifact_root.mkdir(parents=True)
    torch.save(
        {
            "raw_embedding": torch.randn(4, 8),
            "spectral_embedding": torch.randn(4, 2),
            "rmt_pairs": [(0, 1), (2, 3)],
        },
        Path(h1.output_diagnostics_path).with_suffix(".pt"),
    )

    ctx = _runtime()
    original_gate = ctx.model.moe.gate
    summary = run_one_layer_pilot(config, ctx)
    assert summary["candidate_count"] == 1
    assert summary["selected_candidate_count"] == 1
    assert summary["validation_selection"]["slice_count"] == 3
    assert len(summary["baseline_validation_slices"]) == 3
    assert len(summary["selected_validation_results"]) == 1
    assert summary["selected_validation_results"][0]["validation_robustness"]["slice_count"] == 3
    assert ctx.model.moe.gate is original_gate
    records = (artifact_root / "pilot_runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 4
