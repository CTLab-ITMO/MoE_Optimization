from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

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
        "methods": ["random_merge", "usage_prune", "functional_cosine", "rmt_geometry"],
        "compression_fractions": [0.25],
        "seeds": [7, 11],
        "rmt_pairs": [(0, 1), (2, 3), (4, 5)],
        "functional_embedding": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "routing_probs": torch.softmax(torch.randn(10, 6, generator=torch.Generator().manual_seed(3)), dim=-1),
    }
    first = build_candidate_plans(**kwargs)
    second = build_candidate_plans(**kwargs)
    assert first == second
    assert len(first) == 5
    for plan in first:
        validate_candidate_plan(plan, num_experts=6)


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
        max_eval_batches=1,
    )
    profile = replace(config.profile, warmup_steps=0, measure_steps=1)
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
    batches = [{"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids}]
    for split in ("validation", "test"):
        split_dir = trace_root / split
        split_dir.mkdir(parents=True)
        torch.save(batches, split_dir / "token_batches.pt")
    artifact_root.mkdir(parents=True)
    torch.save(
        {
            "raw_embedding": torch.randn(4, 8),
            "rmt_pairs": [(0, 1), (2, 3)],
        },
        Path(h1.output_diagnostics_path).with_suffix(".pt"),
    )

    ctx = _runtime()
    original_gate = ctx.model.moe.gate
    summary = run_one_layer_pilot(config, ctx)
    assert summary["candidate_count"] == 1
    assert summary["selected_candidate_count"] == 1
    assert ctx.model.moe.gate is original_gate
    records = (artifact_root / "pilot_runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 4
