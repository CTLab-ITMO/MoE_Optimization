from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from common.moe_utils import MoeHandle
from moe_opt_core.adapters import HFModelAdapter
from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import TraceSource
from moe_opt_core.experiment.data_planning import split_text_rows
from moe_opt_runner.runtime import RuntimeContext
from moe_opt_runner.tracing import collect_traces_from_batches


class TinyMoe(nn.Module):
    def __init__(self, hidden_dim: int, experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(experts)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.gate(hidden)
        return hidden


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.moe = TinyMoe(hidden_dim=4, experts=3)

    def forward(self, input_ids, attention_mask=None):
        return self.moe(self.embedding(input_ids))


def _runtime() -> RuntimeContext:
    model = TinyModel()
    handle = MoeHandle(layer_idx=0, moe=model.moe, gate=model.moe.gate, experts=model.moe.experts, kind="tiny")
    return RuntimeContext(
        model=model,
        adapter=HFModelAdapter(model=model),
        handles_by_layer={0: handle},
    )


def test_split_text_rows_is_deterministic_and_conservative() -> None:
    rows = [f"row-{idx}" for idx in range(20)]
    first = split_text_rows(rows, fractions=(0.6, 0.2, 0.2), seed=7)
    second = split_text_rows(rows, fractions=(0.6, 0.2, 0.2), seed=7)
    assert first == second
    assert sum(len(values) for values in first.values()) == len(rows)
    assert set().union(*(set(values) for values in first.values())) == set(rows)


def test_collect_traces_from_batches_writes_real_gate_rows(tmp_path: Path) -> None:
    config = load_experiment_config("configs/experiments/h123_refactor_baseline.json")
    h1 = replace(config.h1, layer=0, mlp_inputs_path=str(tmp_path / "h1.pt"))
    h3 = replace(config.h3, routing_probs_path=str(tmp_path / "routing.pt"))
    calibration = replace(
        config.calibration,
        source=TraceSource.HF_DATASET.value,
        output_dir=str(tmp_path / "traces"),
    )
    config = replace(config, layers=[0], h1=h1, h3=h3, calibration=calibration)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "labels": torch.tensor([[1, 2, 3], [4, 5, -100]]),
    }
    contracts = collect_traces_from_batches(
        config,
        _runtime(),
        {"calibration": [batch], "validation": [batch], "test": [batch]},
    )
    assert len(contracts) == 3
    assert all(contract.rows == 5 for contract in contracts)
    trace = torch.load(tmp_path / "traces" / "calibration" / "layer_0.pt")
    assert trace["hidden_states"].shape == (5, 4)
    assert torch.allclose(trace["routing_probs"].sum(dim=1), torch.ones(5))
    assert (tmp_path / "h1.pt").exists()
    assert (tmp_path / "routing.pt").exists()
