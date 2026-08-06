from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from common.moe_utils import get_moe_top_k
from moe_opt_core.experiment import ExpectedExperimentFailure, ExperimentFailureCode
from moe_opt_core.experiment.runtime_validation import (
    build_runtime_contract,
    infer_routed_expert_count,
    infer_shared_expert_count,
)
from moe_opt_core.experiment.trace_validation import validate_trace_payload


class TinyQwenMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(2, 2) for _ in range(60)])
        self.shared_expert = nn.Linear(2, 2)
        self.top_k = 4


def _runtime_contract():
    return build_runtime_contract(
        layer=0,
        kind="qwen",
        routed_experts=60,
        shared_experts=4,
        configured_routed_top_k=4,
        requested_routed_top_k=4,
        effective_routed_top_k=4,
        top_k_updated=False,
        expected_routed_experts=60,
        expected_shared_experts=4,
        strict=True,
    )


def test_qwen_architecture_inference_separates_routed_and_shared_experts() -> None:
    moe = TinyQwenMoe()
    config = SimpleNamespace(moe_intermediate_size=8, shared_expert_intermediate_size=32)
    assert infer_routed_expert_count(moe, config) == 60
    assert infer_shared_expert_count(moe, config) == 4
    assert get_moe_top_k(SimpleNamespace(), kind="qwen") == 4


def test_strict_runtime_contract_rejects_wrong_expert_count() -> None:
    with pytest.raises(ExpectedExperimentFailure) as failure:
        build_runtime_contract(
            layer=0,
            kind="qwen",
            routed_experts=59,
            shared_experts=4,
            configured_routed_top_k=4,
            requested_routed_top_k=4,
            effective_routed_top_k=4,
            top_k_updated=False,
            expected_routed_experts=60,
            expected_shared_experts=4,
            strict=True,
        )
    assert failure.value.code == ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH


def test_trace_validation_rejects_shared_expert_indices() -> None:
    probabilities = torch.softmax(torch.randn(3, 60), dim=-1)
    payload = {
        "metadata": {
            "source": "hf_dataset",
            "layer": 0,
            "split": "calibration",
            "routed_experts": 60,
            "shared_experts": 4,
            "routed_top_k": 4,
        },
        "routing_probs": probabilities,
        "selected_experts": torch.tensor([[0, 1, 2, 60], [1, 2, 3, 4], [2, 3, 4, 5]]),
    }
    with pytest.raises(ExpectedExperimentFailure) as failure:
        validate_trace_payload(payload, _runtime_contract(), require_real_traces=True)
    assert failure.value.code == ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH


def test_trace_validation_accepts_routed_top4_probabilities() -> None:
    probabilities = torch.softmax(torch.randn(3, 60), dim=-1)
    selected = torch.topk(probabilities, k=4, dim=-1).indices
    payload = {
        "metadata": {
            "source": "hf_dataset",
            "layer": 0,
            "split": "calibration",
            "routed_experts": 60,
            "shared_experts": 4,
            "routed_top_k": 4,
        },
        "routing_probs": probabilities,
        "selected_experts": selected,
    }
    metadata = validate_trace_payload(payload, _runtime_contract(), require_real_traces=True)
    assert metadata.routed_top_k == 4
