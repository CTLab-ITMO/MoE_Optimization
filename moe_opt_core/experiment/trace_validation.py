from __future__ import annotations

from typing import Mapping

import torch

from moe_opt_core.experiment.contracts import MoeLayerRuntimeContract, TracePayloadMetadata, TraceSource
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode


def validate_trace_payload(
    payload: Mapping[str, object],
    runtime: MoeLayerRuntimeContract,
    *,
    require_real_traces: bool,
) -> TracePayloadMetadata:
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.REAL_TRACES_REQUIRED,
            "Trace payload has no architecture metadata",
            context={"layer": runtime.layer},
        )

    try:
        metadata = TracePayloadMetadata(
            source=TraceSource(str(raw_metadata["source"])),
            layer=int(raw_metadata["layer"]),
            split=str(raw_metadata["split"]),
            routed_experts=int(raw_metadata["routed_experts"]),
            shared_experts=int(raw_metadata["shared_experts"]),
            routed_top_k=int(raw_metadata["routed_top_k"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.REAL_TRACES_REQUIRED,
            "Trace payload contains invalid architecture metadata",
            context={"layer": runtime.layer},
        ) from exc

    if require_real_traces and metadata.source == TraceSource.SYNTHETIC:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.REAL_TRACES_REQUIRED,
            "The one-layer pilot requires non-synthetic traces",
            context={"layer": runtime.layer},
        )

    expected = {
        "layer": runtime.layer,
        "routed_experts": runtime.routed_experts,
        "shared_experts": runtime.shared_experts,
        "routed_top_k": runtime.effective_routed_top_k,
    }
    observed = {
        "layer": metadata.layer,
        "routed_experts": metadata.routed_experts,
        "shared_experts": metadata.shared_experts,
        "routed_top_k": metadata.routed_top_k,
    }
    if observed != expected:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            "Trace architecture does not match the loaded MoE layer",
            context={"expected": expected, "observed": observed},
        )

    probabilities = payload.get("routing_probs")
    selected = payload.get("selected_experts")
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.REAL_TRACES_REQUIRED,
            "Trace routing_probs must be a rank-2 tensor",
        )
    if probabilities.shape[1] != runtime.routed_experts:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            "Trace probability width does not match routed experts",
        )
    if not isinstance(selected, torch.Tensor) or selected.ndim != 2:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.REAL_TRACES_REQUIRED,
            "Trace selected_experts must be a rank-2 tensor",
        )
    if selected.shape != (probabilities.shape[0], runtime.effective_routed_top_k):
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            "Trace selected_experts has an invalid shape",
        )
    if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= runtime.routed_experts):
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            "Trace selected_experts contains a shared or unknown expert index",
        )
    if selected.shape[1] > 1:
        sorted_selected = torch.sort(selected, dim=1).values
        if torch.any(sorted_selected[:, 1:] == sorted_selected[:, :-1]):
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.REAL_TRACES_REQUIRED,
                "Trace selected_experts contains duplicate routed experts",
            )
    row_sums = probabilities.float().sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.REAL_TRACES_REQUIRED,
            "Trace routing probabilities are not normalized",
        )
    return metadata
