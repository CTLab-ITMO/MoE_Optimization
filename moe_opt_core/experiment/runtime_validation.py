from __future__ import annotations

from typing import Any

from moe_opt_core.experiment.contracts import MoeLayerRuntimeContract
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode


def _positive_int_attribute(source: Any, names: tuple[str, ...]) -> int | None:
    if source is None:
        return None
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    return None


def infer_configured_routed_expert_count(model_config: Any | None) -> int | None:
    return _positive_int_attribute(
        model_config,
        ("num_experts", "num_local_experts", "n_routed_experts"),
    )


def infer_configured_shared_expert_count(model_config: Any | None) -> int | None:
    explicit = _positive_int_attribute(
        model_config,
        ("num_shared_experts", "shared_expert_count", "n_shared_experts"),
    )
    if explicit is not None:
        return explicit
    shared_width = _positive_int_attribute(model_config, ("shared_expert_intermediate_size",))
    routed_width = _positive_int_attribute(model_config, ("moe_intermediate_size",))
    if shared_width is None or routed_width is None:
        return None
    ratio = shared_width / routed_width
    rounded = int(round(ratio))
    if rounded > 0 and abs(ratio - rounded) < 1e-6:
        return rounded
    return None


def infer_configured_routed_top_k(model_config: Any | None) -> int | None:
    return _positive_int_attribute(
        model_config,
        ("num_experts_per_tok", "num_experts_per_token", "top_k"),
    )


def infer_routed_expert_count(moe: Any, model_config: Any | None = None) -> int:
    experts = getattr(moe, "experts", None)
    if experts is not None:
        try:
            count = len(experts)
        except TypeError:
            count = 0
        if count > 0:
            return int(count)

    configured = infer_configured_routed_expert_count(model_config)
    if configured is None:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            "Could not determine the routed expert count",
        )
    return configured


def infer_shared_expert_count(moe: Any, model_config: Any | None = None) -> int:
    configured = infer_configured_shared_expert_count(model_config)
    if configured is not None:
        return configured
    if getattr(moe, "shared_expert", None) is None:
        return 0
    return 1


def build_runtime_contract(
    *,
    layer: int,
    kind: str,
    routed_experts: int,
    shared_experts: int,
    configured_routed_top_k: int,
    requested_routed_top_k: int,
    effective_routed_top_k: int,
    top_k_updated: bool,
    expected_routed_experts: int | None,
    expected_shared_experts: int | None,
    strict: bool,
) -> MoeLayerRuntimeContract:
    violations: dict[str, object] = {}
    if requested_routed_top_k > routed_experts:
        violations["requested_routed_top_k"] = requested_routed_top_k
    if effective_routed_top_k != requested_routed_top_k:
        violations["effective_routed_top_k"] = effective_routed_top_k
    if expected_routed_experts is not None and routed_experts != expected_routed_experts:
        violations["routed_experts"] = routed_experts
    if expected_shared_experts is not None and shared_experts != expected_shared_experts:
        violations["shared_experts"] = shared_experts

    if strict and violations:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
            f"MoE architecture mismatch at layer {layer}",
            context={
                "layer": layer,
                "kind": kind,
                "violations": violations,
                "expected_routed_experts": expected_routed_experts,
                "expected_shared_experts": expected_shared_experts,
                "requested_routed_top_k": requested_routed_top_k,
            },
        )

    return MoeLayerRuntimeContract(
        layer=int(layer),
        kind=str(kind),
        routed_experts=int(routed_experts),
        shared_experts=int(shared_experts),
        configured_routed_top_k=int(configured_routed_top_k),
        requested_routed_top_k=int(requested_routed_top_k),
        effective_routed_top_k=int(effective_routed_top_k),
        top_k_updated=bool(top_k_updated),
    )
