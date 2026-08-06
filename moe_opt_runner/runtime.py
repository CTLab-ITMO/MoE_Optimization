from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from moe_opt_core.adapters import HFModelAdapter
from moe_opt_core.config import ExperimentConfig
from moe_opt_core.experiment.contracts import MoeLayerRuntimeContract
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode
from moe_opt_core.experiment.runtime_validation import (
    build_runtime_contract,
    infer_routed_expert_count,
    infer_shared_expert_count,
)


@dataclass
class RuntimeContext:
    model: object
    adapter: HFModelAdapter
    handles_by_layer: Dict[int, object]
    tokenizer: Optional[object] = None
    runtime_contracts_by_layer: Dict[int, MoeLayerRuntimeContract] = field(default_factory=dict)


def _resolve_torch_dtype(token: str):
    import torch

    t = token.lower()
    if t == "auto":
        return "auto"
    if t == "fp16":
        return torch.float16
    if t == "bf16":
        return torch.bfloat16
    if t == "fp32":
        return torch.float32
    return None


def build_runtime_context(config: ExperimentConfig) -> RuntimeContext:
    if config.model.load_source != "hf":
        raise ValueError(f"Unsupported model.load_source: {config.model.load_source}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {
        "device_map": config.model.device_map,
        "local_files_only": config.model.local_files_only,
        "trust_remote_code": config.model.trust_remote_code,
    }
    dtype = _resolve_torch_dtype(config.model.torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)
    tokenizer = None
    if config.calibration.source == "hf_dataset" or config.pilot.enabled:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
    adapter = HFModelAdapter(model=model, kind=config.model.kind)

    handles = adapter.discover_moe_layers()
    handles_by_layer = {h.layer_idx: h for h in handles}

    target_layers: Set[int] = set(config.layers)
    target_layers.add(config.h1.layer)
    target_layers.add(config.h2.layer)
    if config.pilot.enabled:
        target_layers.add(config.pilot.layer)

    runtime_contracts: Dict[int, MoeLayerRuntimeContract] = {}
    model_config = getattr(model, "config", None)
    for layer in target_layers:
        h = handles_by_layer.get(layer)
        if h is None:
            if config.runtime_strict_architecture:
                raise ExpectedExperimentFailure(
                    ExperimentFailureCode.LAYER_UNAVAILABLE,
                    f"Configured MoE layer {layer} was not discovered",
                    context={"available_layers": sorted(handles_by_layer)},
                )
            continue
        routed_experts = infer_routed_expert_count(h.moe, model_config)
        shared_experts = infer_shared_expert_count(h.moe, model_config)
        configured_top_k = adapter.get_top_k(h.moe, model_config=model_config)
        top_k_updated = False
        if configured_top_k != config.runtime_top_k:
            top_k_updated = adapter.set_top_k_if_possible(h.moe, config.runtime_top_k)
        effective_top_k = adapter.get_top_k(h.moe, model_config=model_config)
        runtime_contracts[layer] = build_runtime_contract(
            layer=layer,
            kind=h.kind,
            routed_experts=routed_experts,
            shared_experts=shared_experts,
            configured_routed_top_k=configured_top_k,
            requested_routed_top_k=config.runtime_top_k,
            effective_routed_top_k=effective_top_k,
            top_k_updated=top_k_updated,
            expected_routed_experts=config.runtime_expected_routed_experts,
            expected_shared_experts=config.runtime_expected_shared_experts,
            strict=config.runtime_strict_architecture,
        )

    return RuntimeContext(
        model=model,
        adapter=adapter,
        handles_by_layer=handles_by_layer,
        tokenizer=tokenizer,
        runtime_contracts_by_layer=runtime_contracts,
    )
