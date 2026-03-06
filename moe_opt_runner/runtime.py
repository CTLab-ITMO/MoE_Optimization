from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

from moe_opt_core.adapters import HFModelAdapter
from moe_opt_core.config import ExperimentConfig


@dataclass
class RuntimeContext:
    model: object
    adapter: HFModelAdapter
    handles_by_layer: Dict[int, object]


def _resolve_torch_dtype(token: str):
    import torch

    t = token.lower()
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

    from transformers import AutoModelForCausalLM

    kwargs = {
        "device_map": config.model.device_map,
        "local_files_only": config.model.local_files_only,
        "trust_remote_code": config.model.trust_remote_code,
    }
    dtype = _resolve_torch_dtype(config.model.torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)
    adapter = HFModelAdapter(model=model, kind=config.model.kind)

    handles = adapter.discover_moe_layers()
    handles_by_layer = {h.layer_idx: h for h in handles}

    target_layers: Set[int] = set(config.layers)
    target_layers.add(config.h1.layer)
    target_layers.add(config.h2.layer)

    for layer in target_layers:
        h = handles_by_layer.get(layer)
        if h is None:
            continue
        adapter.set_top_k_if_possible(h.moe, config.runtime_top_k)

    return RuntimeContext(model=model, adapter=adapter, handles_by_layer=handles_by_layer)
