"""Utilities for working with different MoE implementations (Mixtral-like, Qwen-like).
It provides:
  - best-effort discovery of MoE modules inside a HF CausalLM model
  - access to router/gate weights
  - spike-projection denoising for expert-importance vectors (RMT-motivated)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import torch
import torch.nn as nn


@dataclass
class MoeHandle:
    """A thin handle pointing to a MoE block inside a transformer layer."""
    layer_idx: int
    moe: nn.Module
    gate: nn.Module
    experts: nn.Module
    kind: str  # e.g. "qwen", "mixtral", "unknown"


def _is_linear(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear)


def _get_first_attr(obj: Any, names: Iterable[str]) -> Optional[Any]:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def discover_moe_layers(model: nn.Module) -> List[MoeHandle]:
    """Discovery of MoE blocks in common HF architectures.

    Supported patterns:
      - Qwen1.5-MoE / Qwen3-MoE: layer.mlp has attributes:
            gate, experts (ModuleList), shared_expert, shared_expert_gate (optional)
      - Mixtral: layer.block_sparse_moe with attributes:
            gate, experts (ModuleList), num_experts_per_tok (optional)

    Returns a list of MoeHandle.
    """
    handles: List[MoeHandle] = []

    layers = _get_first_attr(
        model,
        [
            "model.layers",            # Qwen-style (model.model.layers in some)
            "model.model.layers",      # nested
            "transformer.h",           # GPT-like
            "layers",                  # fallback
        ],
    )

    # If nested path, resolve manually
    if layers is None:
        # common HF: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h

    if layers is None:
        raise RuntimeError("Could not locate transformer layers for MoE discovery.")

    for i, layer in enumerate(layers):
        # Qwen-like: layer.mlp is MoE
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate"):
            handles.append(
                MoeHandle(
                    layer_idx=i,
                    moe=layer.mlp,
                    gate=layer.mlp.gate,
                    experts=layer.mlp.experts,
                    kind="qwen",
                )
            )
            continue

        # Mixtral-like: layer.block_sparse_moe is MoE
        if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts") and hasattr(layer.block_sparse_moe, "gate"):
            handles.append(
                MoeHandle(
                    layer_idx=i,
                    moe=layer.block_sparse_moe,
                    gate=layer.block_sparse_moe.gate,
                    experts=layer.block_sparse_moe.experts,
                    kind="mixtral",
                )
            )
            continue

        # fallback: search shallow for modules with gate+experts
        for child_name, child in layer.named_children():
            if hasattr(child, "experts") and hasattr(child, "gate"):
                handles.append(MoeHandle(i, child, child.gate, child.experts, kind=f"unknown:{child_name}"))

    return handles


def get_gate_weight(gate: nn.Module) -> torch.Tensor:
    """Return router/gate weight as a tensor of shape [num_experts, hidden_dim]."""
    if _is_linear(gate):
        return gate.weight.detach()
    # some routers wrap a linear inside
    inner = _get_first_attr(gate, ["weight", "linear", "proj", "router"])
    if isinstance(inner, torch.Tensor):
        return inner.detach()
    if isinstance(inner, nn.Module) and _is_linear(inner):
        return inner.weight.detach()
    raise RuntimeError(f"Unsupported gate module type: {type(gate)}")



def get_moe_top_k(moe: nn.Module, model_config: Any | None = None, kind: str | None = None, default: int | None = None) -> int:
    """Best-effort extraction of the *runtime* top-k used by a MoE block.

    IMPORTANT:
      - Do NOT confuse with num_experts.
      - Some HF MoE blocks expose `num_experts_per_tok` (Mixtral), others `top_k` (Qwen3),
        or store it only in model config / init kwargs.

    Returns an int >= 1.
    """
    candidates = [
        'num_experts_per_tok',
        'num_experts_per_token',
        'top_k',
        'k',
    ]
    for name in candidates:
        if hasattr(moe, name):
            v = getattr(moe, name)
            if isinstance(v, int):
                return max(1, v)
            try:
                import torch
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    return max(1, int(v.item()))
            except Exception:
                pass

    if model_config is not None:
        for name in ['num_experts_per_tok', 'top_k']:
            if hasattr(model_config, name):
                v = getattr(model_config, name)
                if isinstance(v, int):
                    return max(1, v)

    # Heuristic defaults
    if default is not None:
        return max(1, int(default))
    if kind == 'qwen':
        return 8
    if kind == 'mixtral':
        return 2
    return 2