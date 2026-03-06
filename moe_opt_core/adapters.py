from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

import torch
import torch.nn as nn

from common.moe_utils import MoeHandle, discover_moe_layers, get_gate_weight, get_moe_top_k


class ModelAdapter(Protocol):
    def discover_moe_layers(self) -> List[MoeHandle]:
        ...

    def get_top_k(self, moe: nn.Module, model_config: Any | None = None, default: int | None = None) -> int:
        ...

    def set_top_k_if_possible(self, moe: nn.Module, k: int) -> bool:
        ...

    def get_gate_weight(self, gate: nn.Module) -> torch.Tensor:
        ...

    def get_experts(self, moe: nn.Module) -> List[nn.Module]:
        ...


@dataclass
class HFModelAdapter:
    model: nn.Module
    kind: Optional[str] = None

    def discover_moe_layers(self) -> List[MoeHandle]:
        return discover_moe_layers(self.model)

    def get_top_k(self, moe: nn.Module, model_config: Any | None = None, default: int | None = None) -> int:
        return get_moe_top_k(moe, model_config=model_config, kind=self.kind, default=default)

    def set_top_k_if_possible(self, moe: nn.Module, k: int) -> bool:
        candidates = ["num_experts_per_tok", "num_experts_per_token", "top_k", "k"]
        for name in candidates:
            if hasattr(moe, name):
                try:
                    setattr(moe, name, int(k))
                    return True
                except Exception:
                    continue
        return False

    def get_gate_weight(self, gate: nn.Module) -> torch.Tensor:
        return get_gate_weight(gate)

    def get_experts(self, moe: nn.Module) -> List[nn.Module]:
        experts = getattr(moe, "experts", None)
        if experts is None:
            raise RuntimeError("MoE block has no 'experts' field")
        return list(experts)
