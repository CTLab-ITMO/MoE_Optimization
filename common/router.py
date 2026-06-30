"""Canonical MoE router interface shared by co-activation capture and inference.

A gate's output appears in two forms across the codebase:
- **tuple** ``(logits, routing_weights, selected_experts)`` — inference/fusion layer.
- **bare-logits tensor** — capture hooks; softmaxed and top-k'd here.

:class:`RouterAdapter` decodes either form into a :class:`RouterDecision`.
:class:`Qwen15RouterAdapter` is the in-tree implementation:
- tuple: unpacked as ``(out[0], out[1], out[-1])``.
- tensor: ``softmax(logits)`` → ``topk`` (softmax is monotonic, so this selects
  the same experts as topk on raw logits).

Qwen1.5-MoE uses ``norm_topk_prob=False``, so ``normalize_topk`` defaults to ``False``.

Library code only — never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RouterDecision:
    """Decoded output of a MoE router for one token batch.

    - ``selected_experts``: top-k expert indices (used by all call-sites).
    - ``routing_weights``: gate weights for mixing expert outputs — top-k softmax
      slice (optionally renormalized) for bare-logits gates, or ``out[1]`` for tuple gates.
    - ``logits``: raw router logits when available.
    """

    selected_experts: torch.Tensor
    routing_weights: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class RouterAdapter(ABC):
    """Abstract base: turns a gate's raw output into a :class:`RouterDecision`.

    Subclass to support a new gate layout (different tuple arity, sparse gate, etc.).
    """

    @abstractmethod
    def route(self, gate_output, *, top_k: int) -> RouterDecision:
        """Decode ``gate_output`` (tuple or bare logits) into a :class:`RouterDecision`."""
        raise NotImplementedError()


class Qwen15RouterAdapter(RouterAdapter):
    """Router canonicalizer for Qwen1.5-MoE (and Qwen/Mixtral-shaped gates).

    Handles both tuple gates (inference) and bare-logits tensors (capture).
    ``normalize_topk`` mirrors ``norm_topk_prob``; Qwen1.5-MoE sets it ``False``.
    """

    def __init__(self, normalize_topk: bool = False):
        self.normalize_topk = normalize_topk

    def route(self, gate_output, *, top_k: int) -> RouterDecision:
        if isinstance(gate_output, tuple):
            # Tuple gate: (logits, routing_weights, selected_experts); capture paths use only out[-1].
            return RouterDecision(
                selected_experts=gate_output[-1],
                routing_weights=gate_output[1],
                logits=gate_output[0],
            )
        if not isinstance(gate_output, torch.Tensor):
            raise TypeError(
                "Qwen15RouterAdapter.route expects a tuple gate output or a bare "
                f"logits Tensor, got {type(gate_output).__name__}"
            )
        # Bare-logits path: softmax then topk. Softmax is monotonic so topk(softmax(logits))
        # selects the same experts as topk(raw logits).
        logits = gate_output
        routing_weights_full = logits.softmax(dim=-1, dtype=torch.float)
        top_weights, selected = torch.topk(routing_weights_full, top_k, dim=-1)
        if self.normalize_topk:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return RouterDecision(
            selected_experts=selected,
            routing_weights=top_weights,
            logits=logits,
        )
