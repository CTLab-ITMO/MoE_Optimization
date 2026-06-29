"""Router-output canonicalizer shared by co-activation capture and inference.

A MoE gate's output is decoded in three places that historically each had their
own ``if isinstance(out, tuple)`` branch:

* **Co-activation capture** (bare-logits branch)::

      routing_weights = router_logits.softmax(dim=1, dtype=torch.float)
      _, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)

  Only ``selected_experts`` is then used (to count co-activations).

* **Top-k selection capture** (bare-logits branch)::

      _, selected = router_output.topk(TOP_K, dim=-1)   # topk on RAW logits

  Note: no softmax. ``softmax`` is strictly monotonic, so ``topk`` over the
  softmax and ``topk`` over the raw logits select the *same* experts for the
  same logits -- the two paths agree by construction.

* **Inference / fused layer** (``common.fusion_layer``, 3-tuple branch)::

      _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

  i.e. the gate returns ``(logits, routing_weights, selected_experts)`` and both
  the weights and the selection are used downstream.

:class:`RouterAdapter` folds all three into one ``route`` contract returning a
:class:`RouterDecision`. :class:`Qwen15RouterAdapter` is the in-tree
implementation; its default behaviour reproduces every path above exactly:

* a **tuple** gate output is unpacked as ``(logits, routing_weights,
  selected_experts) = (out[0], out[1], out[-1])`` -- both the inference path
  (uses all three) and the capture paths (consume only ``out[-1]``);
* a **bare-logits** tensor is softmaxed and top-k'd. ``softmax(dim=-1)`` on a 2D
  ``[tokens, experts]`` tensor is identical to ``softmax(dim=1)`` (for 2D
  ``dim=1`` *is* the last dim), and by the monotonicity argument above the
  selected experts also match a raw-logit ``topk``.

Qwen1.5-MoE has ``norm_topk_prob=False``, so the top-k weights are *not*
renormalized by default (``normalize_topk=False``); the flag is exposed for
models/configs that do renormalize.

The variadic ``**kwargs`` lives only on the *abstract* signature; concrete
adapters take explicit, named parameters so a typo is caught rather than
silently swallowed.

Library code only -- this module never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RouterDecision:
    """Canonical decoded output of a MoE router for one batch of tokens.

    * ``selected_experts`` -- the top-k expert indices (the only field the
      capture paths consume).
    * ``routing_weights`` -- the per-selected-expert gate weights the fused layer
      mixes the expert outputs with. For a bare-logits gate these are the top-k slice
      of ``softmax(logits)`` (optionally renormalized); for a tuple gate they are
      the gate's own ``out[1]``.
    * ``logits`` -- the raw router logits when available (``out[0]`` for a tuple
      gate, the input tensor for a bare-logits gate).
    """

    selected_experts: torch.Tensor
    routing_weights: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class RouterAdapter(ABC):
    """Abstract canonicalizer turning a gate's output into a :class:`RouterDecision`.

    The frozen public contract is :meth:`route`. A new router layout (a different
    tuple arity, a sparse gate, ...) is a new subclass, never a change to this
    base. ``**kwargs`` is the forward-compatible interface slot only; concrete
    adapters take explicit named parameters.
    """

    @abstractmethod
    def route(self, gate_output, *, top_k: int, **kwargs) -> RouterDecision:
        """Decode ``gate_output`` (tuple or bare logits) into a :class:`RouterDecision`."""
        raise NotImplementedError()


class Qwen15RouterAdapter(RouterAdapter):
    """Router canonicalizer for Qwen1.5-MoE (and Qwen/Mixtral-shaped gates).

    Reproduces the three inline call-sites exactly; see the module
    docstring. ``normalize_topk`` mirrors ``norm_topk_prob`` -- Qwen1.5-MoE sets
    it to ``False``, hence the default.
    """

    def __init__(self, normalize_topk: bool = False):
        self.normalize_topk = normalize_topk

    def route(self, gate_output, *, top_k: int) -> RouterDecision:
        if isinstance(gate_output, tuple):
            # Inference layout (logits, routing_weights, selected_experts); the
            # capture tuple branches consume only out[-1].
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
        # Bare-logits path. softmax(dim=-1) == softmax(dim=1) for a 2D
        # [tokens, experts] tensor; topk(softmax(logits)) selects the same experts
        # as topk(raw logits) because softmax is monotonic.
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
