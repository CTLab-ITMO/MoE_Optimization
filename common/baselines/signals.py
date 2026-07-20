"""Offline calibration signals for the REAP / HC-SMoE baselines.

Standalone copies of the per-expert saliency / characteristic-activation /
selection-frequency computers (and the private stage-1 loaders they need),
verbatim from ``common.pairing`` so the baselines own their signal code. All
three are computed on CPU from stage-1 artifacts alone; stage 1 is never touched.
"""
import os
from typing import Optional

import torch

from common import layout


def _load_expert(path: str, device: torch.device):
    """Load an expert as an ``AdvancedMLP``, tolerating both on-disk formats.

    Tries versioned ``SavedExpert`` (safetensors) first, then falls back to a raw
    pickle. Mirrors ``_load_coactivation_matrix``.
    """
    from common.artifacts import SavedExpert

    try:
        return SavedExpert.load(path, device=device).mlp
    except Exception:
        return torch.load(path, map_location=device)


def _load_gate(path: str, device: torch.device):
    """Load a layer's saved router gate submodule (pickled ``nn.Linear``)."""
    return torch.load(path, weights_only=False).to(device)


def _gate_logits(gate, x: torch.Tensor) -> torch.Tensor:
    """Return the full expert-logit vector from a gate module.

    A bare gate submodule returns raw logits; some families wrap them in the
    ``(logits, weights, selected)`` tuple (RouterAdapter convention) -- take
    element 0 in that case. Mirrors :class:`common.router.Qwen15RouterAdapter`.
    """
    out = gate(x)
    return out[0] if isinstance(out, tuple) else out


def _load_mlp_input_rows(path: str) -> torch.Tensor:
    """Load captured MLP inputs (list of rows, 3-D, or 2-D) as a flat ``[N, H]`` tensor."""
    raw = torch.load(path)
    inputs = torch.stack(raw, dim=0) if isinstance(raw, list) else raw
    if inputs.dim() == 3:
        inputs = inputs.reshape(-1, inputs.shape[-1])
    return inputs


def _load_topk_selections(path: str) -> torch.Tensor:
    """Load per-token top-k selections (list of ``[k]`` tensors) as a ``[N, k]`` long tensor.

    Token order matches :func:`_load_mlp_input_rows` -- stage 1 appends one row and
    one selection per token in the same hook, so index ``t`` refers to the same token.
    """
    raw = torch.load(path)
    selections = torch.stack([sel.flatten() for sel in raw], dim=0) if isinstance(raw, list) else raw
    if selections.dim() == 3:
        selections = selections.reshape(-1, selections.shape[-1])
    return selections.long()


def _subsample_indices(
    n: int, num_samples: Optional[int], generator: Optional[torch.Generator] = None
) -> Optional[torch.Tensor]:
    """Return a random ``num_samples``-subset of ``range(n)``, or ``None`` if uncapped.

    ``None`` means "use every token" (either ``num_samples is None`` or ``n <= num_samples``).
    Callers that subsample several token-aligned tensors must apply the same index to all.
    """
    if num_samples is None or n <= num_samples:
        return None
    return torch.randperm(n, generator=generator)[:num_samples]


def _run_expert_batched(expert, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Run ``expert`` over ``inputs`` in batches; return the concatenated ``[N, H]`` float outputs."""
    outs = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            outs.append(expert(inputs[start : start + batch_size]).float())
    return torch.cat(outs, dim=0)


def _require_file(path: str, method: str) -> None:
    """Raise a clear :class:`FileNotFoundError` naming the calibration signal that needs ``path``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{method} requires the stage-1 artifact {path!r}")


def compute_reap_saliency(
    layer: int,
    data_dir: str,
    n_experts: int,
    device: torch.device,
    num_samples: int = 4096,
    batch_size: int = 512,
    generator: Optional[torch.Generator] = None,
    split: str = layout.TRAIN_SPLIT,
) -> torch.Tensor:
    """REAP router-weighted expert-activation saliency, one score per expert.

    ``S_e`` is the *mean over the tokens routed to* ``e`` (``e`` in the token's
    top-k) of ``softmax(gate(x_t))[e] * ||expert_e(x_t)||_2`` -- the mean of
    per-token products, not the product of means. The routing mask comes from the
    saved top-k selections (family-agnostic; no re-decode); ``gate_value`` is the
    full softmax over all experts (for Qwen1.5 ``norm_topk_prob=False``, so the
    top-k slice is not renormalized -- matching the reference on this model).
    A never-routed expert scores 0 (dropped first by the pruning grouping).

    Aggregation is in float64; the ``[E]`` result is float32.
    """
    inputs_path = layout.mlp_inputs_path(data_dir, split, layer)
    topk_path = layout.activations_all_path(data_dir, split, layer)
    gate_path = layout.gate_path(data_dir, layer)
    _require_file(inputs_path, "REAP saliency")
    _require_file(topk_path, "REAP saliency")
    _require_file(gate_path, "REAP saliency")

    inputs = _load_mlp_input_rows(inputs_path)
    selections = _load_topk_selections(topk_path)
    if selections.shape[0] != inputs.shape[0]:
        raise ValueError(
            f"REAP saliency: {selections.shape[0]} top-k selections but "
            f"{inputs.shape[0]} MLP-input rows for layer {layer}; token alignment broken"
        )
    idx = _subsample_indices(inputs.shape[0], num_samples, generator)
    if idx is not None:
        inputs = inputs[idx]
        selections = selections[idx]
    inputs = inputs.to(device)

    # [N, E] boolean: token t is routed to expert e iff e is in its top-k.
    routed = torch.nn.functional.one_hot(selections, num_classes=n_experts).sum(dim=1) > 0
    gate = _load_gate(gate_path, device)
    with torch.no_grad():
        gate_probs = _gate_logits(gate, inputs).softmax(dim=-1, dtype=torch.float).cpu()

    saliency = torch.zeros(n_experts, dtype=torch.float64)
    for e in range(n_experts):
        expert = _load_expert(layout.expert_path(data_dir, layer, e), device)
        expert.eval()
        norms = _run_expert_batched(expert, inputs, batch_size).norm(dim=-1).cpu().double()
        mask_e = routed[:, e]
        count = int(mask_e.sum())
        if count:
            products = norms * gate_probs[:, e].double()
            saliency[e] = products[mask_e].sum() / count
        del expert
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return saliency.to(torch.float32)


def compute_characteristic_activation(
    layer: int,
    data_dir: str,
    n_experts: int,
    device: torch.device,
    num_samples: int = 4096,
    batch_size: int = 512,
    generator: Optional[torch.Generator] = None,
    split: str = layout.TRAIN_SPLIT,
) -> torch.Tensor:
    """HC-SMoE characteristic activation: the mean *dense* output vector per expert.

    Every expert is run on *every* calibration token regardless of routing (not
    co-activation, not routed-only); the per-expert mean over tokens is the
    clustering feature. Aggregation is in float64; the ``[E, H]`` result is float32.
    """
    inputs_path = layout.mlp_inputs_path(data_dir, split, layer)
    _require_file(inputs_path, "HC-SMoE characteristic activation")

    inputs = _load_mlp_input_rows(inputs_path)
    idx = _subsample_indices(inputs.shape[0], num_samples, generator)
    if idx is not None:
        inputs = inputs[idx]
    inputs = inputs.to(device)

    means: list[torch.Tensor] = []
    for e in range(n_experts):
        expert = _load_expert(layout.expert_path(data_dir, layer, e), device)
        expert.eval()
        means.append(_run_expert_batched(expert, inputs, batch_size).double().mean(dim=0).cpu())
        del expert
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.stack(means, dim=0).to(torch.float32)


def compute_selection_frequencies(
    layer: int,
    data_dir: str,
    n_experts: int,
    split: str = layout.TRAIN_SPLIT,
) -> torch.Tensor:
    """Per-expert top-k selection counts over the calibration set (``[E]`` long).

    ``bincount`` of the raw top-k indices. This is exactly the diagonal of the
    :class:`~common.artifacts.CoActivationMatrix` (stage 1 increments
    ``matrix[e, e]`` once per top-k membership); the identity is asserted by test.
    Read straight from disk -- never subsampled -- so it stays a faithful count.
    """
    topk_path = layout.activations_all_path(data_dir, split, layer)
    _require_file(topk_path, "selection frequencies")
    selections = _load_topk_selections(topk_path)
    counts = torch.bincount(selections.reshape(-1), minlength=n_experts)
    return counts[:n_experts].long()
