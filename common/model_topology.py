"""Discover which decoder layers of a causal-LM are MoE (router + experts).

Some models (DeepSeek, Qwen3-MoE, Granite-MoE) mix dense and sparse layers.
Blindly indexing ``.mlp.experts`` on a dense layer raises ``AttributeError``.
This module replaces blind ``range(N)`` with actual discovery: a layer is MoE
iff its ``mlp`` has both an ``experts`` collection and a router ``gate``. On
Qwen1.5-MoE all 24 layers are MoE, so discovery is behaviour-neutral there.

Accessors are ``hasattr``-based and never raise on a non-MoE layer. No
experiment config is imported.
"""

from typing import Iterable, Optional


def _decoder_layers(model):
    """Return ``model.model.layers`` or raise if absent.

    Raises rather than returning ``[]`` so passing the wrong model object fails
    loudly instead of silently producing a no-op run.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "discover_moe_layers expects a causal-LM whose decoder stack lives at "
            "model.model.layers (the Qwen/Llama-family layout); got an object "
            f"without it: {type(model).__name__}"
        )
    return layers


def is_moe_layer(layer) -> bool:
    """Return ``True`` iff ``layer.mlp`` has both ``experts`` and a router ``gate``.

    Dense layers have ``gate_proj``/``up_proj``/``down_proj`` but no ``experts``
    or router ``gate`` -> returns ``False``. Missing ``mlp`` or sub-attributes
    also returns ``False`` rather than raising.
    """
    if not hasattr(layer, "mlp"):
        return False
    mlp = layer.mlp
    return hasattr(mlp, "experts") and hasattr(mlp, "gate")


def discover_moe_layers(
    model, selected: Optional[Iterable[int]] = None
) -> list[int]:
    """Return indices of MoE decoder layers in layer order.

    Scans ``model.model.layers`` via :func:`is_moe_layer`. If ``selected`` is
    given, returns the intersection in the order of ``selected`` (out-of-range or
    non-MoE indices are silently skipped). If ``None``, returns all MoE indices
    in ascending order.

    Raises ``AttributeError`` if the model has no ``model.model.layers``.
    """
    layers = _decoder_layers(model)
    moe_indices = [i for i, layer in enumerate(layers) if is_moe_layer(layer)]
    if selected is None:
        return moe_indices
    moe_set = set(moe_indices)
    return [i for i in selected if i in moe_set]
