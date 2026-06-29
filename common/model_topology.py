"""Discover which decoder layers of a causal-LM are MoE (router + experts).

Stages 1/4/5 used to assume *every* decoder layer is a sparse MoE block: they
trimmed the stack with ``for i in range(MAX_LAYERS, len(model.model.layers)):
del model.model.layers[MAX_LAYERS]`` and then indexed ``model.model.layers[i].mlp.gate``
/ ``.experts`` blindly over ``LAYERS = range(N)`` (§3.2). That crashes on hybrid
dense+sparse models (DeepSeek, Qwen3-MoE, Granite-MoE) whose early layers carry a
plain dense MLP (``gate_proj``/``up_proj``/``down_proj``, no ``experts`` and no
router ``gate``): indexing ``.mlp.experts`` on such a layer raises ``AttributeError``.

This module replaces the blind ``range(N)`` with a *discovery* of the MoE layers:
a layer is MoE iff its ``mlp`` exposes both an ``experts`` collection and a router
``gate``. For ``Qwen/Qwen1.5-MoE`` every one of the 24 decoder layers is MoE, so
discovery returns exactly ``range(24)`` and is behaviour-neutral there (likewise on
:class:`common.testing.fakes.FakeCausalLM`, where every layer is MoE).

The accessors are deliberately defensive (``hasattr``-based, never raising on a
malformed layer) so probing a non-MoE / bare object reports ``False`` instead of
crashing -- the whole point is to *not* crash on a hybrid stack.

Library code only -- this module never imports an experiment ``config``.
"""

from typing import Iterable, Optional


def _decoder_layers(model):
    """Return ``model.model.layers`` or raise a clear error if absent.

    Contract: a model without the standard ``model.model.layers`` decoder stack
    (the Qwen/Llama-family layout the whole pipeline targets) is a usage error,
    not an empty model -- so this *raises* rather than returning ``[]``, which
    would silently turn a wrong model object into a no-op run.
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
    """Return ``True`` iff ``layer`` exposes a MoE block.

    A MoE decoder layer has ``layer.mlp`` carrying both an ``experts`` collection
    and a router ``gate``. A *dense* layer's ``mlp`` (e.g. ``Qwen2MoeMLP`` /
    ``LlamaMLP``) has ``gate_proj``/``up_proj``/``down_proj`` but neither
    ``experts`` nor a router ``gate`` -> ``False``. Defensive by design: a layer
    (or bare object) missing ``mlp`` or the expected sub-attributes returns
    ``False`` rather than raising.
    """
    if not hasattr(layer, "mlp"):
        return False
    mlp = layer.mlp
    return hasattr(mlp, "experts") and hasattr(mlp, "gate")


def discover_moe_layers(
    model, selected: Optional[Iterable[int]] = None
) -> list[int]:
    """Indices of the decoder layers that are MoE, in layer order.

    Scans ``model.model.layers`` and keeps every index whose layer satisfies
    :func:`is_moe_layer`. When ``selected`` is given the result is the
    intersection with it, **preserving the order of** ``selected`` (so
    ``selected=[1, 0]`` yields ``[1, 0]``) and silently skipping any selected
    index that is out of range or points at a non-MoE (dense) layer. When
    ``selected`` is ``None`` every MoE index is returned in ascending layer order.

    Raises ``AttributeError`` (via :func:`_decoder_layers`) if the model has no
    ``model.model.layers`` decoder stack.
    """
    layers = _decoder_layers(model)
    moe_indices = [i for i, layer in enumerate(layers) if is_moe_layer(layer)]
    if selected is None:
        return moe_indices
    moe_set = set(moe_indices)
    return [i for i in selected if i in moe_set]
