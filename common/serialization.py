"""Versioned artifact (de)serializer over ``safetensors``.

This is the serialization contract every pipeline artifact will be built on
(the meaningful artifact *classes* themselves land in a later step). The
contract, in one sentence: an artifact on disk is a **``state_dict`` of tensors
plus a JSON header of metadata** -- it is never a pickled ``nn.Module``.

Why not ``torch.save`` of a module:

* pickled modules break the moment a class is renamed/moved;
* ``torch>=2.6`` loads with ``weights_only=True`` and refuses arbitrary pickles;
* a pickle is opaque to other tools and a security risk.

``safetensors`` gives us exactly the contract for free: tensors in a flat,
typed container (bf16-exact, no pickle) and a string metadata slot for the JSON
header, with lazy per-key access (:func:`peek`/:func:`load_header` read the
header and tensor shapes without materializing any weights).

Versioning & migration:

* every artifact header carries :data:`FORMAT_VERSION`;
* ``version > FORMAT_VERSION`` (artifact written by a newer reader) is an
  explicit :class:`FormatVersionError`, never a silent misread;
* within-format header upgrades are an additive registry
  (:func:`register_upgrader`, applied by :func:`load`);
* the legacy pickled ``.pt`` files (v0: a pickled ``AdvancedMLP`` or a raw
  ``(i, j)`` pair list) are bridged into the v1 envelope by
  :func:`migrate_v0_to_v1` / :func:`load_legacy`.

Library code only -- this module never imports an experiment ``config``.
"""

import json
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from common.mlp import AdvancedMLP
from common.numeric import _default_device

# Current on-disk artifact format version. Bump when the header/tensor contract
# changes and register an upgrader (see register_upgrader).
FORMAT_VERSION = 1

# safetensors metadata key holding the JSON header, and the header fields the
# serializer owns.
_HEADER_KEY = "header"
VERSION_FIELD = "format_version"
PAYLOAD_TYPE_FIELD = "payload_type"
PROVENANCE_FIELD = "provenance"

# Payload-type tags for the artifacts the v0->v1 migrator understands.
PAYLOAD_ADVANCED_MLP = "advanced_mlp"
PAYLOAD_PAIRS = "pairs"


class SerializationError(Exception):
    """Base error for the artifact (de)serializer."""


class FormatVersionError(SerializationError):
    """Raised when an artifact was written by a newer format than this reader."""


@dataclass(frozen=True)
class Artifact:
    """A loaded artifact: a tensor ``state_dict`` plus its JSON metadata header.

    ``header`` always carries :data:`VERSION_FIELD`; it may carry
    :data:`PAYLOAD_TYPE_FIELD` and a :data:`PROVENANCE_FIELD` slot.
    """

    tensors: dict[str, torch.Tensor]
    header: dict


@dataclass(frozen=True)
class ArtifactInfo:
    """Inventory of an artifact read without materializing any weights."""

    header: dict
    tensors: dict[str, dict]  # name -> {"shape": [...], "dtype": "F32"}


# --------------------------------------------------------------------------- #
# Provenance slot (§3.6). The slot exists from v1; later stages populate it
# fully (commit / config-hash / dataset fingerprint).
# --------------------------------------------------------------------------- #
def make_provenance(
    *,
    commit: Optional[str] = None,
    seed: Optional[int] = None,
    model_id: Optional[str] = None,
    dataset: Optional[str] = None,
    config_hash: Optional[str] = None,
    upstream_hash: Optional[str] = None,
) -> dict:
    """Build a provenance header slot, dropping unset (``None``) fields."""
    fields = {
        "commit": commit,
        "seed": seed,
        "model_id": model_id,
        "dataset": dataset,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
    }
    return {key: value for key, value in fields.items() if value is not None}


# --------------------------------------------------------------------------- #
# Core save / load.
# --------------------------------------------------------------------------- #
def _validate_tensors(tensors: dict) -> None:
    if not isinstance(tensors, dict):
        raise SerializationError(f"tensors must be a dict, got {type(tensors).__name__}")
    for key, value in tensors.items():
        if not torch.is_tensor(value):
            raise SerializationError(
                f"artifact tensor {key!r} is a {type(value).__name__}, not a Tensor; "
                "the serializer stores state_dicts, never pickled objects"
            )


def _prepare_tensors(tensors: dict) -> dict:
    # safetensors needs detached, contiguous tensors and rejects shared storage.
    return {key: value.detach().contiguous() for key, value in tensors.items()}


def save(
    path,
    tensors: dict,
    header: Optional[dict] = None,
    *,
    payload_type: Optional[str] = None,
    provenance: Optional[dict] = None,
) -> None:
    """Write ``tensors`` + JSON ``header`` to ``path`` as a versioned safetensors.

    ``tensors`` must be a dict of tensors (a ``state_dict``); passing an
    ``nn.Module`` or any non-tensor value is an error -- this serializer never
    pickles. The current :data:`FORMAT_VERSION` is always stamped into the header.
    """
    _validate_tensors(tensors)
    full_header = dict(header or {})
    full_header[VERSION_FIELD] = FORMAT_VERSION
    if payload_type is not None:
        full_header[PAYLOAD_TYPE_FIELD] = payload_type
    if provenance is not None:
        full_header[PROVENANCE_FIELD] = provenance
    save_file(
        _prepare_tensors(tensors),
        str(path),
        metadata={_HEADER_KEY: json.dumps(full_header)},
    )


def _read_header(path) -> dict:
    with safe_open(str(path), framework="pt") as handle:
        raw = handle.metadata() or {}
    return json.loads(raw.get(_HEADER_KEY, "{}"))


def _header_version(header: dict) -> int:
    if VERSION_FIELD not in header:
        warnings.warn(
            f"artifact header has no {VERSION_FIELD!r}; assuming current "
            f"version {FORMAT_VERSION}",
            RuntimeWarning,
            stacklevel=3,
        )
        return FORMAT_VERSION
    return int(header[VERSION_FIELD])


def _check_not_newer(version: int, path) -> None:
    if version > FORMAT_VERSION:
        raise FormatVersionError(
            f"{path}: artifact format version {version} is newer than this "
            f"reader supports ({FORMAT_VERSION}); upgrade the code"
        )


def load_header(path) -> dict:
    """Read just the JSON header (no weights). Raises on a too-new version."""
    header = _read_header(path)
    _check_not_newer(_header_version(header), path)
    return header


def peek(path) -> ArtifactInfo:
    """Inventory an artifact -- header + tensor shapes/dtypes -- without loading weights."""
    with safe_open(str(path), framework="pt") as handle:
        raw = handle.metadata() or {}
        specs = {}
        for name in handle.keys():
            sl = handle.get_slice(name)
            specs[name] = {"shape": list(sl.get_shape()), "dtype": sl.get_dtype()}
    header = json.loads(raw.get(_HEADER_KEY, "{}"))
    _check_not_newer(_header_version(header), path)
    return ArtifactInfo(header=header, tensors=specs)


def load(path) -> Artifact:
    """Load an artifact, applying any registered header upgraders to reach CURRENT."""
    header = _read_header(path)
    version = _header_version(header)
    _check_not_newer(version, path)
    tensors = load_file(str(path))
    return _apply_upgraders(Artifact(tensors=tensors, header=header), version)


# --------------------------------------------------------------------------- #
# Within-format header upgraders (v_n -> v_{n+1}), additive registry.
# --------------------------------------------------------------------------- #
_UPGRADERS: dict[int, Callable[[Artifact], Artifact]] = {}


def register_upgrader(from_version: int):
    """Register an upgrader bringing a v``from_version`` artifact to the next version.

    Collision is an error (no silent override); the chain is applied by :func:`load`.
    """

    def decorator(func: Callable[[Artifact], Artifact]) -> Callable[[Artifact], Artifact]:
        if from_version in _UPGRADERS:
            raise SerializationError(f"upgrader from version {from_version} already registered")
        _UPGRADERS[from_version] = func
        return func

    return decorator


def _apply_upgraders(artifact: Artifact, version: int) -> Artifact:
    while version < FORMAT_VERSION:
        upgrader = _UPGRADERS.get(version)
        if upgrader is None:
            raise SerializationError(
                f"no registered upgrader from format version {version} "
                f"(target {FORMAT_VERSION})"
            )
        artifact = upgrader(artifact)
        version = int(artifact.header.get(VERSION_FIELD, version + 1))
    return artifact


# --------------------------------------------------------------------------- #
# AdvancedMLP <-> artifact (the expert payload).
# --------------------------------------------------------------------------- #
def mlp_to_artifact(mlp: AdvancedMLP) -> Artifact:
    """Serialize an :class:`AdvancedMLP` as ``state_dict`` tensors + a header.

    The header records exactly what ``state_dict`` alone cannot: the layer
    shape (``in/hid/out``), the activation, and whether the optional ``down``
    layer and ``U`` unmerge matrix are present.
    """
    tensors = {key: value for key, value in mlp.state_dict().items()}
    header = {
        "in_features": int(mlp.in_features),
        "hid_features": int(mlp.hid_features),
        "out_features": int(mlp.out_features),
        "activation": type(mlp.act).__name__,
        "has_down": mlp.down is not None,
        "has_U": mlp.U is not None,
    }
    return Artifact(tensors=tensors, header=header)


def artifact_to_mlp(artifact: Artifact, device: Optional[torch.device] = None) -> AdvancedMLP:
    """Reconstruct an :class:`AdvancedMLP` forward-identically from an artifact."""
    if device is None:
        device = _default_device()
    header = artifact.header
    activation = header.get("activation", "SiLU")
    if activation != "SiLU":
        # The header records the activation; AdvancedMLP is SiLU-only, so honour
        # it explicitly instead of silently rebuilding SiLU for a non-SiLU
        # artifact. (Configurable activations arrive with the model adapters.)
        raise SerializationError(
            f"artifact activation {activation!r} is not supported by AdvancedMLP (SiLU only)"
        )
    mlp = AdvancedMLP(
        int(header["in_features"]),
        int(header["hid_features"]),
        int(header["out_features"]),
        device=device,
    )
    tensors = {key: value.to(device) for key, value in artifact.tensors.items()}
    # Drop ``down`` before loading so the module's expected keys match the
    # state (an LS-fused expert has no ``down``); load weights; then attach the
    # optional ``U`` unmerge matrix (kept out of load_state_dict so strict=True
    # holds for the gate/up/down weights).
    if not header.get("has_down", "down.weight" in tensors):
        mlp.down = None
    state = {key: value for key, value in tensors.items() if key != "U"}
    mlp.load_state_dict(state, strict=True)
    if header.get("has_U", "U" in tensors):
        mlp.U = torch.nn.Parameter(tensors["U"])
    return mlp


# --------------------------------------------------------------------------- #
# pairs list <-> artifact (the legacy pairing payload).
# --------------------------------------------------------------------------- #
def pairs_to_artifact(pairs) -> Artifact:
    """Wrap a legacy ``[(i, j), ...]`` pair list as a metadata-only artifact."""
    encoded = [[int(i), int(j)] for (i, j) in pairs]
    return Artifact(tensors={}, header={"pairs": encoded, "num_pairs": len(encoded)})


def artifact_to_pairs(artifact: Artifact) -> list:
    """Recover the ``[(i, j), ...]`` pair list from a pairs artifact."""
    return [tuple(pair) for pair in artifact.header.get("pairs", [])]


# --------------------------------------------------------------------------- #
# Legacy pickle (v0) -> v1 envelope migration.
# --------------------------------------------------------------------------- #
def _is_pairs_list(obj) -> bool:
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        return isinstance(obj, (list, tuple))  # empty list is a (degenerate) pairs list
    return all(
        isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in obj
    )


def migrate_v0_to_v1(legacy_obj) -> Artifact:
    """Wrap a legacy pickled v0 object into a v1 :class:`Artifact`.

    Understands the two artifacts the old pipeline pickled: an
    :class:`AdvancedMLP` expert and a raw ``[(i, j), ...]`` pair list. The
    returned artifact carries its :data:`PAYLOAD_TYPE_FIELD` and is byte-free of
    pickle; :func:`save` will stamp the current format version onto it.
    """
    if isinstance(legacy_obj, AdvancedMLP):
        artifact = mlp_to_artifact(legacy_obj)
        artifact.header[PAYLOAD_TYPE_FIELD] = PAYLOAD_ADVANCED_MLP
        return artifact
    if _is_pairs_list(legacy_obj):
        artifact = pairs_to_artifact(legacy_obj)
        artifact.header[PAYLOAD_TYPE_FIELD] = PAYLOAD_PAIRS
        return artifact
    raise SerializationError(
        f"no v0->v1 migration for legacy object of type {type(legacy_obj).__name__}"
    )


def load_legacy(path) -> Artifact:
    """Load a legacy pickled ``.pt`` and migrate it to a v1 :class:`Artifact`."""
    legacy_obj = torch.load(str(path), weights_only=False)
    return migrate_v0_to_v1(legacy_obj)
