"""Single source of truth for the on-disk artifact layout under ``DATA_DIR``.

Every stage reads and writes intermediate ``.pt`` tensors under a per-experiment
``DATA_DIR``. The subfolder names and filename templates used to be duplicated
between the writing stage and every reader (including :mod:`common.pairing` and
:mod:`common.fusion_layer`), so renaming an artifact meant grepping the whole
pipeline. This module centralizes that layout: writers, ``common`` readers and
the test builders all import these helpers, and the ``expert_*.pt`` glob is
derived from the same prefix as the per-expert load path.

Paths are built with forward slashes to match the literals the stages already
used (forward slashes resolve identically to ``os.path.join`` on every platform
for file I/O), so this is a behaviour-neutral move.

Library code only -- this module never imports an experiment ``config``.
"""

import json
import os

# Subfolder names under DATA_DIR.
EXPERTS_DIRNAME = "experts"
PAIRS_DIRNAME = "pairs"
MLP_INPUTS_DIRNAME = "mlp_inputs"
ACTIVATIONS_DIRNAME = "activations"
FUSION_EXPERTS_DIRNAME = "fusion_experts"
# New-module grouping prefix (replaces ``pairs/`` for the versioned ``Partition``
# artifact). Reserved here so the new and legacy layouts never share a directory.
GROUPS_DIRNAME = "groups"

# Diagnostic plots live under DATA_DIR (never CWD-relative ``plots/``). Earlier
# plotting wrote to ``plots/fusion/<timestamp>`` in the cwd; scoping them under
# DATA_DIR keeps different experiments from colliding.
PLOTS_DIRNAME = "plots"
FUSION_SUBDIR = "fusion"

# DATA_DIR layout version. A loader validates it and refuses a directory written
# under a different layout, so the new (groups/, safetensors) and legacy (pairs/,
# pickle) trees can never be silently mixed.
LAYOUT_VERSION = 1
LAYOUT_VERSION_FILENAME = "_layout_version"

# Fixed filenames inside an ``experts/<layer>`` directory.
GATE_FILENAME = "gate.pt"
SHARED_EXPERT_FILENAME = "shared_expert.pt"
SHARED_EXPERT_GATE_FILENAME = "shared_expert_gate.pt"

# Split tags used in split-scoped filenames.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"


# --------------------------------------------------------------------------- #
# experts/<layer>/...
# --------------------------------------------------------------------------- #
def experts_dir(data_dir: str, layer: int) -> str:
    """Directory holding one layer's saved experts and gates."""
    return f"{data_dir}/{EXPERTS_DIRNAME}/{layer}"


def expert_path(data_dir: str, layer: int, i: int) -> str:
    """Path of the ``i``-th saved expert in a layer."""
    return f"{experts_dir(data_dir, layer)}/expert_{i}.pt"


def expert_glob(data_dir: str, layer: int) -> str:
    """Glob matching every saved ``expert_*.pt`` of a layer.

    Derived from the same prefix as :func:`expert_path`, so the glob count and
    the load path can never drift apart.
    """
    return f"{experts_dir(data_dir, layer)}/expert_*.pt"


def gate_path(data_dir: str, layer: int) -> str:
    return f"{experts_dir(data_dir, layer)}/{GATE_FILENAME}"


def shared_expert_path(data_dir: str, layer: int) -> str:
    return f"{experts_dir(data_dir, layer)}/{SHARED_EXPERT_FILENAME}"


def shared_expert_gate_path(data_dir: str, layer: int) -> str:
    return f"{experts_dir(data_dir, layer)}/{SHARED_EXPERT_GATE_FILENAME}"


# --------------------------------------------------------------------------- #
# pairs/<layer>.pt
# --------------------------------------------------------------------------- #
def pairs_dir(data_dir: str) -> str:
    return f"{data_dir}/{PAIRS_DIRNAME}"


def pairs_path(data_dir: str, layer: int) -> str:
    """Path of a layer's saved expert pairing."""
    return f"{pairs_dir(data_dir)}/{layer}.pt"


# --------------------------------------------------------------------------- #
# mlp_inputs/<split>_<layer>.pt
# --------------------------------------------------------------------------- #
def mlp_inputs_dir(data_dir: str) -> str:
    return f"{data_dir}/{MLP_INPUTS_DIRNAME}"


def mlp_inputs_path(data_dir: str, split: str, layer: int) -> str:
    """Path of a layer's captured MLP-block inputs for a data split."""
    return f"{mlp_inputs_dir(data_dir)}/{split}_{layer}.pt"


# --------------------------------------------------------------------------- #
# activations/...
# --------------------------------------------------------------------------- #
def activations_dir(data_dir: str) -> str:
    return f"{data_dir}/{ACTIVATIONS_DIRNAME}"


def activation_matrix_path(data_dir: str, split: str, layer: int) -> str:
    """Path of a layer's EXPERTS x EXPERTS co-activation matrix for a split."""
    return f"{activations_dir(data_dir)}/{split}_matrix_{layer}.pt"


def activations_all_path(data_dir: str, split: str, layer: int) -> str:
    """Path of a layer's per-example top-k selection list for a split."""
    return f"{activations_dir(data_dir)}/{split}_all_{layer}.pt"


def pair_activations_path(data_dir: str, split: str, layer: int) -> str:
    """Path of a layer's per-pair activating-example index lists for a split."""
    return f"{activations_dir(data_dir)}/pairs_{split}_{layer}.pt"


# --------------------------------------------------------------------------- #
# fusion_experts/<method>/<layer>/<i>.pt
# --------------------------------------------------------------------------- #
def fusion_method_dir(method: str) -> str:
    """``DATA_DIR``-relative subdirectory of one fusion method's experts.

    The relative form (no ``data_dir``) is what :func:`get_fusion_layer_by_index`
    receives as its ``experts_dir`` argument.
    """
    return f"{FUSION_EXPERTS_DIRNAME}/{method}"


def fusion_experts_dir(data_dir: str, method: str, layer: int) -> str:
    return f"{data_dir}/{fusion_method_dir(method)}/{layer}"


def fusion_expert_path(data_dir: str, method: str, layer: int, i: int) -> str:
    """Path of the fused expert for one ``(method, layer, group index)``."""
    return f"{fusion_experts_dir(data_dir, method, layer)}/{i}.pt"


# --------------------------------------------------------------------------- #
# groups/<layer>.pt  (new-module Partition; replaces pairs/<layer>.pt)
# --------------------------------------------------------------------------- #
def groups_dir(data_dir: str) -> str:
    return f"{data_dir}/{GROUPS_DIRNAME}"


def group_path(data_dir: str, layer: int) -> str:
    """Path of a layer's saved grouping ``Partition`` (new versioned artifact)."""
    return f"{groups_dir(data_dir)}/{layer}.pt"


def group_activations_path(data_dir: str, split: str, layer: int) -> str:
    """Path of a layer's per-group activating-example index (new versioned artifact).

    The ``GroupActivationIndex`` counterpart of :func:`pair_activations_path`: it
    lives under the new ``groups/`` prefix so it never collides with the legacy
    ``activations/pairs_*`` artifact.
    """
    return f"{groups_dir(data_dir)}/{split}_activations_{layer}.pt"


# --------------------------------------------------------------------------- #
# plots/<subdir>  (DATA_DIR-scoped diagnostic plots)
# --------------------------------------------------------------------------- #
def plots_dir(data_dir: str, subdir: str = "") -> str:
    """DATA_DIR-scoped directory for diagnostic plots (never CWD-relative)."""
    base = f"{data_dir}/{PLOTS_DIRNAME}"
    return f"{base}/{subdir}" if subdir else base


# --------------------------------------------------------------------------- #
# DATA_DIR layout version.
# --------------------------------------------------------------------------- #
class LayoutVersionError(Exception):
    """Raised when a ``DATA_DIR`` was written under a different layout version."""


def layout_version_path(data_dir: str) -> str:
    return f"{data_dir}/{LAYOUT_VERSION_FILENAME}"


def write_layout_version(data_dir: str, version: int = LAYOUT_VERSION) -> None:
    """Stamp ``DATA_DIR`` with its layout version."""
    os.makedirs(data_dir, exist_ok=True)
    with open(layout_version_path(data_dir), "w", encoding="utf-8") as handle:
        json.dump({"layout_version": int(version)}, handle)


def read_layout_version(data_dir: str):
    """Return the stamped layout version, or ``None`` if unstamped."""
    path = layout_version_path(data_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return int(json.load(handle)["layout_version"])


def validate_layout_version(data_dir: str, expected: int = LAYOUT_VERSION) -> None:
    """Raise if ``DATA_DIR`` is stamped with a different layout version.

    An unstamped directory (``None``) is allowed -- it predates versioning -- so
    this is a guard against *mismatch*, not against absence.
    """
    found = read_layout_version(data_dir)
    if found is not None and found != expected:
        raise LayoutVersionError(
            f"{data_dir}: layout version {found} != expected {expected}"
        )
