"""Centralized path helpers for all on-disk artifacts under ``DATA_DIR``.

All pipeline stages import these helpers so each artifact path is defined once;
renaming a subfolder means editing this file, not the whole pipeline.
Paths use forward slashes (equivalent to os.path.join on all platforms).
No experiment ``config`` is imported here.
"""

import json
import os

# Subfolder names under DATA_DIR.
EXPERTS_DIRNAME = "experts"
PAIRS_DIRNAME = "pairs"
MLP_INPUTS_DIRNAME = "mlp_inputs"
ACTIVATIONS_DIRNAME = "activations"
FUSION_EXPERTS_DIRNAME = "fusion_experts"
# Grouping prefix for versioned Partition artifacts (replaces pairs/).
GROUPS_DIRNAME = "groups"

# Plots are scoped under DATA_DIR, not CWD, so parallel experiments don't collide.
PLOTS_DIRNAME = "plots"
FUSION_SUBDIR = "fusion"

# Stamped into DATA_DIR; loaders reject mismatches so new (groups/) and legacy
# (pairs/) artifact trees are never silently mixed.
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

    Uses the same prefix as expert_path so glob count and load path stay in sync.
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
    """DATA_DIR-relative subdirectory for one fusion method's experts."""
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
    """Path of a layer's per-group activating-example index (GroupActivationIndex).

    Stored under groups/ so it never collides with the legacy activations/pairs_* file.
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
    """Raise if DATA_DIR is stamped with a different layout version.

    Unstamped directories (None) are allowed; this guards against mismatch, not absence.
    """
    found = read_layout_version(data_dir)
    if found is not None and found != expected:
        raise LayoutVersionError(
            f"{data_dir}: layout version {found} != expected {expected}"
        )
