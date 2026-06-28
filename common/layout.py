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

# Subfolder names under DATA_DIR.
EXPERTS_DIRNAME = "experts"
PAIRS_DIRNAME = "pairs"
MLP_INPUTS_DIRNAME = "mlp_inputs"
ACTIVATIONS_DIRNAME = "activations"
FUSION_EXPERTS_DIRNAME = "fusion_experts"

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
