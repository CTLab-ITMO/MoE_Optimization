"""Centralized path helpers for all on-disk artifacts under ``DATA_DIR``.

The common loaders use these helpers so each artifact path is defined once.
Paths use forward slashes (equivalent to os.path.join on all platforms).
No experiment ``config`` is imported here.
"""

# Subfolder names under DATA_DIR.
EXPERTS_DIRNAME = "experts"
PAIRS_DIRNAME = "pairs"
MLP_INPUTS_DIRNAME = "mlp_inputs"
ACTIVATIONS_DIRNAME = "activations"

# Fixed filenames inside an ``experts/<layer>`` directory.
GATE_FILENAME = "gate.pt"
SHARED_EXPERT_FILENAME = "shared_expert.pt"
SHARED_EXPERT_GATE_FILENAME = "shared_expert_gate.pt"

# Split tag used in split-scoped filenames.
TRAIN_SPLIT = "train"


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
