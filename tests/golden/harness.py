"""End-to-end golden chain-driver for the fusion pipeline.

Runs the value-producing stages of a fusion-pipeline module on the importable
FakeMoE zoo (``common.testing``) in a temporary ``DATA_DIR``, then extracts the
order-invariant aggregates that the golden snapshot guards:

* the exact ``activation_correlation`` ``Partition`` (pair list per layer,
  preserving tuple orientation ``(pair[0], pair[1])`` and the ``i.pt`` order);
* per-layer reconstruction MSE / RMSE / Cosine / R2 (median over pairs) for a
  representative set of fusion methods, computed from the real stage-6 fused
  artifacts via the same ``test_model_on_data`` the pipeline uses;
* stage-9 ``accuracy_list`` (single-replacement mode) per method;
* a registry snapshot (names + order of fusion methods / metrics / pairings,
  plus the ``best_metrics`` direction map).

The driver is parametrized over the module under test (``ModuleSpec``) so the
legacy ``Qwen_MoE_experts_fusion_experiment`` and the new ``experts_fusion``
copy are held to the same ``golden_v1.json``.

FakeMoE bridge: the fake router gate (``FakeRouterGate``) returns bare logits,
whereas ``FusionLayer`` (stage 9) expects a gate returning a 3-tuple. After
stage 3 the driver overwrites the saved ``gate.pt`` with a deterministic
``FakeFusionGate`` -- harness glue for the fake model only; the experts and
fused artifacts feeding stage 9 are the real chained outputs.
"""
import glob
import importlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common.testing import (
    N_EXPERTS,
    TOP_K,
    FakeFusionGate,
    FakeQADataset,
    make_fake_datasets,
    make_fake_transformers,
)
from common.testing.fakes import N_LAYERS

ROOT = Path(__file__).resolve().parents[2]

# Golden run parameters (small + deterministic; see module docstring).
GOLDEN_SEED = 1234
GOLDEN_LAYERS = list(range(N_LAYERS))
GOLDEN_SIZE = 8
# Representative subset: deterministic methods (tight) + one stochastic (loose).
GOLDEN_FUSIONS = ["Average", "ZipIt", "Procrustes", "TIES", "Pruning", "Fisher"]
RECON_METRICS = ["MSE", "RMSE", "Cosine", "R2"]
# Frozen pairing-method name set (literals in common.pairing.get_pairs; moved to
# common.names in S1 -- the snapshot guards the set/order across that move).
PAIRING_METHODS = ["activation_correlation", "random", "output_similarity"]


@dataclass(frozen=True)
class ModuleSpec:
    label: str
    stage_dir: Path
    config_module: str


LEGACY = ModuleSpec(
    "legacy", ROOT / "Qwen_MoE_experts_fusion_experiment", "config"
)
NEW = ModuleSpec("experts_fusion", ROOT / "experts_fusion", "experts_fusion.config")


def _golden_overrides(data_dir: str) -> dict:
    return {
        "DATA_DIR": data_dir,
        "LAYERS": GOLDEN_LAYERS,
        "EXPERTS": N_EXPERTS,
        "TOP_K": TOP_K,
        "device": torch.device("cpu"),
        "SEED": GOLDEN_SEED,
        "EVOLUTION_STEPS": 0,
        "PAIRING_METHOD": "activation_correlation",
        "TRAIN_SIZE": GOLDEN_SIZE,
        "VALIDATION_SIZE": GOLDEN_SIZE,
        "TEST_SIZE": GOLDEN_SIZE,
        "FILTER_FUSIONS": list(GOLDEN_FUSIONS),
        "FILTER_METRICS": [],
        "USE_DEFAULT_EXPERTS": False,
        "REPLACE_LAYERS_SEQUANTIAL": False,
        "REPLACE_BEST_LAYERS_SEQUANTIAL": False,
        "REPLACE_LAYERS_SMART_SEQUANTIAL": False,
    }


def _substitute_gates(data_dir: str) -> None:
    """Replace each layer's saved ``gate.pt`` with a deterministic FakeFusionGate.

    Stage-9 FakeMoE bridge -- see module docstring.
    """
    torch.manual_seed(GOLDEN_SEED)
    for layer in GOLDEN_LAYERS:
        gate = FakeFusionGate()
        torch.save(gate, os.path.join(data_dir, "experts", str(layer), "gate.pt"))


def _compute_reconstruction(data_dir: str) -> dict:
    """Per-layer median (over pairs) reconstruction metrics for each method."""
    from common.mlp_utils import test_model_on_data

    out = {method: {} for method in GOLDEN_FUSIONS}
    for layer in GOLDEN_LAYERS:
        experts = [
            torch.load(
                os.path.join(data_dir, "experts", str(layer), f"expert_{i}.pt"),
                weights_only=False,
            )
            for i in range(N_EXPERTS)
        ]
        val = torch.load(os.path.join(data_dir, "mlp_inputs", f"validation_{layer}.pt"))
        pairs = torch.load(os.path.join(data_dir, "pairs", f"{layer}.pt"))
        for method in GOLDEN_FUSIONS:
            collected = {metric: [] for metric in RECON_METRICS}
            for i, (a, b) in enumerate(pairs):
                fused = torch.load(
                    os.path.join(data_dir, "fusion_experts", method, str(layer), f"{i}.pt"),
                    weights_only=False,
                )
                report = test_model_on_data(fused, experts[a], experts[b], val)
                for metric in RECON_METRICS:
                    collected[metric].append(report[metric])
            out[method][str(layer)] = {
                metric: _finite_median(collected[metric]) for metric in RECON_METRICS
            }
    return out


def _finite_median(values: list) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _read_partition(data_dir: str) -> dict:
    out = {}
    for layer in GOLDEN_LAYERS:
        pairs = torch.load(os.path.join(data_dir, "pairs", f"{layer}.pt"))
        out[str(layer)] = [[int(a), int(b)] for (a, b) in pairs]
    return out


def _read_accuracy(workdir: Path) -> dict:
    prefix = "accuracy_list_single_"
    out = {}
    pattern = str(workdir / "test_result_*" / "accuracy" / f"{prefix}*.pt")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)[len(prefix):-len(".pt")]
        out[name] = [float(x) for x in torch.load(path)]
    return out


def registry_snapshot() -> dict:
    """Names + order of fusion methods / metrics / pairings (config-free)."""
    from common.fusion import build_fusion_methods
    from common.metrics import METRICS, best_metrics

    return {
        "fusion_methods": list(build_fusion_methods().keys()),
        "metrics": [metric.name for metric in METRICS],
        "best_metrics": {name: bool(flag) for name, flag in best_metrics.items()},
        "pairing_methods": list(PAIRING_METHODS),
    }


def run_chain(spec: ModuleSpec, tmp_root: Path, monkeypatch) -> dict:
    """Run stages 1-6 + 9 of ``spec``'s module on FakeMoE; return aggregates."""
    import runpy

    import seaborn

    data_dir = tmp_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    workdir = tmp_root / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = importlib.import_module(spec.config_module)
    for key, value in _golden_overrides(str(data_dir)).items():
        monkeypatch.setattr(cfg, key, value, raising=False)

    monkeypatch.setattr(seaborn, "kdeplot", lambda *a, **k: None)
    monkeypatch.setitem(__import__("sys").modules, "transformers", make_fake_transformers())
    monkeypatch.setitem(__import__("sys").modules, "datasets", make_fake_datasets())

    from common import qa_eval
    monkeypatch.setattr(qa_eval, "load_dataset", lambda *a, **k: FakeQADataset())
    qa_eval._DATA_CACHE.clear()

    monkeypatch.chdir(workdir)

    def stage(name: str) -> None:
        runpy.run_path(str(spec.stage_dir / name), run_name="__main__")

    stage("1_generate_activation_matrix.py")
    stage("2_find_best_matches.py")
    stage("3_save_experts.py")
    _substitute_gates(str(data_dir))
    stage("4_write_mlp_inputs.py")
    stage("5_activations.py")
    stage("6_fusion.py")
    reconstruction = _compute_reconstruction(str(data_dir))
    stage("9_test.py")
    accuracy = _read_accuracy(workdir)

    return {
        "seed": GOLDEN_SEED,
        "partition": _read_partition(str(data_dir)),
        "reconstruction": reconstruction,
        "accuracy": accuracy,
    }
