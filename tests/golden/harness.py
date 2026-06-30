"""End-to-end golden chain-driver for the fusion pipeline.

Runs the value-producing stages of a fusion-pipeline module on the importable
FakeMoE zoo (``common.testing``) in a temporary ``DATA_DIR``, then extracts the
order-invariant aggregates that the golden snapshot guards:

* the exact ``activation_correlation`` ``Partition`` (pair list per layer,
  preserving tuple orientation ``(pair[0], pair[1])`` and the group order);
* per-layer reconstruction MSE / RMSE / Cosine / R2 (median over pairs) for a
  representative set of fusion methods, computed from the real fused artifacts
  via the same ``test_model_on_data`` the pipeline uses;
* stage ``accuracy_list`` (single-replacement mode) per method;
* a registry snapshot (names + order of fusion methods / metrics / pairings,
  plus the ``best_metrics`` direction map).

The driver is parametrized over the module under test (:class:`ModuleSpec`) so
the legacy ``Qwen_MoE_experts_fusion_experiment`` (nine stages, pickled
artifacts) and the new ``experts_fusion`` (four stages, versioned safetensors
artifacts) are held to the same ``golden_v1.json``. The two modules differ only
in their stage list, where the FakeFusionGate substitution happens, how the
pairing / fused artifacts are stored, and where the results land -- captured by
the :class:`ModuleSpec` fields.

FakeMoE bridge: the fake router gate (``FakeRouterGate``) returns bare logits,
whereas ``FusionLayer`` (the eval stage) expects a gate returning a 3-tuple.
After the expert-saving stage the driver overwrites the saved ``gate.pt`` with a
deterministic ``FakeFusionGate`` -- harness glue for the fake model only.
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

# Artifact-format tags: the legacy module stores pickled pairs/AdvancedMLP, the
# new module the versioned Partition (groups/) + FusedExpert (safetensors).
FORMAT_LEGACY = "legacy"
FORMAT_V1 = "v1"

LEGACY_STAGES = (
    "1_generate_activation_matrix.py",
    "2_find_best_matches.py",
    "3_save_experts.py",
    "4_write_mlp_inputs.py",
    "5_activations.py",
    "6_fusion.py",
    "9_test.py",
)
NEW_STAGES = (
    "1_preprocess.py",
    "2_grouping.py",
    "3_fusion.py",
    "4_evaluate.py",
)


@dataclass(frozen=True)
class ModuleSpec:
    label: str
    stage_dir: Path
    config_module: str
    stages: tuple
    # Stage filename after which the FakeFusionGate is substituted (the stage
    # that has just written gate.pt and before the stage that reads it for eval).
    substitute_after: str
    artifact_format: str
    # ``True`` -> results land under DATA_DIR (§1.4); ``False`` -> under the cwd.
    results_under_data_dir: bool


LEGACY = ModuleSpec(
    "legacy",
    ROOT / "Qwen_MoE_experts_fusion_experiment",
    "config",
    LEGACY_STAGES,
    substitute_after="3_save_experts.py",
    artifact_format=FORMAT_LEGACY,
    results_under_data_dir=False,
)
NEW = ModuleSpec(
    "experts_fusion",
    ROOT / "experts_fusion",
    "experts_fusion.config",
    NEW_STAGES,
    substitute_after="1_preprocess.py",
    artifact_format=FORMAT_V1,
    results_under_data_dir=True,
)


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
        # Legacy spelling (the old module) + corrected spelling (the new module);
        # monkeypatch sets both with raising=False so each module gets its own.
        "REPLACE_LAYERS_SEQUANTIAL": False,
        "REPLACE_BEST_LAYERS_SEQUANTIAL": False,
        "REPLACE_LAYERS_SMART_SEQUANTIAL": False,
        "REPLACE_LAYERS_SEQUENTIAL": False,
        "REPLACE_BEST_LAYERS_SEQUENTIAL": False,
        "REPLACE_LAYERS_SMART_SEQUENTIAL": False,
    }


def _substitute_gates(data_dir: str) -> None:
    """Replace each layer's saved ``gate.pt`` with a deterministic FakeFusionGate.

    FakeMoE eval bridge -- see module docstring. Both modules save the gate at
    ``experts/<layer>/gate.pt``.
    """
    torch.manual_seed(GOLDEN_SEED)
    for layer in GOLDEN_LAYERS:
        gate = FakeFusionGate()
        torch.save(gate, os.path.join(data_dir, "experts", str(layer), "gate.pt"))


def _load_pairs(data_dir: str, layer: int, spec: ModuleSpec):
    """Per-layer pairing as a ``[(a, b), ...]`` list, in either artifact format."""
    if spec.artifact_format == FORMAT_V1:
        from common import layout
        from common.artifacts import Partition

        return Partition.load(layout.group_path(data_dir, layer)).to_pairs()
    return torch.load(os.path.join(data_dir, "pairs", f"{layer}.pt"))


def _load_fused(data_dir: str, method: str, layer: int, index: int, spec: ModuleSpec):
    """Load one fused expert as an ``AdvancedMLP``, in either artifact format."""
    if spec.artifact_format == FORMAT_V1:
        from common import layout
        from common.fused_group import FusedExpert

        return FusedExpert.load(
            layout.fusion_expert_path(data_dir, method, layer, index)
        ).mlp
    return torch.load(
        os.path.join(data_dir, "fusion_experts", method, str(layer), f"{index}.pt"),
        weights_only=False,
    )


def _compute_reconstruction(data_dir: str, spec: ModuleSpec) -> dict:
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
        pairs = _load_pairs(data_dir, layer, spec)
        for method in GOLDEN_FUSIONS:
            collected = {metric: [] for metric in RECON_METRICS}
            for i, (a, b) in enumerate(pairs):
                fused = _load_fused(data_dir, method, layer, i, spec)
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


def _read_partition(data_dir: str, spec: ModuleSpec) -> dict:
    out = {}
    for layer in GOLDEN_LAYERS:
        pairs = _load_pairs(data_dir, layer, spec)
        out[str(layer)] = [[int(a), int(b)] for (a, b) in pairs]
    return out


def _read_accuracy(results_root: Path) -> dict:
    prefix = "accuracy_list_single_"
    out = {}
    pattern = str(results_root / "test_result_*" / "accuracy" / f"{prefix}*.pt")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)[len(prefix):-len(".pt")]
        out[name] = [float(x) for x in torch.load(path)]
    return out


def registry_snapshot() -> dict:
    """Names + order of fusion methods / metrics / pairings (config-free)."""
    from common.fusion import build_fusion_methods
    from common.grouping import GROUPING_REGISTRY
    from common.metrics import METRICS, best_metrics

    return {
        "fusion_methods": list(build_fusion_methods().keys()),
        "metrics": [metric.name for metric in METRICS],
        "best_metrics": {name: bool(flag) for name, flag in best_metrics.items()},
        "pairing_methods": list(GROUPING_REGISTRY.names()),
    }


def run_chain(spec: ModuleSpec, tmp_root: Path, monkeypatch) -> dict:
    """Run ``spec``'s value-producing stages on FakeMoE; return aggregates."""
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
        if name == spec.substitute_after:
            _substitute_gates(str(data_dir))

    for name in spec.stages:
        if name == spec.stages[-1]:
            # The last (eval) stage produces accuracy; compute reconstruction
            # from the fused artifacts just before it.
            reconstruction = _compute_reconstruction(str(data_dir), spec)
        stage(name)

    results_root = data_dir if spec.results_under_data_dir else workdir
    accuracy = _read_accuracy(results_root)

    return {
        "seed": GOLDEN_SEED,
        "partition": _read_partition(str(data_dir), spec),
        "reconstruction": reconstruction,
        "accuracy": accuracy,
    }
