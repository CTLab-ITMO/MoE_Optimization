"""Provenance, config snapshot, run manifest, and idempotency helpers.

Three concerns:
* **Config snapshot** -- stage 1 writes ``config_snapshot.json``; every later
  stage calls ``assert_config_matches_snapshot`` and fails with a per-key diff
  if the config drifted.
* **Run manifest** -- ``RunManifest`` (seed, commit, model id, dataset
  fingerprint, config hash) written alongside the artifacts.
* **Idempotency** -- ``is_fresh`` checks whether a fused artifact's stored
  upstream hash still matches; if so, the work item is skipped on re-run.

Library-only: never imports experiment config. Callers pass a plain dict.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional

from common import serialization

CONFIG_SNAPSHOT_FILENAME = "config_snapshot.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    """Return the current git commit hash, or None on any error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

# Keys excluded from the snapshot/hash; changing these must not invalidate artifacts.
EPHEMERAL_CONFIG_KEYS = frozenset({"device"})

# Keys consumed only by stage 4 (evaluation): the REPLACE_* layer-replacement modes,
# the diagnostic INSPECT_FUSION_METHOD, and the A/B USE_DEFAULT_EXPERTS toggle. They
# never change what stages 1-3 produce, so they are excluded from the stage-3 fusion
# idempotency hash (see ``fusion_config_hash``): flipping an eval flag must not
# invalidate already-fused experts. Unlike EPHEMERAL_CONFIG_KEYS these stay in the
# config snapshot -- they are real, recorded config that stage 4 reads.
EVAL_ONLY_CONFIG_KEYS = frozenset(
    {
        "REPLACE_LAYERS_SEQUENTIAL",
        "REPLACE_BEST_LAYERS_SEQUENTIAL",
        "REPLACE_LAYERS_SMART_SEQUENTIAL",
        "INSPECT_FUSION_METHOD",
        "USE_DEFAULT_EXPERTS",
    }
)


# --------------------------------------------------------------------------- #
# Value stabilization + canonicalization (reproducible across processes).
# --------------------------------------------------------------------------- #
def _stable(value: Any) -> Any:
    """Convert a config value to a JSON-safe, process-stable form.

    Scalars pass through; containers recurse; torch dtypes/devices and types
    become their string name. Unknown objects become their class name (not
    repr, which contains memory addresses that differ between processes).
    """
    import torch

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return type(value).__name__


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def canonical_config(config: dict, *, exclude: frozenset = frozenset()) -> dict:
    """Return an order-stable, ephemeral-free, value-stabilized config copy.

    ``EPHEMERAL_CONFIG_KEYS`` are always dropped; ``exclude`` drops additional
    keys, used to scope the hash to one stage (e.g. eval-only flags for stage 3).
    """
    dropped = EPHEMERAL_CONFIG_KEYS | exclude
    return {
        key: _stable(config[key])
        for key in sorted(config)
        if key not in dropped
    }


def config_hash(config: dict, *, exclude: frozenset = frozenset()) -> str:
    """Content hash of the canonical config (stable across processes).

    ``exclude`` drops extra keys before hashing (see ``canonical_config``).
    """
    return hashlib.sha256(
        _canonical_json(canonical_config(config, exclude=exclude)).encode("utf-8")
    ).hexdigest()


def fusion_config_hash(config: dict) -> str:
    """Stage-3 idempotency hash: ``config_hash`` with eval-only keys excluded.

    Fused experts are fully determined by the fusion-relevant config, so changing
    an evaluation flag (see ``EVAL_ONLY_CONFIG_KEYS``) leaves this hash -- and thus
    ``fused_upstream_hash`` -- unchanged, and stage 3 skips already-fused groups.
    """
    return config_hash(config, exclude=EVAL_ONLY_CONFIG_KEYS)


# --------------------------------------------------------------------------- #
# Config snapshot.
# --------------------------------------------------------------------------- #
class ConfigMismatchError(Exception):
    """Raised when the live config drifts from the written ``config_snapshot.json``."""


def config_snapshot_path(data_dir: str) -> str:
    return f"{data_dir}/{CONFIG_SNAPSHOT_FILENAME}"


def write_config_snapshot(data_dir: str, config: dict) -> dict:
    """Write the canonical config snapshot to ``DATA_DIR`` and return it."""
    snapshot = {"config": canonical_config(config), "config_hash": config_hash(config)}
    os.makedirs(data_dir, exist_ok=True)
    with open(config_snapshot_path(data_dir), "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
    return snapshot


def read_config_snapshot(data_dir: str) -> Optional[dict]:
    """Return the written snapshot, or ``None`` if no snapshot has been written."""
    path = config_snapshot_path(data_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_config_matches_snapshot(
    data_dir: str, config: dict, *, keys: Optional[list] = None
) -> None:
    """Raise ConfigMismatchError with a per-key diff if the live config drifted from the snapshot.

    ``keys`` restricts comparison to specific fields. Missing snapshot is a no-op.
    """
    snapshot = read_config_snapshot(data_dir)
    if snapshot is None:
        return
    saved = snapshot.get("config", {})
    live = canonical_config(config)
    compare = set(keys) if keys is not None else set(saved) | set(live)
    diffs = []
    for key in sorted(compare):
        saved_value = saved.get(key, "<absent>")
        live_value = live.get(key, "<absent>")
        if _canonical_json(saved_value) != _canonical_json(live_value):
            diffs.append(f"  {key}: snapshot={saved_value!r} != live={live_value!r}")
    if diffs:
        raise ConfigMismatchError(
            f"config drift vs {CONFIG_SNAPSHOT_FILENAME}:\n" + "\n".join(diffs)
        )


# --------------------------------------------------------------------------- #
# Run manifest.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunManifest:
    """Per-run provenance written next to the artifacts."""

    seed: Optional[int] = None
    commit: Optional[str] = None
    model_id: Optional[str] = None
    dataset: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    config_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def save(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        with open(
            f"{data_dir}/{RUN_MANIFEST_FILENAME}", "w", encoding="utf-8"
        ) as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, data_dir: str) -> Optional["RunManifest"]:
        path = f"{data_dir}/{RUN_MANIFEST_FILENAME}"
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


# --------------------------------------------------------------------------- #
# Fingerprints.
# --------------------------------------------------------------------------- #
def _safe_len(obj) -> Optional[int]:
    try:
        return len(obj)
    except TypeError:
        return None


def dataset_fingerprint(
    dataset, *, n: Optional[int] = None, seed: Optional[int] = None
) -> str:
    """Return a short stable hash identifying this dataset selection.

    Folds in size, seed, and the HF ``_fingerprint`` when available, so two
    runs over the same data and selection produce the same fingerprint.
    """
    parts = [f"n={n}", f"seed={seed}", f"len={_safe_len(dataset)}"]
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None:
        parts.append(f"fp={fingerprint}")
    info = getattr(dataset, "info", None)
    version = getattr(info, "version", None)
    if version is not None:
        parts.append(f"ver={version}")
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def fused_upstream_hash(
    *, config_hash: str, layer: int, method: str, group: int, extra: Optional[str] = None
) -> str:
    """Hash identifying one (layer, method, group) work item for skip-if-done.

    Combines config hash and work-item coordinates; does not read expert weights
    (they are fully determined by the config). ``config_hash`` should be the
    fusion-scoped ``fusion_config_hash`` so eval-only flags do not invalidate the
    item. Stored in the artifact so ``is_fresh`` can decide whether to skip
    recomputation.
    """
    parts = [f"cfg={config_hash}", f"layer={layer}", f"method={method}", f"group={group}"]
    if extra is not None:
        parts.append(f"extra={extra}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def header_hash(path: str) -> str:
    """Hash of an artifact's JSON header (no weights), for upstream fingerprints."""
    header = serialization.load_header(path)
    return hashlib.sha256(_canonical_json(header).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Idempotency / resume.
# --------------------------------------------------------------------------- #
def is_fresh(path: str, upstream_hash: str) -> bool:
    """Return True if the artifact at ``path`` is up to date with ``upstream_hash``.

    Checks the stored provenance hash; returns False on missing file, missing
    hash field, or any read error (triggers recompute, never crashes).
    """
    if not os.path.isfile(path):
        return False
    try:
        header = serialization.load_header(path)
    except Exception:
        return False
    provenance = header.get(serialization.PROVENANCE_FIELD, {})
    return provenance.get("upstream_hash") == upstream_hash
