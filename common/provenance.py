"""Provenance, config snapshot, run manifest and idempotency.

The pipeline must be able to (a) record *how* each artifact was produced,
(b) refuse to run a stage against a ``DATA_DIR`` written under a different
config, and (c) skip a work item that is already up to date. This module owns
those three concerns:

* **Config snapshot** -- the first stage writes a canonical ``config_snapshot.json`` of
  the run's config (ephemeral keys such as ``device`` dropped, values stabilized
  so the hash is reproducible across processes). Every later stage calls
  :func:`assert_config_matches_snapshot` and fails fast with a per-key diff if
  the live config drifted.
* **Run manifest** -- a per-run :class:`RunManifest` (seed, commit, model id,
  dataset fingerprint, config hash) written alongside the artifacts.
* **Fingerprints + idempotency** -- :func:`dataset_fingerprint` for the dataset
  provenance and :func:`fused_upstream_hash` / :func:`is_fresh` for skip-if-done
  resume: a fused artifact carries its upstream hash in the serializer's
  provenance slot, and a re-run skips it when the hash still matches.

Library code only -- this module never imports an experiment ``config``. The
experiment builds the plain value map and passes it in; the canonicalization,
hashing, on-disk JSON and the freshness check live here.
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
    """Best-effort current git commit hash for provenance.

    Returns ``None`` outside a repository, when git is unavailable, or on any
    error -- provenance is recorded when it can be, never a hard dependency.
    """
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

# Config keys that are environment-specific and excluded from the canonical
# snapshot and its hash -- a device move must not invalidate the artifacts.
EPHEMERAL_CONFIG_KEYS = frozenset({"device"})


# --------------------------------------------------------------------------- #
# Value stabilization + canonicalization (reproducible across processes).
# --------------------------------------------------------------------------- #
def _stable(value: Any) -> Any:
    """Map a config value to a JSON-safe, process-stable representation.

    Plain scalars pass through; containers recurse; ``torch`` dtypes/devices and
    types become their names; any other object becomes its *class name* (never
    its ``repr``, whose memory address would make the hash non-reproducible).
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


def canonical_config(config: dict) -> dict:
    """Return an order-stable, ephemeral-free, value-stabilized config copy."""
    return {
        key: _stable(config[key])
        for key in sorted(config)
        if key not in EPHEMERAL_CONFIG_KEYS
    }


def config_hash(config: dict) -> str:
    """Content hash of the canonical config (stable across processes)."""
    return hashlib.sha256(_canonical_json(canonical_config(config)).encode("utf-8")).hexdigest()


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
    """Raise :class:`ConfigMismatchError` with a per-key diff on config drift.

    ``keys`` optionally restricts the comparison to the fields a stage depends
    on. An absent snapshot (nothing written yet) is allowed -- nothing to compare.
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
    """Stable fingerprint of a dataset selection for the provenance header.

    Best-effort across HF datasets (``_fingerprint`` / revision) and the fake
    QA dataset: folds in the HF fingerprint when present plus the selection size
    and seed, so two runs over the same data + selection agree.
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
    """Cheap upstream hash for one fused ``(layer, method, group)`` work item.

    Folds in the run's config hash (model / data / seed / methods -> the experts
    and calibration inputs) and the work-item coordinates, plus an optional
    ``extra`` (e.g. the partition header hash). It deliberately does **not** read
    the (potentially large) expert weights: those are fully determined by the
    config, which is already in the hash. Stored in the artifact's provenance
    slot so :func:`is_fresh` can decide skip-vs-recompute.
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
    """True iff ``path`` exists and its provenance ``upstream_hash`` matches.

    The skip-if-done predicate: a fused artifact records its upstream hash
    in the serializer provenance slot, so a re-run with unchanged upstream finds
    a matching hash and skips the recompute. A missing file, a header without the
    hash, or any read error all read as *not fresh* (recompute), never a crash.
    """
    if not os.path.isfile(path):
        return False
    try:
        header = serialization.load_header(path)
    except Exception:
        return False
    provenance = header.get(serialization.PROVENANCE_FIELD, {})
    return provenance.get("upstream_hash") == upstream_hash
