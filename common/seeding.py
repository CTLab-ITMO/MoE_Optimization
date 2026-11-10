"""Deterministic seeding utilities for the fusion pipeline.

Two helpers on top of the global ``set_seed``:

* ``seed_for`` — derives a stable integer seed from ``(base_seed, stage, **coords)``
  using SHA-256, so the result is invariant to traversal order. SHA-256 (not
  Python's ``hash()``) ensures the same value across processes regardless of
  ``PYTHONHASHSEED``.
* ``generator_for`` — wraps ``seed_for`` in a ``torch.Generator`` ready to pass
  to torch samplers.

Library code: never imports experiment ``config``.
"""

import hashlib
import random

import numpy as np
import torch

# Mask to 63 bits: torch.Generator.manual_seed requires a non-negative 64-bit int.
_SEED_MASK = (1 << 63) - 1


def set_seed(seed: int) -> None:
    """Seed the global Python / NumPy / torch RNGs (and CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for(*, seed=None, stage: str, **coords) -> int:
    """Derive a stable integer seed for one unit of work.

    Hashes ``(seed, stage, **coords)`` with SHA-256. The same arguments always
    produce the same value regardless of call order. ``None`` coords are skipped,
    so a missing coord and an explicit ``None`` coord are equivalent.
    """
    parts = [f"seed={seed}", f"stage={stage}"]
    parts += [f"{key}={coords[key]}" for key in sorted(coords) if coords[key] is not None]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


def generator_for(
    *, seed=None, stage: str, device=None, **coords
) -> torch.Generator:
    """Return a ``torch.Generator`` seeded by ``seed_for``.

    Pass as ``generator=`` to torch samplers to get deterministic, order-independent
    draws. ``device`` defaults to CPU and must match the sampler's device.
    """
    generator = torch.Generator(device=device if device is not None else "cpu")
    generator.manual_seed(seed_for(seed=seed, stage=stage, **coords))
    return generator
