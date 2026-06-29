"""Determinism and seeding for the fusion pipeline.

Two layers:

* :func:`set_seed` -- the base, global seeding kept for backward compatibility.
* :func:`seed_for` / :func:`generator_for` -- *derived local seeds per unit of
  work*. A stochastic operation seeds its own :class:`torch.Generator` from
  ``hash(SEED, stage, layer, group, method, ...)`` instead of consuming the
  global RNG stream. Because the seed depends only on the *coordinates* of the
  work item (and the base seed), not on how many random draws happened before
  it, the result is **invariant to the traversal order** of the pipeline. That
  avoids the trap where ``set_seed`` once-per-stage plus random draws inside
  nested loops would make every stochastic method order-dependent.

The hash is :mod:`hashlib`-based on purpose: Python's builtin ``hash()`` is
salted per process (``PYTHONHASHSEED``) and would make seeds non-reproducible
across runs.

Library code only -- this module never imports an experiment ``config``.
"""

import hashlib
import random

import numpy as np
import torch

# torch.Generator.manual_seed accepts a non-negative 64-bit int. We mask the
# digest to 63 bits so the value is always a positive, in-range Python int.
_SEED_MASK = (1 << 63) - 1


def set_seed(seed: int) -> None:
    """Seed the global Python / NumPy / torch RNGs (and CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for(*, seed=None, stage: str, **coords) -> int:
    """Derive a stable, order-independent seed for one unit of work.

    ``seed`` is the run-level base seed (e.g. ``config.SEED``); ``stage`` names
    the consumer (e.g. ``"fusion"``, ``"pairing"``); ``**coords`` are the
    coordinates of the work item (``layer=...``, ``group=...``, ``method=...``).
    The same ``(seed, stage, coords)`` always maps to the same value, and any
    change in any coordinate changes it -- so distinct work items get distinct,
    reproducible seeds regardless of the order in which they are processed.

    ``None`` coordinates are skipped, so ``seed_for(seed=1, stage="fusion",
    method="Fisher")`` and ``seed_for(seed=1, stage="fusion", layer=None,
    method="Fisher")`` agree.
    """
    parts = [f"seed={seed}", f"stage={stage}"]
    parts += [f"{key}={coords[key]}" for key in sorted(coords) if coords[key] is not None]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


def generator_for(
    *, seed=None, stage: str, device=None, **coords
) -> torch.Generator:
    """Return a :class:`torch.Generator` seeded by :func:`seed_for`.

    Pass the result as the ``generator=`` argument of a torch sampler
    (``torch.randn``, ``torch.randperm``, ...) to make that draw deterministic
    and order-independent. ``device`` defaults to CPU; it must match the device
    the sampler draws on.
    """
    generator = torch.Generator(device=device if device is not None else "cpu")
    generator.manual_seed(seed_for(seed=seed, stage=stage, **coords))
    return generator
