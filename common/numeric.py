"""Shared numerical constants and utilities for the fusion pipeline.

* ``EPS`` -- stability epsilon for divisions, logs, and norm denominators;
* ``_default_device()`` -- returns CUDA if available, else CPU;
* ``solve_linear()`` -- ridge least-squares solver used by both mlp and fusion.
  Lives here (not in fusion) to avoid a circular import: fusion imports mlp.

Never imports an experiment config.
"""

import torch

# Numerical-stability epsilon guarding divisions, logs and norm denominators.
EPS = 1e-8


def _default_device() -> torch.device:
    """Return CUDA device if available, else CPU.

    Called at runtime (not import time) so tests can patch ``torch.cuda.is_available``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solve_linear(x: torch.Tensor, y: torch.Tensor, lambda_ridge: float = 1e-3) -> torch.Tensor:
    """Ridge least-squares: return W minimizing ``||x W^T - y|| (+ l2)``."""
    d_in = x.shape[1]
    xtx = x.T @ x + lambda_ridge * torch.eye(d_in, device=x.device, dtype=x.dtype)
    rhs = x.T @ y
    return torch.linalg.solve(xtx, rhs).T
