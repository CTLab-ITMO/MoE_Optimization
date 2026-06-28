"""Shared numerical constants and device fallback for the fusion pipeline.

Single source of truth for the small numerical knobs that used to be duplicated
across modules:

* ``EPS`` -- the numerical-stability epsilon guarding divisions, logs and norm
  denominators (was a bare ``1e-8`` in :mod:`common.metrics`, :mod:`common.pairing`
  and :mod:`common.fusion`);
* ``_default_device()`` -- the ``cuda``-if-available device fallback that used to
  be re-derived as ``torch.device("cuda" if ... else "cpu")`` in
  :mod:`common.mlp`, :mod:`common.mlp_utils` and :mod:`common.pairing`;
* ``solve_linear()`` -- the ridge least-squares solver shared by the
  least-squares unmerge refit (:meth:`common.mlp.AdvancedMLP.refit_unmerge`) and
  re-exported from :mod:`common.fusion` for the ``LS_*`` adapter. It lives here
  (not in :mod:`common.fusion`) so :mod:`common.mlp` can use it without importing
  :mod:`common.fusion` -- which would be a circular import (fusion imports mlp).

Library code only -- this module never imports an experiment ``config``.
"""

import torch

# Numerical-stability epsilon guarding divisions, logs and norm denominators.
EPS = 1e-8


def _default_device() -> torch.device:
    """Return the CUDA device when available, otherwise CPU.

    Evaluated at call time so callers see the current CUDA availability (tests
    patch ``torch.cuda.is_available`` before the pipeline runs).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solve_linear(x: torch.Tensor, y: torch.Tensor, lambda_ridge: float = 1e-3) -> torch.Tensor:
    """Ridge least-squares: return W minimizing ``||x W^T - y|| (+ l2)``."""
    d_in = x.shape[1]
    xtx = x.T @ x + lambda_ridge * torch.eye(d_in, device=x.device, dtype=x.dtype)
    rhs = x.T @ y
    return torch.linalg.solve(xtx, rhs).T
