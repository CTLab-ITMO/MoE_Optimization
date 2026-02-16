import math

import torch
import torch.nn as nn

def _random_omega(n: int, proj_dim: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # Rademacher is cheap and stable
    omega = torch.empty((n, proj_dim), device=device)
    omega.bernoulli_(0.5, generator=g)
    omega = omega * 2 - 1
    omega = omega / math.sqrt(float(n))
    return omega
@torch.no_grad()
def expert_functional_sketch(
    expert: nn.Module,
    X: torch.Tensor,          # [N, d]
    omega: torch.Tensor,      # [N, p]
    batch: int,) -> torch.Tensor:
    """Compute sketch A = Y^T * Omega, where Y = expert(X), without storing Y fully.

    Returns:
      vec(A) with shape [d * p] (float32)
    """
    N, d = X.shape
    p = omega.shape[1]
    A = torch.zeros((d, p), device=X.device, dtype=torch.float32)
    for s in range(0, N, batch):
        xb = X[s : s + batch]
        ob = omega[s : s + batch]
        yb = expert(xb)  # expected [B, d]
        if isinstance(yb, tuple):
            yb = yb[0]
        yb = yb.to(torch.float32)
        # accumulate Y^T * Omega
        A += yb.t() @ ob.to(torch.float32)
    return A.reshape(-1)