from __future__ import annotations

from typing import Dict
import torch

def covariance_shrinkage(
    X: torch.Tensor,  # [N, E]
    alpha: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a shrunk covariance estimate.

    Sigma_hat = (1-alpha) S + alpha * mu I
    where mu = trace(S)/E and S is sample covariance.
    """
    X = X.detach().float()
    X = X - X.mean(dim=0, keepdim=True)
    N, E = X.shape
    S = (X.t() @ X) / max(N - 1, 1)
    mu = torch.trace(S) / float(E)
    Sigma = (1 - alpha) * S + alpha * mu * torch.eye(E, device=S.device, dtype=S.dtype)
    # jitter
    Sigma = Sigma + eps * torch.eye(E, device=S.device, dtype=S.dtype)
    return Sigma

def whitened_mean_bias(
    X: torch.Tensor,  # [N, E]
    alpha: float = 0.2,
    lam: float = 1e-3,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute a denoised bias direction from per-sample importance.

    b_raw = (Sigma_hat + lam I)^{-1} * mean
    """
    Xf = X.detach().float()
    m = Xf.mean(dim=0)  # [E]
    Sigma = covariance_shrinkage(Xf, alpha=alpha, eps=eps)
    E = Sigma.shape[0]
    A = Sigma + lam * torch.eye(E, device=Sigma.device, dtype=Sigma.dtype)
    b = torch.linalg.solve(A, m.unsqueeze(1)).squeeze(1)
    return {"bias": b, "mean": m, "Sigma": Sigma}
