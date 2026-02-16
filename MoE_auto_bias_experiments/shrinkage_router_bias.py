"""Router-bias denoising(shrinkage).

Replace z-normalization of expert-importance vectors.
Two denoising algo:

(A) Covariance shrinkage on per-sample importance matrix X_imp [N, E]:
    - Ledoit-Wolf style shrinkage to mu I
    - 'whitened mean' as bias direction

(B) Spike projection using the router gate weight W_gate [E, d]:
    - compute Gram G = (W W^T)/d
    - estimate a bulk edge (MP-inspired, robust via median eig)
    - project importance into the spike subspace; optionally Wiener-shrink.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import torch


def router_spike_basis(
    gate_weight: torch.Tensor,  # [E, d]
    mp_factor: float = 1.25,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    W = gate_weight.detach().float()
    E, d = W.shape
    G = (W @ W.t()) / float(d)
    eigs, vecs = torch.linalg.eigh(G)  # asc
    eigs = eigs.flip(0)
    vecs = vecs.flip(1)
    gamma = E / float(d)
    med = torch.median(eigs).item()
    edge = med * (1.0 + math.sqrt(max(gamma, 1e-8))) ** 2 * mp_factor
    mask = eigs > edge
    k = int(mask.sum().item())
    if k < 1:
        k = 1
        mask = torch.zeros_like(eigs, dtype=torch.bool)
        mask[0] = True
    Q = vecs[:, mask]  # [E, k]
    return Q, eigs, float(edge)


def denoise_importance(
    importance: torch.Tensor,      # [E]
    gate_weight: torch.Tensor,     # [E, d]
    mp_factor: float = 1.25,
    method: str = "wiener",
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Spike-projection denoising for an importance vector."""
    v = importance.detach().float().flatten()
    Q, eigs, edge = router_spike_basis(gate_weight, mp_factor=mp_factor)
    coords = Q.t() @ v
    if method == "none":
        coords_dn = coords
    elif method == "soft":
        tau = math.sqrt(edge)
        coords_dn = torch.sign(coords) * torch.clamp(coords.abs() - tau, min=0.0)
    else:
        lam = eigs[: Q.shape[1]]
        scale = torch.clamp((lam - edge) / (lam + eps), min=0.0, max=1.0)
        coords_dn = coords * scale
    v_dn = Q @ coords_dn
    return {
        "denoised": v_dn.to(importance.dtype),
        "proj": (Q @ coords).to(importance.dtype),
        "k": torch.tensor(Q.shape[1]),
        "edge": torch.tensor(edge),
        "eigs": eigs,
    }
