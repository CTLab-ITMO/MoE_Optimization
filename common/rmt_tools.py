from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import torch

def mp_bulk_edge_from_eigs(eigs: torch.Tensor, gamma: float, mp_factor: float = 1.0) -> float:
    """Heuristic MP-like bulk edge estimate from eigenvalues.

    For real MoE routers the i.i.d. assumptions do not hold. Use robust estimator:
    sigma^2 ~= median(eigs) edge of detection ~= sigma^2 * (1 + sqrt(gamma))^2
    and multiply by mp_factor (>=1) for conservatism.

    Args:
      eigs: eigenvalues of a covariance/Gram matrix (1D tensor, sorted or unsorted)
      gamma: aspect ratio (p/n). For Gram=[E x E] from weight [E x d], gamma = E/d.
    """
    eigs = eigs.detach().flatten()
    med = torch.median(eigs).item()
    edge = med * (1.0 + math.sqrt(max(gamma, 1e-8))) ** 2
    return float(mp_factor * edge)


def spike_projection_basis(gate_weight: torch.Tensor, mp_factor: float = 1.25) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute spike subspace basis from gate weights.

    Returns:
      Q: [E, k] eigenvectors for spikes
      eigs: [E] eigenvalues (sorted desc)
      edge: bulk edge threshold used
    """
    W = gate_weight.detach().float()  # [E, d]
    E, d = W.shape
    # Gram in expert space (E x E)
    G = (W @ W.t()) / float(d)
    eigs, vecs = torch.linalg.eigh(G)  # asc
    eigs = eigs.flip(0)
    vecs = vecs.flip(1)
    gamma = E / float(d)
    edge = mp_bulk_edge_from_eigs(eigs, gamma=gamma, mp_factor=mp_factor)
    mask = eigs > edge
    k = int(mask.sum().item())
    if k < 1:
        # Always keep at least 1 direction to avoid degenerate behaviour
        k = 1
        mask = torch.zeros_like(eigs, dtype=torch.bool)
        mask[0] = True
    Q = vecs[:, mask]  # [E, k]
    return Q, eigs, edge


def denoise_importance_spike_proj(
    importance: torch.Tensor,
    gate_weight: torch.Tensor,
    mp_factor: float = 1.25,
    shrink: str = "wiener",
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Denoise an expert-importance vector using spike projection in router space.

    Args:
      importance: [E] vector (any dtype)
      gate_weight: [E, d] router weight
      mp_factor: multiplier for bulk edge
      shrink: "none" | "wiener" | "soft"
    Returns:
      dict with fields:
        denoised: [E] vector
        proj: [E] projected vector
        k: scalar tensor, spike rank
        edge: scalar tensor
        eigs: [E] eigenvalues
    """
    v = importance.detach().float().flatten()
    Q, eigs, edge = spike_projection_basis(gate_weight, mp_factor=mp_factor)
    proj = Q @ (Q.t() @ v)

    if shrink == "none":
        den = proj
    elif shrink == "soft":
        # soft-threshold in projected coordinates
        coords = Q.t() @ v
        # tau tied to bulk edge
        tau = math.sqrt(edge)
        coords = torch.sign(coords) * torch.clamp(coords.abs() - tau, min=0.0)
        den = Q @ coords
    else:
        # Wiener-like shrinkage along spike eigendirections:
        # scale_i = max(0, (lambda_i - edge) / lambda_i)
        # This suppresses near-edge spikes.
        lam = eigs[: Q.shape[1]]
        scale = torch.clamp((lam - edge) / (lam + eps), min=0.0, max=1.0)
        coords = Q.t() @ v
        den = Q @ (coords * scale)

    return {
        "denoised": den.to(importance.dtype),
        "proj": proj.to(importance.dtype),
        "k": torch.tensor(Q.shape[1]),
        "edge": torch.tensor(edge),
        "eigs": eigs,
    }

@torch.no_grad()
def spike_filter_embeddings(Xe: torch.Tensor, mp_factor: float = 1.5) -> dict:
    """Spike-filter the expert embeddings by eigendecomposition of Xe Xe^T.

    Xe: [E, D] expert embeddings (float32)
    Returns:
      denoised embeddings, spike rank, bulk edge, eigenvalues.
    """
    E, D = Xe.shape
    G = (Xe @ Xe.t()) / float(D)
    eigs, vecs = torch.linalg.eigh(G)  # asc
    eigs = eigs.flip(0)
    vecs = vecs.flip(1)
    gamma = E / float(D)
    med = torch.median(eigs).item()
    edge = med * (1 + math.sqrt(max(gamma, 1e-8))) ** 2 * mp_factor
    mask = eigs > edge
    k = int(mask.sum().item())
    if k < 1:
        k = 1
        mask = torch.zeros_like(eigs, dtype=torch.bool)
        mask[0] = True
    Q = vecs[:, mask]  # [E, k]
    # denoised embedding in spike subspace
    coords = Q.t() @ Xe
    Xe_dn = (Q @ coords)
    return {"Xe_dn": Xe_dn, "k": k, "edge": edge, "eigs": eigs}