import math

import torch
import torch.nn as nn
from tqdm import tqdm


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
    batch: int,
) -> torch.Tensor:
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


@torch.no_grad()
def build_functional_sketch(
    experts,  # ModuleList
    mlp_inputs: torch.Tensor,  # [N, d]
    proj_dim: int = 8,
    batch_size: int = 64,
    seed: int = 123,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Return X of shape [E, N*proj_dim] where row e is a random-projection sketch."""
    device = torch.device(device)
    E = len(experts)
    N, d = mlp_inputs.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn((d, proj_dim), generator=g, dtype=torch.float32) / (d**0.5)

    X = torch.empty((E, N * proj_dim), dtype=torch.float32)
    for e_idx in tqdm(range(E), desc="Sketch experts"):
        expert = experts[e_idx]
        try:
            first_parameter = next(expert.parameters())
            original_device = first_parameter.device
            expert_dtype = first_parameter.dtype
        except StopIteration:
            original_device = torch.device("cpu")
            expert_dtype = torch.float32
        execution_device = original_device if original_device.type != "meta" else device
        moved = execution_device != original_device
        if moved:
            expert = expert.to(execution_device)
        R_device = R.to(execution_device)
        cursor = 0
        for s in range(0, N, batch_size):
            x = mlp_inputs[s : s + batch_size].to(device=execution_device, dtype=expert_dtype)
            y = expert(x)  # [B, d]
            y = y.float() @ R_device  # [B, proj_dim]
            b = y.shape[0]
            X[e_idx, cursor : cursor + b * proj_dim] = y.reshape(-1).cpu()
            cursor += b * proj_dim
        if moved:
            experts[e_idx] = expert.to(original_device)

    # center each feature dimension across experts
    X = X - X.mean(dim=0, keepdim=True)
    return X
