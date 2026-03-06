from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

import torch


@dataclass
class H3PipelineResult:
    topk: torch.Tensor
    routing_error_budget: torch.Tensor
    diagnostics: Dict[str, float]


@torch.no_grad()
def dynamic_topk_cumprob(
    probs: torch.Tensor,
    tau: float,
    k_min: int,
    k_max: int,
) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs shape [T,E], got {tuple(probs.shape)}")
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    k_max = min(k_max, probs.shape[1])
    csum = torch.cumsum(sorted_probs[:, :k_max], dim=-1)
    k = (csum < tau).sum(dim=-1) + 1
    return torch.clamp(k, min=k_min, max=k_max)


@torch.no_grad()
def routing_error_budget(probs: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    csum = torch.cumsum(sorted_probs, dim=-1)
    idx = (k - 1).clamp(min=0).unsqueeze(1)
    captured = torch.gather(csum, 1, idx).squeeze(1)
    return 1.0 - captured


@torch.no_grad()
def load_balance_coefficient_of_variation(probs: torch.Tensor) -> float:
    usage = probs.mean(dim=0)
    mean = usage.mean().item()
    std = usage.std().item()
    if mean <= 1e-12:
        return 0.0
    return float(std / mean)


@torch.no_grad()
def run_h3_pipeline(
    routing_probs_path: str,
    output_diagnostics_path: str,
    tau: float,
    k_min: int,
    k_max: int,
) -> H3PipelineResult:
    probs = torch.load(routing_probs_path, map_location="cpu")
    if probs.ndim != 2:
        raise ValueError(f"Routing probs must be 2D [T,E], got {tuple(probs.shape)}")
    probs = probs.float()
    if probs.min().item() < 0.0:
        probs = torch.softmax(probs, dim=-1)

    k = dynamic_topk_cumprob(probs, tau=tau, k_min=k_min, k_max=k_max)
    delta = routing_error_budget(probs, k)
    diag = {
        "tau": float(tau),
        "k_mean": float(k.float().mean().item()),
        "k_min": float(k.min().item()),
        "k_max": float(k.max().item()),
        "delta_mean": float(delta.mean().item()),
        "delta_max": float(delta.max().item()),
        "load_balance_cv": load_balance_coefficient_of_variation(probs),
    }

    out = Path(output_diagnostics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    return H3PipelineResult(topk=k, routing_error_budget=delta, diagnostics=diag)
