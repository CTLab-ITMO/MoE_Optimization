from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

from common.projection_tools import build_functional_sketch
from common.rmt_tools import spike_denoised_embedding


@dataclass
class H1PipelineResult:
    pairs: List[Tuple[int, int]]
    clusters: List[List[int]]
    diagnostics: Dict[str, object]


@torch.no_grad()
def collect_mlp_inputs(path: str | Path, n_inputs: int, device: str = "cpu") -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MLP inputs not found: {p}")
    raw = torch.load(p, map_location="cpu")
    if isinstance(raw, list):
        if len(raw) == 0:
            raise ValueError("MLP inputs list is empty")
        x = torch.stack([t for t in raw[:n_inputs]], dim=0)
    elif isinstance(raw, torch.Tensor):
        x = raw[:n_inputs]
    else:
        raise ValueError(f"Unsupported mlp input container: {type(raw)}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D inputs [N,d], got shape {tuple(x.shape)}")
    return x.to(device)


@torch.no_grad()
def _cosine_sim_matrix(emb: torch.Tensor) -> torch.Tensor:
    n = emb / torch.linalg.norm(emb, dim=1, keepdim=True).clamp_min(1e-12)
    return n @ n.t()


@torch.no_grad()
def greedy_pairs_from_similarity(sim: torch.Tensor) -> List[Tuple[int, int]]:
    n = sim.shape[0]
    used = torch.zeros(n, dtype=torch.bool, device=sim.device)
    s = sim.clone()
    s.fill_diagonal_(-1e9)
    out: List[Tuple[int, int]] = []
    for _ in range(n // 2):
        masked = s.clone()
        masked[used, :] = -1e9
        masked[:, used] = -1e9
        flat = int(torch.argmax(masked).item())
        i = flat // n
        j = flat % n
        if masked[i, j].item() < -1e8:
            break
        used[i] = True
        used[j] = True
        out.append((i, j))
    return out


@torch.no_grad()
def _kmeans_torch(x: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    if k <= 1:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    idx = torch.randperm(x.shape[0], device=x.device)[:k]
    centers = x[idx].clone()
    labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = torch.cdist(x, centers)
        labels = torch.argmin(dist, dim=1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = x[mask].mean(dim=0)
    return labels


@torch.no_grad()
def pair_or_cluster(embedding: torch.Tensor, mode: str = "pair", cluster_count: int = 0) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    if mode == "pair":
        pairs = greedy_pairs_from_similarity(_cosine_sim_matrix(embedding))
        return pairs, []

    k = cluster_count if cluster_count > 0 else max(1, min(embedding.shape[0], embedding.shape[1]))
    labels = _kmeans_torch(embedding, k=k)
    clusters: List[List[int]] = []
    for c in range(k):
        members = torch.where(labels == c)[0].tolist()
        if members:
            clusters.append(members)
    return [], clusters


@torch.no_grad()
def _projection_mean(u_list: Sequence[torch.Tensor], rank: int) -> torch.Tensor:
    p = torch.zeros((u_list[0].shape[0], u_list[0].shape[0]), dtype=u_list[0].dtype, device=u_list[0].device)
    for u in u_list:
        p = p + u @ u.t()
    p = p / float(len(u_list))
    evals, evecs = torch.linalg.eigh(p)
    return evecs[:, -rank:]


@torch.no_grad()
def merge_weight_matrices_baseline(weights: Sequence[torch.Tensor], usage: Sequence[float] | None = None) -> torch.Tensor:
    if not weights:
        raise ValueError("No weights to merge")
    if usage is None:
        usage = [1.0 / len(weights)] * len(weights)
    out = torch.zeros_like(weights[0])
    for a, w in zip(usage, weights):
        out = out + float(a) * w
    return out


@torch.no_grad()
def merge_weight_matrices_geometry(
    weights: Sequence[torch.Tensor],
    usage: Sequence[float] | None = None,
    rank: int = 64,
    shrink_tau: float = 0.0,
) -> torch.Tensor:
    if not weights:
        raise ValueError("No weights to merge")
    if usage is None:
        usage = [1.0 / len(weights)] * len(weights)
    max_rank = min(weights[0].shape)
    rank = max(1, min(rank, max_rank))

    u_list: List[torch.Tensor] = []
    for w in weights:
        u, _, _ = torch.linalg.svd(w, full_matrices=False)
        u_list.append(u[:, :rank])

    u_star = _projection_mean(u_list, rank=rank)
    merged = torch.zeros_like(weights[0])
    for a, w in zip(usage, weights):
        wp = u_star @ (u_star.t() @ w)
        merged = merged + float(a) * wp

    if shrink_tau > 0:
        u, s, vt = torch.linalg.svd(merged, full_matrices=False)
        s2 = torch.clamp(s - shrink_tau, min=0.0)
        merged = (u * s2) @ vt
    return merged


@torch.no_grad()
def _merge_param_tensor(tensors: Sequence[torch.Tensor], method: str, rank: int, shrink_tau: float) -> torch.Tensor:
    if tensors[0].ndim < 2:
        return merge_weight_matrices_baseline([t.detach() for t in tensors])
    mats = [t.detach().reshape(t.shape[0], -1) for t in tensors]
    if method == "baseline":
        merged = merge_weight_matrices_baseline(mats)
    else:
        merged = merge_weight_matrices_geometry(mats, rank=rank, shrink_tau=shrink_tau)
    return merged.reshape(tensors[0].shape)


@torch.no_grad()
def merge_cluster_modules(
    experts: Sequence[nn.Module],
    cluster: Sequence[int],
    method: str,
    rank: int,
    shrink_tau: float,
) -> Dict[str, torch.Tensor]:
    if not cluster:
        raise ValueError("Cluster is empty")
    state_dicts = [experts[i].state_dict() for i in cluster]
    keys = list(state_dicts[0].keys())
    merged: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        merged[key] = _merge_param_tensor(tensors, method=method, rank=rank, shrink_tau=shrink_tau)
    return merged


@torch.no_grad()
def run_h1_with_experts(
    experts: Sequence[nn.Module],
    mlp_inputs: torch.Tensor,
    mode: str = "pair",
    proj_dim: int = 8,
    batch_size: int = 64,
    seed: int = 123,
    mp_factor: float = 1.1,
    cluster_count: int = 0,
) -> H1PipelineResult:
    x = build_functional_sketch(
        list(experts),
        mlp_inputs=mlp_inputs,
        proj_dim=proj_dim,
        batch_size=batch_size,
        seed=seed,
        device=str(mlp_inputs.device),
    )
    emb, info = spike_denoised_embedding(x, mp_factor=mp_factor, min_rank=1)
    pairs, clusters = pair_or_cluster(emb, mode=mode, cluster_count=cluster_count)
    diagnostics = {
        "rank": int(info["rank"]),
        "tau": float(info["tau"]),
        "lambda_plus": float(info["lambda_plus"]),
        "eigvals": info["eigvals"].tolist(),
    }
    return H1PipelineResult(pairs=pairs, clusters=clusters, diagnostics=diagnostics)


@torch.no_grad()
def run_h1_pipeline(
    experts: Sequence[nn.Module],
    mlp_inputs_path: str,
    n_inputs: int,
    mode: str,
    proj_dim: int,
    batch_size: int,
    seed: int,
    mp_factor: float,
    cluster_count: int,
    output_pairs_path: str,
    output_diagnostics_path: str,
) -> H1PipelineResult:
    mlp_inputs = collect_mlp_inputs(mlp_inputs_path, n_inputs=n_inputs, device="cpu")
    result = run_h1_with_experts(
        experts=experts,
        mlp_inputs=mlp_inputs,
        mode=mode,
        proj_dim=proj_dim,
        batch_size=batch_size,
        seed=seed,
        mp_factor=mp_factor,
        cluster_count=cluster_count,
    )

    pairs_path = Path(output_pairs_path)
    diag_path = Path(output_diagnostics_path)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(result.pairs, pairs_path)
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(result.diagnostics, f, indent=2)
    return result


