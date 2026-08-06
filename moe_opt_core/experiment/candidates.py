from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import torch

from moe_opt_core.experiment.contracts import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
)


def _pair_count(num_experts: int, compression_fraction: float) -> int:
    requested = max(1, int(math.ceil(num_experts * compression_fraction)))
    return min(num_experts // 2, requested)


def _partition_from_pairs(num_experts: int, pairs: Iterable[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    used: set[int] = set()
    for raw_pair in pairs:
        pair = tuple(int(idx) for idx in raw_pair)
        if len(pair) != 2 or pair[0] == pair[1]:
            continue
        if any(idx < 0 or idx >= num_experts or idx in used for idx in pair):
            continue
        groups.append(pair)
        used.update(pair)
    groups.extend((idx,) for idx in range(num_experts) if idx not in used)
    return tuple(groups)


def _random_pairs(num_experts: int, count: int, seed: int) -> tuple[tuple[int, int], ...]:
    indices = list(range(num_experts))
    random.Random(seed).shuffle(indices)
    return tuple((indices[2 * idx], indices[2 * idx + 1]) for idx in range(count))


def _greedy_pairs_from_score(score: torch.Tensor, count: int) -> tuple[tuple[int, int], ...]:
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("pair score must be a square matrix")
    score = score.float().clone()
    score.fill_diagonal_(-torch.inf)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        masked = score.clone()
        if used:
            idx = torch.tensor(sorted(used), dtype=torch.long, device=masked.device)
            masked[idx, :] = -torch.inf
            masked[:, idx] = -torch.inf
        flat = int(torch.argmax(masked).item())
        first, second = divmod(flat, masked.shape[1])
        if not torch.isfinite(masked[first, second]):
            break
        pairs.append((first, second))
        used.update((first, second))
    return tuple(pairs)


def _cosine_similarity(embedding: torch.Tensor) -> torch.Tensor:
    normalized = embedding.float()
    normalized = normalized / torch.linalg.norm(normalized, dim=1, keepdim=True).clamp_min(1e-12)
    return (normalized @ normalized.t() + 1.0) * 0.5


def _radial_similarity(embedding: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(embedding.float(), embedding.float())
    off_diagonal = distances[~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)]
    positive = off_diagonal[off_diagonal > 1e-12]
    scale = torch.median(positive) if positive.numel() else distances.new_tensor(1.0)
    return torch.exp(-distances / scale.clamp_min(1e-12))


def _usage_safety_similarity(usage: torch.Tensor) -> torch.Tensor:
    normalized = usage.float() / usage.float().max().clamp_min(1e-12)
    return 1.0 - 0.5 * (normalized.unsqueeze(0) + normalized.unsqueeze(1))


def _greedy_cosine_pairs(embedding: torch.Tensor, count: int) -> tuple[tuple[int, int], ...]:
    return _greedy_pairs_from_score(_cosine_similarity(embedding), count)


def _hybrid_pairs(
    spectral_embedding: torch.Tensor,
    functional_embedding: torch.Tensor,
    usage: torch.Tensor,
    count: int,
    weights: Sequence[float],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[str, float], ...]]:
    if len(weights) != 3 or any(float(weight) < 0.0 for weight in weights):
        raise ValueError("hybrid_weights must contain three non-negative values")
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        raise ValueError("hybrid_weights must contain at least one positive value")
    num_experts = functional_embedding.shape[0]
    if spectral_embedding.shape[0] != num_experts or usage.numel() != num_experts:
        raise ValueError("hybrid pairing inputs must use the same expert dimension")
    rmt_weight, functional_weight, usage_weight = (float(weight) / total for weight in weights)
    rmt_similarity = _radial_similarity(spectral_embedding)
    functional_similarity = _cosine_similarity(functional_embedding)
    usage_safety = _usage_safety_similarity(usage)
    score = (
        rmt_weight * rmt_similarity
        + functional_weight * functional_similarity
        + usage_weight * usage_safety
    )
    pairs = _greedy_pairs_from_score(score, count)

    def selected_mean(matrix: torch.Tensor) -> float:
        if not pairs:
            return 0.0
        values = torch.stack([matrix[first, second] for first, second in pairs])
        return float(values.mean().item())

    diagnostics = (
        ("mean_hybrid_score", selected_mean(score)),
        ("mean_rmt_similarity", selected_mean(rmt_similarity)),
        ("mean_functional_similarity", selected_mean(functional_similarity)),
        ("mean_usage_safety", selected_mean(usage_safety)),
    )
    return pairs, diagnostics


def validate_candidate_plan(plan: CandidatePlanContract, num_experts: int) -> None:
    flattened = [idx for group in plan.groups for idx in group]
    if any(idx < 0 or idx >= num_experts for idx in flattened):
        raise ValueError(f"Candidate {plan.candidate_id} contains an invalid expert index")
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"Candidate {plan.candidate_id} assigns an expert more than once")
    if plan.action == CandidateAction.MERGE and set(flattened) != set(range(num_experts)):
        raise ValueError(f"Merge candidate {plan.candidate_id} must cover every expert")
    if not plan.groups:
        raise ValueError(f"Candidate {plan.candidate_id} has no active expert groups")


def build_candidate_plans(
    *,
    num_experts: int,
    layer: int,
    methods: Sequence[str],
    compression_fractions: Sequence[float],
    seeds: Sequence[int],
    rmt_pairs: Sequence[Sequence[int]],
    spectral_embedding: torch.Tensor,
    functional_embedding: torch.Tensor,
    routing_probs: torch.Tensor,
    hybrid_weights: Sequence[float] = (0.5, 0.35, 0.15),
) -> tuple[CandidatePlanContract, ...]:
    plans: list[CandidatePlanContract] = []
    usage = routing_probs.float().mean(dim=0)
    if usage.numel() != num_experts:
        raise ValueError("routing_probs expert dimension does not match the model")

    for method_token in methods:
        method = CandidateMethod(method_token)
        method_seeds = tuple(seeds) if method == CandidateMethod.RANDOM_MERGE else (int(seeds[0]),)
        for fraction in compression_fractions:
            count = _pair_count(num_experts, float(fraction))
            for seed in method_seeds:
                diagnostics: tuple[tuple[str, float], ...] = ()
                if method == CandidateMethod.RANDOM_MERGE:
                    groups = _partition_from_pairs(num_experts, _random_pairs(num_experts, count, seed))
                    action = CandidateAction.MERGE
                elif method == CandidateMethod.FUNCTIONAL_COSINE:
                    groups = _partition_from_pairs(
                        num_experts,
                        _greedy_cosine_pairs(functional_embedding, count),
                    )
                    action = CandidateAction.MERGE
                elif method == CandidateMethod.RMT_GEOMETRY:
                    groups = _partition_from_pairs(num_experts, tuple(rmt_pairs)[:count])
                    action = CandidateAction.MERGE
                elif method == CandidateMethod.RMT_FUNCTIONAL_USAGE:
                    hybrid_pairs, diagnostics = _hybrid_pairs(
                        spectral_embedding,
                        functional_embedding,
                        usage,
                        count,
                        hybrid_weights,
                    )
                    groups = _partition_from_pairs(
                        num_experts,
                        hybrid_pairs,
                    )
                    action = CandidateAction.MERGE
                else:
                    prune_count = min(num_experts - 1, max(1, int(math.ceil(num_experts * float(fraction)))))
                    pruned = set(torch.argsort(usage)[:prune_count].tolist())
                    groups = tuple((idx,) for idx in range(num_experts) if idx not in pruned)
                    action = CandidateAction.PRUNE

                plan = CandidatePlanContract(
                    candidate_id=f"{method.value}-f{float(fraction):.3f}-s{int(seed)}",
                    layer=int(layer),
                    method=method,
                    action=action,
                    compression_fraction=float(fraction),
                    groups=groups,
                    seed=int(seed),
                    diagnostics=diagnostics,
                )
                validate_candidate_plan(plan, num_experts=num_experts)
                plans.append(plan)
    return tuple(plans)
