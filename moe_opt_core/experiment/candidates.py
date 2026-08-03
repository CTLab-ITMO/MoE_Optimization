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


def _greedy_cosine_pairs(embedding: torch.Tensor, count: int) -> tuple[tuple[int, int], ...]:
    normalized = embedding.float()
    normalized = normalized / torch.linalg.norm(normalized, dim=1, keepdim=True).clamp_min(1e-12)
    similarity = normalized @ normalized.t()
    similarity.fill_diagonal_(-torch.inf)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        masked = similarity.clone()
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
    functional_embedding: torch.Tensor,
    routing_probs: torch.Tensor,
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
                )
                validate_candidate_plan(plan, num_experts=num_experts)
                plans.append(plan)
    return tuple(plans)
