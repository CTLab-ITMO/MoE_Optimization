from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, Sequence

import torch

from moe_opt_core.experiment.contracts import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
    MergeWeightingPolicy,
)
from moe_opt_core.experiment.merge_weighting import (
    build_group_merge_weights,
    functional_ridge_token_aligned,
    merge_weight_diagnostics,
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


def _usage_guided_pairs(
    functional_embedding: torch.Tensor,
    usage: torch.Tensor,
    count: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[str, float], ...]]:
    num_experts = functional_embedding.shape[0]
    if usage.numel() != num_experts:
        raise ValueError("usage-guided pairing inputs must use the same expert dimension")
    ordered = sorted(range(num_experts), key=lambda idx: (float(usage[idx].item()), idx))
    sources = tuple(ordered[:count])
    available_partners = set(ordered[count:])
    similarity = _cosine_similarity(functional_embedding)
    pairs: list[tuple[int, int]] = []
    for source in sources:
        if not available_partners:
            break
        partner = max(
            available_partners,
            key=lambda idx: (float(similarity[source, idx].item()), -idx),
        )
        pairs.append((source, partner))
        available_partners.remove(partner)

    if pairs:
        source_usage = torch.stack([usage[source] for source, _ in pairs]).mean()
        partner_usage = torch.stack([usage[partner] for _, partner in pairs]).mean()
        pair_similarity = torch.stack(
            [similarity[source, partner] for source, partner in pairs]
        ).mean()
    else:
        source_usage = usage.new_tensor(0.0)
        partner_usage = usage.new_tensor(0.0)
        pair_similarity = usage.new_tensor(0.0)
    diagnostics = (
        ("mean_source_usage", float(source_usage.item())),
        ("mean_partner_usage", float(partner_usage.item())),
        ("mean_functional_similarity", float(pair_similarity.item())),
    )
    return tuple(pairs), diagnostics


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
    if plan.group_merge_weights:
        if plan.action != CandidateAction.MERGE:
            raise ValueError("Only merge candidates may define group merge weights")
        if len(plan.group_merge_weights) != len(plan.groups):
            raise ValueError("Group merge weights must align with candidate groups")
        for group, weights in zip(plan.groups, plan.group_merge_weights):
            if len(weights) != len(group):
                raise ValueError("Each merge weight vector must align with its expert group")
            if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
                raise ValueError("Group merge weights must be finite and non-negative")
            if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Each group merge weight vector must sum to one")
    elif plan.merge_weighting != MergeWeightingPolicy.UNIFORM:
        raise ValueError("Non-uniform merge weighting requires explicit group weights")


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
    spectral_embeddings: Mapping[int, torch.Tensor] | None = None,
    hybrid_weight_variants: Sequence[tuple[str, Sequence[float]]] = (),
    spectral_rank_schedule: Sequence[tuple[str, float, int]] = (),
    merge_weighting_methods: Sequence[str] = (),
    merge_weighting_variants: Sequence[tuple] = (),
) -> tuple[CandidatePlanContract, ...]:
    plans: list[CandidatePlanContract] = []
    usage = routing_probs.float().mean(dim=0)
    if usage.numel() != num_experts:
        raise ValueError("routing_probs expert dimension does not match the model")
    usage_values = tuple(float(value) for value in usage.detach().cpu().tolist())

    explicit_rank_variants = tuple(
        (int(rank), value) for rank, value in (spectral_embeddings or {}).items()
    )
    for rank, value in explicit_rank_variants:
        if value.ndim != 2 or value.shape != (num_experts, rank):
            raise ValueError(
                "spectral rank embedding must have shape "
                f"({num_experts}, {rank}), got {tuple(value.shape)}"
            )
    explicit_weight_variants = tuple(
        (str(name), tuple(float(weight) for weight in weights))
        for name, weights in hybrid_weight_variants
    )
    explicit_rank_schedule = tuple(
        (CandidateMethod(method), float(fraction), int(rank))
        for method, fraction, rank in spectral_rank_schedule
    )
    weighted_methods = {CandidateMethod(method) for method in merge_weighting_methods}
    explicit_merge_weighting_variants = tuple(
        (
            str(item[0]),
            MergeWeightingPolicy(item[1]),
            float(item[2]),
            float(item[3]) if len(item) > 3 else 0.0,
        )
        for item in merge_weighting_variants
    )
    if weighted_methods and not explicit_merge_weighting_variants:
        raise ValueError("merge weighting methods require at least one weighting variant")
    if explicit_merge_weighting_variants and not weighted_methods:
        raise ValueError("merge weighting variants require at least one target method")

    for method_token in methods:
        method = CandidateMethod(method_token)
        method_seeds = tuple(seeds) if method == CandidateMethod.RANDOM_MERGE else (int(seeds[0]),)
        weight_variants = (
            explicit_weight_variants
            if explicit_weight_variants and method == CandidateMethod.RMT_FUNCTIONAL_USAGE
            else ((None, tuple(float(weight) for weight in hybrid_weights)),)
        )
        for fraction in compression_fractions:
            count = _pair_count(num_experts, float(fraction))
            rank_variants = (
                explicit_rank_variants
                if explicit_rank_variants
                and method
                in {CandidateMethod.RMT_GEOMETRY, CandidateMethod.RMT_FUNCTIONAL_USAGE}
                else ((None, spectral_embedding),)
            )
            scheduled_ranks = tuple(
                rank
                for scheduled_method, scheduled_fraction, rank in explicit_rank_schedule
                if scheduled_method == method
                and math.isclose(
                    scheduled_fraction,
                    float(fraction),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            method_has_schedule = any(
                scheduled_method == method
                for scheduled_method, _, _ in explicit_rank_schedule
            )
            if method_has_schedule:
                if len(scheduled_ranks) != 1:
                    raise ValueError(
                        "spectral rank schedule must contain one rank per method and fraction"
                    )
                scheduled_rank = scheduled_ranks[0]
                matching_embeddings = tuple(
                    item for item in explicit_rank_variants if item[0] == scheduled_rank
                )
                if len(matching_embeddings) != 1:
                    raise ValueError("scheduled spectral rank has no matching embedding")
                rank_variants = matching_embeddings
            for rank, ranked_embedding in rank_variants:
                for weight_name, variant_weights in weight_variants:
                    for seed in method_seeds:
                        diagnostics: tuple[tuple[str, float], ...] = ()
                        if method == CandidateMethod.RANDOM_MERGE:
                            groups = _partition_from_pairs(
                                num_experts,
                                _random_pairs(num_experts, count, seed),
                            )
                            action = CandidateAction.MERGE
                        elif method == CandidateMethod.USAGE_PRUNE:
                            prune_count = min(
                                num_experts - 1,
                                max(1, int(math.ceil(num_experts * float(fraction)))),
                            )
                            pruned = set(torch.argsort(usage)[:prune_count].tolist())
                            groups = tuple((idx,) for idx in range(num_experts) if idx not in pruned)
                            action = CandidateAction.PRUNE
                        elif method == CandidateMethod.USAGE_GUIDED_MERGE:
                            usage_pairs, diagnostics = _usage_guided_pairs(
                                functional_embedding,
                                usage,
                                count,
                            )
                            groups = _partition_from_pairs(num_experts, usage_pairs)
                            action = CandidateAction.MERGE
                        elif method == CandidateMethod.FUNCTIONAL_COSINE:
                            groups = _partition_from_pairs(
                                num_experts,
                                _greedy_cosine_pairs(functional_embedding, count),
                            )
                            action = CandidateAction.MERGE
                        elif method == CandidateMethod.RMT_GEOMETRY:
                            pairs = (
                                _greedy_cosine_pairs(ranked_embedding, count)
                                if rank is not None
                                else tuple(rmt_pairs)[:count]
                            )
                            groups = _partition_from_pairs(num_experts, pairs)
                            action = CandidateAction.MERGE
                        elif method == CandidateMethod.RMT_FUNCTIONAL_USAGE:
                            hybrid_pairs, diagnostics = _hybrid_pairs(
                                ranked_embedding,
                                functional_embedding,
                                usage,
                                count,
                                variant_weights,
                            )
                            groups = _partition_from_pairs(num_experts, hybrid_pairs)
                            action = CandidateAction.MERGE
                        else:
                            raise AssertionError(f"Unhandled candidate method: {method.value}")

                        candidate_merge_variants = (
                            explicit_merge_weighting_variants
                            if method in weighted_methods
                            else ((None, MergeWeightingPolicy.UNIFORM, 0.0, 0.0),)
                        )
                        for (
                            merge_variant_name,
                            merge_policy,
                            minimum_weight,
                            ridge_lambda,
                        ) in candidate_merge_variants:
                            variant_diagnostics = diagnostics
                            id_parts = [method.value]
                            if rank is not None:
                                id_parts.append(f"r{rank}")
                                variant_diagnostics += (
                                    ("spectral_pairing_rank", float(rank)),
                                )
                            if weight_name is not None:
                                id_parts.append(f"w{weight_name}")
                                variant_diagnostics += (
                                    ("hybrid_rmt_weight", float(variant_weights[0])),
                                    ("hybrid_functional_weight", float(variant_weights[1])),
                                    ("hybrid_usage_weight", float(variant_weights[2])),
                                )

                            group_merge_weights: tuple[tuple[float, ...], ...] = ()
                            if merge_variant_name is not None:
                                id_parts.append(f"mw{merge_variant_name}")
                                group_merge_weights = build_group_merge_weights(
                                    groups,
                                    usage_values,
                                    policy=merge_policy,
                                    minimum_weight=minimum_weight,
                                    ridge_lambda=ridge_lambda,
                                    functional_embedding=functional_embedding,
                                    routing_probs=routing_probs,
                                )
                                variant_diagnostics += merge_weight_diagnostics(
                                    group_merge_weights
                                )
                                variant_diagnostics += (
                                    ("merge_weight_minimum", float(minimum_weight)),
                                    ("merge_ridge_lambda", float(ridge_lambda)),
                                )
                                if merge_policy == MergeWeightingPolicy.FUNCTIONAL_RIDGE:
                                    assert functional_embedding is not None
                                    assert routing_probs is not None
                                    variant_diagnostics += (
                                        (
                                            "merge_ridge_token_aligned",
                                            float(
                                                functional_ridge_token_aligned(
                                                    functional_embedding,
                                                    routing_probs,
                                                )
                                            ),
                                        ),
                                        (
                                            "merge_ridge_routing_rows",
                                            float(routing_probs.shape[0]),
                                        ),
                                        (
                                            "merge_ridge_functional_features",
                                            float(functional_embedding.shape[1]),
                                        ),
                                    )

                            candidate_prefix = "-".join(id_parts)
                            plan = CandidatePlanContract(
                                candidate_id=(
                                    f"{candidate_prefix}-f{float(fraction):.3f}-s{int(seed)}"
                                ),
                                layer=int(layer),
                                method=method,
                                action=action,
                                compression_fraction=float(fraction),
                                groups=groups,
                                seed=int(seed),
                                diagnostics=variant_diagnostics,
                                merge_weighting=merge_policy,
                                group_merge_weights=group_merge_weights,
                            )
                            validate_candidate_plan(plan, num_experts=num_experts)
                            plans.append(plan)
    return tuple(plans)
