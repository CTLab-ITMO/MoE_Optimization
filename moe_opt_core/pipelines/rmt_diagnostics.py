from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FiniteSizeSpectralResult:
    signal_embedding: torch.Tensor
    eigenvalues: torch.Tensor
    null_edge: float
    r_detect: int
    r_stable: int
    r_useful: int
    bootstrap_max_angle_median_deg: float
    heldout_max_angle_deg: float | None
    heldout_angles_deg: tuple[float, ...]
    heldout_passed: bool | None


def _expert_gram(embedding: torch.Tensor) -> torch.Tensor:
    centered = embedding.float() - embedding.float().mean(dim=0, keepdim=True)
    return centered @ centered.t() / max(1, centered.shape[1])


def _descending_eigensystem(embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(_expert_gram(embedding))
    return eigenvalues.flip(0), eigenvectors.flip(1)


def spectral_embeddings_at_ranks(
    calibration_embedding: torch.Tensor,
    ranks: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    if not ranks:
        return {}
    if len(set(ranks)) != len(ranks):
        raise ValueError("spectral ranks must be unique")
    eigenvalues, eigenvectors = _descending_eigensystem(calibration_embedding)
    max_rank = eigenvectors.shape[1]
    result: dict[int, torch.Tensor] = {}
    for rank in ranks:
        if rank < 1 or rank > max_rank:
            raise ValueError(f"spectral rank must be in [1, {max_rank}], got {rank}")
        result[rank] = eigenvectors[:, :rank] * torch.sqrt(
            eigenvalues[:rank].clamp_min(0.0)
        ).unsqueeze(0)
    return result


def resolve_spectral_rank_grid(
    calibration_embedding: torch.Tensor,
    *,
    fixed_ranks: tuple[int, ...] = (),
    useful_rank: int | None = None,
    mp_rank: int | None = None,
    include_useful_rank: bool = False,
    include_mp_rank: bool = False,
    explained_variance_thresholds: tuple[float, ...] = (),
) -> tuple[int, ...]:
    _, eigenvectors = _descending_eigensystem(calibration_embedding)
    max_rank = int(eigenvectors.shape[1])
    ranks = {int(rank) for rank in fixed_ranks}
    if include_useful_rank:
        if useful_rank is None:
            raise ValueError("useful_rank is required by the spectral rank policy")
        ranks.add(int(useful_rank))
    if include_mp_rank:
        if mp_rank is None:
            raise ValueError("mp_rank is required by the spectral rank policy")
        ranks.add(int(mp_rank))

    eigenvalues, _ = _descending_eigensystem(calibration_embedding)
    positive = eigenvalues.clamp_min(0.0)
    total = float(positive.sum().item())
    for threshold in explained_variance_thresholds:
        value = float(threshold)
        if not 0.0 < value <= 1.0:
            raise ValueError("explained variance thresholds must be in (0, 1]")
        if total <= 0.0:
            ranks.add(1)
            continue
        cumulative = torch.cumsum(positive, dim=0) / total
        selected = int(torch.searchsorted(cumulative, value).item()) + 1
        ranks.add(min(selected, max_rank))

    invalid = sorted(rank for rank in ranks if rank < 1 or rank > max_rank)
    if invalid:
        raise ValueError(f"spectral ranks must be in [1, {max_rank}]: {invalid}")
    return tuple(sorted(ranks))


def _max_principal_angle_deg(first: torch.Tensor, second: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(first.t() @ second).clamp(0.0, 1.0)
    angles = torch.arccos(singular_values)
    return float(torch.rad2deg(angles).max().item())


def _empirical_null_edge(
    embedding: torch.Tensor,
    *,
    trials: int,
    quantile: float,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu = embedding.detach().float().cpu()
    maxima = []
    for _ in range(max(1, trials)):
        signs = torch.randint(0, 2, cpu.shape, generator=generator, dtype=torch.int8)
        signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
        eigenvalues, _ = _descending_eigensystem(cpu * signs)
        maxima.append(eigenvalues[0])
    return float(torch.quantile(torch.stack(maxima), quantile).item())


def finite_size_spectral_diagnostics(
    calibration_embedding: torch.Tensor,
    *,
    validation_embedding: torch.Tensor | None = None,
    null_trials: int = 32,
    bootstrap_trials: int = 32,
    null_quantile: float = 0.95,
    stability_max_angle_deg: float = 30.0,
    seed: int = 123,
    min_rank: int = 1,
) -> FiniteSizeSpectralResult:
    eigenvalues, eigenvectors = _descending_eigensystem(calibration_embedding)
    null_edge = _empirical_null_edge(
        calibration_embedding,
        trials=null_trials,
        quantile=null_quantile,
        seed=seed,
    )
    r_detect = max(min_rank, int((eigenvalues > null_edge).sum().item()))
    r_detect = min(r_detect, eigenvectors.shape[1])

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    cpu = calibration_embedding.detach().float().cpu()
    reference = eigenvectors.detach().float().cpu()
    angle_samples: dict[int, list[float]] = {rank: [] for rank in range(1, r_detect + 1)}
    for _ in range(max(1, bootstrap_trials)):
        columns = torch.randint(0, cpu.shape[1], (cpu.shape[1],), generator=generator)
        _, bootstrap_vectors = _descending_eigensystem(cpu.index_select(1, columns))
        for rank in range(1, r_detect + 1):
            angle_samples[rank].append(
                _max_principal_angle_deg(reference[:, :rank], bootstrap_vectors[:, :rank])
            )

    medians = {
        rank: float(torch.tensor(values).median().item())
        for rank, values in angle_samples.items()
    }
    stable = [rank for rank, angle in medians.items() if angle <= stability_max_angle_deg]
    r_stable = max(stable) if stable else min_rank
    r_stable = min(r_stable, r_detect)

    heldout_angle = None
    heldout_angles: tuple[float, ...] = ()
    heldout_passed = None
    r_useful = r_stable
    if validation_embedding is not None:
        _, validation_vectors = _descending_eigensystem(validation_embedding)
        angle_values = []
        useful = []
        for rank in range(1, r_stable + 1):
            angle = _max_principal_angle_deg(
                reference[:, :rank],
                validation_vectors.detach().float().cpu()[:, :rank],
            )
            angle_values.append(angle)
            if angle <= stability_max_angle_deg:
                useful.append(rank)
        r_useful = max(useful) if useful else min_rank
        heldout_angles = tuple(angle_values)
        heldout_angle = heldout_angles[r_useful - 1]
        heldout_passed = bool(useful)

    selected_rank = max(min_rank, min(r_useful, eigenvectors.shape[1]))
    signal_embedding = eigenvectors[:, :selected_rank] * torch.sqrt(
        eigenvalues[:selected_rank].clamp_min(0.0)
    ).unsqueeze(0)
    return FiniteSizeSpectralResult(
        signal_embedding=signal_embedding,
        eigenvalues=eigenvalues,
        null_edge=null_edge,
        r_detect=r_detect,
        r_stable=r_stable,
        r_useful=r_useful,
        bootstrap_max_angle_median_deg=medians[r_stable],
        heldout_max_angle_deg=heldout_angle,
        heldout_angles_deg=heldout_angles,
        heldout_passed=heldout_passed,
    )
