import pytest
import torch

from moe_opt_core.pipelines.rmt_diagnostics import (
    finite_size_spectral_diagnostics,
    resolve_spectral_rank_grid,
    spectral_embeddings_at_ranks,
)


def test_finite_size_diagnostics_are_deterministic_and_rank_ordered() -> None:
    generator = torch.Generator().manual_seed(4)
    factors = torch.randn((8, 2), generator=generator)
    loadings = torch.randn((2, 64), generator=generator)
    calibration = factors @ loadings + 0.02 * torch.randn((8, 64), generator=generator)
    validation = factors @ loadings + 0.02 * torch.randn((8, 64), generator=generator)
    first = finite_size_spectral_diagnostics(
        calibration,
        validation_embedding=validation,
        null_trials=8,
        bootstrap_trials=8,
        seed=9,
    )
    second = finite_size_spectral_diagnostics(
        calibration,
        validation_embedding=validation,
        null_trials=8,
        bootstrap_trials=8,
        seed=9,
    )
    assert first.r_useful <= first.r_stable <= first.r_detect
    assert first.r_detect >= 1
    assert first.r_useful == second.r_useful
    assert first.heldout_max_angle_deg is not None
    assert first.heldout_passed is True
    assert len(first.heldout_angles_deg) == first.r_stable
    assert torch.allclose(first.signal_embedding, second.signal_embedding)


def test_spectral_embeddings_at_ranks_are_nested_and_deterministic() -> None:
    embedding = torch.randn(6, 12, generator=torch.Generator().manual_seed(17))
    first = spectral_embeddings_at_ranks(embedding, (1, 4))
    second = spectral_embeddings_at_ranks(embedding, (1, 4))
    assert first[1].shape == (6, 1)
    assert first[4].shape == (6, 4)
    assert torch.allclose(first[1], second[1])
    assert torch.allclose(first[4], second[4])
    assert torch.allclose(first[1].abs(), first[4][:, :1].abs())


def test_spectral_embeddings_reject_invalid_rank_grid() -> None:
    embedding = torch.randn(4, 8)
    with pytest.raises(ValueError, match="unique"):
        spectral_embeddings_at_ranks(embedding, (1, 1))
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        spectral_embeddings_at_ranks(embedding, (5,))


def test_rank_grid_combines_fixed_detected_and_explained_variance_policies() -> None:
    embedding = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5]))
    ranks = resolve_spectral_rank_grid(
        embedding,
        fixed_ranks=(2,),
        useful_rank=1,
        mp_rank=3,
        include_useful_rank=True,
        include_mp_rank=True,
        explained_variance_thresholds=(0.95,),
    )
    assert ranks == tuple(sorted(set(ranks)))
    assert {1, 2, 3}.issubset(ranks)
    assert max(ranks) <= 4
