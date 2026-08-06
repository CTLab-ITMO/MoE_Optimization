import torch

from moe_opt_core.pipelines.rmt_diagnostics import finite_size_spectral_diagnostics


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
