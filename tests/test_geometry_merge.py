from pathlib import Path
import torch

from moe_opt_core.pipelines.h1_merge import (
    collect_mlp_inputs,
    merge_weight_matrices_baseline,
    merge_weight_matrices_geometry,
)


def test_merge_weight_baseline_and_geometry() -> None:
    w1 = torch.randn(16, 8)
    w2 = torch.randn(16, 8)
    b = merge_weight_matrices_baseline([w1, w2])
    g = merge_weight_matrices_geometry([w1, w2], rank=4, shrink_tau=0.0)
    assert b.shape == w1.shape
    assert g.shape == w1.shape


def test_collect_mlp_inputs_invalid_shape(tmp_path: Path) -> None:
    p = tmp_path / "bad.pt"
    torch.save(torch.randn(2, 3, 4), p)
    try:
        collect_mlp_inputs(p, n_inputs=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-2D inputs")
