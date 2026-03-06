from pathlib import Path

from moe_opt_core.pipelines.h2_bias import run_h2_pipeline


def test_h2_failures_missing_inputs(tmp_path: Path) -> None:
    out_bias = tmp_path / "bias.pt"
    out_diag = tmp_path / "diag.json"
    try:
        run_h2_pipeline(
            mode="covariance",
            output_bias_path=str(out_bias),
            output_diagnostics_path=str(out_diag),
            importance_matrix_path="",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing importance_matrix_path")
