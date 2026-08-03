from moe_opt_runner.preflight import build_preflight_report


def test_preflight_reports_runtime_without_loading_model() -> None:
    report = build_preflight_report("configs/experiments/h123_refactor_baseline.json")
    assert report["config"] == "h123_refactor_baseline"
    assert "torch" in report["dependencies"]
    assert "cuda_available" in report
