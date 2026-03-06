import json
from pathlib import Path

from moe_opt_runner.main import run_experiment_from_config


def test_runner_with_disabled_stages_writes_manifest(tmp_path: Path) -> None:
    cfg_path = Path("configs/experiments/h123_refactor_baseline.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["artifacts"]["root_dir"] = str(tmp_path / "artifacts")
    cfg["artifacts"]["manifest_path"] = str(tmp_path / "artifacts" / "manifest.json")
    cfg["stages"]["h1"]["enabled"] = False
    cfg["stages"]["h2"]["enabled"] = False
    cfg["stages"]["h3"]["enabled"] = False
    constants = {
        "training": str(Path("configs/constants/constant_repo_training.json").resolve()),
        "routing": str(Path("configs/constants/constant_repo_routing.json").resolve()),
        "visualization": str(Path("configs/constants/constant_repo_visualization.json").resolve()),
        "io": str(Path("configs/constants/constant_repo_io.json").resolve()),
    }
    cfg["constants"] = constants
    local_cfg = tmp_path / "cfg.json"
    local_cfg.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    out = run_experiment_from_config(str(local_cfg))
    assert "stages" in out
    assert Path(cfg["artifacts"]["manifest_path"]).exists()
