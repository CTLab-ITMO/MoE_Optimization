from pathlib import Path

from moe_opt_core.artifacts import ArtifactSpec, load_artifact_spec, save_artifact_spec


def test_artifact_roundtrip(tmp_path: Path) -> None:
    spec = ArtifactSpec(
        schema_version="1.0",
        config_hash="abc",
        root_dir="artifacts",
        paths={"pairs": "pairs.pt"},
    )
    out = tmp_path / "manifest.json"
    save_artifact_spec(spec, out)
    loaded = load_artifact_spec(out)
    assert loaded.schema_version == spec.schema_version
    assert loaded.paths["pairs"] == "pairs.pt"
