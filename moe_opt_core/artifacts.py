from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ArtifactSpec:
    schema_version: str
    config_hash: str
    root_dir: str
    paths: Dict[str, str]


def validate_artifact_spec_dict(data: Dict[str, Any]) -> None:
    for key in ["schema_version", "config_hash", "root_dir", "paths"]:
        if key not in data:
            raise ValueError(f"ArtifactSpec missing key: {key}")
    if not isinstance(data["paths"], dict):
        raise ValueError("ArtifactSpec.paths must be a dict")


def save_artifact_spec(spec: ArtifactSpec, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(spec), f, indent=2, ensure_ascii=False)


def load_artifact_spec(path: str | Path) -> ArtifactSpec:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    validate_artifact_spec_dict(data)
    return ArtifactSpec(
        schema_version=str(data["schema_version"]),
        config_hash=str(data["config_hash"]),
        root_dir=str(data["root_dir"]),
        paths={str(k): str(v) for k, v in data["paths"].items()},
    )


def write_artifact_manifest(
    *,
    schema_version: str,
    config_hash: str,
    root_dir: str | Path,
    manifest_path: str | Path,
    paths: Mapping[str, str | Path | None],
) -> ArtifactSpec:
    normalized_paths: Dict[str, str] = {}
    for key, value in paths.items():
        if value is None:
            continue
        rendered = str(value).strip()
        if not rendered:
            continue
        normalized_paths[str(key)] = rendered

    spec = ArtifactSpec(
        schema_version=str(schema_version),
        config_hash=str(config_hash),
        root_dir=str(root_dir),
        paths=normalized_paths,
    )
    save_artifact_spec(spec, manifest_path)
    return spec


def write_run_summary(
    *,
    root_dir: str | Path,
    summary_name: str,
    payload: Mapping[str, Any],
) -> Path:
    summary_path = Path(root_dir) / str(summary_name)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    return summary_path
