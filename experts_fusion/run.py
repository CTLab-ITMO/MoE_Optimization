"""Run the full fusion pipeline (stages 1..9) from the repository root.

Invoke from the repo root::

    python -m experts_fusion.run

Each stage is a script named ``<n>_*.py`` executed in numeric order via
``runpy`` in this single process. Stages import their config with
``from experts_fusion.config import ...``, so the repo root must be importable
as a package parent -- guaranteed when launched with ``-m`` from the root.

Pass an explicit list of stage numbers to run a subset, e.g.::

    python -m experts_fusion.run 1 2 3
"""
import runpy
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent


def stage_files(only: list[str] | None = None) -> list[Path]:
    stages = sorted(STAGE_DIR.glob("[0-9]_*.py"))
    if only:
        wanted = set(only)
        stages = [s for s in stages if s.name.split("_", 1)[0] in wanted]
    return stages


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    for stage in stage_files(argv or None):
        print(f"=== {stage.name} ===", flush=True)
        runpy.run_path(str(stage), run_name="__main__")


if __name__ == "__main__":
    main()
