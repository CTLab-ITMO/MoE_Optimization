"""Run the four-stage fusion pipeline from the repository root.

    python -m experts_fusion.run          # all stages
    python -m experts_fusion.run 1 2      # subset
    python -m experts_fusion.run --config winogrande

Stages (each a ``<n>_*.py`` script run in-process via ``runpy``):
1_preprocess, 2_grouping, 3_fusion, 4_evaluate.
Launch with ``-m`` from the root so package imports resolve correctly.
"""
import argparse
import os
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
    parser = argparse.ArgumentParser(prog="experts_fusion.run")
    parser.add_argument(
        "--config",
        default=None,
        help="config name under experts_fusion/configs/ (default: 'default')",
    )
    parser.add_argument("stages", nargs="*", help="stage numbers to run (default: all)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # Must be set before any stage runs so experts_fusion.config picks the right file.
    if args.config:
        os.environ["EXPERTS_FUSION_CONFIG"] = args.config

    for stage in stage_files(args.stages or None):
        print(f"=== {stage.name} ===", flush=True)
        runpy.run_path(str(stage), run_name="__main__")


if __name__ == "__main__":
    main()
