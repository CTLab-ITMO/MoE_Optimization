"""Run the full fusion pipeline (the four stages) from the repository root.

Invoke from the repo root::

    python -m experts_fusion.run

The pipeline is four stages -- ``1_preprocess`` (model load, weight extraction,
single-pass capture of co-activation / MLP inputs / top-k), ``2_grouping``
(partition + per-group activation join), ``3_fusion`` (fuse each group by every
method) and ``4_evaluate`` (load-once accuracy + diagnostics). Each is a script
named ``<n>_*.py`` executed in numeric order via ``runpy`` in this single
process. Stages import their config with ``from experts_fusion.config import
...``, so the repo root must be importable as a package parent -- guaranteed
when launched with ``-m`` from the root.

Pass an explicit list of stage numbers to run a subset, and ``--config`` to pick
a config under ``experts_fusion/configs/`` (default ``default``), e.g.::

    python -m experts_fusion.run 1 2
    python -m experts_fusion.run --config winogrande
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

    # Set before importing/running any stage so experts_fusion.config resolves to
    # the chosen config (the stages run in this one process via runpy).
    if args.config:
        os.environ["EXPERTS_FUSION_CONFIG"] = args.config

    for stage in stage_files(args.stages or None):
        print(f"=== {stage.name} ===", flush=True)
        runpy.run_path(str(stage), run_name="__main__")


if __name__ == "__main__":
    main()
