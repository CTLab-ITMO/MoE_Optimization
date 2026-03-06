#!/usr/bin/env python
"""Thin wrapper to run JSON-driven experiments from legacy folder."""

from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from moe_opt_runner.main import run_experiment_from_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    out = run_experiment_from_config(args.config)
    print(out)
