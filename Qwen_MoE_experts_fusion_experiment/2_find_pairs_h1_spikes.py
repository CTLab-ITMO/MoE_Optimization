#!/usr/bin/env python
"""Legacy thin wrapper: run H1 spike-based pairing from JSON config."""

from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
if _PARENT not in sys.path:
    sys.path.append(_PARENT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    from moe_opt_runner.legacy import find_pairs_from_config

    out = find_pairs_from_config(args.config)
    print(out)


if __name__ == "__main__":
    main()
