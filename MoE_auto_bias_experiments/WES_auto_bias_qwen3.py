#!/usr/bin/env python
"""Qwen3 auto-bias wrapper."""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(__file__)
if _THIS not in sys.path:
    sys.path.append(_THIS)

from auto_bias_runner import main


if __name__ == "__main__":
    main(default_profile="qwen3")
