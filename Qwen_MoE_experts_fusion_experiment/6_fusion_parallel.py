#!/usr/bin/env python
"""Distributed thin wrapper for fusion runner."""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(__file__)
if _THIS not in sys.path:
    sys.path.append(_THIS)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from fusion_runner import main


if __name__ == "__main__":
    main()
