"""Make the golden harness modules importable by the golden test.

``test_golden.py`` imports the sibling ``harness`` and ``tolerances`` modules by
bare name. Under ``--import-mode=importlib`` pytest does not add a test file's
directory to ``sys.path`` automatically, so add this directory here (conftest
runs before the tests in its directory are imported).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
