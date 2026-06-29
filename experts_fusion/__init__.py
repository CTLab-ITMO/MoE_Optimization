"""Experts-fusion pipeline (run from the repository root).

A package launched from the repo root via ``python -m experts_fusion.run`` (or
per stage through the same runner). Stages import shared primitives from
:mod:`common` and their own configuration from :mod:`experts_fusion.config`;
there is no ``sys.path`` manipulation and no reliance on the module directory
being the cwd.
"""
