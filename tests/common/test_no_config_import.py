"""Architecture test: ``common/`` never imports an experiment ``config``.

``common`` is library code shared by every pipeline and by neighbouring
projects, so it must not depend on a specific experiment's ``config`` module
(principle 1.3). This AST-scans every ``common/**/*.py`` and fails on any import
of a bare ``config`` or a ``*.config`` module.
"""

import ast
from pathlib import Path

import pytest

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"


def _is_config_module(name: str | None) -> bool:
    if not name:
        return False
    return name == "config" or name.split(".")[-1] == "config"


def _config_imports(tree: ast.AST) -> list[str]:
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_config_module(alias.name):
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Ignore relative imports (node.level > 0): they stay inside common.
            if node.level == 0 and _is_config_module(node.module):
                offenders.append(node.module or "")
    return offenders


@pytest.mark.parametrize(
    "path",
    sorted(COMMON_DIR.rglob("*.py")),
    ids=lambda p: str(p.relative_to(COMMON_DIR)),
)
def test_common_module_does_not_import_config(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = _config_imports(tree)
    assert not offenders, f"{path} imports experiment config: {offenders}"
