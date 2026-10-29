import ast
from pathlib import Path

import rho.loop


def test_loop_static_import_check() -> None:
    tree = ast.parse(Path(rho.loop.__file__).read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "Grade" not in imported_names
    assert not any(module.startswith("rho.reporting") for module in imported_modules)
    assert "Grade" not in vars(rho.loop)
