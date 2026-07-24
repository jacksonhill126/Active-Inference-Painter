"""AST-level documentation contracts for library, chapter, and script methods."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    REPO_ROOT / "src" / "active_inference",
    REPO_ROOT / "chapters",
    REPO_ROOT / "extras",
    REPO_ROOT / "scripts",
)
PUBLIC_MODULES = (
    "active_inference",
    "active_inference.core",
    "active_inference.estimators",
    "active_inference.utils",
    "active_inference.visualizations",
)


def _python_files() -> list[Path]:
    """Return all Python implementation files covered by the docstring contract."""
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    return sorted(files)


def _missing_docstrings(path: Path) -> list[str]:
    """Return non-dunder classes/functions in ``path`` that lack docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        if ast.get_docstring(node, clean=False):
            continue
        rel = path.relative_to(REPO_ROOT)
        missing.append(f"{rel}:{node.lineno} {node.name}")
    return missing


def test_every_non_dunder_method_and_class_has_docstring() -> None:
    """Require local docs for every implementation def/class, including helpers."""
    missing: list[str] = []
    for path in _python_files():
        missing.extend(_missing_docstrings(path))
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)


def test_public_all_exports_have_nontrivial_docstrings() -> None:
    """Require documented public API symbols exported through package ``__all__``."""
    failures: list[str] = []
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for symbol in getattr(module, "__all__", ()):
            obj = getattr(module, symbol)
            doc = inspect.getdoc(obj) or ""
            words = [word for word in doc.replace("``", " ").split() if word.strip()]
            if len(words) < 8:
                failures.append(f"{module_name}.{symbol}: {doc!r}")
    assert not failures, "Public exports need non-trivial docstrings:\n" + "\n".join(failures)
