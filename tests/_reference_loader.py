"""Loader for the vendored textbook active-inference implementation.

``Reference material/src/active_inference/__init__.py`` (and its ``core``
sub-package ``__init__``) eagerly import the whole textbook chain, which pulls
in ``nbformat``. This project does not declare ``nbformat`` as a dependency, so
``import active_inference`` fails before any oracle module can be reached.

Instead of installing the reference package we register two *namespace stubs*
(``active_inference`` and ``active_inference.core``) whose ``__path__`` points
at the vendored source tree. ``importlib`` then resolves individual leaf
modules such as ``active_inference.core.pomdp`` directly, without executing
either package ``__init__``. The reference tree is read-only measurement
apparatus: nothing under ``src/active_painter/`` imports it.

The module name starts with an underscore so pytest does not collect it as a
test module. Test modules import it as ``from _reference_loader import ...``
(pytest puts ``tests/`` on ``sys.path`` because there is no ``tests/__init__.py``).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import ModuleType

import pytest

# Resolved from this file, never from the current working directory, so the
# harness behaves the same under `pytest`, `pytest tests/...`, and IDE runners.
_REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = _REPO_ROOT / "Reference material" / "src" / "active_inference"

# The oracle modules we load all live under core/; pomdp.py is the discrete
# POMDP oracle and its presence is the availability signal for the whole tree.
REFERENCE_AVAILABLE: bool = (REFERENCE_ROOT / "core" / "pomdp.py").is_file()

requires_reference = pytest.mark.skipif(
    not REFERENCE_AVAILABLE,
    reason=f"Vendored reference implementation is absent at {REFERENCE_ROOT}",
)

_STUB_MARKER = "_active_painter_reference_stub"

_STUB_PACKAGES: tuple[tuple[str, Path], ...] = (
    ("active_inference", REFERENCE_ROOT),
    ("active_inference.core", REFERENCE_ROOT / "core"),
)


def _register_stubs() -> None:
    """Install the namespace stubs once, idempotently.

    If ``sys.modules`` already holds one of the names -- either our own marked
    stub from an earlier call or a genuinely installed ``active_inference`` --
    it is left completely alone. Rebuilding a stub mid-session would reset
    ``__path__`` underneath modules already imported through it.
    """

    for name, path in _STUB_PACKAGES:
        existing = sys.modules.get(name)
        if existing is not None:
            continue
        package = types.ModuleType(name)
        package.__path__ = [str(path)]  # type: ignore[attr-defined]
        package.__package__ = name
        setattr(package, _STUB_MARKER, True)
        sys.modules[name] = package


def load_reference_module(dotted_suffix: str) -> ModuleType:
    """Import ``active_inference.<dotted_suffix>`` from the vendored tree.

    Skips (never raises) when the reference tree is absent, so a pruned local
    checkout reports skips rather than collection errors.
    """

    if not REFERENCE_AVAILABLE:
        pytest.skip(f"Vendored reference implementation is absent at {REFERENCE_ROOT}")
    _register_stubs()
    return importlib.import_module(f"active_inference.{dotted_suffix}")
