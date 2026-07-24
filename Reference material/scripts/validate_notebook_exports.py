"""Validate exported Jupyter notebooks against discovery inventory.

Run::

    python scripts/validate_notebook_exports.py
    python scripts/validate_notebook_exports.py --root output/notebooks

Exits non-zero when a expected notebook is missing or its section count
drifts from the live orchestrator inventory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import nbformat  # noqa: E402

from active_inference.menu.runner import (  # noqa: E402
    discover_chapters,
    discover_demo_scripts,
    discover_demos,
    discover_extra_scripts,
    discover_extras,
    discover_scripts,
)
from active_inference.utils.notebooks import (  # noqa: E402
    chapter_notebook_path,
    demo_notebook_path,
    default_notebook_dir,
    extra_notebook_path,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_notebook_dir(),
        help="Notebook output root (default: output/notebooks/)",
    )
    parser.add_argument("--chapters", nargs="*", type=int, default=None)
    parser.add_argument("--extras", nargs="*", default=None)
    parser.add_argument("--demos", nargs="*", default=None)
    parser.add_argument("--no-chapters", action="store_true")
    parser.add_argument("--no-extras", action="store_true")
    parser.add_argument("--no-demos", action="store_true")
    return parser.parse_args()


def _section_headers(notebook: nbformat.NotebookNode) -> list[str]:
    """Collect orchestrator section stems from level-2 markdown headers."""
    headers: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            if line.startswith("## `") and line.endswith("`"):
                headers.append(line.removeprefix("## `").removesuffix("`"))
    return headers


def _validate_notebook(path: Path, expected_stems: list[str]) -> list[str]:
    """Validate one notebook file and return a list of error messages."""
    errors: list[str] = []
    if not path.is_file():
        errors.append(f"missing notebook: {path}")
        return errors
    notebook = nbformat.read(path, as_version=4)
    if notebook.nbformat < 4:
        errors.append(f"{path}: nbformat {notebook.nbformat} < 4")
    headers = _section_headers(notebook)
    expected = [Path(stem).stem if "/" in stem else stem for stem in expected_stems]
    for stem in expected:
        if stem not in headers:
            errors.append(f"{path}: missing section `{stem}`")
    if len(headers) < len(expected):
        errors.append(
            f"{path}: {len(headers)} sections found, expected at least {len(expected)}"
        )
    return errors


def main() -> int:
    """Validate notebook exports and return an exit code."""
    args = parse_args()
    root = args.root
    errors: list[str] = []

    if not args.no_chapters:
        chapter_numbers = (
            args.chapters
            if args.chapters is not None
            else [entry.number for entry in discover_chapters()]
        )
        for number in chapter_numbers:
            scripts = discover_scripts(number, include_interactive=True)
            stems = [entry.path.stem for entry in scripts]
            errors.extend(
                _validate_notebook(chapter_notebook_path(number, root=root), stems)
            )

    if not args.no_extras:
        topics = (
            args.extras
            if args.extras is not None
            else [entry.slug for entry in discover_extras()]
        )
        for topic in topics:
            scripts = discover_extra_scripts(topic, include_interactive=True)
            stems = [entry.path.stem for entry in scripts]
            errors.extend(_validate_notebook(extra_notebook_path(topic, root=root), stems))

    if not args.no_demos:
        slugs = (
            args.demos
            if args.demos is not None
            else [entry.slug for entry in discover_demos()]
        )
        for slug in slugs:
            scripts = discover_demo_scripts(slug, include_interactive=True)
            stems = [entry.path.stem for entry in scripts]
            errors.extend(_validate_notebook(demo_notebook_path(slug, root=root), stems))

    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    print(f"Validated notebook exports under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
