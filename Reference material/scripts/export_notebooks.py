"""Export chapter, extras, and demo orchestrators as Jupyter notebooks.

Run::

    python scripts/export_notebooks.py
    python scripts/export_notebooks.py --chapters 1 2 3
    python scripts/export_notebooks.py --no-extras --no-demos
    python scripts/export_notebooks.py --clean

Notebooks land in ``output/notebooks/``. Requires ``nbformat`` (installed via
``uv sync --extra notebooks`` or the default ``dev`` group).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from active_inference.menu.runner import discover_chapters, discover_demos, discover_extras  # noqa: E402
from active_inference.utils.notebooks import (  # noqa: E402
    default_notebook_dir,
    export_all_notebooks,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    discovered_chapters = [entry.number for entry in discover_chapters()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chapters",
        nargs="+",
        type=int,
        default=discovered_chapters,
        choices=discovered_chapters,
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        default=None,
        help="Extras topic slugs; pass without values for all extras",
    )
    parser.add_argument(
        "--demos",
        nargs="*",
        default=None,
        help="Demo slugs; pass without values for all demos",
    )
    parser.add_argument("--no-chapters", action="store_true")
    parser.add_argument("--no-extras", action="store_true")
    parser.add_argument("--no-demos", action="store_true")
    parser.add_argument(
        "--no-animations",
        action="store_true",
        help="Skip animation run cells; sections are documented only",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated .ipynb files before exporting",
    )
    return parser.parse_args()


def clean_notebook_dir(root: Path = default_notebook_dir()) -> int:
    """Remove generated notebooks while preserving README/AGENTS docs."""
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob("*.ipynb"):
        path.unlink()
        removed += 1
    return removed


def main() -> None:
    """Export notebooks for the selected targets."""
    args = parse_args()
    if args.clean:
        removed = clean_notebook_dir()
        print(f"Removed {removed} notebook(s)")

    extras: list[str] | None = None
    if not args.no_extras and args.extras is not None:
        extras = [entry.slug for entry in discover_extras()] if len(args.extras) == 0 else args.extras

    demos: list[str] | None = None
    if not args.no_demos and args.demos is not None:
        demos = [entry.slug for entry in discover_demos()] if len(args.demos) == 0 else args.demos

    result = export_all_notebooks(
        chapters=args.chapters,
        extras=extras,
        demos=demos,
        include_chapters=not args.no_chapters,
        include_extras=not args.no_extras,
        include_demos=not args.no_demos,
        include_animations=not args.no_animations,
    )
    print(
        f"Exported {result.total} notebook(s): "
        f"{len(result.chapter_paths)} chapter, "
        f"{len(result.extra_paths)} extras, "
        f"{len(result.demo_paths)} demo"
    )


if __name__ == "__main__":
    main()
