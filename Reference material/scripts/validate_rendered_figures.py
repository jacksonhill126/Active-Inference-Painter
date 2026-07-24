"""Validate regenerated chapter PNG/GIF artifacts.

The checks are intentionally mechanical: they catch corrupt files, tiny accidental
exports, blank canvases, and single-frame GIFs before visual review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("output/figures"),
                   help="Directory containing rendered PNG/GIF artifacts")
    p.add_argument("--min-width", type=int, default=160)
    p.add_argument("--min-height", type=int, default=100)
    p.add_argument("--min-bytes", type=int, default=250)
    p.add_argument("--require-variation", type=float, default=1.0,
                   help="Minimum grayscale standard deviation for representative frames")
    return p.parse_args()


def _variation(img: Image.Image) -> float:
    """Support this repository command-line validation or rendering script."""
    arr = np.asarray(img.convert("L"), dtype=float)
    return float(np.std(arr))


def _check_image(path: Path, args: argparse.Namespace) -> list[str]:
    """Check one generated artifact and return any validation failure."""
    problems: list[str] = []
    if path.stat().st_size < args.min_bytes:
        problems.append(f"too small ({path.stat().st_size} bytes)")
    try:
        with Image.open(path) as img:
            if img.width < args.min_width or img.height < args.min_height:
                problems.append(f"tiny dimensions ({img.width}x{img.height})")
            if path.suffix.lower() == ".gif":
                frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
                if len(frames) < 2:
                    problems.append("GIF has fewer than 2 frames")
                # Sample first/middle/last to catch blank animations without decoding everything twice.
                sample = [frames[0], frames[len(frames) // 2], frames[-1]] if frames else []
                if sample and max(_variation(frame) for frame in sample) < args.require_variation:
                    problems.append("sampled GIF frames appear blank")
            elif _variation(img) < args.require_variation:
                problems.append("image appears blank")
    except Exception as exc:  # pragma: no cover - defensive, exercised through CLI
        problems.append(f"cannot open: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run this repository command-line tool and return its exit status."""
    args = parse_args() if argv is None else parse_args_from(argv)
    root = args.root
    paths = sorted([*root.rglob("*.png"), *root.rglob("*.gif")])
    if not paths:
        print(f"No PNG/GIF artifacts found under {root}")
        return 1
    failures: list[tuple[Path, list[str]]] = []
    for path in paths:
        problems = _check_image(path, args)
        if problems:
            failures.append((path, problems))
    if failures:
        print("Rendered artifact validation failed:")
        for path, problems in failures:
            print(f"- {path}: {', '.join(problems)}")
        return 1
    print(f"Validated {len(paths)} rendered artifacts under {root}")
    return 0


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options from an explicit argument sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/figures"))
    parser.add_argument("--min-width", type=int, default=160)
    parser.add_argument("--min-height", type=int, default=100)
    parser.add_argument("--min-bytes", type=int, default=250)
    parser.add_argument("--require-variation", type=float, default=1.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
