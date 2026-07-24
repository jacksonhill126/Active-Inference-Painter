#!/usr/bin/env python3
"""Validate chapter NPZ+JSON raw-data exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the raw-data validator command line."""
    parser = argparse.ArgumentParser(
        description="Validate output/data/chapter_NN NPZ+JSON sidecars.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/data"),
        help="Root output/data directory to validate.",
    )
    parser.add_argument(
        "--chapters",
        type=int,
        nargs="*",
        default=None,
        help="Chapter numbers that must have at least one valid export.",
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        default=None,
        help="Extras topic slugs that must have at least one valid export.",
    )
    parser.add_argument(
        "--demos",
        nargs="*",
        default=None,
        help="Demo topic slugs that must have at least one valid export.",
    )
    return parser.parse_args(argv)


def _chapter_dir(root: Path, chapter: int) -> Path:
    """Return the expected raw-data directory for one chapter."""
    return root / f"chapter_{chapter:02d}"


def _extra_dir(root: Path, topic: str) -> Path:
    """Return the expected raw-data directory for one extras topic."""
    return root / "extras" / topic


def _demo_dir(root: Path, slug: str) -> Path:
    """Return the expected raw-data directory for one demo topic."""
    return root / "demo" / slug


def _pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Return JSON/NPZ pairs discovered in ``directory``."""
    return sorted(
        (json_path.with_suffix(".npz"), json_path)
        for json_path in directory.glob("*.json")
        if json_path.with_suffix(".npz").exists()
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load a raw-data JSON manifest as a dictionary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return payload


def _validate_array(name: str, arr: np.ndarray, spec: dict[str, Any], json_path: Path) -> None:
    """Validate one NPZ array against its JSON shape/dtype manifest."""
    if arr.size == 0:
        raise ValueError(f"{json_path}: {name} is empty")
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{json_path}: {name} must be numeric and non-object")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{json_path}: {name} contains non-finite values")
    expected_shape = list(spec.get("shape", []))
    if expected_shape != list(arr.shape):
        raise ValueError(
            f"{json_path}: {name} shape mismatch "
            f"{expected_shape!r} != {list(arr.shape)!r}"
        )
    expected_dtype = spec.get("dtype")
    if expected_dtype != str(arr.dtype):
        raise ValueError(
            f"{json_path}: {name} dtype mismatch {expected_dtype!r} != {str(arr.dtype)!r}"
        )


def _validate_pair(npz_path: Path, json_path: Path) -> None:
    """Validate one NPZ/JSON raw-data sidecar pair."""
    manifest = _load_manifest(json_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{json_path}: schema_version must be 1")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError(f"{json_path}: arrays manifest must be a non-empty object")
    with np.load(npz_path) as data:
        npz_names = set(data.files)
        manifest_names = set(arrays)
        if npz_names != manifest_names:
            raise ValueError(
                f"{json_path}: array names mismatch "
                f"{sorted(manifest_names)!r} != {sorted(npz_names)!r}"
            )
        for name in sorted(npz_names):
            spec = arrays[name]
            if not isinstance(spec, dict):
                raise ValueError(f"{json_path}: {name} manifest must be an object")
            _validate_array(name, data[name], spec, json_path)


def _validate_directory(directory: Path) -> list[str]:
    """Validate all sidecar pairs in one chapter directory and collect errors."""
    errors: list[str] = []
    json_paths = set(directory.glob("*.json"))
    npz_paths = set(directory.glob("*.npz"))
    for npz_path in sorted(npz_paths):
        json_path = npz_path.with_suffix(".json")
        if json_path not in json_paths:
            errors.append(f"{npz_path}: missing JSON manifest")
    for json_path in sorted(json_paths):
        npz_path = json_path.with_suffix(".npz")
        if npz_path not in npz_paths:
            errors.append(f"{json_path}: missing NPZ arrays")
            continue
        try:
            _validate_pair(npz_path, json_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate raw-data exports and return a process exit code."""
    args = parse_args(argv)
    root = args.root
    errors: list[str] = []
    directories: list[Path] = []

    if args.chapters or args.extras or args.demos:
        if args.chapters:
            for chapter in args.chapters:
                directory = _chapter_dir(root, chapter)
                if not directory.exists():
                    errors.append(f"{directory}: missing chapter_{chapter:02d} data directory")
                else:
                    directories.append(directory)
                    if not _pairs(directory):
                        errors.append(f"{directory}: no NPZ+JSON raw-data pairs")
        if args.extras:
            for topic in args.extras:
                directory = _extra_dir(root, topic)
                if not directory.exists():
                    errors.append(f"{directory}: missing extras topic {topic!r} data directory")
                else:
                    directories.append(directory)
                    if not _pairs(directory):
                        errors.append(f"{directory}: no NPZ+JSON raw-data pairs")
        if args.demos:
            for slug in args.demos:
                directory = _demo_dir(root, slug)
                if not directory.exists():
                    errors.append(f"{directory}: missing demo topic {slug!r} data directory")
                else:
                    directories.append(directory)
                    if not _pairs(directory):
                        errors.append(f"{directory}: no NPZ+JSON raw-data pairs")
    else:
        directories = sorted(path for path in root.glob("chapter_*") if path.is_dir())
        extras_root = root / "extras"
        if extras_root.is_dir():
            directories.extend(sorted(path for path in extras_root.iterdir() if path.is_dir()))
        demo_root = root / "demo"
        if demo_root.is_dir():
            directories.extend(sorted(path for path in demo_root.iterdir() if path.is_dir()))
        if not directories:
            errors.append(f"{root}: no chapter, extras, or demo data directories found")

    for directory in directories:
        if directory.exists():
            errors.extend(_validate_directory(directory))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    total_pairs = sum(len(_pairs(directory)) for directory in directories if directory.exists())
    print(f"Validated {total_pairs} raw-data export pair(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
